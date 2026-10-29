import time

import numpy as np
from scipy import stats

import datetime
from functools import partial

# other parts of the project
# this is too big, you should try and split it or rearrange it somehow
# maybe you can move some "guardian" functions to a separate module
from tools import (sample_via_importance_resampling, support_width, support_width_axis, descriptive_fns_sentinel,
                   StrangeAlphaException, match_x_to_grid_with_cubic_1d, match_x_to_grid_and_val_1d,
                   compute_gaussian_kernel_2d, match_x_to_grid_and_val_2d, compute_gaussian_kernel_1d,
                   compute_l2_wasserstein_1d, compute_kl_divergence, wasserstein_from_pdfs_1d, NegativeProbabilityError,
                   estimate_bounding_radius_2d, comparative_fns_sentinel, sliced_wasserstein_2d)
from kde import (compute_kde, kde_sampled_pdf_func, compute_kde_scott,
                 compute_kde_scott_2d, sample_led_kde_direct_raw, _alpha_func_direct_1d, _alpha_func_fast_approx_1d,
                 _alpha_func_direct_2d, sample_led_kde_direct_raw_fast_is, _alpha_func_direct_1d_numba,
                 compute_kde_gaussian_1d_numba, compute_kde_gaussian_2d_numba, kde_sampled_pdf_func_2d,
                 _alpha_func_direct_2d_numba, compute_kde_binned_fft)
from logops import print_statistics_for_forget, print_statistics_for_retain, build_progress_bar

descriptive_functions = {'mean': np.mean, 'sd': np.std, 'median': np.median,
                         'max___density': np.max, 'max': np.max, 'support_width': support_width,
                         'mean_ax0': lambda obs: np.mean(obs[:, 0]), 'mean_ax1': lambda obs: np.mean(obs[:, 1]),
                         'sd_ax0': lambda obs: np.std(obs[:, 0]), 'sd_ax1': lambda obs: np.std(obs[:, 1]),
                         'median_ax0': lambda obs: np.median(obs[:, 0]), 'median_ax1': lambda obs: np.median(obs[:, 0]),
                         'support_width_ax0': partial(support_width_axis, axis=0), 'support_width_ax1': partial(support_width_axis, axis=1),
                         'radius_2d': estimate_bounding_radius_2d}

comparative_functions = {'kl_divergence': compute_kl_divergence, 'sliced_wasserstein_2d': sliced_wasserstein_2d,
                         'l2_wasserstein': compute_l2_wasserstein_1d, 'l2_wasserstein___density': wasserstein_from_pdfs_1d}

def update_forgetting(old_obs, new_obs):
	return new_obs

def update_retaining(old_obs, new_obs):
	return np.concatenate([old_obs, new_obs], axis=0)

def run_recurrent_led_1d(n_trials, initial_observations, grid, dx, grid_size,
                          k_scale, t_scale, n_samples, descriptive_metrics, comparative_metrics, radius,
                          alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq, cands_multiplier,
                          fast, led_is, update_fn, print_fn, verbose, interpolation, **kwargs):
	"""

	internal function for pipelines.

	See run_recurrent_led_with_data_forgetting_1d and run_recurrent_led_with_data_retaining_1d which are actual interfaces
	for this function for the docstring.

	"""
	all_pdfs = []
	statistics = []
	additional_info = {}

	obs = initial_observations
	k_sampler = stats.norm(loc=0, scale=np.sqrt(k_scale)).rvs
	t_sampler = stats.norm(loc=0, scale=np.sqrt(t_scale)).rvs

	_radius = grid_size * dx if radius is None else radius
	kernel_arr = compute_gaussian_kernel_1d(_radius, dx, k_scale)
	alpha_func_obj = _alpha_func_direct_1d if not fast else _alpha_func_direct_1d_numba
	sampler_fn = sample_led_kde_direct_raw_fast_is if led_is else sample_led_kde_direct_raw # TODO think about abandoning this and always choosing the 1st one
	matcher_fn = match_x_to_grid_with_cubic_1d if interpolation == 'cubic' else match_x_to_grid_and_val_1d
	pbar_length = max(50, int(n_trials / verbosity_freq))
	t0 = datetime.datetime.now()

	comparative_functions['l2_wasserstein___density'] = partial(wasserstein_from_pdfs_1d, grid=grid)
	starting_density = compute_kde_scott(grid, initial_observations)

	descriptive_metrics = descriptive_fns_sentinel(descriptive_metrics, 1)
	comparative_metrics = comparative_fns_sentinel(comparative_metrics, 1)
	print("Metrics: ", descriptive_metrics + comparative_metrics)
	for i in range(n_trials):
		try:
			led_samples = sampler_fn(
				n_samples=n_samples, observations=obs, grid=grid, t_scale=t_scale,
				t_sampler=t_sampler, k_sampler=k_sampler, k_kernel_arr=kernel_arr,
				alpha_log_eps=alpha_log_eps, alpha_denom_eps=alpha_denom_eps, dx=dx, cands_multiplier=cands_multiplier,
				alpha_func_obj=alpha_func_obj, matcher_fn=matcher_fn, ndim=1
			)
		except (NegativeProbabilityError, StrangeAlphaException) as err:
			print(f"Iteration {i}")
			if hasattr(err, 'alpha_value'):
				additional_info.update({"strange_alpha": err.alpha_value})
			if hasattr(err, 'min_prob_value'):
				additional_info.update({"negative_proba": err.min_prob_value})
			if i > 100:
				print("we will proceed with saving the results anyway")
			else:
				exit()

		obs = update_fn(obs, led_samples)
		local_led_pdf = compute_kde_scott(x=grid, observations=obs)
		all_pdfs.append(local_led_pdf)

		stat = print_fn(local_pdf=local_led_pdf, local_samples=obs, metrics=descriptive_metrics + comparative_metrics,
		                i=i, desc_func_dict=descriptive_functions,
		                verbosity_freq=verbosity_freq, verbose=verbose, n_samples=n_samples, grid=grid,
		                comp_func_dict=comparative_functions, orig_samples=initial_observations,
		                orig_density=starting_density)
		statistics.append(stat)

		if not verbose:
			print(build_progress_bar(i, n_trials, pbar_length), end="\r")

	t1 = datetime.datetime.now() # my IDE complains about duplication but idk how to not be repetitive here
	additional_info.update({"time_in_seconds": (t1 - t0).total_seconds()})
	if return_final_samples:
		return (led_samples, np.asarray(all_pdfs, dtype=float), statistics, additional_info)
	print("\n\n")
	return np.asarray(all_pdfs, dtype=float), statistics, additional_info

def run_recurrent_led_with_data_forgetting_1d(n_trials, initial_observations, grid, dx, grid_size,
                          k_scale, t_scale, n_samples, descriptive_metrics, comparative_metrics, radius,
                          alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq, cands_multiplier,
                          fast, led_is, verbose, interpolation, **kwargs):
	"""
	runs n_trials of data forgetting scheme with LED-KDE.

	some of the arguments are redundant, but this is because this is not supposed to be front-facing function, it's
	called in run.py where we compute all of them

	n_trials : int
	initial_observations : np.ndarray 1-dimensional
	grid : np.ndarray 1-dimensional, evenly spaced grid (this is not checked !)
	dx : float, distance between grid points
	grid_size : size of the grid
	k_scale : float, variance of the external gaussian K
	t_scale : float, variance of the internal gaussian L
	n_samples : int, number of samples resampled at every step
	descriptive_metrics : list of strings, keys of dictionary descriptive_functions. Can be :
	  mean, sd, support_width, max, max___density
    kernel : str, irrelevant for now (29/09)
    radius : while creating kernel for convolution, its size will be capped at this value. If None then we will apply
      kernel of the same size as grid
    alpha_log_eps : float, epsilon to put in log in the computation of the \alpha value
    alpha_denom_eps : float, epsilon to put in the denominator in computation of the \alpha value
    return_final_samples : bool, if True returns samples of the final round.
    verbosity_freq : int, verbosity frequency
    fast : bool, if True will apply fast KDE (approximation by binning)
    led_is : bool, if True importance sampling adapted ot LED
    verbose : bool, if True will print values of the statistics during sampling. If False only a progress bar will be displayed
	"""
	return run_recurrent_led_1d(n_trials, initial_observations, grid, dx, grid_size,
	                            k_scale, t_scale, n_samples, descriptive_metrics, comparative_metrics, radius,
	                            alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq, cands_multiplier,
	                            fast, led_is, verbose=verbose, update_fn=update_forgetting, print_fn=print_statistics_for_forget,
	                            interpolation=interpolation)

def run_recurrent_led_with_data_retaining_1d(n_trials, initial_observations, grid, dx, grid_size,
                          k_scale, t_scale, n_samples, descriptive_metrics, comparative_metrics, radius,
                          alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq, cands_multiplier,
                          fast, led_is, verbose, interpolation, **kwargs):
	"""
	runs scheme with LED-KDE with data retaining.

	some of the arguments are redundant, but this is because this is not supposed to be front-facing function, it's
	called in run.py where we compute all of them

	n_trials : int
	initial_observations : np.ndarray 1-dimensional
	grid : np.ndarray 1-dimensional, evenly spaced grid (this is not checked !)
	dx : float, distance between grid points
	grid_size : size of the grid
	k_scale : float, variance of the external gaussian K
	t_scale : float, variance of the internal gaussian L
	n_samples : int, number of samples resampled at every step.
	descriptive_metrics : list of strings, keys of dictionary descriptive_functions. Can be :
	  mean, sd, suppord_width, max, max___density
    kernel : str, (irrelevant for now 29/09)
    radius : while creating kernel for convolution, its size will be capped at this value. If None then we will apply
      kernel of the same size as grid
    alpha_log_eps : float, epsilon to put in log in the computation of the \alpha value
    alpha_denom_eps : float, epsilon to put in the denominator in computation of the \alpha value
    return_final_samples : bool, if True returns samples of the final round.
    verbosity_freq : int, verbosity frequency
    fast : bool, if True will apply fast KDE (approximation by binning)
    led_is : importance sampling adapted ot LED
    verbose : bool, if True will print values of the statistics during sampling. If False only a progress bar will be displayed
	"""
	return run_recurrent_led_1d(n_trials, initial_observations, grid, dx, grid_size,
	                            k_scale, t_scale, n_samples, descriptive_metrics, comparative_metrics, radius,
	                            alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq, cands_multiplier,
	                            fast, led_is, verbose=verbose, update_fn=update_retaining, print_fn=print_statistics_for_retain,
	                            interpolation=interpolation)


def run_recurrent_led_2d(n_trials, initial_observations, grid, dx, grid_size,
                                              k_scale, t_scale, n_samples, descriptive_metrics, comparative_metrics, radius,
                                              alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq,
                                              interpolation, verbose, seed, sampler_fn, update_fn, print_fn, fast, *args, **kwargs):
	"""

	"""
	all_pdfs = []
	statistics = []
	additional_info = {}

	obs = initial_observations
	k_sampler = stats.multivariate_normal(mean=[0, 0], cov=np.eye(2) * k_scale, seed=seed).rvs
	t_sampler = stats.multivariate_normal(mean=[0, 0], cov=np.eye(2) * t_scale, seed=seed).rvs
	radius = dx * grid_size
	kernel_arr = compute_gaussian_kernel_2d(grid, radius, k_scale)
	matcher_fn = partial(match_x_to_grid_and_val_2d, method=interpolation)
	pbar_length = max(50, int(n_trials / verbosity_freq))
	descriptive_metrics = descriptive_fns_sentinel(descriptive_metrics, 2)
	comparative_metrics = comparative_fns_sentinel(comparative_metrics, 2)
	starting_density = compute_kde_scott_2d(grid, initial_observations)
	debug = 'debug' in kwargs and kwargs['debug']
	if debug:
		print(k_sampler(5))
		print("\n")
		print(t_sampler(5))
	t0 = datetime.datetime.now()
	for i in range(n_trials):
		try:
			led_samples = sampler_fn(
				n_samples=n_samples, observations=obs, grid=grid, t_scale=t_scale,
				t_sampler=t_sampler, k_sampler=k_sampler, k_kernel_arr=kernel_arr,
				alpha_log_eps=alpha_log_eps, alpha_denom_eps=alpha_denom_eps, dx=dx,
				alpha_func_obj=_alpha_func_direct_2d_numba, matcher_fn=matcher_fn, ndim=2, seed=seed, debug=debug
			)
		except StrangeAlphaException as err:
			print(f"Iteration {i}")
			if len(additional_info) == 0:
				additional_info = {"strange_alpha": err.alpha_value}
			if i > 100:
				print("we will proceed with saving the results anyway")
			break
		if 'debug' in kwargs and kwargs['debug']:
			print('statistics of sampled observations: ',
			      led_samples[:, 0].mean(), led_samples[:, 1].mean(), led_samples[:, 0].std(), led_samples[:, 1].std())
		obs = update_fn(obs, led_samples)
		local_led_pdf = compute_kde_scott_2d(x=grid, observations=obs)
		all_pdfs.append(local_led_pdf)

		stat = print_fn(local_led_pdf, obs, descriptive_metrics + comparative_metrics,
		                                   desc_func_dict=descriptive_functions, i=i,
		                                   verbosity_freq=verbosity_freq, orig_samples=initial_observations,
		                                   comp_func_dict=comparative_functions, verbose=verbose,
		                                    orig_density=starting_density, n_samples=n_samples, grid=grid)
		statistics.append(stat)
		if not verbose:
			print(build_progress_bar(i, n_trials, pbar_length), end="\r")

	t1 = datetime.datetime.now() # my IDE complains that it's duplicated but I'm not sure how to not make it so
	additional_info.update({"time_in_seconds": (t1 - t0).total_seconds()})
	if return_final_samples:
		return (led_samples, np.asarray(all_pdfs, dtype=float), statistics, additional_info)
	return np.asarray(all_pdfs, dtype=float), statistics, additional_info

def run_recurrent_led_with_data_forgetting_2d(n_trials, initial_observations, grid, dx, grid_size,
                                              k_scale, t_scale, n_samples, descriptive_metrics, comparative_metrics, radius,
                                              alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq,
                                              interpolation, verbose, seed, fast, *args, **kwargs):

	return run_recurrent_led_2d(n_trials, initial_observations, grid, dx, grid_size,
                              k_scale, t_scale, n_samples, descriptive_metrics, comparative_metrics, radius,
                              alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq,
                              interpolation, verbose, seed, sampler_fn=sample_led_kde_direct_raw_fast_is,
                                update_fn=update_forgetting, print_fn=print_statistics_for_forget, fast=fast, **kwargs)

def run_recurrent_led_with_data_retaining_2d(n_trials, initial_observations, grid, dx, grid_size,
                                              k_scale, t_scale, n_samples, descriptive_metrics, comparative_metrics, radius,
                                              alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq,
                                              interpolation, verbose, seed, fast, *args, **kwargs):

	return run_recurrent_led_2d(n_trials, initial_observations, grid, dx, grid_size,
	                            k_scale, t_scale, n_samples, descriptive_metrics, comparative_metrics, radius,
	                            alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq,
	                            interpolation, verbose, sampler_fn=sample_led_kde_direct_raw_fast_is,
	                            update_fn=update_retaining, print_fn=print_statistics_for_retain, seed=seed, fast=fast)


# TODO refactor to be similar to the previous one, let the pdf thing happen inside sampler
def run_recurrent_kde(n_trials, starting_dist, grid, k_scale, n_candidates, n_samples,
                      descriptive_metrics, comparative_metrics,
                      verbosity_freq, return_final_samples, update_fn, print_fn, verbose, fast_kde, dimension=1, *args, **kwargs):


	current_obs = []

	all_dists = []
	statistics = []

	if dimension == 1:
		q_dist = stats.norm(loc=0, scale=5)
	elif dimension == 2:
		q_dist = stats.multivariate_normal(mean=[0, 0], cov=np.eye(2) * 2)
	else:
		raise ValueError("dimension \notin \{ 1, 2 \}")

	def custom_proposal_pdf(x):
		return q_dist.pdf(x)

	def custom_proposal_sampler(n):
		return q_dist.rvs(size=n)

	pbar_length = max(50, int(n_trials / verbosity_freq))
	t0 = time.perf_counter()
	if dimension == 1:
		# kde_func = compute_kde if not fast_kde else compute_kde_gaussian_1d_numba # TODO change to FFT
		kde_func = compute_kde_binned_fft if fast_kde else compute_kde_gaussian_1d_numba
		starting_kde = compute_kde_gaussian_1d_numba(x=grid, observations=starting_dist,
		                           var=k_scale)
		current_dist = kde_sampled_pdf_func(grid=grid, kde=starting_kde)  # TODO review
	elif dimension == 2:
		kde_func = compute_kde_gaussian_2d_numba # TODO change to FFT if fast_kde
		starting_kde = compute_kde_gaussian_2d_numba(x=grid, observations=starting_dist,
		                           norm_scale=k_scale)
		current_dist = kde_sampled_pdf_func_2d(grid=grid, kde=starting_kde)
	else:
		raise ValueError("dimension \notin \{ 1, 2 \}")
	for i in range(n_trials):
		# first sample through IS
		# then use this IS to compute new distribution with KDE

		samples_from_current_pdf, ess = sample_via_importance_resampling(
			target_pdf=current_dist,
			proposal_pdf=custom_proposal_pdf,
			proposal_sampler=custom_proposal_sampler,
			n_candidates=n_candidates, n_samples=n_samples)


		current_obs = update_fn(current_obs, samples_from_current_pdf)
		kde_from_samples = kde_func(x=grid, observations=current_obs,
		                               norm_scale=k_scale)  # produces values of kde on grid, inferred from current_obs

		all_dists.append(kde_from_samples)
		current_dist = kde_sampled_pdf_func(grid=grid, kde=kde_from_samples) # a function that will be put in the sampler in the next round
		stat = print_fn(local_pdf=kde_from_samples, local_samples=current_obs, metrics=descriptive_metrics + comparative_metrics,
		                i=i, desc_func_dict=descriptive_functions,
		                verbosity_freq=verbosity_freq, verbose=verbose, n_samples=n_samples, grid=grid,
		                comp_func_dict=comparative_functions, orig_samples=starting_dist, orig_density=starting_kde)
		statistics.append(stat)

		if not verbose:
			print(build_progress_bar(i, n_trials, pbar_length), end="\r")

	t1 = time.perf_counter()
	if return_final_samples:
		return (samples_from_current_pdf, np.asarray(all_dists, dtype=float), statistics, {"time_in_seconds": t1-t0}) # the last one to match the LED

	return np.asarray(all_dists, dtype=float), statistics, {"time_in_seconds": t1-t0}

def run_recurrent_kde_with_forgetting(n_trials, starting_dist, grid,
	k_scale=0.04, n_candidates=1000, n_samples=1000, scale_is_variance=False,
	descriptive_metrics=['mean', 'sd', 'max', 'max___density'], comparative_metrics=['kl_d'],
	                                  verbosity_freq=10, return_final_samples=False, verbose=False,
	                                  fast_kde=True, **kwargs):
	"""

	n_trials : int
	starting_dist : a function that computes density value
	grid : a grid of values on which we will evaluate densities

	"""
	return run_recurrent_kde(n_trials=n_trials, starting_dist=starting_dist, grid=grid,
	                         k_scale=k_scale, n_candidates=n_candidates, n_samples=n_samples, comparative_metrics=comparative_metrics,
	                         descriptive_metrics=descriptive_metrics, print_fn=print_statistics_for_forget,
	                         update_fn=update_forgetting, return_final_samples=return_final_samples, verbosity_freq=verbosity_freq,
	                         verbose=verbose, fast_kde=fast_kde, **kwargs)


def run_recurrent_kde_with_retaining(n_trials, starting_dist, grid,
				k_scale=0.04, n_candidates=1000, n_samples=1000, scale_is_variance=False,
				descriptive_metrics=['mean', 'sd', 'max', 'max___density'], comparative_metrics=['kl_d'],
	                                 verbosity_freq=10, return_final_samples=False, verbose=False, fast_kde=True, **kwargs):
	"""

	n_trials : int
	starting_dist : a function that computes density value
	grid : a grid of values on which we will evaluate densities

	"""
	return run_recurrent_kde(n_trials=n_trials, starting_dist=starting_dist, grid=grid,
	                         k_scale=k_scale, n_candidates=n_candidates, n_samples=n_samples, comparative_metrics=comparative_metrics,
	                         descriptive_metrics=descriptive_metrics, print_fn=print_statistics_for_retain,
	                         update_fn=update_retaining, verbose=verbose, return_final_samples=return_final_samples,
	                         verbosity_freq=verbosity_freq, fast_kde=fast_kde, dimension=1, **kwargs)


def run_recurrent_kde_with_data_forgetting_2d(n_trials, starting_dist, grid,
									k_scale=0.04, n_candidates=1000, n_samples=1000, scale_is_variance=False,
									descriptive_metrics=['mean', 'sd', 'max', 'max___density'],
					                comparative_metrics=['sliced_wassersteind_2d', 'kl_d'], fast_kde=True,
					                verbosity_freq=10, return_final_samples=False, verbose=False, *args, **kwargs):

	return run_recurrent_kde(n_trials=n_trials, starting_dist=starting_dist, grid=grid,
	                         k_scale=k_scale, n_candidates=n_candidates, n_samples=n_samples, comparative_metrics=comparative_metrics,
	                         descriptive_metrics=descriptive_metrics, print_fn=print_statistics_for_retain,
	                         update_fn=update_retaining, verbose=verbose, return_final_samples=return_final_samples,
	                         verbosity_freq=verbosity_freq, fast_kde=fast_kde, dimension=2)

def run_recurrent_kde_with_data_retaining_2d(n_trials, starting_dist, grid,
				k_scale=0.04, n_candidates=1000, n_samples=1000, scale_is_variance=False,
				descriptive_metrics=['mean', 'sd', 'max', 'max___density'], comparative_metrics=['kl_d'],
                 verbosity_freq=10, return_final_samples=False, verbose=False, fast_kde=True, **kwargs):
	"""

	n_trials : int
	starting_dist : a function that computes density value
	grid : a grid of values on which we will evaluate densities

	"""
	return run_recurrent_kde(n_trials=n_trials, starting_dist=starting_dist, grid=grid,
	                         k_scale=k_scale, n_candidates=n_candidates, n_samples=n_samples, comparative_metrics=comparative_metrics,
	                         descriptive_metrics=descriptive_metrics, print_fn=print_statistics_for_retain,
	                         update_fn=update_retaining, verbose=verbose, return_final_samples=return_final_samples,
	                         verbosity_freq=verbosity_freq, fast_kde=fast_kde, dimension=2, **kwargs)