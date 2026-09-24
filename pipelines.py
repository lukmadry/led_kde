import numpy as np
from scipy import stats

import datetime
import argparse

# only necessary for logging
import os

# other parts of the project
import distributions
from tools import (sample_via_importance_resampling, support_width,
                   StrangeAlphaException, print_statistics, match_x_to_grid_and_val_1d, exact_gaussian_1d,
                   compute_gaussian_kernel_2d, match_x_to_grid_and_val_2d)
from kde import (compute_kde, kde_sampled_pdf_func, sample_led_kde, compute_kde_scott,
                 compute_kde_scott_2d, sample_led_kde_direct_raw, _alpha_func_direct_1d, _alpha_func_fast_approx_1d,
                 _alpha_func_direct_2d)

descriptive_functions = {'mean': np.mean, 'sd': np.std, 'median': np.median,
                         'max___density': np.max, 'max': np.max, 'support_width': support_width}

# kernel is always gaussian, maybe this can be changed, I'd like to avoid feature creep
def run_recurrent_led_with_data_forgetting_1d(n_trials, initial_observations, grid, dx, grid_size,
                                              k_scale, t_scale, n_samples, descriptive_metrics, kernel, radius,
                                              alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq, fast):
	"""

	"""
	all_pdfs = []
	statistics = []
	additional_info = {}

	obs = initial_observations
	k_sampler = stats.norm(loc=0, scale=np.sqrt(k_scale)).rvs
	t_sampler = stats.norm(loc=0, scale=np.sqrt(t_scale)).rvs

	if radius is None:
		kernel_arr = np.arange(-grid_size * dx, grid_size * dx + 0.5 * dx, step=dx)
	else:
		kernel_arr = np.arange(-radius, radius + 0.5*dx, step=dx)
	kernel_arr = exact_gaussian_1d(kernel_arr, var=k_scale)
	alpha_func_obj = _alpha_func_direct_1d if not fast else _alpha_func_fast_approx_1d
	t0 = datetime.datetime.now()
	for i in range(n_trials):
		try:
			led_samples = sample_led_kde_direct_raw(
				n_samples=n_samples, observations=obs, grid=grid, t_scale=t_scale,
				t_sampler=t_sampler, k_sampler=k_sampler, k_kernel_arr=kernel_arr,
				alpha_log_eps=alpha_log_eps, alpha_denom_eps=alpha_denom_eps, dx=dx,
				alpha_func_obj=alpha_func_obj, matcher_fn=match_x_to_grid_and_val_1d
			)
		except StrangeAlphaException as err:
			print(f"Iteration {i}")
			if len(additional_info) == 0:
				additional_info = {"strange_alpha": err.alpha_value}
			if i > 100:
				print("we will proceed with saving the results anyway")
			break

		local_led_pdf = compute_kde_scott(x=grid, observations=led_samples)
		all_pdfs.append(local_led_pdf)

		stat = print_statistics(local_led_pdf, led_samples, descriptive_metrics, descriptive_functions, i,
		                        verbosity_freq)
		statistics.append(stat)
		obs = led_samples
	t1 = datetime.datetime.now() # my IDE complains about duplication but idk how to not be repetitive here
	additional_info.update({"time_in_seconds": (t1 - t0).total_seconds()})
	if return_final_samples:
		return (led_samples, np.asarray(all_pdfs, dtype=float), statistics, additional_info)
	return np.asarray(all_pdfs, dtype=float), statistics, additional_info


def run_recurrent_led_with_data_forgetting_2d(n_trials, initial_observations, grid, dx, grid_size,
                                              k_scale, t_scale, n_samples, descriptive_metrics, kernel, radius,
                                              alpha_log_eps, alpha_denom_eps, return_final_samples, verbosity_freq, fast):
	"""

	"""
	all_pdfs = []
	statistics = []
	additional_info = {}

	obs = initial_observations
	k_sampler = stats.multivariate_normal(mean=[0, 0], cov=np.eye(2) * k_scale).rvs
	t_sampler = stats.multivariate_normal(mean=[0, 0], cov=np.eye(2) * t_scale).rvs

	kernel_arr = compute_gaussian_kernel_2d(grid, radius, k_scale)
	t0 = datetime.datetime.now()
	for i in range(n_trials):
		try:
			led_samples = sample_led_kde_direct_raw(
				n_samples=n_samples, observations=obs, grid=grid, t_scale=t_scale,
				t_sampler=t_sampler, k_sampler=k_sampler, k_kernel_arr=kernel_arr,
				alpha_log_eps=alpha_log_eps, alpha_denom_eps=alpha_denom_eps, dx=dx,
				alpha_func_obj=_alpha_func_direct_2d, matcher_fn=match_x_to_grid_and_val_2d
			)
		except StrangeAlphaException as err:
			print(f"Iteration {i}")
			if len(additional_info) == 0:
				additional_info = {"strange_alpha": err.alpha_value}
			if i > 100:
				print("we will proceed with saving the results anyway")
			break

		local_led_pdf = compute_kde_scott_2d(x=grid, observations=led_samples)
		all_pdfs.append(local_led_pdf)

		stat = print_statistics(local_led_pdf, led_samples, descriptive_metrics, descriptive_functions, i,
		                        verbosity_freq)
		statistics.append(stat)
		obs = led_samples
	t1 = datetime.datetime.now() # my IDE complains that it's duplicated but I'm not sure how to not make it so
	additional_info.update({"time_in_seconds": (t1 - t0).total_seconds()})
	if return_final_samples:
		return (led_samples, np.asarray(all_pdfs, dtype=float), statistics, additional_info)
	return np.asarray(all_pdfs, dtype=float), statistics, additional_info


# this also to be reworked in a similar way
def run_recurrent_led_with_data_retaining(n_trials, initial_observations, grid, t_scale=0.02,
                                          kde_scale=0.04, n_candidates=1000, n_samples=1000, scale_is_variance=False,
                                          kernel='gaussian',
                                          descriptive_metrics=['mean', 'sd', 'support_width', 'max___density'],
                                          comparative_metrics=['kl_d'],
                                          verbose=True, verbosity_freq=10, return_final_samples=False,
                                          alpha_filter_denom=True,
                                          sampling_method='standard'):
	all_pdfs = []
	statistics = []
	additional_info = None

	all_obs = initial_observations
	if scale_is_variance:
		t_scale = np.sqrt(t_scale)
		kde_scale = np.sqrt(kde_scale)

	t0 = datetime.datetime.now()
	for i in range(n_trials):
		led_samples = sample_led_kde(
			n_samples=n_samples,
			observations=all_obs,
			grid=grid,
			t_scale=t_scale,
			k_scale=kde_scale,
			verbose=verbose,
			alpha_filter_denom=alpha_filter_denom,
			kernel=kernel,
			sampling_method=sampling_method,
			debug=False,
		)

		all_obs = np.concatenate([all_obs, led_samples])
		local_led_pdf = compute_kde_scott(x=grid, observations=led_samples)
		all_pdfs.append(local_led_pdf)
		stat = print_statistics(local_led_pdf, led_samples, descriptive_metrics, descriptive_functions, i,
		                        verbosity_freq)
		statistics.append(stat)

	t1 = datetime.datetime.now()
	if additional_info is None:
		additional_info = {'time_in_seconds': (t1 - t0).total_seconds()}
	else:
		additional_info.update({"time_in_seconds": (t1 - t0).total_seconds()})

	return (led_samples, np.asarray(all_pdfs, dtype=float), statistics, additional_info)



def run_recurrent_kde_with_forgetting(n_trials, starting_dist, grid,
	kde_scale=0.04, n_candidates=1000, n_samples=1000, scale_is_variance=False,
	descriptive_metrics=['mean', 'sd', 'max', 'max___density'], comparative_metrics=['kl_d'], verbosity_freq=10, return_final_samples=False):
	"""

	n_trials : int
	starting_dist : a function that computes density value
	grid : a grid of values on which we will evaluate densities

	"""

	current_dist = starting_dist

	all_dists = []
	statistics = []

	q_dist = stats.norm(loc=0, scale=5)

	def custom_proposal_pdf(x):
		return q_dist.pdf(x)

	def custom_proposal_sampler(n):
		return q_dist.rvs(size=n)

	for i in range(n_trials):
		# first sample through IS
		# then use this IS to compute new distribution with KDE
		# TODO to rethink the order, the first one is kind of redundant, I think ?

		samples_from_current_pdf, ess = sample_via_importance_resampling(
			target_pdf=current_dist,
			proposal_pdf=custom_proposal_pdf,
			proposal_sampler=custom_proposal_sampler,
			n_candidates=n_candidates, n_samples=n_samples)

		kde_from_samples = compute_kde(x=grid, observations=samples_from_current_pdf,
		                               norm_scale=kde_scale)  # produces values of kde on grid, inferred from samples_from_current_pdf

		all_dists.append(kde_from_samples)

		current_dist = kde_sampled_pdf_func(grid=grid, kde=kde_from_samples) # a function that will be put in the sampler in the next round
		stat = print_statistics(kde_from_samples, samples_from_current_pdf, descriptive_metrics, descriptive_functions, i,
		                        verbosity_freq)
		statistics.append(stat)

	if return_final_samples:
		return (samples_from_current_pdf, np.asarray(all_dists, dtype=float), statistics)
	return np.asarray(all_dists, dtype=float), statistics
