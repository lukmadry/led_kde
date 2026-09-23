import numpy as np
from scipy import stats

import datetime
import argparse

# only necessary for logging
import os

# other parts of the project
import distributions
from tools import (sample_via_importance_resampling, kl_divergence_observations, support_width,
                   StrangeAlphaException, print_statistics, run_dimension_consistency_check)
from kde import compute_kde, kde_sampled_pdf_func, sample_led_kde, compute_kde_scott, compute_kde_2d
from logops import log_results, find_datetime_with_this_params, PARAMETER_KEYS

q_dist = stats.norm(loc=0, scale=5)

# TODO: separate 1d and 2d

# they are not very well adapted to ndim > 1, to be changed
descriptive_functions = {'mean': np.mean, 'sd': np.std, 'median': np.median,
                         'max___density': np.max, 'max': np.max, 'support_width': support_width}
comparative_functions = {'kl_d': kl_divergence_observations}
# I'm not really using comparative metrics given that it appears to change shape a lot for anything apart from a simple Gaussian

def custom_proposal_pdf(x):
	return q_dist.pdf(x)

def custom_proposal_sampler(n):
	return q_dist.rvs(size=n)

# these two can be put under some common function or a class where only the core is going to be changed
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

def run_recurrent_kde_with_retaining(**kwargs):
	raise NotImplementedError("KDE with retaining not implemented yet")

# TODO instead of doing so many checks for dimensions inside you should rather write two different variants, one for 1d and one for 2d
def run_recurrent_led_kde_with_forgetting(n_trials, initial_observations, grid, t_scale=0.02,
	kde_scale=0.04, n_candidates=1000, n_samples=1000, scale_is_variance=False, kernel='gaussian',
	descriptive_metrics=['mean', 'sd', 'support_width', 'max___density'], comparative_metrics=['kl_d'],
	verbose=True, verbosity_freq=10, return_final_samples=False, alpha_filter_denom=False, ndim=1,
    sampling_method='standard', interpolation=True, alpha_log_eps=1e-10, alpha_denom_eps=1e-3,
                                          debug=False):
	"""

	grid:
	"""
	all_pdfs = []
	statistics = []
	additional_info = None

	obs = initial_observations
	if scale_is_variance:
		t_scale = np.sqrt(t_scale)
		kde_scale = np.sqrt(kde_scale)

	t0 = datetime.datetime.now()
	# TODO move precomputing kernel for \alpha here
	print("about to starting sampling")
	for i in range(n_trials):

		try:
			led_samples = sample_led_kde(
							n_samples=n_samples,
							observations=obs,
							grid=grid,
							t_scale=t_scale,
							k_scale=kde_scale,
							verbose=verbose,
							alpha_filter_denom=alpha_filter_denom,
							kernel=kernel,
							sampling_method=sampling_method,
							debug=debug,
							interpolation=interpolation,
							alpha_denom_eps=alpha_denom_eps,
							alpha_log_eps=alpha_log_eps,
							)
		except StrangeAlphaException as err:
			print(f"Iteration {i}")
			if additional_info is None:
				additional_info = {"strange_alpha": err.alpha_value}
			if i > 100:
				print("we will proceed with saving the results anyway")
			break
		if ndim == 1:
			local_led_pdf = compute_kde_scott(x=grid, observations=led_samples)
		elif ndim == 2:
			bandwidth = 1/len(led_samples)**(1/6)
			local_led_pdf = compute_kde_2d(x=grid, observations=led_samples, variance=bandwidth, keep_shape=True)
		all_pdfs.append(local_led_pdf)

		stat = print_statistics(local_led_pdf, led_samples, descriptive_metrics, descriptive_functions, i, verbosity_freq)
		statistics.append(stat)
		obs = led_samples

	t1 = datetime.datetime.now()
	if additional_info is None:
		additional_info = {'time_in_seconds' : (t1 - t0).total_seconds()}
	else:
		additional_info.update({"time_in_seconds": (t1-t0).total_seconds()})
	if return_final_samples:
		return (led_samples, np.asarray(all_pdfs, dtype=float), statistics, additional_info)
	return np.asarray(all_pdfs, dtype=float), statistics, additional_info

def run_recurrent_led_with_data_retaining(n_trials, initial_observations, grid, t_scale=0.02,
	kde_scale=0.04, n_candidates=1000, n_samples=1000, scale_is_variance=False, kernel='gaussian',
	descriptive_metrics=['mean', 'sd', 'support_width', 'max___density'], comparative_metrics=['kl_d'],
	verbose=True, verbosity_freq=10, return_final_samples=False, alpha_filter_denom=True,
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
		additional_info = {'time_in_seconds' : (t1 - t0).total_seconds()}
	else:
		additional_info.update({"time_in_seconds": (t1-t0).total_seconds()})

	return (led_samples, np.asarray(all_pdfs, dtype=float), statistics, additional_info)

if __name__ == '__main__':
	# TODO add help arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--n", required=True, type=int, help="number of iterations")
	parser.add_argument("--lower_bd", default=-5, type=float, help="lower bound on the grid") # possibly multiple values if dim > 1, but to be changed later on
	parser.add_argument("--upper_bd", default=5, type=float, help="upper bound on the grid")
	parser.add_argument("--method",
						choices=['led', 'kde', 'led_retain'], type=str, default='led',
						help=("method of sampling. Choose led for LED-KDE, kde for KDE and led_retain with LED-KDE "
							  "in a data retaining scheme"))

	# some technical arguments
	parser.add_argument("--grid_size", default=500, type=int, help="size of a grid on which we will sample distribution")
	parser.add_argument("--alpha_filter_denom",
						action=argparse.BooleanOptionalAction, default=False,
						help="if true alpha we will not add small eps to the denominator in alpha, but rather choose"
							 "values where denominator is not zero. Don't touch if you don't know the code very well")
	parser.add_argument("--alpha_log_eps", type=float, default=1e-10)
	parser.add_argument("--alpha_denom_eps", type=float, default=1e-3)
	parser.add_argument("--with_interpolation", action=argparse.BooleanOptionalAction, default=True,
	                    help="don't touch, it seems to be important to keep it at True, otherwise there is a bias towards negative values")
	parser.add_argument("--with_fft_cutoff",
						action=argparse.BooleanOptionalAction, default=False, help="experimental, don't touch")
	parser.add_argument("--seed", type=int)

	# only relevant for retrieving past experiments
	parser.add_argument("--search", action=argparse.BooleanOptionalAction, default=False,
						help="add this parameter if you want to check if similar experiments have already been done. "
							 "Displays a list of folders in results where experiments with matching keywords have been found."
							 "Displays a folder if at least half of a keywords match + the ones in full_agreement_keys. See --keys and --full_agreement_keys")
	parser.add_argument("--keys", nargs='+', help="give a list of keys that will be taken into account "
	                                              "when performing a search. By default PARAMETER_KEYS in logops.py")
	parser.add_argument("--full_agreement_keys", nargs='+', help="give a list of keys that have to match perfectly"
	                                                             "for the experiment to be considered a match. "
																 "By default only method is considered")

	# distribution arguments
	parser.add_argument("--dist_type", default='gaussian',
						choices=['gaussian', 'uniform'], help="basic building distribution of an initial mixture")

	# one dimensional arguments
	parser.add_argument("--centers", nargs='+',
						type=float, help="centers of the initial mixture distribution")
	parser.add_argument("--scales", nargs='+',
						type=float, help="if dist_type is gaussian, then scales (sddevs) of gaussians. "
										 "Irrelevant if dist_type is uniform. "
										 "Can be a float or a list of floats the same legth as centers")
	parser.add_argument("--widths", nargs='+', type=float,
						help="if dist_type is uniform, then widths of uniforms. Irrelevant if dist_type is gaussian. "
							 "Can be a float or a list of floats the same legth as centers.")

	# two-dimensional arguments
	parser.add_argument("--centers_x", nargs='+', type=float)
	parser.add_argument("--centers_y", nargs='+', type=float)
	parser.add_argument("--scales_x", nargs='+', type=float)
	parser.add_argument("--scales_y", nargs='+', type=float)
	parser.add_argument("--covariances", nargs='+', type=float)
	parser.add_argument("--widths_x", nargs='+', type=float)
	parser.add_argument("--widths_y", nargs='+', type=float)

	parser.add_argument("--weights", nargs='+',
						type=float, help="weights of mixtures. Has to be the same length as centers")


	# kde or led-kde arguments
	parser.add_argument("--kernel", default='gaussian',
						choices=['gaussian', 'uniform'], help="kernel for LED-KDE. ")
	parser.add_argument("--sampling_method", default='standard',
						choices=['standard', 'is'], help="sampling method for LED-KDE. Stick to standard, "
														 "IS was for debugging")
	parser.add_argument("--t_scale", default=0.02, type=float)
	parser.add_argument("--k_scale", default=0.04, type=float)
	parser.add_argument("--scale_is_variance",
						action=argparse.BooleanOptionalAction, default=False) # probably deprecated
	parser.add_argument("--n_candidates", default=10000, type=int,
						help="useful for KDE, ignored for everything else. Number of candidates in the standard IS scheme")
	parser.add_argument("--n_samples", default=10000, type=int,
						help="Number of samples chosen each round. Choose a small value if method = led_retains")
	parser.add_argument("--n_init_samples", default=1000, type=int,
						help="number of initial samples, only relevant for data retaining scheme")

	# arguments related to logging
	parser.add_argument("--verbosity_freq", default=25, type=int)
	parser.add_argument("--dump_intermediary_dist", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--track_git",
						action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--save_gif",
						action=argparse.BooleanOptionalAction, default=True, help="will save GIF of evolution if true")
	parser.add_argument("--save_distributions", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--gif_simulation_step", default=20,
						type=int, help="if equal to k, will save to gif every k steps. For smooth but costly gif choose 1")

	args = parser.parse_args()
	args_dict = vars(args)
	res_filepath = os.path.join("results", "record.jsonl")
	if args.search:
		smaller_dict = {key: value for key, value in args_dict.items() if key in PARAMETER_KEYS}
		kwargs_param_search = dict(desired_values=smaller_dict, results_filepath=res_filepath)
		if args.keys is not None:
			kwargs_param_search['important_keys'] = args.keys
		if args.full_agreement_keys is not None:
			kwargs_param_search['full_agreement_keys'] = args.full_agreement_keys
		print("Here's the list of datetimes of matching experiments found in directory results.")
		print("Float number next to it indicates proportion of desired parameters that agree. These parameters are:")
		print(smaller_dict)
		print("Note that you can change them by passing arguments to this script, using arguments keys and full_agreement_keys.\n")
		find_datetime_with_this_params(**kwargs_param_search)
		exit()

	dim = run_dimension_consistency_check(args)
	if dim == 1:
		grid = np.linspace(args.lower_bd, args.upper_bd, num=args.grid_size)
	else:
		print("detected two dimensions. Note that this path is still experimental")
		grid_x, grid_y = np.linspace(args.lower_bd, args.upper_bd, num=args.grid_size), np.linspace(args.lower_bd, args.upper_bd, num=args.grid_size)
		grid = np.meshgrid(grid_x, grid_y)
		grid = np.dstack(grid)

	# TODO to be made more concise, a lot of repetition
	run_kwargs = dict(
		n_trials=args.n, 
		grid=grid, 
		kde_scale=args.k_scale,
		scale_is_variance=args.scale_is_variance,
		verbosity_freq=args.verbosity_freq,
		return_final_samples=True,
		n_candidates=args.n_candidates,
		n_samples=args.n_samples,
		kernel=args.kernel)

	if args.dist_type == 'gaussian' and dim == 1:
		true_mixture = distributions.GaussianMixture(
			centers=args.centers,
			scales=args.scales,
			weights=args.weights,
			scale_is_variance=args.scale_is_variance)
	elif args.dist_type == 'gaussian' and dim == 2:
		raise NotImplementedError
	elif args.dist_type == 'uniform' and dim == 1:
		true_mixture = distributions.UniformMixture(
			centers=args.centers,
			widths=args.widths,
			weights=args.weights,
		)
	elif args.dist_type == 'uniform' and dim == 2:
		centers = list(zip(args.centers_x, args.centers_y))
		widths = list(zip(args.widths_x, args.widths_y))
		true_mixture = distributions.UniformBoxMixture(
			centers=centers,
			shapes=widths,
			weights=args.weights
		)

	print(f"Testing {args.method} with t: {args.t_scale}, k: {args.k_scale}, N: {args.n_samples}")
	if args.method == 'kde':
		true_samples = true_mixture.rvs(args.n_samples)
		print(f"True distribution:\nMean: {np.mean(true_samples):.4f}. Std: {np.std(true_samples):.4f}. Max - min: {support_width(true_samples)}\n")

		run_kwargs.update(dict(starting_dist=true_mixture.pdf))
		samples_last_pdf, all_dist, stat = run_recurrent_kde_with_forgetting(**run_kwargs)
	elif args.method == 'led':
		init_obs = true_mixture.rvs(args.n_samples)
		if dim == 1:
			print(f"True distribution:\nMean: {np.mean(init_obs):.4f}. Std: {np.std(init_obs):.4f}. Max - min: {support_width(init_obs)}\n")

		run_kwargs.update(dict(
			t_scale=args.t_scale,
			initial_observations=init_obs,
			alpha_filter_denom=args.alpha_filter_denom,
			alpha_log_eps=args.alpha_log_eps,
			alpha_denom_eps=args.alpha_denom_eps,
			sampling_method=args.sampling_method,
			interpolation=args.with_interpolation,
			ndim=dim))
		final_sample, all_dist, stat, new_info = run_recurrent_led_kde_with_forgetting(**run_kwargs)
	elif args.method == 'led_retain':
		init_obs = true_mixture.rvs(args.n_init_samples)

		run_kwargs.update(dict(
			t_scale=args.t_scale,
			initial_observations=init_obs,
			alpha_filter_denom=args.alpha_filter_denom,
			sampling_method=args.sampling_method,))
		final_sample, all_dist, stat, new_info = run_recurrent_led_with_data_retaining(**run_kwargs)

	args_dict['new_info'] = new_info
	print(f"Time passed: {new_info['time_in_seconds']:.4f} seconds")
	init_dist = true_mixture.pdf(grid)
	log_results(all_distributions=all_dist,
	            statistics=stat,
	            results_dirpath="results",
	            results_filepath=res_filepath,
	            params=args_dict,
	            init_dist=init_dist,
	            simulation_step=args.gif_simulation_step,
	            grid=grid,
	            ndim=dim)
