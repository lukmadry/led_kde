import numpy as np

import datetime
import argparse

# only necessary for logging
import os

# other parts of the project
import distributions
from tools import support_width, run_dimension_consistency_check
from logops import log_results, find_datetime_with_this_params, PARAMETER_KEYS
from pipelines import (run_recurrent_led_with_data_forgetting_1d, run_recurrent_kde_with_forgetting,
                       run_recurrent_kde_with_retaining,
                       run_recurrent_led_with_data_retaining_1d, run_recurrent_led_with_data_forgetting_2d)


if __name__ == '__main__':
	# TODO add help arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--n", required=True, type=int, help="number of iterations")
	parser.add_argument("--lower_bd", default=-5, type=float, help="lower bound on the grid") # possibly multiple values if dim > 1, but to be changed later on
	parser.add_argument("--upper_bd", default=5, type=float, help="upper bound on the grid")
	parser.add_argument("--method",
						choices=['led_forget', 'kde_forget', 'led_retain', 'kde_retain'], type=str, default='led_forget',
						help=("method of sampling. Choose led for LED-KDE, kde for KDE and led_retain with LED-KDE "
							  "in a data retaining scheme"))

	# some technical arguments
	parser.add_argument("--grid_size", default=500, type=int, help="size of a grid on which we will sample distribution")
	parser.add_argument("--alpha_filter_denom",
						action=argparse.BooleanOptionalAction, default=False,
						help="if true alpha we will not add small eps to the denominator in alpha, but rather choose"
							 "values where denominator is not zero. Don't touch if you don't know the code very well")
	parser.add_argument("--alpha_log_eps", type=float, default=1e-10,
	                    help="technical parameter for numerical stability of log in the computation of alpha. Best not touch")
	parser.add_argument("--alpha_denom_eps", type=float, default=1e-3,
	                    help="technical parameter for numerical stability in denominator in the computation of alpha. Best not touch")
	parser.add_argument("--with_interpolation", action=argparse.BooleanOptionalAction, default=True,
	                    help="don't touch, it seems to be important to keep it at True, otherwise there is a bias towards negative values")
	parser.add_argument("--with_fft_cutoff",
						action=argparse.BooleanOptionalAction, default=False,
						help="allows to use smaller kernel array in Gaussian computations. "
						     "Empirically, it does not seem to influence much, except for very dense grids, "
						     "for grid_size = 1000 and n_samples = 1e4 on my machine we gain 0.5-1 second per 25 trials."
						     "The effect is negligible for grid_size = 500, or even grid_size=1000 and n_samples=1e3")
	parser.add_argument("--fast_kde", action=argparse.BooleanOptionalAction, default=False,
	                    help="use FFT-binning to compute KDE. Use with caution as it is approximation, not the exact computation."
	                         "For 25 trials, n_samples = 1e4, grid_size = 1000 it wins 4-5 seconds")
	parser.add_argument("--fast_is", action=argparse.BooleanOptionalAction, default=False,
	                    help="activates parallelized importance sampling in LED. However I'm not really an expert on importance sampling, "
	                         "so I'm not sure how to run a diagnosis to make sure that it is correct. I will try to run empirical"
	                         "tests, but I cannot do much more. This is the reason why I'm leaving this as an option.")
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
						choices=['gaussian', 'uniform'], help="kernel for LED-KDE. Not fully implemented yet")
	parser.add_argument("--sampling_method", default='standard',
						choices=['standard', 'is'], help="sampling method for LED-KDE. Stick to standard, "
														 "IS was for debugging. Deprecated")
	parser.add_argument("--t_scale", default=0.02, type=float)
	parser.add_argument("--k_scale", default=0.04, type=float)
	parser.add_argument("--scale_is_variance",
						action=argparse.BooleanOptionalAction, default=False) # probably deprecated
	parser.add_argument("--cands_multiplier", default=100, type=int)
	parser.add_argument("--n_samples", default=1000, type=int,
						help="Number of samples chosen each round. Choose a small value (~50, 100) if method = led_retains")
	parser.add_argument("--n_init_samples", default=1000, type=int,
						help="number of initial samples, only relevant for data retaining scheme")

	# arguments related to logging
	parser.add_argument("--log_results", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--verbosity_freq", default=50, type=int)
	parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
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
		dx = grid[1] - grid[0]
	else:
		grid_x, grid_y = np.linspace(args.lower_bd, args.upper_bd, num=args.grid_size), np.linspace(args.lower_bd, args.upper_bd, num=args.grid_size)
		dx = grid_x[1] - grid_x[0]
		grid = np.meshgrid(grid_x, grid_y)
		grid = np.dstack(grid)

	# TODO to be made more concise, a lot of repetition
	run_kwargs = dict(
		n_trials=args.n, 
		grid=grid, 
		k_scale=args.k_scale,
		verbose=args.verbose,
		descriptive_metrics=['mean', 'sd', 'support_width', 'max___density'],
		comparative_metrics=['l2_wasserstein', 'kl_divergence'],
		verbosity_freq=args.verbosity_freq,
		return_final_samples=True,
		n_samples=args.n_samples,
		fast=args.fast_kde)

	setup_message = (f"Dimension: {dim}.\nSpeed-ups activated:\nKDE with FFT-binning (1D only): {args.fast_kde}\n"
	                    f"FFT with cutoff in the kernel (if 2D, False here is overridden): {args.with_fft_cutoff}\n"
	                    f"Vectorised importance sampling (1D only): {args.fast_is}\n")
	print(setup_message)
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

	print(f"Testing {args.method} with t: {args.t_scale}, k: {args.k_scale}, N: {args.n_samples}, "
	      f"grid_size: {args.grid_size}, n of trials: {args.n}")
	radius = 60 * args.k_scale if args.with_fft_cutoff or dim == 2 else None
	if args.method.startswith('led'):
		run_kwargs.update(dict(
			t_scale=args.t_scale,
			dx=dx,
			grid_size=args.grid_size,
			alpha_log_eps=args.alpha_log_eps,
			alpha_denom_eps=args.alpha_denom_eps,
			cands_multiplier=args.cands_multiplier,
			led_is=args.fast_is))

	if args.method.startswith('kde'):
		true_samples = true_mixture.rvs(args.n_samples)
		print(f"True distribution:\nMean: {np.mean(true_samples):.4f}. Std: {np.std(true_samples):.4f}. Max - min: {support_width(true_samples)}\n")

		run_kwargs.update(dict(
			starting_dist=true_samples,
            n_candidates=args.n_samples*args.cands_multiplier)
		)
		if args.method == 'kde_forget':
			samples_last_pdf, all_dist, stat, new_info = run_recurrent_kde_with_forgetting(**run_kwargs)
		else:
			samples_last_pdf, all_dist, stat, new_info = run_recurrent_kde_with_retaining(**run_kwargs)
	elif args.method == 'led_forget':
		init_obs = true_mixture.rvs(args.n_samples)
		if dim == 1:
			print(f"True distribution:\nMean: {np.mean(init_obs):.4f}. Std: {np.std(init_obs):.4f}. Max - min: {support_width(init_obs)}\n")
		run_kwargs.update(dict(initial_observations=init_obs, radius=radius))
		if dim == 1:
			final_sample, all_dist, stat, new_info = run_recurrent_led_with_data_forgetting_1d(**run_kwargs)
		else:
			final_sample, all_dist, stat, new_info = run_recurrent_led_with_data_forgetting_2d(**run_kwargs)
	elif args.method == 'led_retain':
		init_obs = true_mixture.rvs(args.n_init_samples)
		run_kwargs.update(dict(initial_observations=init_obs, radius=radius))
		final_sample, all_dist, stat, new_info = run_recurrent_led_with_data_retaining_1d(**run_kwargs)

	args_dict['new_info'] = new_info
	print(f"\nTime passed from t0 to t1: {new_info['time_in_seconds']:.4f} seconds.")

	# t0 is defined as the moment after all the preparations are done. "
	# t1 is defined as the moment right after the final sampling step is done, see t0 and t1 inside run_recurrent_...\n\n")
	init_dist = true_mixture.pdf(grid)
	if args.log_results:
		log_results(all_distributions=all_dist,
		            statistics=stat,
		            results_dirpath="results",
		            results_filepath=res_filepath,
		            params=args_dict,
		            init_dist=init_dist,
		            simulation_step=args.gif_simulation_step,
		            grid=grid,
		            ndim=dim)
