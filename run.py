import numpy as np
from scipy import stats
from scipy.stats import norm

import datetime
import argparse

# only necessary for logging
import os

# other parts of the project
import distributions
from tools import sample_via_importance_resampling, kl_divergence_observations, support_width
from kde import compute_kde, kde_sampled_pdf_func, sample_led_kde, compute_kde_scott
from logops import log_results

# TODO find a better of collecting info on experiments that you have already done
q_dist = stats.norm(loc=0, scale=5)

descriptive_functions = {'mean': np.mean, 'sd': np.std, 'median': np.median, 'max': np.max, 'support_width': support_width}
comparative_functions = {'kl_d': kl_divergence_observations}

def custom_proposal_pdf(x):
	return q_dist.pdf(x)

def custom_proposal_sampler(n):
	return q_dist.rvs(size=n)

# these two can be put under some common function or a class where only the core is going to be changed

def run_recurrent_kde_with_forgetting(n_trials, starting_dist, grid, 
	kde_scale=0.04, n_candidates=1000, n_samples=1000, scale_is_variance=False,
	descriptive_metrics=['mean', 'sd', 'max'], comparative_metrics=['kl_d'], verbosity_freq=10, return_final_samples=False):
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

		stat = {}
		stat['is_ess'] = ess
		for s in descriptive_metrics:
			stat[s] = descriptive_functions[s](kde_from_samples)

		statistics.append(stat)
		if (1+i) % verbosity_freq == 0:
			s = f"Trial nr: {1+i}. Descriptive statistics: "
			for key, val in stat.items():
				s += f"{key}: {val:.3f} |"
			s += "\n"
			print(s)

	if return_final_samples:
		return (samples_from_current_pdf, np.asarray(all_dists, dtype=float), statistics)
	return np.asarray(all_dists, dtype=float), statistics

# TODO for some reason it doesn't appear to be working very well on 2D data, that is a spiral
def run_recurrent_led_kde_with_forgetting(n_trials, initial_observations, grid, t_scale=0.02,
	kde_scale=0.04, n_candidates=1000, n_samples=1000, scale_is_variance=False,
	descriptive_metrics=['mean', 'sd', 'support_width'], comparative_metrics=['kl_d'],
	verbose=True, verbosity_freq=10, return_final_samples=False):
	"""

	"""

	all_pdfs = []
	statistics = []

	obs = initial_observations
	if scale_is_variance:
		t_scale = np.sqrt(t_scale)
		kde_scale = np.sqrt(kde_scale)

	for i in range(n_trials):
		
		led_samples = sample_led_kde(
						n_samples=n_samples, 
						observations=obs,
						grid=grid,
						t_scale=t_scale,
						k_scale=kde_scale,
						verbose=verbose
						)
		local_led_pdf = compute_kde_scott(x=grid, observations=led_samples)
		all_pdfs.append(local_led_pdf)
		stat = {}
		# TODO find a way to put the following in a single function to remove duplication
		for s in descriptive_metrics:
			stat[s] = descriptive_functions[s](led_samples)

		statistics.append(stat)
		if (1+i) % verbosity_freq == 0:
			s = f"Trial nr: {1+i}.\nDescriptive statistics: "
			for key, val in stat.items():
				s += f"{key}: {val:.3f} |"
			s += "\n"
			print(s)

		obs = led_samples

	if return_final_samples:
		return (led_samples, np.asarray(all_pdfs, dtype=float), statistics)
	return np.asarray(all_pdfs, dtype=float), statistics


# TODO will have to be extended to treat other distributions than gaussian mixtures
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--n", required=True, type=int)
	parser.add_argument("--dim", default=1, type=int)
	parser.add_argument("--lower_bd", default=-5, type=float) # possibly multiple values if dim > 1, but to be changed later on
	parser.add_argument("--upper_bd", default=5, type=float)
	parser.add_argument("--grid_size", default=500, type=int)
	parser.add_argument("--method", choices=['led', 'kde'], type=str, default='kde')

	# distribution arguments
	parser.add_argument("--centers", nargs='+', default=[-1, 1], type=float)
	parser.add_argument("--scales", nargs='+', default=[0.35, 0.35], type=float)

	# kde or led-kde arguments
	parser.add_argument("--t_scale", default=0.02, type=float)
	parser.add_argument("--k_scale", default=0.04, type=float)
	parser.add_argument("--scale_is_variance", action=argparse.BooleanOptionalAction, default=False) # probably deprecated
	parser.add_argument("--n_candidates", default=10000, type=int)
	parser.add_argument("--n_samples", default=10000, type=int)

	# arguments related to logging
	parser.add_argument("--verbosity_freq", default=25, type=int)
	parser.add_argument("--dump_intermediary_dist", action=argparse.BooleanOptionalAction, default=False)
	parser.add_argument("--track_git", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--save_gif", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--save_distributions", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--gif_simulation_step", default=20, type=int)

	args = parser.parse_args()

	log_dir = os.path.join("results" , datetime.datetime.now().strftime("%A_%d_%B_%Y_%I:%M%p"))
	grid = np.linspace(args.lower_bd, args.upper_bd, num=args.grid_size)
	
	# TODO to be made more concise, a lot of repetition
	run_kwargs = dict(
		n_trials=args.n, 
		grid=grid, 
		kde_scale=args.k_scale,
		scale_is_variance=args.scale_is_variance,
		verbosity_freq=args.verbosity_freq,
		return_final_samples=True,
		n_candidates=args.n_candidates,
		n_samples=args.n_samples)

	true_mixture = distributions.GaussianMixture(
		centers=args.centers,
		scales=args.scales,
		scale_is_variance=args.scale_is_variance)

	if args.method == 'kde':
		true_samples = true_mixture.rvs(args.n_samples)
		print(f"True distribution:\nMean: {np.mean(true_samples):.4f}. Std: {np.std(true_samples):.4f}. Max - min: {support_width(true_samples)}\n")

		run_kwargs.update(dict(starting_dist=true_mixture.pdf))
		if args.return_final_samples:
			samples_last_pdf, all_dist, stat = run_recurrent_kde_with_forgetting(**run_kwargs)
		else:
			all_dist, stat = run_recurrent_kde_with_forgetting(**run_kwargs)


	elif args.method == 'led':
		init_obs = true_mixture.rvs(args.n_samples)
		print(f"True distribution:\nMean: {np.mean(init_obs):.4f}. Std: {np.std(init_obs):.4f}. Max - min: {support_width(init_obs)}\n")

		run_kwargs.update(dict(t_scale=args.t_scale, initial_observations=init_obs))
		final_sample, all_dist, stat = run_recurrent_led_kde_with_forgetting(**run_kwargs)

	if not os.path.exists(log_dir):
		os.makedirs(log_dir, exist_ok=True)

	args_dict = vars(args)
	init_dist = true_mixture.pdf(grid)
	log_results(all_distributions=all_dist,
	            statistics=stat,
	            results_dirpath="results",
	            results_filepath=os.path.join("results", "record.jsonl"),
	            params=args_dict,
	            init_dist=init_dist,
	            simulation_step=args.gif_simulation_step)

