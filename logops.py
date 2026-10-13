import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import json
import os
import datetime
import h5py

from tools import get_git_commit_gitpython
from kde import compute_kde_scott

def build_progress_bar(i, M, length=50, p_sign='@'):
	prop = int(float(i)/float(M)* length)
	s = "[" + p_sign * prop + '_' * (length - prop) + "]" + f" {100*float(i/M):.2f}%"
	return s

# we may need some 2d version of this
def print_statistics_for_1d_forget(local_pdf, local_samples, metrics, desc_func_dict,
                                   i, verbosity_freq, verbose=True, prefix_stat=None,
                                   comp_func_dict=None, orig_samples=None,
                                   orig_density=None, *args, **kwargs):
	"""
	Helper function used in the main pass to print statistics of the current distribution.

	Return a dictionary with values of statistics of pdf and samples.

	local_pdf : np.ndarray of pdf
	local_samples : np.ndarray of samples
	desc_func_dict : dictionary of format str : Callable . If string ends with ___density then
			Callable will be applied to local_pdf, if not it will be applied to local_samples
	metrics : iterable with strings of metrics names. Has to be a subset of keys in desc_func_dict | comp_func_dict
	i : index to display
	verbosity_freq : dictates verbosity, if 1+i is divisible by this value, will print descriptive statistics
	"""
	stat = {}
	comp_func_dict = {} if comp_func_dict is None else comp_func_dict
	N = max(len(local_samples), len(orig_samples))
	for s in metrics:
		if s in desc_func_dict:
			if s.endswith("___density"):
				key = s.split("___")[0] + " of density"
				val = desc_func_dict[s](local_pdf)
			else:
				key = s
				val = desc_func_dict[s](local_samples)
			if prefix_stat is not None:
				key = prefix_stat + "___" + key
			stat[key] = val
		if s in comp_func_dict:
			if s.endswith("___density"):
				key = s.split("___")[0] + " of density"
				val = comp_func_dict[s](local_pdf, orig_density)
			else:
				val = comp_func_dict[s](local_samples, orig_samples, N)
				key = s
			stat[key] = val

	if verbose and (1 + i) % verbosity_freq == 0:
		s = f"Trial nr: {1 + i}.\nDescriptive statistics: "
		for key, val in stat.items():
			s += f"{key}: {val:.3f} |"
		s += "\n"
		print(s)

	return stat

def print_statistics_for_1d_retain(local_pdf, local_samples, metrics,
                                   desc_func_dict, i, verbosity_freq,
                                   n_samples, grid, verbose, comp_func_dict, orig_samples, orig_density):
	new_stats = print_statistics_for_1d_forget(local_pdf, local_samples, metrics, desc_func_dict, i,
	                                           verbosity_freq, verbose=verbose,
	                                           orig_samples=orig_samples, prefix_stat="A_",
	                                           comp_func_dict=comp_func_dict, orig_density=orig_density)
	if n_samples > 50:
		new_samples = local_samples[-n_samples:]
		new_pdf = compute_kde_scott(x=grid, observations=local_samples)
		gen_stats = print_statistics_for_1d_forget(new_pdf, new_samples, metrics, desc_func_dict, i,
		                                           verbosity_freq, verbose=verbose, prefix_stat="G_",
		                                           orig_samples=orig_samples)
		new_stats.update(gen_stats)
	return new_stats

def append_to_jsonl(filepath: str, new_record: dict) -> None:
	with open(filepath, mode="a", encoding="utf-8") as f:
		f.write(json.dumps(new_record) + "\n")

def log_results(all_distributions,
                statistics,
                results_dirpath,
                results_filepath,
                params,
                init_dist, y_bounds=None, simulation_step=100, grid=None, ndim=1):
	"""

	Saves the results in the following format:
	1) results_filepath is a global file with the arguments and final statistics available for a full consultation
	Each run is given a unique identifier which is a string from a current datetime
	2) in a folder "results" it stores:
	a) a plot of initial and final distribution
	b) a GIF of distribution evolution
	c) plots of descriptive statistics of all distributions (depending on what was collected)

	"""
	dtime_sting = datetime.datetime.now().strftime("%d_%B_%Y_%I:%M%p")
	log_dir = os.path.join(results_dirpath, dtime_sting)
	if os.path.exists(log_dir):
		log_dir = log_dir + "_" + str(np.random.randint(100)) # TODO replace with a hash of params
	os.makedirs(log_dir, exist_ok=False)
	print("saving params metadata")
	params.update({"datetime": dtime_sting})
	if params['track_git']:
		git_info = get_git_commit_gitpython()
		params.update(git_info)
	append_to_jsonl(results_filepath, params)
	with open(os.path.join(log_dir, "kwargs.json"), "w") as f:
		json.dump(params, f)

	stat_names = list(statistics[0].keys())
	Ns = len(statistics)
	stat_unrolled = {key: [] for key in stat_names}
	print("saving statistics logs") # are we going to save them the same when sampling new stuff ?
	for i in range(Ns):
		for key in stat_names:
			stat_unrolled[key].append(statistics[i][key])

	# TODO fix number of rows, one too big for (2,3), we have (3,3) instead
	Ns = len(stat_names)
	square_root_int = int(np.sqrt(Ns))
	addon = int((Ns - square_root_int**2) > 0)
	height = square_root_int + addon
	width = square_root_int + addon
	fig_stat, axs_stat = plt.subplots(height, width, figsize=(10, 10), sharex=True, sharey=False)
	fig_stat.suptitle(f"Plots of statistics of run {dtime_sting}")
	for i, key in enumerate(stat_names):
		ih = i // height
		iw = i % height
		axs_stat[ih, iw].plot(stat_unrolled[key])
		axs_stat[ih, iw].set_title(key)

	fig_stat.savefig(os.path.join(log_dir, "statistics_plots.png"))

	print("saving the picture of the final distribution")
	if ndim == 1:
		fig_final, ax_final = plt.subplots()
		if grid is not None:
			ax_final.plot(grid, init_dist, color='orange')
		else:
			ax_final.plot(init_dist, color='orange')
		if grid is not None:
			ax_final.plot(grid, all_distributions[-1], color='blue')
		else:
			ax_final.plot(all_distributions[-1], color='blue')
		fig_final.suptitle("Orange - original distribution, blue - a final distribution")
		fig_final.savefig(os.path.join(log_dir, "density_plots_led.png"))

		if params['save_gif']:
			print("saving a gif of distribution evolution")
			fig_gif, ax_gif = plt.subplots()
			ax_gif.set_title("Distribution evolution")
			line, = ax_gif.plot([], [])
			if y_bounds is None:
				_y = range(len(all_distributions[0]))
			elif isinstance(y_bounds, (list, tuple)) and len(y_bounds) == 2:
				_y = np.linspace(y_bounds[0], y_bounds[1], num=len(all_distributions[0]))
			else:
				raise TypeError("y_bounds must be a list or tuple of size 2")

			ax_gif.set_xlim(_y[0], _y[-1])
			# Determine global min/max across all frames so y-axis remains stable
			y_min = min(np.min(d) for d in all_distributions)
			y_max = max(np.max(d) for d in all_distributions)
			ax_gif.set_ylim(y_min, y_max)

			def update(frame):
				line.set_data(_y, all_distributions[frame]) # some internal mechanics of animation demand that we pass both x and y
				ax_gif.set_title(f"Distribution at time {frame}")
				return (line,)

			anim = FuncAnimation(
				fig_gif, update, range(0, len(all_distributions), simulation_step),
			)

			anim.save(os.path.join(log_dir, "evolution.gif"), writer='pillow', fps=10)
			plt.close(fig_gif)
	elif ndim == 2:
		fig_final, axes_final = plt.subplots(2)
		axes_final[0].contourf(grid[:, :, 0], grid[:, :, 1], init_dist)
		axes_final[1].contourf(grid[:, :, 0], grid[:, :, 1], all_distributions[-1])
		fig_final.suptitle("Top : original distributions, bottom : final distribution")
		fig_final.savefig(os.path.join(log_dir, "density_plots_led.png"))

	if params['save_distributions']:
		filename = os.path.join(log_dir, "distribution_logs.hdf5")
		with h5py.File(filename, "w") as f:
			f.create_dataset("distribution_grids", all_distributions.shape, data=all_distributions)

	print("Logs saved successfully")
	print(f"Folder: {log_dir}\n")

def log_results_2d():
	pass

# does not work very well will have to reworked
PARAMETER_KEYS = ['t_scale', 'k_scale', 'method', 'centers', 'scales', 'n_samples', 'n', 'dist_type']
def find_datetime_with_this_params(desired_values, results_filepath,
                                   important_keys=PARAMETER_KEYS, full_agreement_keys=['method'],
                                   min_agreement: float=0.5):
	for key in important_keys:
		assert(key in PARAMETER_KEYS)

	for key in full_agreement_keys:
		assert(key in PARAMETER_KEYS)

	with open(results_filepath, "r") as f:
		all_experiments = f.readlines()
	all_experiments = [json.loads(exp[:-1]) for exp in all_experiments]

	all_keys = set(desired_values.keys()) | set(important_keys)
	less_important_keys = all_keys - set(full_agreement_keys)
	less_important_keys = list(less_important_keys)

	good_datetimes = []
	for exp in all_experiments:
		major_disagreement = False
		for key in full_agreement_keys:
			if key not in exp:
				continue
			if exp[key] != desired_values[key]:
				major_disagreement = True

		if major_disagreement:
			continue

		agreement = 0
		matches = []
		for key in less_important_keys:
			if key not in exp:
				continue
			if exp[key] == desired_values[key]:
				agreement += 1
				matches.append(key)

		agreement /= len(less_important_keys)
		if agreement > min_agreement:
			good_datetimes.append((exp['datetime'], agreement, matches))

	for dt in good_datetimes:
		print(f"Datetime: {dt[0]}. Agreement rate: {agreement}. Matches: {matches}")

