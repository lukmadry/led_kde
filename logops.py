import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import json
import os
import datetime
import h5py

from tools import get_git_commit_gitpython

def append_to_jsonl(filepath: str, new_record: dict) -> None:
	with open(filepath, mode="a", encoding="utf-8") as f:
		f.write(json.dumps(new_record) + "\n")

def log_results(all_distributions,
                statistics,
                results_dirpath,
                results_filepath,
                params,
                init_dist, y_bounds=None, simulation_step=100):
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

	with open("kwargs.json", "w") as f:
		json.dump(params, f)

	print("saving statistics logs")
	stat_names = list(statistics[0].keys())
	Ns = len(statistics)
	stat_unrolled = {key: [] for key in stat_names}
	for i in range(Ns):
		for key in stat_names:
			stat_unrolled[key].append(statistics[i][key])

	Ns = len(stat_names)
	square_root_int = int(np.sqrt(Ns))
	addon = int((Ns - square_root_int**2) > 0)
	height = square_root_int + addon
	width = square_root_int + addon

	fig_stat, axs_stat = plt.subplots(height, width, sharex=True, sharey=False)
	fig_stat.suptitle(f"Plots of statistics of run {dtime_sting}")
	for i, key in enumerate(stat_names):
		ih = i // height
		iw = i % height
		axs_stat[ih, iw].plot(stat_unrolled[key])
		axs_stat[ih, iw].set_title(key)

	fig_stat.savefig(os.path.join(log_dir, "statistics_plots.png"))

	print("saving the picture of the final distribution")
	fig_final, ax_final = plt.subplots()
	ax_final.plot(init_dist, color='orange')
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

	if params['save_distributions']:
		filename = os.path.join(log_dir, "distribution_logs.hdf5")
		with h5py.File(filename, "w") as f:
			f.create_dataset("distribution_grids", (params['n'], params['grid_size']), data=all_distributions)

	print("Logs saved successfully")

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

