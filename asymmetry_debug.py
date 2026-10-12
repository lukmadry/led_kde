import numpy as np
from scipy import stats
from kde import _alpha_func_direct_1d, sample_led_kde_direct_raw_fast_is, alpha_func
from pipelines import run_recurrent_led_with_data_forgetting_1d
from tools import match_x_to_grid_and_val_1d, compute_gaussian_kernel_1d
import matplotlib.pyplot as plt
from distributions import GaussianMixture
import json
import time

# TODO investigate whether the shift depends on the center of the starting_gauss
def run_pipeline(t_scale, n_candidates_multiplier, grid_size, n_trials=5000):

	k_scale = 2 * t_scale

	n_samples = 1000

	k_sampler = stats.norm(loc=0, scale=np.sqrt(k_scale)).rvs
	t_sampler = stats.norm(loc=0, scale=np.sqrt(t_scale)).rvs

	grid = np.linspace(-5, 5, num=grid_size)
	real_means_obs = []
	means_alpha_dist = []
	after_sampling_obs_mean = []

	real_median_obs = []
	after_sampling_obs_median = []

	dx = grid[1] - grid[0]
	_radius = grid.shape[0] * dx
	kernel_arr = compute_gaussian_kernel_1d(_radius, dx, k_scale)

	ess_values = []
	for _ in range(n_trials):
		starting_gauss = GaussianMixture(centers=np.random.uniform(-1, 1), scales=np.random.uniform(0.2, 1))
		observations = starting_gauss.rvs(n_samples)
		alpha_grid = _alpha_func_direct_1d(grid, observations, t_scale, kernel_arr, dx)

		n_candidates = n_candidates_multiplier * n_samples
		u = t_sampler(size=n_candidates)
		w = k_sampler(size=n_candidates)
		idx = np.random.randint(low=0, high=n_samples, size=n_candidates)

		X = observations[idx] - u - w
		alph_weights = match_x_to_grid_and_val_1d(X, grid, alpha_grid, interpolation=True)
		alph_weights /= np.sum(alph_weights)
		resampled_indices = np.random.choice(
			n_candidates, size=n_samples, replace=True, p=alph_weights
		)
		ess = 1 / np.sum(alph_weights ** 2)
		ess_values.append(ess)

		new_obs = X[resampled_indices]

		real_means_obs.append(np.mean(observations))
		means_alpha_dist.append(np.sum(grid * alpha_grid) * (grid[1] - grid[0]))
		after_sampling_obs_mean.append(np.mean(new_obs))

		real_median_obs.append(np.median(observations))
		after_sampling_obs_median.append(np.median(new_obs))

	bigger = 0
	for rmean, amean in zip(real_means_obs, means_alpha_dist):
		bigger += amean > rmean

	bigger_dist = 0
	for bmean, aft_mean in zip(real_means_obs, after_sampling_obs_mean):
		bigger_dist += aft_mean > bmean

	bigger_median = 0
	for rmedian, aft_median in zip(real_median_obs, after_sampling_obs_median):
		bigger_median += aft_median > rmedian

	ess_min = np.min(ess_values)
	ess_max = np.max(ess_values)
	ess_median = np.median(ess_values)
	return bigger, bigger_dist, bigger_median, ess_min, ess_max, ess_median

t_vals = [0.001, 0.002, 0.01, 0.02, 0.03, 0.04]
cands_multip = [10, 100]
grid_size = [500, 1000, 2000, 4000, 10000]

results = []
with open("asymmetry_debug.json", "r") as f:
	results = json.load(f)

for t_val in t_vals:
	for cands_multiplier in cands_multip:
		for gs in grid_size:
			already_tested = False
			for datum in results:
				if datum["t_val"] == t_val and datum['cands_multiplier'] == cands_multiplier and datum['grid_size'] == gs:
					already_tested = True
			if already_tested:
				print(f"t_val: {t_val}, cands_multiplier: {cands_multiplier}, grid_size: {gs} already tested")
				continue

			t0 = time.perf_counter()
			bigger, bigger_dist, bigger_median, ess_min, ess_max, ess_median = run_pipeline(t_val, cands_multiplier, gs)
			t1 = time.perf_counter()
			results.append({"t_val": t_val, "cands_multiplier": cands_multiplier, "grid_size": gs,
			                "bigger_mean": bigger, "bigger_dist": bigger_dist, "bigger_median": bigger_median,
			                "ess_min": ess_min, "ess_max": ess_max, "ess_median": ess_median})
			print(f"t_val: {t_val}, cands_multiplier: {cands_multiplier}, grid_size: {gs} | median: {bigger_median}")
			print(f"Time passed on computation : {t1 - t0}\n")
			with open("asymmetry_debug.json", "w") as f:
				json.dump(results, f, default=str) # default to protect against non-serialisable because of numpy stuff

