import unittest
import numpy as np
import time

from kde import alpha_func, compute_led_kde, compute_kde, sample_led_kde
from tools import exact_gaussian_1d, match_x_to_grid_and_val, match_x_to_grid_and_val_1d
from functools import partial
from distributions import GaussianMixture, UniformMixture
from run import run_recurrent_led_kde_with_forgetting, run_recurrent_led_with_data_retaining

# we should check its approximate symmetry if the input distribution is symmetric
class TestAlphaFunc(unittest.TestCase):
	def test_one_dim_alpha_func(self):
		x = np.linspace(-5, 5, 100)
		xdim = np.random.randn(2, 100)
		obs = np.random.randn(100)*0.1

		alpha_func(x, obs, t_scale=0.01, k_scale=0.02)
		try:
			alpha_func(xdim, obs, t_scale=0.01, k_scale=0.02)
			alpha_func(x, obs, t_scale=0.01, k_scale=0.02, kernel="exponential")
		except ValueError:
			pass

class TestKDE(unittest.TestCase):
	def test_kde(self):
		x = np.linspace(-5, 5, 100)
		obs = np.random.randn(100)*0.1
		compute_kde(x, obs, 0.01)
		compute_kde(x[0], obs, 0.01)
		k_pdf = partial(exact_gaussian_1d, var=0.02)
		compute_led_kde(x, obs, 0.01, k_pdf)

class TestMatching(unittest.TestCase):
	def test_matching_1d(self, grid_size=101):
		x = np.linspace(-5, 5, grid_size)
		y = np.power(x, 2)
		u = np.random.uniform(low=-5, high=5, size=grid_size)
		# just testing if it works
		res_inter = match_x_to_grid_and_val_1d(x=u, grid=x, values_grid=y, interpolation=True)
		self.assertTrue(isinstance(res_inter, np.ndarray))
		self.assertEqual(res_inter.ndim, u.ndim)
		self.assertEqual(res_inter.shape, x.shape)
		res_no_interp = match_x_to_grid_and_val_1d(x=u, grid=x, values_grid=y, interpolation=False)
		self.assertTrue(isinstance(res_no_interp, np.ndarray))
		self.assertEqual(res_no_interp.ndim, u.ndim)
		self.assertEqual(res_no_interp.shape, x.shape)
		try:
			match_x_to_grid_and_val_1d(x=np.asarray([-7, 7], float), grid=x, values_grid=y, raise_error_if_outside=True)
		except ValueError:
			pass

		res = match_x_to_grid_and_val_1d(1.23, grid=x, values_grid=y, interpolation=False)
		self.assertAlmostEqual(res, 1.2**2)

	def test_matching_routing(self):

		xg = np.linspace(-5, 5, 11)
		u = np.asarray([-3.45, 3.23])
		res = match_x_to_grid_and_val(x=xg, grid=u, values_grid=u, interpolation=False)
		self.assertTrue(isinstance(res, np.ndarray))
		self.assertTrue(res.ndim == 1)

		with self.assertRaises(expected_exception=IndexError):
			match_x_to_grid_and_val(x=xg.reshape(-1, 1), grid=u, values_grid=u, interpolation=False)

class TestGaussianMixture(unittest.TestCase):
	def test_gmixture_creation(self):
		GaussianMixture(centers=0, scales=2)
		GaussianMixture(centers=[-2, 2], scales=1)
		self.good_g = GaussianMixture(centers=[-2, 2], scales=[1, 1])

	def test_weights(self):
		g = GaussianMixture(centers=[-2, 2], scales=[1, 1])
		self.assertTrue(np.allclose(g.weights, [1,1]))

	def test_wrong_sizes_errors(self):
		with self.assertRaises(ValueError):
			GaussianMixture(centers=[-2, 2], scales=[2,2,2])
		with self.assertRaises(ValueError):
			GaussianMixture(centers=[-3,3], scales=1, weights=[3,4,5])
		with self.assertRaises(TypeError):
			GaussianMixture(centers=[-3, 3])

	def test_absurd_scale(self):
		with self.assertRaises(ValueError):
			GaussianMixture(centers=[-2, 2], scales=-2)

	def test_proba_methods(self):
		g = GaussianMixture(centers=[-2, 2], scales=[1, 1])
		g.rvs(100)
		g.pdf(np.linspace(-5, 5, 100))

def test_gauss_input_led_sampling_end_to_end(n):
	centers = [-2, 2]
	scales = [0.35, 0.35]
	weights = None
	scale_is_variance = True
	grid = np.linspace(-10, 10, 500)
	n_samples = 200

	true_mixture = GaussianMixture(
		centers=centers,
		scales=scales,
		weights=weights,
		scale_is_variance=scale_is_variance)
	init_obs = true_mixture.rvs(n_samples)

	run_kwargs = dict(
		n_trials=n,
		grid=grid,
		kde_scale=0.02,
		scale_is_variance=scale_is_variance,
		verbosity_freq=10,
		return_final_samples=True,
		n_candidates=10000,
		n_samples=1000,
		kernel='gaussian',
		t_scale=0.01,
		initial_observations=init_obs,
		alpha_filter_denom=True,
		sampling_method='standard')
	final_sample, all_dist, stat, new_info = run_recurrent_led_kde_with_forgetting(**run_kwargs)

def test_gauss_input_led_retain_sampling_end_to_end(n):
	centers = [-2, 2]
	scales = [0.35, 0.35]
	weights = None
	scale_is_variance = True
	grid = np.linspace(-10, 10, 500)
	n_samples = 1000

	true_mixture = GaussianMixture(
		centers=centers,
		scales=scales,
		weights=weights,
		scale_is_variance=scale_is_variance)
	init_obs = true_mixture.rvs(n_samples)

	run_kwargs = dict(
		n_trials=n,
		grid=grid,
		kde_scale=0.02,
		scale_is_variance=scale_is_variance,
		verbosity_freq=10,
		return_final_samples=True,
		n_candidates=10000,
		n_samples=50,
		kernel='gaussian',
		t_scale=0.01,
		initial_observations=init_obs,
		alpha_filter_denom=True,
		sampling_method='standard')
	final_sample, all_dist, stat, new_info = run_recurrent_led_with_data_retaining(**run_kwargs)

def test_uniform_input_led_sampling_end_to_end(n):
	pass

# I guess that you shouldn't do this because it's not really a unit test but I don't have the time explore pytest
class TestFullRun(unittest.TestCase):
	def test_full_run(self):
		test_gauss_input_led_sampling_end_to_end(10)

	def test_full_run_retain(self):
		test_gauss_input_led_retain_sampling_end_to_end(10)

# profiling
def profile_function(func, runs = 5, verbose=False, *args, **kwargs):
	"""Executes a function multiple times and prints runtime statistics."""
	durations = []
	for _ in range(runs):
		start = time.perf_counter()
		func(*args, **kwargs)
		end = time.perf_counter()
		durations.append(end - start)

	avg_time = sum(durations) / len(durations)
	min_time = min(durations)
	max_time = max(durations)

	if verbose:
		print(f"[{func.__name__}] ({runs} runs)")
		print(f"  Avg: {avg_time * 1000:.2f} ms | Min: {min_time * 1000:.2f} ms | Max: {max_time * 1000:.2f} ms\n")
	return (avg_time, min_time, max_time)

unif_mixture = UniformMixture(centers=[-2, 2], widths=[1, 1])

# TODO abstract it away, no need to write a separate loop for all of them
def profile_kde(runs = 10, grid_sizes = [100, 500, 1000], sample_sizes = [1000, 5000, 10000]):
	average_times = []
	for g in grid_sizes:
		for s in sample_sizes:
			obs = unif_mixture.rvs(s)
			# print(f"Grid size {g}, sample size {s}")
			x_grid = np.linspace(-5, 5, g)
			avg, mint, maxt = profile_function(compute_kde, runs=runs, x=x_grid, observations=obs, norm_scale=0.01, kernel='gaussian')
			average_times.append((g, s, avg))
	return average_times

def profile_kde_cutoff(runs = 10, grid_sizes = [100, 500, 1000], sample_sizes = [1000, 5000, 10000]):
	average_times = []
	for g in grid_sizes:
		for s in sample_sizes:
			obs = unif_mixture.rvs(s)
			# print(f"Grid size {g}, sample size {s}")
			x_grid = np.linspace(-5, 5, g)
			avg, mint, maxt = profile_function(compute_kde, runs=runs, x=x_grid,
											   observations=obs, norm_scale=0.01, kernel='gaussian_cutoff')
			average_times.append((g, s, avg))

	return average_times

def profile_alpha_func(runs=10, grid_sizes = [100, 500, 1000], sample_sizes = [1000, 5000, 10000]):
	average_times_alpha = []
	for g in grid_sizes:
		for s in sample_sizes:
			obs = unif_mixture.rvs(s)
			x_grid = np.linspace(-5, 5, g)
			avg, _, _ = profile_function(alpha_func, runs=runs, grid=x_grid, observations=obs, t_scale=0.01, k_scale=0.02)
			average_times_alpha.append((g, s, avg))
	return average_times_alpha

def profile_sampling_led(runs=10, init_s_size=1000, grid_sizes=[100, 500, 1000],
						 sample_sizes=[10, 25, 100, 1000, 5000]):
	obs = unif_mixture.rvs(init_s_size)
	avg_times = []
	for g in grid_sizes:
		xgrid = np.linspace(-5, 5, g)
		alpha = alpha_func(grid=xgrid, observations=obs, t_scale=0.01, k_scale=0.02, filter_denom=False)
		for s in sample_sizes:
			res, _, _ = profile_function(sample_led_kde,
							 n_samples=s, observations=obs,
							 t_scale=0.01, k_scale=0.02, alpha_grid=alpha, grid=xgrid)
			avg_times.append((g, s, res))

	return avg_times

if __name__ == '__main__':
	K = 10
	print("Profiling KDE")
	kde_avg_times = profile_kde(K)
	print("Profiling KDE with cutoff")
	kde_cutoff_avg_times = profile_kde_cutoff(K)
	print("Profiling alpha function")
	alpha_avg_times = profile_alpha_func(K)
	print("Profiling LED sampling")
	led_sampling = profile_sampling_led(K)

	for fname, arr in zip(['kde', 'kde_cutoff', 'alpha', 'led'],
						  [kde_avg_times, kde_cutoff_avg_times, alpha_avg_times, led_sampling]):
		print(fname)
		for g, s, avg in arr:
			print(f"Grid size: {g}, sample size: {s}, average time: {1000*avg:.4f}ms")

	unittest.main()

