import unittest
import numpy as np
import time
from scipy.stats import norm

from src.kde import (compute_led_kde, compute_kde, compute_kde_gaussian_numba_wrapper, compute_kde_binned_fft, compute_kde_1d_sklearn,
                     _kde_2d_kernel, compute_kde_2d, compute_kde_2d_fftbin)
from src.sampling import sample_led_kde, sample_led_kde_direct_raw_fast_is
from src.alpha import alpha_func, _alpha_func_direct_1d_numba
from src.tools import exact_gaussian_1d, match_x_to_grid_and_val_1d, compute_gaussian_kernel_1d
from functools import partial
from src.distributions import GaussianMixture, UniformMixture, UniformBoxMixture

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
	def test_matching_1d_shapes(self, grid_size=101):
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

	def test_matching_1d_value(self, grid_size=101):
		x = np.linspace(-5, 5, grid_size)
		y = np.power(x, 2)
		res = match_x_to_grid_and_val_1d(1.23, grid=x, values_grid=y, interpolation=False)
		self.assertAlmostEqual(res, 1.2**2)

		propx = np.array([0.53, 0.74, 0.88, 1.23])
		res = match_x_to_grid_and_val_1d(propx, grid=x, values_grid=y, interpolation=True)
		self.assertTrue(np.allclose(res, np.array([0.2533, 0.496 , 0.6536, 1.4475])))

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


class TestApproximations(unittest.TestCase):

	unif_mixture = UniformMixture(centers=[-2, 2], widths=[1, 1])
	gauss_mixture = GaussianMixture(centers=[-2, 2], scales=[1, 1])
	var = 1e-1
	xgrid = np.linspace(-5, 5, num=1001)
	dx = xgrid[1] - xgrid[0]

	unif_box_mixture = UniformBoxMixture(centers=list(zip([-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2],
	                                                    [-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1,1,1,1])),
	                                     shapes=[[1, 1]], weights=[10, 5, 1, 5, 10, 1, 3, 9, 3, 1, 5, 10, 1, 10, 7])

	grid_x, grid_y = np.linspace(-5, 5, num=500 | 1), np.linspace(-5, 5, num=500 | 1)
	dxx = grid_x[1] - grid_x[0]
	grid = np.meshgrid(grid_x, grid_y)
	grid = np.dstack(grid)

	def test_kde_binning(self):

		unif_samples = self.unif_mixture.rvs(10000)
		kde_gaussians_gt = compute_kde_gaussian_numba_wrapper(self.xgrid, unif_samples, self.var)
		kde_binned = compute_kde_binned_fft(self.xgrid, unif_samples, self.var, self.dx)
		self.assertTrue(np.allclose(kde_gaussians_gt, kde_binned, rtol=1e-4, atol=1e-4))

	def test_kde_sklearn(self):

		unif_samples = self.unif_mixture.rvs(1000)
		gaus_samples = self.gauss_mixture.rvs(1000)
		kde_unif_gt = compute_kde_gaussian_numba_wrapper(self.xgrid, unif_samples, self.var)
		kde_gaus_gt = compute_kde_gaussian_numba_wrapper(self.xgrid, gaus_samples, self.var)

		kde_sklearn_unif = compute_kde_1d_sklearn(self.xgrid, unif_samples, self.var)
		kde_sklearn_gaus = compute_kde_1d_sklearn(self.xgrid, gaus_samples, self.var)

		er_unif = np.max(np.abs(kde_unif_gt - kde_sklearn_unif))
		er_gaus = np.max(np.abs(kde_gaus_gt - kde_sklearn_gaus))

		self.assertTrue(er_unif < 1e-10)
		self.assertTrue(er_gaus < 1e-10)


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
unif_box_mixture = UniformBoxMixture(centers=list(zip([-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2],
	                                                    [-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1,1,1,1])),
	                                     shapes=[[1, 1]], weights=[10, 5, 1, 5, 10, 1, 3, 9, 3, 1, 5, 10, 1, 10, 7])

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

def profile_kde_sklearn_1d(runs = 10, grid_sizes = [100, 500, 1000], sample_sizes = [1000, 5000, 10000]):
	average_times = []
	for g in grid_sizes:
		for s in sample_sizes:
			obs = unif_mixture.rvs(s)
			# print(f"Grid size {g}, sample size {s}")
			x_grid = np.linspace(-5, 5, g)
			avg, mint, maxt = profile_function(compute_kde_1d_sklearn, runs=runs, x=x_grid, observations=obs, variance=0.01)
			average_times.append((g, s, avg))
	return average_times

def profile_kde_2d_numba(runs = 10, grid_sizes = [100, 500, 1000], sample_sizes = [1000, 5000, 10000]):
	average_times = []
	for g in grid_sizes:
		grid_x, grid_y = np.linspace(-5, 5, num=g), np.linspace(-5, 5,num=g)
		grid = np.meshgrid(grid_x, grid_y)
		grid = np.dstack(grid)

		for s in sample_sizes:
			obs = unif_box_mixture.rvs(s)
			# print(f"Grid size {g}, sample size {s}")
			avg, mint, maxt = profile_function(_kde_2d_kernel, runs=runs, x=grid, obs=obs, variance=0.01)
			average_times.append((g, s, avg))
	return average_times

def profile_kde_2d_sklearn(runs = 10, grid_sizes = [100, 500, 1000], sample_sizes = [1000, 5000, 10000]):
	average_times = []
	for g in grid_sizes:
		grid_x, grid_y = np.linspace(-5, 5, num=g), np.linspace(-5, 5,num=g)
		grid = np.meshgrid(grid_x, grid_y)
		grid = np.dstack(grid)

		for s in sample_sizes:
			obs = unif_box_mixture.rvs(s)
			# print(f"Grid size {g}, sample size {s}")
			avg, mint, maxt = profile_function(compute_kde_2d, runs=runs, x=grid, observations=obs, variance=0.01)
			average_times.append((g, s, avg))
	return average_times


def profile_kde_2d_fftbin(runs = 10, grid_sizes = [100, 500, 1000], sample_sizes = [1000, 5000, 10000]):
	average_times = []
	for g in grid_sizes:
		grid_x, grid_y = np.linspace(-5, 5, num=g), np.linspace(-5, 5,num=g)
		grid = np.meshgrid(grid_x, grid_y)
		grid = np.dstack(grid)

		for s in sample_sizes:
			obs = unif_box_mixture.rvs(s)
			# print(f"Grid size {g}, sample size {s}")
			avg, mint, maxt = profile_function(compute_kde_2d_fftbin, runs=runs, x=grid, observations=obs, variance=0.01)
			average_times.append((g, s, avg))
	return average_times


def profile_kde_numba(runs = 10, grid_sizes = [100, 500, 1000], sample_sizes = [1000, 5000, 10000]):
	average_times = []
	# warmup to precompile
	C = compute_kde_gaussian_numba_wrapper(np.linspace(-5, 5, 100), unif_mixture.rvs(1000), var=0.01)
	for g in grid_sizes:
		for s in sample_sizes:
			obs = unif_mixture.rvs(s)
			# print(f"Grid size {g}, sample size {s}")
			x_grid = np.linspace(-5, 5, g)
			avg, mint, maxt = profile_function(compute_kde_gaussian_numba_wrapper, runs=runs, x=x_grid, observations=obs, var=0.01)
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
			dx = x_grid[1] - x_grid[0]
			k_kernel_arr = compute_gaussian_kernel_1d(10, dx=dx, var=0.02)
			avg, _, _ = profile_function(_alpha_func_direct_1d_numba, runs=runs, k_kernel_arr=k_kernel_arr, grid=x_grid,
			                             observations=obs, t_scale=0.01, dx=dx)
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
			res, _, _ = profile_function(sample_led_kde, runs=runs,
							 n_samples=s, observations=obs,
							 t_scale=0.01, k_scale=0.02, alpha_grid=alpha, grid=xgrid)
			avg_times.append((g, s, res))

	return avg_times


def profile_sampling_led_fast(runs=10, init_s_size=1000, grid_sizes=[100, 500, 1000],
						 sample_sizes=[10, 25, 100, 1000, 5000]):
	obs = unif_mixture.rvs(init_s_size)
	avg_times = []
	k_sampler = norm(loc=0, scale=0.02).rvs
	t_sampler = norm(loc=0, scale=0.01).rvs
	for g in grid_sizes:
		xgrid = np.linspace(-5, 5, g)
		dx = xgrid[1] - xgrid[0]
		k_kernel_arr = compute_gaussian_kernel_1d(10, dx=dx, var=0.02)
		alpha_grid = _alpha_func_direct_1d_numba(grid=xgrid, observations=obs, t_scale=0.01, k_kernel_arr=k_kernel_arr,
		                                    dx=dx)
		for s in sample_sizes:
			res, _, _ = profile_function(sample_led_kde_direct_raw_fast_is, runs=runs, t_sampler=t_sampler, dx=dx,
							 n_samples=s, observations=obs, k_sampler=k_sampler, alpha_grid=alpha_grid,
							 t_scale=0.01, k_kernel_arr=k_kernel_arr, grid=xgrid)
			avg_times.append((g, s, res))

	return avg_times

if __name__ == '__main__':
	import sys
	if len(sys.argv) > 1 and sys.argv[1] == '--profile':
		K = 10
		print("Profiling fast LED sampling")
		led_sampling_fast = profile_sampling_led_fast(K)
		print("Profiling KDE")
		kde_avg_times = profile_kde(K)
		print("Profiling KDE with numba")
		kde_numba_avg_times = profile_kde_numba(K)
		print("Profiling KDE with sklearn")
		kde_sklearn = profile_kde_sklearn_1d(K)
		print("Profiling KDE 2d with fftbin")
		kde_2d_fftbin = profile_kde_2d_fftbin(K)
		print("Profiling KDE with cutoff")
		kde_cutoff_avg_times = profile_kde_cutoff(K)
		print("Profiling alpha function")
		alpha_avg_times = profile_alpha_func(K)
		print("Profiling LED sampling")
		led_sampling = profile_sampling_led(K)

		for fname, arr in zip(['kde', 'kde_cutoff', 'kde_numba',  'alpha',
		                       'led', 'kde_sklearn', 'led_fast_is', 'kde_2d_fftbin'],
							  [kde_avg_times, kde_cutoff_avg_times, kde_numba_avg_times, alpha_avg_times,
							   led_sampling, kde_sklearn, led_sampling_fast, kde_2d_fftbin]):
			print(fname)
			for g, s, avg in arr:
				print(f"Grid size: {g}, sample size: {s}, average time: {1000*avg:.4f}ms")
	else:
		unittest.main()

