import unittest
import numpy as np

from kde import alpha_func, compute_led_kde, compute_kde
from tools import exact_gaussian_1d, match_x_to_grid_and_val, match_x_to_grid_and_val_1d
from functools import partial
from distributions import GaussianMixture
from run import run_recurrent_led_kde_with_forgetting

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
		n_samples=1000,
		kernel='gaussian',
		t_scale=0.01,
		initial_observations=init_obs,
		alpha_filter_denom=True,
		sampling_method='standard')
	final_sample, all_dist, stat, new_info = run_recurrent_led_kde_with_forgetting(**run_kwargs)

def test_uniform_input_led_sampling_end_to_end(n):
	pass

# I guess that you shouldn't do this because it's not really a unit test but I don't have the time explore pytest
class TestFullRun(unittest.TestCase):
	def test_full_run(self):
		test_gauss_input_led_sampling_end_to_end(10)

if __name__ == '__main__':
	unittest.main()
