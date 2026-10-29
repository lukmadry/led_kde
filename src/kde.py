import numpy as np
import math
from functools import partial
from numbers import Number
import numba as nb

from scipy.stats import norm
from scipy.signal import fftconvolve
from sklearn.neighbors import KernelDensity

from .tools import (convolve_continuous_grid, exact_unif_1d, exact_gaussian_1d,
                    exact_gaussian_1d_cutoff,
                    match_x_to_grid_and_val_1d, match_x_to_grid_and_val_2d, none_sentinel, _interpret_two_dim_grid)

# if you use exact_gaussian_1d_cutoff then it's comparable in speed with gaussian_kde from scipy for N = 1000, and wins for visibly bigger N
# I guess that scipy does a lot of other stuff under the hood
# however it cuts the function a little bit and I haven't tested systematically yet if it makes the difference, so keeping it as a separate option
# TODO move some of the numerical tests into a separate file so that one can always re-test them if needed
# TODO I think this can be entirely replace by a scikit-learn methods, just need to be carefuly about their bandwith parameter

""" 1D KDE """


def compute_kde(x, observations, norm_scale, kernel='gaussian'):
	"""
	Function that computes kde, i.e. for empirical measure mu^N = 1/N sum_{i=1}^N delta_{X_i} the result is
	1/N sum_{i=1}^N K(x - X_i)

	where x is a numerical value. If x is an array, then the result is applied componentwise.

	x : np.ndarray or float, values where KDE is computed
	observations : np.ndarray of dimensions (N,)
	norm_scale : parameter ruling the precision of KDE, equal to variance in Gaussian distribution
				or width of uniform distribution (see parameter kernel)
	kernel : string, possible choices are ['gaussian', 'gaussian_cutoff', 'uniform']
			gaussian_cutoff is experimental

	"""
	N = len(observations)
	val = 0
	if isinstance(x, Number):
		x = np.asarray([x], float)
	if x.ndim != observations.ndim:
		raise ValueError("x and observations should have matching dimensionality")
	if kernel == 'gaussian':
		# the reason why we're not calling other functions here is that they assume that x is on evenly-spaced grid
		for i in range(N):
			val += exact_gaussian_1d(x-observations[i], var=norm_scale)
	elif kernel == 'gaussian_cutoff':
		for i in range(N):
			val += exact_gaussian_1d_cutoff(x-observations[i], var=norm_scale)
	elif kernel == 'uniform':
		for i in range(N):
			val += exact_unif_1d(x-observations[i], width=norm_scale)
	else:
		raise ValueError("kernel has to be either 'gaussian' or 'uniform'")
	return val / N


@nb.njit(parallel=True, fastmath=True)
def compute_kde_gaussian_1d_numba(
	x: np.ndarray, observations: np.ndarray, var: float
) -> np.ndarray:

	M = x.shape[0]
	N = observations.shape[0]

	out = np.zeros(M, dtype=np.float64)

	inv_two_var = 1.0 / (2.0 * var)
	norm_factor = 1.0 / (np.sqrt(2.0 * np.pi * var) * N)
	for j in nb.prange(M):
		x_val = x[j]
		acc = 0.0

		for i in range(N):
			diff = x_val - observations[i]
			acc += np.exp(-diff * diff * inv_two_var)

		out[j] = acc * norm_factor

	return out

def compute_kde_gaussian_numba_wrapper(x, observations, var):
	"""Computes 1D Gaussian Kernel Density Estimation at arbitrary points x.

		Parameters
		----------
		x : np.ndarray (M,)
			1D array of target evaluation points.
		observations : np.ndarray (N,)
			1D array of observed sample points X_i.
		var : float
			Variance (sigma^2) of the Gaussian kernel.

		Returns
		-------
		kde : np.ndarray (M,)
			Componentwise KDE evaluation: (1/N) sum_{i=1}^N K(x - X_i).
		"""
	return compute_kde_gaussian_1d_numba(x, observations, var)

def compute_kde_binned_fft(x, observations, norm_scale, dx):
	"""
	uses binning-fft method to compute kde.

	x: np.ndarray (M, )
	observations: np.ndarray (N, )
	norm_scale: float
	dx: float, step of a grid
	"""
	dx = x[1] - x[0]
	M = len(x)

	edges = np.empty(M + 1, dtype=np.float64)
	edges[0] = x[0] - 0.5 * dx
	edges[1:-1] = 0.5 * (x[:-1] + x[1:])
	edges[-1] = x[-1] + 0.5 * dx

	counts, _ = np.histogram(observations, bins=edges)
	weights = counts.astype(np.float64) / (len(observations) * dx)

	bandwidth = np.sqrt(norm_scale)
	k_radius = int(np.ceil(4.0 * bandwidth / dx))
	k_x = np.arange(-k_radius, k_radius + 1, dtype=np.float64) * dx
	kernel_vals = (
			np.exp(-0.5 * (k_x / bandwidth) ** 2)
			/ (bandwidth * np.sqrt(2.0 * np.pi))
			* dx
	)
	density = fftconvolve(weights, kernel_vals, mode="same")
	return density

# actually we're not using this, but it was a hypothesis which remains in a profiling suite
def compute_kde_1d_sklearn(x, observations, variance):

	kde = KernelDensity(bandwidth=np.sqrt(variance))
	kde.fit(observations.reshape(-1, 1))
	prob_grid = np.exp(kde.score_samples(x.reshape(-1, 1)))  # score_samples yields logprobability

	return prob_grid



def compute_kde_scott(x, observations):
	"""
	computes KDE where the choice of bandwith is done with Scott's method

	x : np.ndarray or a float. If np.ndarray applied componentwise
	observations : np.ndarray (N, ) type float
	"""
	N = len(observations)
	h = N**(-1/5.)
	val = 0
	for i in range(N):
		val += norm.pdf(x - observations[i], scale=h)
	return val/N

def kde_sampled_pdf_wrapper(x, grid, kde):
	if isinstance(x, Number):
		x = [x]
	vals = []
	for v in x:
		vals.append(match_x_to_grid_and_val_1d(v, grid, kde, interpolation=True))
	return vals

def kde_sampled_pdf_func(grid, kde):
	return partial(kde_sampled_pdf_wrapper, grid=grid, kde=kde)


""" 2D KDE """

# a wrapper for KDE from sklearn, on grid of size 400 the old version takes 2 minutes and this one 4 seconds
def compute_kde_2d(x, observations, variance, keep_shape=True):
	if len(observations.shape) != 2:
		raise ValueError("observations must be 2D")

	if len(x.shape) == 3:
		orig_dim = x.shape[:-1]
		x = x.reshape(-1, 2)
	else:
		keep_shape = False
	if isinstance(variance, float):
		bandwidth = np.sqrt(variance)
	else:
		bandwidth = variance # to allow 'scott'
	kde = KernelDensity(bandwidth=bandwidth)
	kde.fit(observations)
	prob_grid = np.exp(kde.score_samples(x.reshape(-1, 2))) # score_samples yields logprobability

	if keep_shape:
		return prob_grid.reshape(orig_dim)
	return prob_grid


# not necessary anymore, keeping it for myself for reference just in case
def compute_kde_2d_by_hand(x, observations, variance,
                    batch_size=5000, keep_shape=True):
	"""
	computes KDE of 2-dimensional observations, returns np.ndarray of shape (M, ).
	Works by slicing x in batches of size batch_size.

	x: np.ndarray of (N, 2) or (N, K, 2). M = N or M = N * K
	observations : np.ndarray of (N, 2)
	variance : float
	batch_size : int. Set it to be as high as you can, speeds up computation.
	keep_shape : bool. If True final result will be reshaped to (N, K). Only used if x was passed in 3dim form
	"""
	if len(observations.shape) != 2:
		raise ValueError("observations must be 2D")

	if len(x.shape) == 3:
		orig_dim = x.shape[:-1]
		x = x.reshape(-1, 2)
	else:
		keep_shape = False # in this case it will be impossible to reach orig_dim which won't exist

	norm_factor = 1.0 / (2.0 * np.pi * variance)
	M = x.shape[0]
	estimates_on_x = np.empty(M, dtype=float)
	for idx in range(0, M, batch_size):
		end_idx = max(idx + batch_size, M)
		x_batch = x[idx:end_idx]

		diff = x_batch[:, None, :] - observations[None, :, :] # this will have shape (batch_size, N, 2)
		sq_dist = np.sum(diff**2, axis=-1)
		kernel_vals = norm_factor * np.exp(-sq_dist / (variance * 2.0))
		estimates_on_x[idx:end_idx] = np.mean(kernel_vals, axis=1)

	if keep_shape:
		return estimates_on_x.reshape(*orig_dim)
	return np.asarray(estimates_on_x, float)

@nb.njit(parallel=True, fastmath=True)
def _kde_2d_kernel(x: np.ndarray, observations: np.ndarray, variance: float) -> np.ndarray:
	M = x.shape[0]
	N = observations.shape[0]

	out = np.empty(M, dtype=np.float64)

	inv_two_var = 1.0 / (2.0 * variance)
	norm_factor = 1.0 / (2.0 * np.pi * variance * N)

	for j in nb.prange(M):
		x0 = x[j, 0]
		x1 = x[j, 1]
		acc = 0.0

		for i in range(N):
			dx0 = x0 - observations[i, 0]
			dx1 = x1 - observations[i, 1]
			sq_dist = dx0 * dx0 + dx1 * dx1
			acc += np.exp(-sq_dist * inv_two_var)

		out[j] = acc * norm_factor

	return out

# wrapper for the function above to turn on IDE autocompletion
def compute_kde_gaussian_2d_numba(x, observations, norm_scale=None, keep_shape=True, variance=None):
	none_sentinel(norm_scale, variance) # backwards compatibility
	if norm_scale is None:
		norm_scale = variance

	if len(x.shape) == 3:
		orig_dim = x.shape[:-1]
		x = x.reshape(-1, 2)
	else:
		keep_shape = False

	kde_vals = _kde_2d_kernel(x, observations, norm_scale)
	if keep_shape:
		return kde_vals.reshape(orig_dim)
	return kde_vals

def compute_kde_scott_2d(x, observations, keep_shape=True, fast=False):
	"""
	wrapper for compute_kde_2d that itself wraps the sklearn... (artefact from previous no-sklearn variant)
	"""
	N = observations.shape[0]
	sddev = N**(-1/6.)

	if not fast:
		return compute_kde_gaussian_2d_numba(x, observations, sddev**2, keep_shape=keep_shape)
	else:
		return compute_kde_2d_fftbin(x, observations, sddev**2, keep_shape=keep_shape)


def kde_sampled_pdf_func_2d(grid, kde):
	def fn(x):
		return match_x_to_grid_and_val_2d(x, grid, kde, interpolation=True)

	return fn

def fast_fft_kde_2d_standalone(
	observations: np.ndarray,
	grid: np.ndarray,
	bandwidth: float,
) -> np.ndarray:
	"""Computes 2D Gaussian KDE rapidly via histogram binning and FFT convolution
	"""
	grid, x_grid, y_grid = _interpret_two_dim_grid(grid)

	nx, ny = len(x_grid), len(y_grid)
	dx = x_grid[1] - x_grid[0]
	dy = y_grid[1] - y_grid[0]

	x_edges = np.empty(nx + 1, dtype=np.float64)
	x_edges[0] = x_grid[0] - 0.5 * dx
	x_edges[1:-1] = 0.5 * (x_grid[:-1] + x_grid[1:])
	x_edges[-1] = x_grid[-1] + 0.5 * dx

	y_edges = np.empty(ny + 1, dtype=np.float64)
	y_edges[0] = y_grid[0] - 0.5 * dy
	y_edges[1:-1] = 0.5 * (y_grid[:-1] + y_grid[1:])
	y_edges[-1] = y_grid[-1] + 0.5 * dy

	counts, _, _ = np.histogram2d(
		observations[:, 0], observations[:, 1], bins=[x_edges, y_edges]
	)
	weights = counts.T.astype(np.float64) / (len(observations) * dx * dy)

	k_radius_x = int(np.ceil(4.0 * bandwidth / dx))
	k_radius_y = int(np.ceil(4.0 * bandwidth / dy))

	kx = np.arange(-k_radius_x, k_radius_x + 1, dtype=np.float64) * dx
	ky = np.arange(-k_radius_y, k_radius_y + 1, dtype=np.float64) * dy
	KX, KY = np.meshgrid(kx, ky, indexing="ij")

	kernel_vals = (
		np.exp(-0.5 * (KX**2 + KY**2) / (bandwidth**2))
		/ (2.0 * np.pi * bandwidth**2)
		* (dx * dy)
	)

	density = fftconvolve(weights, kernel_vals, mode="same")
	return density

def compute_kde_2d_fftbin(x, observations, variance, keep_shape=True):
	if len(x.shape) != 3:
		raise ValueError("x must be a grid")
	return fast_fft_kde_2d_standalone(grid=x, observations=observations, bandwidth=np.sqrt(variance))


# the difference is that you need to give a ready kernel and we will not question it


# kde with histograms


def compute_led_kde(grid, observations, t_scale, k_pdf, eps=1e-8):
	q_L_approx = compute_kde(x=grid, observations=observations, norm_scale=t_scale)
	k_log_ql_conv = convolve_continuous_grid(k_pdf, np.log(eps+q_L_approx), grid)
	expklog = np.exp(k_log_ql_conv)
	Z = np.sum(expklog) * (grid[1] - grid[0])
	return expklog / Z

def compute_led_kde_2d(pts_list, observations, t_param, k_param, eps=1e-8,
                       dimensions_for_grid=None):
	if len(pts_list.shape) > 2:
		raise ValueError("pts_list must be 2D")

	q_L_approx = compute_kde_2d(x=pts_list, observations=observations, variance=t_param)
	k_norm_factor = 1/(2.0 * np.pi * k_param)
	if dimensions_for_grid is None:
		root = math.sqrt(pts_list.shape[0])
		if root ** 2 != pts_list.shape[0]:
			raise ValueError(f"Length of pts_list is not a square, the root is: {root}")

		dimensions_for_grid = (int(root), int(root), 2)
	elif isinstance(dimensions_for_grid, tuple):
		dimensions_for_grid = (dimensions_for_grid[0], dimensions_for_grid[1], 2)
	else:
		raise ValueError("dimensions_for_grid must be a tuple or left as None")

	pts_list = pts_list.reshape(dimensions_for_grid)
	q_L_approx = q_L_approx.reshape(dimensions_for_grid[:2])

	k_pdf = lambda diffs : k_norm_factor * np.exp( - np.square(diffs).sum(axis=-1) / (2.0 * k_param))
	k_log_ql_conv = convolve_continuous_grid(k_pdf, np.log(eps+q_L_approx), pts_list)
	expklog = np.exp(k_log_ql_conv)
	dx = pts_list[0, 1, 0] - pts_list[0, 0, 0]
	dy = pts_list[1, 0, 1] - pts_list[0, 0, 1]
	Z = np.sum(expklog) * dx * dy
	return expklog / Z

