import numpy as np
import math
from functools import partial
from numbers import Number

import time # for profiling

from scipy.stats import norm, multivariate_normal, uniform
from scipy.signal import fftconvolve

from tools import (convolve_continuous_grid, exact_unif_1d, exact_gaussian_1d,
                   StrangeAlphaException, sample_via_importance_resampling, exact_gaussian_1d_cutoff,
                   match_x_to_grid_and_val_1d, match_x_to_grid_and_val_2d, convolve_cont_grid_1d_with_kernel,
                   convolve_cont_grid_2d_with_kernel)

# if you use exact_gaussian_1d_cutoff then it's comparable in speed with gaussian_kde from scipy for N = 1000, and wins for visibly bigger N
# I guess that scipy does a lot of other stuff under the hood
# however it cuts the function a little bit and I haven't tested systematically yet if it makes the difference, so keeping it as a separate option
# TODO move some of the numerical tests into a separate file so that one can always re-test them if needed
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

def compute_kde_binned_fft(x, observations, norm_scale, dx):
	"""
	uses binning-fft method to compute kde.

	x: np.ndarray (M, )
	observations: np.ndarray (N, )
	norm_scale: float
	dx: float, step of a grid
	"""
	x_grid = np.asarray(x, dtype=np.float64)
	obs = np.asarray(observations, dtype=np.float64)

	M = len(x_grid)

	half_step = dx / 2.0
	bins = np.concatenate(
		[
			x_grid - half_step,
			[x_grid[-1] + half_step],
		]
	)

	weights, _ = np.histogram(obs, bins=bins)
	weights = weights.astype(np.float64) / len(obs)

	kernel_x = np.arange(-(M-1) * dx, M*dx, step=dx) * dx
	kernel_vals = exact_gaussian_1d(kernel_x, norm_scale)

	density = fftconvolve(weights, kernel_vals, mode="same")
	return density

# maybe it's possible to speed it up even more by cutting off v small values
# profiling - it was profiled against a naive version, allocation of matrices seems to win over naive looping
# TODO profile to check which batch_size would be optimal
def compute_kde_2d(x, observations, variance,
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

def compute_kde_scott_2d(x, observations, batch_size=10000, keep_shape=True):
	"""
	wrapper for compute_kde_2d that computes the variance
	"""
	variance = 1/(len(observations)**(1/6.))
	return compute_kde_2d(x, observations, variance=variance, keep_shape=keep_shape, batch_size=batch_size)

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

def alpha_func(grid, observations, t_scale, k_scale, log_eps=1e-10, kernel='gaussian',
               filter_denom=False, denom_eps=1e-3, pos_threshold=1e-6):
	"""
	Computes \alpha from LED-KDE. Two-dimensional is at the moment (22/09) experimental so not described in details
	in a docstring.

	grid : np.ndarray (D, ). Grid with evenly spaced values, typically a result of linspace
	observations : np.ndarray (N, )
	t_scale : float, variance of the first, inner kernel
	k_scale : float, variance of the second, outer kernel
	eps : float, epsilon value which will be added to log for numerical stability
	kernel : str, gaussian or uniform, choice of a kernel. If uniform, then {t,k}_scale are widths
	filter_denom : bool, if True then will give only the values where denominator is positive.
			Most likely will be deprecated
	denom_eps : epsilon added to the denominator

	"""
	# print(grid.shape)
	# print("log epsilon: ", log_eps)
	# print("denom epsilon: ", denom_eps)
	if len(observations.shape) == 2:
		grid_orig_shape = grid.shape
		q_L_approx = compute_kde_2d(x=grid, observations=observations, variance=t_scale, keep_shape=True)
		q_L_approx = q_L_approx.reshape(grid_orig_shape[:2])
		k_norm_factor = 1 / (2.0 * np.pi * k_scale)
		k_pdf = lambda diffs: k_norm_factor * np.exp(- np.square(diffs).sum(axis=-1) / (2.0 * k_scale))
		radius = 50*k_scale
		# print("qlapprox ", q_L_approx.max(), q_L_approx.min(), q_L_approx.std(), q_L_approx.mean())
	else:
		q_L_approx = compute_kde(x=grid, observations=observations, norm_scale=t_scale, kernel=kernel)
		if kernel == 'gaussian_cutoff':
			k_pdf = partial(exact_gaussian_1d_cutoff, var=k_scale)
			# k_norm_factor = 1/ np.sqrt(2.0 * np.pi * k_scale)
			# k_pdf = lambda diffs: k_norm_factor * np.exp(- np.square(diffs) / (2.0 * k_scale))
		elif kernel == 'gaussian':
			k_norm_factor = 1/ np.sqrt(2.0 * np.pi * k_scale)
			k_pdf = lambda diffs: k_norm_factor * np.exp(- np.square(diffs) / (2.0 * k_scale))
		elif kernel == 'uniform':
			k_pdf = partial(exact_unif_1d, width=k_scale)
		else:
			raise ValueError("kernel has to be either gaussian or uniform")
		radius = None
	k_ql_conv = convolve_continuous_grid(f_pdf=k_pdf, g_pdf_grid=q_L_approx, x_grid=grid, radius=radius)
	# print("k_ql_conv ", k_ql_conv.min(), k_ql_conv.max(), k_ql_conv.mean(), k_ql_conv.std())
	k_log_ql_conv = convolve_continuous_grid(f_pdf=k_pdf, g_pdf_grid=np.log(log_eps + q_L_approx), x_grid=grid, radius=radius)
	# print("k_log_ql_conv ", k_log_ql_conv.min(), k_log_ql_conv.max(), k_log_ql_conv.mean(), k_log_ql_conv.std())
	if filter_denom:
		where_pos = np.where(k_ql_conv > pos_threshold)[0]
		return np.exp(k_log_ql_conv[where_pos])/k_ql_conv[where_pos], where_pos
	else:
		return np.exp(k_log_ql_conv)/(denom_eps+k_ql_conv)

# the difference is that you need to give a ready kernel and we will not question it
def _alpha_func_direct_1d(grid, observations, t_scale, k_kernel_arr, dx,
                          log_eps=1e-10, denom_eps=1e-8):
	q_L_approx = compute_kde(x=grid, observations=observations, norm_scale=t_scale, kernel='gaussian')
	k_ql_conv = convolve_cont_grid_1d_with_kernel(base=q_L_approx, kernel=k_kernel_arr) * dx
	k_loq_conv = convolve_cont_grid_1d_with_kernel(base=np.log(log_eps+q_L_approx), kernel=k_kernel_arr) * dx
	return np.exp(k_loq_conv)/(denom_eps+k_ql_conv)

# kde with histograms
def _alpha_func_fast_approx_1d(grid, observations, t_scale, k_kernel_arr, dx,
                            log_eps=1e-10, denom_eps=1e-8):
	q_L_approx = compute_kde_binned_fft(x=grid, observations=observations, norm_scale=t_scale, dx=dx)
	k_ql_conv = convolve_cont_grid_1d_with_kernel(base=q_L_approx, kernel=k_kernel_arr) * dx
	k_loq_conv = convolve_cont_grid_1d_with_kernel(base=np.log(log_eps+q_L_approx), kernel=k_kernel_arr) * dx
	return np.exp(k_loq_conv)/(denom_eps+k_ql_conv)

def _alpha_func_direct_2d(grid, observations, t_scale, k_kernel_arr, dx, log_eps=1e-10, denom_eps=1e-8):
	q_L_approx = compute_kde_2d(x=grid, observations=observations, variance=t_scale)
	k_ql_conv = convolve_cont_grid_2d_with_kernel(base=q_L_approx, kernel=k_kernel_arr) * dx * dx
	k_log_conv = convolve_cont_grid_2d_with_kernel(base=np.log(log_eps+q_L_approx), kernel=k_kernel_arr) * dx * dx
	return np.exp(k_log_conv)/(denom_eps+k_ql_conv)


# TODO change k_pdf into explicit Gaussian computation
def compute_led_kde(grid, observations, t_scale, k_pdf, eps=1e-8):
	q_L_approx = compute_kde(x=grid, observations=observations, norm_scale=t_scale)
	k_log_ql_conv = convolve_continuous_grid(k_pdf, np.log(eps+q_L_approx), grid)
	expklog = np.exp(k_log_ql_conv)
	Z = np.sum(expklog) * (grid[1] - grid[0])
	return expklog / Z

# TODO give the possibility to pass different kernels
# TODO speed up the convolution computation
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

# TODO method: standard does not work very well with 2D. Maybe it's possible to fix.
# TODO add the possibility of precomputing the kernel, no need to recompute it every time
def sample_led_kde(n_samples, observations, grid, t_scale, k_scale,
				   verbose= False, display_every=20, kernel='gaussian',
				   alpha_filter_denom=True, interpolation=True, alpha_log_eps=1e-10,
                   alpha_denom_eps=1e-3,
				   sampling_method='standard', debug=False, alpha_grid=None):
	"""
	Samples observations with the LED-KDE method. If a parameter shares the name with one in alpha_func,
	it is identical to the description there and we will not repeat it here.

	n_samples : number of samples
	verbose : bool, True for printing some progress
	display_every : int will tell you that we sampled k values
	alpha_grid : np.ndarray, you can pass already computed alpha grid, this way computation here will be skipped
	interpolation : bool, whether the values samples will be interpolated to alpha or taken to be equal to values on the grid
				scheme tends to perform better if True
	sampling_method : str, 'standard' samples alpha function with special LED-KDE algorithm,
			'is' samples with importance sampling based on the density computation from alpha
	"""
	N = len(observations)
	ndim = len(observations.shape)
	if debug:
		print(f"Mean: {np.mean(observations):.2f}, std: {np.std(observations):.2f}")
		print(f"Grid shape: {grid.shape}")
	# precompute \alpha here (maybe you can factor it out, it was separated for testing and profiling)
	if alpha_grid is None:
		if alpha_filter_denom:
			alpha_grid, where_pos = alpha_func(grid, observations, t_scale, k_scale,
			                                   filter_denom=alpha_filter_denom)
			grid_pos = grid[where_pos]
		else:
			alpha_grid = alpha_func(grid, observations, t_scale, k_scale, filter_denom=alpha_filter_denom,
			                        denom_eps=alpha_denom_eps, log_eps=alpha_log_eps)
			grid_pos = grid
	else:
		grid_pos = grid
	if np.max(alpha_grid) > 1.01:
		raise StrangeAlphaException(f"something strange happened with alpha_grid, maximum value is {np.max(alpha_grid)}",
									np.max(alpha_grid))
	matcher = match_x_to_grid_and_val_1d if ndim == 1 else match_x_to_grid_and_val_2d
	if sampling_method == 'standard':
		# start = time.perf_counter()
		if ndim == 1 and kernel == 'gaussian':
			k_sampler = norm(loc=0, scale=np.sqrt(k_scale)).rvs
			l_sampler = norm(loc=0, scale=np.sqrt(t_scale)).rvs
			vals = np.zeros(n_samples)
		elif ndim == 1 and kernel == 'uniform':
			k_sampler = uniform(scale=k_scale).rvs
			l_sampler = uniform(scale=t_scale).rvs
			vals = np.zeros(n_samples)
		elif ndim == 2:
			k_sampler = multivariate_normal(mean=[0,0], cov=np.eye(2)*k_scale).rvs
			l_sampler = multivariate_normal(mean=[0,0], cov=np.eye(2)*t_scale).rvs
			vals = np.zeros((n_samples, 2))
		else:
			raise ValueError("ndim, inferred from shape of observations, must be 1 or 2")
		# end = time.perf_counter()
		# print(f"creating samplers take: {1000*(end-start):.3f} ms")

		# start = time.perf_counter()
		for i in range(n_samples):

			reject = True
			while reject:
				I = np.random.randint(N)
				u = l_sampler()
				w = k_sampler()
				R = np.random.uniform()
				X = observations[I] - u - w

				alph_val = matcher(X, grid_pos, alpha_grid, interpolation=interpolation) # technically it works with 2D but not super well
				if R < alph_val:
					vals[i] = X
					reject = False
			if verbose and (1+i) % display_every == 0:
				print(f"Sampled {1+i}/{n_samples} samples.", end="\r")
		# end = time.perf_counter()
		# print(f"Sampling takes: {1000*(end-start):.3f}ms (this part has to be repeated)")

	elif sampling_method == 'is':
		alpha_func_interpolator = partial(matcher, grid=grid_pos, values_grid=alpha_grid, interpolation=True)
		vals, _ = sample_via_importance_resampling(target_pdf=alpha_func_interpolator, n_samples=n_samples)
	else:
		raise ValueError("sampling_method must be either 'standard' or 'is'")
	return vals

# reproduces the function above, but blocks that are repeated often in many runs are taken outside
def sample_led_kde_direct_raw(n_samples, observations, grid, t_scale,
                              t_sampler, k_kernel_arr, k_sampler, dx,
                              alpha_log_eps=1e-10, alpha_denom_eps=1e-3,
                              interpolation=True, alpha_func_obj=_alpha_func_direct_1d,
                              matcher_fn=match_x_to_grid_and_val_1d):
	alpha_grid = alpha_func_obj(grid=grid, observations=observations,
                               t_scale=t_scale, k_kernel_arr=k_kernel_arr,
                               log_eps=alpha_log_eps, denom_eps=alpha_denom_eps, dx=dx)
	if np.max(alpha_grid) > 1.01:
		raise StrangeAlphaException(f"something strange happened with alpha_grid, maximum value is {np.max(alpha_grid)}",
									np.max(alpha_grid))
	vals = np.zeros(n_samples)
	N = len(observations)
	for i in range(n_samples):
		reject = True
		while reject:
			I = np.random.randint(N)
			u = t_sampler()
			w = k_sampler()
			R = np.random.uniform()
			X = observations[I] - u - w

			alph_val = matcher_fn(X, grid, alpha_grid,
			                                interpolation=interpolation)  # technically it works with 2D but not super well
			if R < alph_val:
				vals[i] = X
				reject = False

	return vals