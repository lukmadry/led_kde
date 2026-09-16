import numpy as np
import math
from functools import partial
from numbers import Number

from scipy.stats import norm

from tools import convolve_continuous_grid, match_x_to_grid_and_val, exact_gaussian_1d

def compute_kde(x, observations, norm_scale):
	N = len(observations)
	val = 0
	for i in range(N):
		val += exact_gaussian_1d(x-observations[i], var=norm_scale)
	return val / N

# maybe it's possible to speed it up even more
def compute_kde_2d(x, observations, variance,
                    batch_size=1000):
	if len(observations.shape) != 2:
		raise ValueError("observations must be 2D")

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

	return np.asarray(estimates_on_x, float)

def compute_kde_mc():
	pass

def compute_kde_2d_mc():
	pass

def compute_kde_scott(x, observations):
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
		vals.append(match_x_to_grid_and_val(v, grid, kde))
	return vals

def kde_sampled_pdf_func(grid, kde):
	return partial(kde_sampled_pdf_wrapper, grid=grid, kde=kde)

def alpha_func(grid, observations, t_scale, k_pdf, eps=1e-8,
               filter_denom=True, denom_eps=1e-3, pos_threshold=1e-6):
	q_L_approx = compute_kde(x=grid, observations=observations, norm_scale=t_scale)
	k_ql_conv = convolve_continuous_grid(f_pdf=k_pdf, g_pdf_grid=q_L_approx, x_grid=grid)
	k_log_ql_conv = convolve_continuous_grid(f_pdf=k_pdf, g_pdf_grid=np.log(eps + q_L_approx), x_grid=grid)
	if filter_denom:
		where_pos = np.where(k_ql_conv > pos_threshold)[0]
		return np.exp(k_log_ql_conv[where_pos])/k_ql_conv[where_pos], where_pos
	else:
		return np.exp(k_log_ql_conv)/(denom_eps+k_ql_conv)

# TODO change k_pdf into explicit Gaussian computation
def compute_led_kde(grid, observations, t_scale, k_pdf, eps=1e-8):
	q_L_approx = compute_kde(x=grid, observations=observations, norm_scale=t_scale)
	k_log_ql_conv = convolve_continuous_grid(k_pdf, np.log(eps+q_L_approx), grid)
	expklog = np.exp(k_log_ql_conv)
	Z = np.sum(expklog) * (grid[1] - grid[0])
	return expklog / Z

# TODO give the possibility to pass different kernels
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

def sample_led_kde(n_samples, observations, grid, t_scale, k_pdf,
				   k_sampler, l_sampler, verbose=True, display_every=20, alpha_filter_denom=True):
	N = len(observations)

	# precompute \alpha here
	alpha_grid, where_pos = alpha_func(grid, observations, t_scale, k_pdf, filter_denom=alpha_filter_denom)
	vals = np.zeros(n_samples)
	grid_pos = grid[where_pos]

	# this can be presumably parallelized, although not by that much on my computer without GPU, some far away TODO would be maybe to try and write a kernel for this
	for i in range(n_samples):

		reject = True
		while reject:
			I = np.random.randint(N)
			u = l_sampler()
			w = k_sampler()
			R = np.random.uniform()
			X = observations[I] - u - w

			alph_val = match_x_to_grid_and_val(X, grid_pos, alpha_grid) # this part does not work with more than one dimension yet
			if R < alph_val:
				vals[i] = X
				reject = False

		if verbose and (1+i) % display_every == 0:
			print(f"Sampled {1+i}/{n_samples} samples.", end="\r")

	return vals