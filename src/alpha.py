from functools import partial

import numpy as np

from src.kde import compute_kde_2d, compute_kde, compute_kde_gaussian_1d_numba, compute_kde_binned_fft, \
	compute_kde_gaussian_2d_numba, fast_fft_kde_2d_standalone
from src.tools import exact_gaussian_1d_cutoff, exact_unif_1d, convolve_continuous_grid, \
	convolve_cont_grid_1d_with_kernel, convolve_cont_grid_2d_with_kernel


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


def _alpha_func_direct_1d(grid, observations, t_scale, k_kernel_arr, dx,
                          log_eps=1e-10, denom_eps=1e-8):
	q_L_approx = compute_kde(x=grid, observations=observations, norm_scale=t_scale, kernel='gaussian')
	k_ql_conv = convolve_cont_grid_1d_with_kernel(base=q_L_approx, kernel=k_kernel_arr) * dx
	k_loq_conv = convolve_cont_grid_1d_with_kernel(base=np.log(log_eps+q_L_approx), kernel=k_kernel_arr) * dx
	return np.exp(k_loq_conv)/(denom_eps+k_ql_conv)


def _alpha_func_direct_1d_numba(grid, observations, t_scale, k_kernel_arr, dx,
                          log_eps=1e-10, denom_eps=1e-8):
	q_L_approx = compute_kde_gaussian_1d_numba(x=grid, observations=observations, var=t_scale)
	k_ql_conv = convolve_cont_grid_1d_with_kernel(base=q_L_approx, kernel=k_kernel_arr) * dx
	k_loq_conv = convolve_cont_grid_1d_with_kernel(base=np.log(log_eps+q_L_approx), kernel=k_kernel_arr) * dx
	return np.exp(k_loq_conv)/(denom_eps+k_ql_conv)


def _alpha_func_fast_approx_1d(grid, observations, t_scale, k_kernel_arr, dx,
                            log_eps=1e-10, denom_eps=1e-8):
	q_L_approx = compute_kde_binned_fft(x=grid, observations=observations, norm_scale=t_scale, dx=dx)
	k_ql_conv = convolve_cont_grid_1d_with_kernel(base=q_L_approx, kernel=k_kernel_arr) * dx
	k_loq_conv = convolve_cont_grid_1d_with_kernel(base=np.log(log_eps+q_L_approx), kernel=k_kernel_arr) * dx
	return np.exp(k_loq_conv)/(denom_eps+k_ql_conv)


def _alpha_func_direct_2d(grid, observations, t_scale, k_kernel_arr, dx, log_eps=1e-10, denom_eps=1e-8):
	q_L_approx = compute_kde_2d(x=grid, observations=observations, variance=t_scale)
	k_ql_conv = convolve_cont_grid_2d_with_kernel(base=q_L_approx, kernel=k_kernel_arr) * dx * dx
	k_log_conv = convolve_cont_grid_2d_with_kernel(base=np.log(log_eps+q_L_approx),
	                                               kernel=k_kernel_arr, val_to_pad=np.log(log_eps)) * dx * dx
	return np.exp(k_log_conv)/(denom_eps+k_ql_conv)


def _alpha_func_direct_2d_numba(grid, observations, t_scale, k_kernel_arr, dx, log_eps=1e-10, denom_eps=1e-8):
	q_L_approx = compute_kde_gaussian_2d_numba(x=grid, observations=observations, variance=t_scale)
	k_ql_conv = convolve_cont_grid_2d_with_kernel(base=q_L_approx, kernel=k_kernel_arr) * dx * dx
	k_log_conv = convolve_cont_grid_2d_with_kernel(base=np.log(log_eps+q_L_approx),
	                                               kernel=k_kernel_arr, val_to_pad=np.log(log_eps)) * dx * dx
	return np.exp(k_log_conv)/(denom_eps+k_ql_conv)


def _alpha_func_direct_2d_fast_kde(grid, observations, t_scale, k_kernel_arr, dx, log_eps=1e-10, denom_eps=1e-8):
	q_L_approx = fast_fft_kde_2d_standalone(grid=grid, observations=observations, bandwidth=np.sqrt(t_scale))
	k_ql_conv = convolve_cont_grid_2d_with_kernel(base=q_L_approx, kernel=k_kernel_arr) * dx * dx
	k_log_conv = convolve_cont_grid_2d_with_kernel(base=np.log(log_eps+q_L_approx), kernel=k_kernel_arr, val_to_pad=np.log(log_eps)) * dx * dx
	return np.exp(k_log_conv) / (denom_eps + k_ql_conv)
