from numbers import Number
import time # for profiling

import numpy as np
from scipy import integrate
from scipy.stats import entropy, norm
from scipy.signal import fftconvolve
from scipy.interpolate import interpn, make_interp_spline
from scipy.special import erf
from scipy.integrate import cumulative_trapezoid


# TODO speed up convolutions by discarding values that are essentially zero, may need more specific variant of the code

try:
	import git
except ImportError:
	print("could not import git package")

def none_sentinel(arg1, arg2):
	if arg1 is None and arg2 is None:
		raise ValueError
	if arg1 is not None and arg2 is not None:
		raise ValueError

# checks if all the parameters are consistent with desired dimension
def run_dimension_consistency_check(args):
	dim_one_not_none = 0
	dim_two_not_none = 0

	for arg in [args.centers, args.scales, args.widths]:
		if arg is not None:
			dim_one_not_none += 1
	for arg in [args.centers_x, args.centers_y, args.scales_x, args.scales_y, args.widths_x, args.widths_y]:
		if arg is not None:
			dim_two_not_none += 1

	if dim_one_not_none > 0 and dim_two_not_none > 0:
		raise ValueError(f"you passed arguments for both one and two dimensions. "
						 f"One dimension: centers: {args.centers}, scales: {args.scales}, widths: {args.widths}"
						 f"Two dimensions: centers: {args.centers_x, args.centers_y}, scales: {args.scales_x, args.scales_y}, "
						 f"widths: {args.widths_x, args.widths_y}")
	elif dim_one_not_none > 0 and dim_two_not_none == 0:
		return 1
	elif dim_two_not_none > 0 and dim_one_not_none == 0:
		return 2
	elif dim_one_not_none == 0 and dim_two_not_none == 0:
		raise ValueError("you did not pass arguments for any of possible dimensions.")

def validate_distribution_args(args, dim):
	arguments_passed = 0
	if args.dist_type is not None:
		arguments_passed += 1
	if args.spiral:
		arguments_passed += 1
	if args.chessboard:
		arguments_passed += 1
	if arguments_passed != 1:
		raise ValueError(f"only one of dist_type, spiral, chessboard can be passed, yet you passed {arguments_passed}. "
		                 f"By default spiral and chessboard are set to false. You passed: {args.dist_type}, {args.spiral}, {args.chessboard}")

	if dim == 1 and (args.spiral or args.chessboard):
		raise ValueError(f"Spiral and chessboard are only available in dimension one, "
		                 f"set dist_type to either gaussian or uniform")

def exact_gaussian_1d(x, var):
	"""
	computes 1/sqrt(2 * \pi * var) * exp( - x^2 / (2* var) ).
	Speeds up the computation with respect to using scipy.stats.norm(loc=0, scale=np.sqrt(var)).pdf

	Note that it will be applied componentwise, so that if you want to pass (N, N, 2) where each (i, j, :) is a point
	in 2D, you will get (N, N, 2), not (N, N) with a sum of squares inside exp

	x : np.ndarray of any size
	var : variance of a Gaussian
	"""
	return 1/np.sqrt(2.0*np.pi*var) * np.exp(-np.square(x)/(2.0*var))

def exact_gaussian_1d_cutoff(x, var):
	"""
	Was supposed to speed up the computation even more than exact_gaussian_1d, but you need to assign zeros
	Experimental (23/09)
	"""
	# TODO add normalisation, right now it's not a probability distribution
	far_bool = np.abs(x) > 50*var
	sddev = np.sqrt(2.0*np.pi*var)
	close_vals = 1/sddev * np.exp(-np.square(x[~far_bool])/(2.0*var))
	u = np.zeros_like(x)
	u[~far_bool] = close_vals
	return u

def exact_gaussian_2d(x, var):
	"""

	x: np.ndarray (N, K)
	var: float

	returns np.ndarray (N, K)
	"""

	norm_factor = 1.0 / (2.0 * np.pi * var)
	kernel = x / (-2.0 * var)
	np.exp(kernel, out=kernel)
	kernel *= norm_factor
	return kernel

def exact_unif_1d(x, width):
	return np.where(x > -width & x < width, 2/width, 0)

def _interpret_two_dim_grid(grid):
	if isinstance(grid, (tuple, list)) and len(grid) == 2:  # the case of two 1D grids
		x, y = grid
		X, Y = np.meshgrid(x, y)
		f_grid = np.dstack([X, Y])
	else:
		f_grid = np.asarray(grid, float)
		x = f_grid[0, :, 0]
		y = f_grid[:, 0, 1]
	return f_grid, x, y

def match_x_to_grid_and_val_1d(x, grid, values_grid,
							raise_error_if_outside=False, interpolation=True):
	"""
	for given x, finds the closest values in the grid and returns matching value of a function.
	You can choose to interpolate between two values in the neighbouring indices

	x : float or ndarray, value to match
	grid : np.ndarray, a grid with even spaces (this assumption is NOT checked, but is crucial)
	values_grid : values of the function. Has to have the same shape as grid
	interpolation : bool if False we will give the value matching the index on lhs.
	"""
	grid_step = grid[1] - grid[0]
	grid_min = grid[0]
	grid_max = grid[-1]

	sm = x < grid_min
	bg = x >= grid_max
	if raise_error_if_outside and np.any(sm | bg):
		raise ValueError("x contains elements outside of grid bounds")

	idx = np.array( (x - grid_min) / grid_step, dtype=np.int32)
	idx = np.clip(idx, 0, len(grid) - 2) # this is not optimal, but in our cases of interest \alpha is zero at the edges

	if interpolation: # to be checked, maybe write a test
		weights = x - grid[idx]
		return values_grid[idx] * (1-weights) + values_grid[idx+1] * weights

	return values_grid[idx]

# wrapper for splines from scipy
def match_x_to_grid_with_cubic_1d(x, grid, values_grid, *args, **kwargs):
	splines = make_interp_spline(grid, values_grid)
	return splines(x)

# I think that no interpolation variant will have to be deprecated
def match_x_to_grid_and_val_2d(x, grid, values_grid,
							   raise_error_if_outside=False, method='cubic', interpolation=True):
	"""
	grid :
	"""

	_, xlinspace, ylinspace = _interpret_two_dim_grid(grid)
	if interpolation:
		return interpn(points=(xlinspace, ylinspace), values=values_grid,
		               xi=x, method=method, bounds_error=False) # TODO presumably this wastes time on binary search as well
	else:
		raise NotImplementedError
#
# def match_x_to_grid_and_val(x, grid, values_grid, interpolation):
# 	if isinstance(x, Number) or (isinstance(x, np.ndarray) and ):
# 		return match_x_to_grid_and_val_1d(x, grid, values_grid, raise_error_if_outside=False, interpolation=interpolation)
# 	elif x.shape[0] == 2:
# 		return match_x_to_grid_and_val_2d(x, grid, values_grid, interpolation=interpolation, raise_error_if_outside=False)
# 	else:
# 		raise NotImplementedError("x has to be one or two-dimensional")

def compute_gaussian_kernel_1d(radius, dx, var):
	n = int(np.round(2.0 * radius / dx)) | 1 # bitwise or, moves up to odd value if even, does nothing if already odd
	vals = np.linspace(-radius, radius, num=n)
	return exact_gaussian_1d(vals, var)

def compute_kernel_2d(grid, radius, var, func, ret_ds=False):

	x_grid, x1_1d, x2_1d = _interpret_two_dim_grid(grid)

	y1_1d = x1_1d
	y2_1d = x2_1d

	dx1 = x1_1d[1] - x1_1d[0]
	dx2 = x2_1d[1] - x2_1d[0]
	dy1 = y1_1d[1] - y1_1d[0]
	dy2 = y2_1d[1] - y2_1d[0]

	n1 = int(2*radius / dx1) | 1
	n2 = int(2*radius / dx2) | 1
	d1 = np.linspace(
		-radius, radius, num=n1
	)
	d2 = np.linspace(
		-radius, radius, num=n2
		)
	S = np.empty((n1, n2), dtype=np.float64)
	S[:] = np.square(d1)[None, :]
	S += np.square(d2)[:, None]
	kernel_arr = func(S, var)
	if ret_ds:
		return kernel_arr, dy1, dy2
	return kernel_arr

# wrapper
def compute_gaussian_kernel_2d(grid, radius, var):
	return compute_kernel_2d(grid, radius, var, exact_gaussian_2d, ret_ds=False)


def compute_uniform_kernel(radius):
	pass

# TODO to be sped up for the kernel with small support
# TODO check if you can use scipy.ndimage instead, it's probably optimised beyond what you can obtain here
def convolve_continuous_grid_1d_fft(f_pdf, g_pdf_grid, x_grid, radius=None):
	"""


	"""

	N = len(x_grid)
	dx = x_grid[1] - x_grid[0]
	# start = time.perf_counter()
	if radius is None:
		diffs = np.arange(-(N-1)*dx, (N-1)*dx + 0.5*dx, step=dx)
	else:
		diffs = np.arange(-radius, radius, step=dx)
	f_grid_conv = f_pdf(diffs)
	# end = time.perf_counter()
	# print(f"time to create a kernel: {1000*(end - start):.3f}ms")
	padding_size = len(f_grid_conv) // 2
	padded_base = np.pad(
		g_pdf_grid,
		(padding_size, padding_size),
		mode='constant',
		constant_values=(g_pdf_grid[0], g_pdf_grid[-1]),
	)
	return fftconvolve(padded_base, f_grid_conv, mode='valid') * dx

def convolve_cont_grid_1d_with_kernel(base, kernel):
	"""
	wrapper for fftconvolve with custom padding

	base: np.ndarray, (N, )
	kernel: np.ndarray, (K, ) float
	"""
	padding_size = len(kernel) // 2
	padded_base = np.pad(
		base,
		(padding_size, padding_size),
		mode='constant',
		constant_values=(base[0], base[-1]),
	)

	return fftconvolve(padded_base, kernel, mode='valid')

# TODO once kernel option is done I think you can get rid of this?
def convolve_continuous_grid_2d_fft(f_pdf, g_pdf_grid, x_grid, radius=None, val_to_pad=0):
	f_grid, dy1, dy2 = compute_kernel_2d(x_grid, radius, val_to_pad, f_pdf, ret_ds=True)
	padding_size = len(f_grid) // 2
	padded_base = np.pad(g_pdf_grid, ((padding_size, padding_size), ),
						 mode='constant', constant_values=val_to_pad)

	conv_result = fftconvolve(padded_base, f_grid, mode="valid") * dy1 * dy2
	return conv_result

def convolve_cont_grid_2d_with_kernel(base, kernel, val_to_pad=0):
	"""
	wrapper for fftconvolve with custom padding for 2D

	"""
	padding_size = len(kernel) // 2
	padded_base = np.pad(base, ((padding_size, padding_size),),
	                     mode='constant', constant_values=val_to_pad)
	conv_result = fftconvolve(padded_base, kernel, mode='valid')
	return conv_result


def convolve_continuous_grid(f_pdf, g_pdf_grid, x_grid, radius=None):
	# will redirect to a matching implementation of convolve_... depending on the dimensions of x_grid
	if len(x_grid.shape) == 1:
		return convolve_continuous_grid_1d_fft(f_pdf=f_pdf, g_pdf_grid=g_pdf_grid, x_grid=x_grid, radius=radius)
	elif isinstance(x_grid, (tuple, list)) and len(x_grid) == 2 or len(x_grid.shape) == 3:
		return convolve_continuous_grid_2d_fft(f_pdf=f_pdf, g_pdf_grid=g_pdf_grid, x_grid=x_grid, radius=radius)
	else:
		raise NotImplementedError("convolution on more than two dimensions is not implemented and likely won't be")

def sample_via_importance_resampling(
	target_pdf,
	proposal_pdf=None,
	proposal_sampler=None,
	n_candidates=100_000,
	n_samples=10_000,
):
	"""
	Draws unweighted samples from target_pdf using Sampling-Importance Resampling (SIR).

	Parameters:
		target_pdf: callable, p(x) returning density values for shape (N, D) or (N,)
		proposal_pdf: callable, q(x) returning density values for shape (N, D) or (N,). if None then defaults to normal(0, 5)
		proposal_sampler: callable, draws N samples from q, returning shape (N, D) or (N,)
		n_candidates: int, number of initial draws from proposal q
		n_samples: int, final number of unweighted samples desired from p

	Returns:
		unweighted_samples: array of shape (n_samples, D)
		ess: float, Effective Sample Size before resampling
	"""
	# 1. Sample candidate points from proposal q
	if proposal_pdf is None or proposal_sampler is None:
		proposal_pdf = norm(loc=0, scale=5).pdf
		proposal_sampler = norm(loc=0, scale=5).rvs

	candidates = proposal_sampler(n_candidates)

	# 2. Evaluate likelihood ratios p(x) / q(x)
	p_vals = np.asarray(target_pdf(candidates), dtype=float)
	p_vals = np.maximum(p_vals, 0) # not sure why but there was once underflow of type -1e300. you may be hiding some actual problem
	q_vals = np.asarray(proposal_pdf(candidates), dtype=float)
	q_vals = np.maximum(q_vals, 0)

	# Stabilise log weights against underflow/overflow
	log_w = np.log(p_vals + 1e-300) - np.log(q_vals + 1e-300)
	weights = np.exp(log_w - np.max(log_w))
	norm_weights = weights / np.sum(weights)

	# 3. Effective Sample Size check
	ess = 1.0 / np.sum(norm_weights**2)

	# 4. Multinomial Resampling: draw indices with replacement based on norm_weights
	resampled_indices = np.random.choice(
		n_candidates, size=n_samples, replace=True, p=norm_weights
	)

	return candidates[resampled_indices], ess


def compute_l2_wasserstein_1d(obs_x, obs_y, num_pts):
	if obs_x.ndim > 1 or obs_y.ndim > 1:
		raise ValueError("you passed multidimensional arrays to l2_wasserstein_1d, this will result in nonsense values")
	q = np.linspace(0, 1, num=num_pts)
	dx = q[1] - q[0]
	qx = np.quantile(obs_x, q)
	qy = np.quantile(obs_y, q)
	return np.sqrt(np.trapezoid(y=np.square(qx - qy), x=q, dx=dx))

def wasserstein_from_pdfs_1d(pdf_p, pdf_q, grid, num_quantiles=1000):

	p_norm = pdf_p / np.trapezoid(pdf_p, grid)
	q_norm = pdf_q / np.trapezoid(pdf_q, grid)

	cdf_p = cumulative_trapezoid(p_norm, grid, initial=0.0)
	cdf_q = cumulative_trapezoid(q_norm, grid, initial=0.0)
	cdf_p /= cdf_p[-1]
	cdf_q /= cdf_q[-1]

	q = np.linspace(0.0, 1.0, num_quantiles)
	inv_cdf_p = np.interp(q, cdf_p, grid)
	inv_cdf_q = np.interp(q, cdf_q, grid)

	return np.sqrt(np.trapezoid((inv_cdf_p - inv_cdf_q)**2, q))

# apparently this is a lower bound for wasserstein, not a true L2 wasserstein
# the two below can be computed together maybe ?
# you could refactor it to "projection_estimation" method or something
def sliced_wasserstein_2d(samples_a, samples_b, num_projections=100, num_pts=1000):

	angles = np.random.uniform(0, 2 * np.pi, num_projections)
	projections = np.stack([np.cos(angles), np.sin(angles)], axis=1)

	sw2_sum = 0.0
	for proj in projections:

		proj_a = samples_a @ proj
		proj_b = samples_b @ proj

		w1d = compute_l2_wasserstein_1d(proj_a, proj_b, num_pts)
		sw2_sum += w1d**2

	return np.sqrt(sw2_sum / num_projections)

def estimate_bounding_radius_2d(samples, num_projections):
	angles = np.random.uniform(0, 2 * np.pi, num_projections)
	projections = np.stack([np.cos(angles), np.sin(angles)], axis=1)
	max_support = 0

	for proj in projections:

		proj_theta = samples @ proj

		s_width_projection = support_width(proj_theta)
		max_support = max(max_support, s_width_projection)

	return max_support

def compute_kl_divergence(obs_x, obs_y, nr_bins=50, eps=1e-8):
	min_obs = min(np.min(obs_x), np.min(obs_y))
	max_obs = max(np.max(obs_x), np.max(obs_y))
	bin_edges = np.linspace(min_obs, max_obs, nr_bins)

	p_x = np.histogram(obs_x, bin_edges)[0]
	p_y = np.histogram(obs_y, bin_edges)[0]

	p_x = p_x.astype(float) + eps
	p_y = p_y.astype(float) + eps

	p_x /= p_x.sum()
	p_y /= p_y.sum()

	return np.sum(p_x * np.log(p_x / p_y))

def convert_planar_to_xy(radius, angle):
	x = radius * np.cos(angle)
	y = radius * np.sin(angle)
	return x, y

def get_git_commit_gitpython():
	try:
		repo = git.Repo(search_parent_directories=True)
		sha = repo.head.object.hexsha
		is_dirty = repo.is_dirty()
		return {"commit_hash": sha[:7], "is_dirty": is_dirty}
	except git.InvalidGitRepositoryError:
		return None

# statistics of distributions that we report
def support_width(observations):
	return np.max(observations) - np.min(observations)

def support_width_axis(observations, axis=0):
	return np.max(observations[:, axis]) - np.min(observations[:, axis])

def descriptive_fns_sentinel(metric_list, dimension):
	revised_metric_list = []
	if dimension == 1:
		for m in metric_list:
			if '_ax' in m:
				print("In a one-dimensional setting we will not need axis-aware metrics, removing ", m)
			else:
				revised_metric_list.append(m)
	if dimension == 2:
		for m in metric_list:
			if '_ax' not in m:
				revised_metric_list.extend([m + '_ax0', m + '_ax1'])

	return revised_metric_list

def comparative_fns_sentinel(metric_list, dimension):
	revised_metric_list = []
	if dimension == 1:
		for m in metric_list:
			if not m.endswith('2d'):
				revised_metric_list.append(m)
	elif dimension == 2:
		for m in metric_list:
			if m.endswith('___density'):
				print("Density based metrics are not supported for 2d")
				continue
			elif 'wasserstein' in m and '2d' not in m:
				revised_metric_list.append('sliced_wasserstein_2d')
			else:
				revised_metric_list.append(m)
	return revised_metric_list

def nr_of_peaks(pdf_on_a_grid):
	pass

class StrangeAlphaException(Exception):
	def __init__(self, message, alpha_value):
		super().__init__(message)
		self.alpha_value = alpha_value
		print(message)

class NegativeProbabilityError(Exception):
	def __init__(self, message, min_prob_value):
		super().__init__(message)
		self.min_prob_value = min_prob_value
		print(message)