from numbers import Number
import time # for profiling

import numpy as np
from scipy import integrate
from scipy.stats import entropy, norm
from scipy.signal import fftconvolve
from scipy.interpolate import interpn
from scipy.special import erf

# TODO speed up convolutions by discarding values that are essentially zero, may need more specific variant of the code

try:
	import git
except ImportError:
	print("could not import git package")


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

def exact_gaussian_1d(x, var):
	"""
	Speeds up the computation with respect to using scipy.stats.norm(loc=0, scale=np.sqrt(var)).pdf
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
	return 1/(2.0*np.pi*var) * np.exp(-np.sum(np.square(x))/(2.0*var))

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

def print_statistics(local_pdf, local_samples, metrics, func_dict, i, verbosity_freq):
	"""
	Helper function used in the main pass to print statistics of the current distribution.

	Return a dictionary with values of statistics of pdf and samples.

	local_pdf : np.ndarray of pdf
	local_samples : np.ndarray of samples
	func_dict : dictionary of format str : Callable . If string ends with ___density then
			Callable will be applied to local_pdf, if not it will be applied to local_samples
	metrics : iterable with strings of metrics names. Has to be a subset of keys in func_dict
	i : index to display
	verbosity_freq : dictates verbosity, if 1+i is divisible by this value, will print descriptive statistics
	"""
	stat = {}
	for s in metrics:
		if s.endswith("___density"):
			new_name = s.split("___")[0] + " of density"
			stat[new_name] = func_dict[s](local_pdf)
		else:
			stat[s] = func_dict[s](local_samples)

	if (1 + i) % verbosity_freq == 0:
		s = f"Trial nr: {1 + i}.\nDescriptive statistics: "
		for key, val in stat.items():
			s += f"{key}: {val:.3f} |"
		s += "\n"
		print(s)

	return stat

# AI use : some (most) of these functions have been written with Gemini Flash
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

    if interpolation:
        return np.interp(x, grid, values_grid)

    idx = np.array( (x - grid_min) / grid_step, dtype=np.int32)
    idx = np.clip(idx, 0, len(grid) - 1)
    return values_grid[idx]

# I think that no interpolation variant will have to be deprecated
def match_x_to_grid_and_val_2d(x, grid, values_grid,
                               raise_error_if_outside=False, interpolation=True):
	"""
	grid :
	"""

	_, xlinspace, ylinspace = _interpret_two_dim_grid(grid)
	if interpolation:
		return interpn(points=(xlinspace, ylinspace), values=values_grid, xi=x)
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
	vals = np.arange(-radius, radius, step=dx)
	return exact_gaussian_1d(vals, var)

def compute_gaussian_kernel_2d(radius):
	pass

def compute_uniform_kernel(radius):
	pass

# TODO to be sped up for the kernel with small support
# TODO check if you can use scipy.ndimage instead, it's probably optimised beyond what you can obtain here
def convolve_continuous_grid_1d_fft(f_pdf, g_pdf_grid, x_grid, radius=None):
	"""


	"""

	N = len(x_grid)
	dx = x_grid[1] - x_grid[0]
	start = time.perf_counter()
	if radius is None:
		diffs = np.arange(-(N-1)*dx, (N-1)*dx + 0.5*dx, step=dx)
	else:
		diffs = np.arange(-radius, radius, step=dx)
	f_grid_conv = f_pdf(diffs)
	end = time.perf_counter()
	print(f"time to create a kernel: {1000*(end - start):.3f}ms")
	padding_size = len(f_grid_conv) // 2
	padded_base = np.pad(
		g_pdf_grid,
		(padding_size, padding_size),
		mode='constant',
		constant_values=(g_pdf_grid[0], g_pdf_grid[-1]),
	)
	return fftconvolve(padded_base, f_grid_conv, mode='valid') * dx

def convolve_cont_grid_1d_with_kernel(base, kernel):
	padding_size = len(kernel) // 2
	padded_base = np.pad(
		base,
		(padding_size, padding_size),
		mode='constant',
		constant_values=(base[0], base[-1]),
	)
	return fftconvolve(padded_base, kernel, mode='valid')

# TODO give the possibility to pass the kernel
def convolve_continuous_grid_2d_fft(f_pdf, g_pdf_grid, x_grid, radius=None, val_to_pad=0):
	x_grid, x1_1d, x2_1d = _interpret_two_dim_grid(x_grid)

	y1_1d = x1_1d
	y2_1d = x2_1d

	# artefacts of the old code with y_bounds
	dx1 = x1_1d[1] - x1_1d[0]
	dx2 = x2_1d[1] - x2_1d[0]
	dy1 = y1_1d[1] - y1_1d[0]
	dy2 = y2_1d[1] - y2_1d[0]

	if radius is None:
		d1 = np.arange(
			-(len(y1_1d)-1)*dy1, (len(x1_1d)-1)*dx1 + 0.5*dx1, step=dx1
		)
		d2 = np.arange(
			-(len(y2_1d)-1)*dy2, (len(x2_1d)-1)*dx2 + 0.5*dx2, step=dx2
		)
	else:
		d1 = np.arange(
			-radius, radius + dx1, step=dx1
		)
		d2 = np.arange(
			-radius, radius + dx2, step=dx2
		)
	D1, D2 = np.meshgrid(d1, d2)
	diff_stack = np.dstack([D1, D2])
	f_grid = f_pdf(diff_stack)
	padding_size = len(f_grid) // 2
	padded_base = np.pad(g_pdf_grid, ((padding_size, padding_size), ),
	                     mode='constant', constant_values=val_to_pad)

	conv_result = fftconvolve(padded_base, f_grid, mode="valid") * dy1 * dy2
	return conv_result

def convolve_cont_grid_2d_with_kernel():
	pass

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
	print(candidates.shape)
	p_vals = np.asarray(target_pdf(candidates), dtype=float)
	q_vals = np.asarray(proposal_pdf(candidates), dtype=float)

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

def kl_divergence_observations(x, y, bins=50, eps=1e-12):
	"""Computes the KL divergence D_KL(P || Q) between two sets of continuous observations

	x and y by binning them into a shared histogram.

	Parameters:
		x: 1D array-like, observations from target distribution P
		y: 1D array-like, observations from reference distribution Q
		bins: int or sequence, number of histogram bins or explicit bin edges
		eps: float, small value added to avoid zero probabilities (q(i) = 0)

	Returns:
		kl_div: float, D_KL(P || Q)
		edges: 1D array, the bin edges used for the shared histogram
	"""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)

	# 1. Determine common bin edges across both datasets to align supports
	if isinstance(bins, int):
		min_val = min(np.min(x), np.min(y))
		max_val = max(np.max(x), np.max(y))
		bin_edges = np.linspace(min_val, max_val, bins + 1)
	else:
		bin_edges = np.asarray(bins)

	# Compute raw frequency counts
	hist_x, _ = np.histogram(x, bins=bin_edges)
	hist_y, _ = np.histogram(y, bins=bin_edges)

	# 2. Add epsilon smoothing to handle empty bins and convert to valid probability distributions
	p = hist_x.astype(float) + eps
	q = hist_y.astype(float) + eps

	p /= np.sum(p)
	q /= np.sum(q)

	# 3. Compute KL divergence using scipy.stats.entropy
	kl_div = entropy(p, q)

	return kl_div, bin_edges

def kl_divergence_samples_vs_grid(
	samples, grid, q_pdf, eps=1e-12, normalize_q=True
):
	"""Computes D_KL(P || Q) where P is given by 1D continuous samples

	and Q is given by a pre-evaluated density grid.

	Parameters:
		samples: 1D array-like, random observations from P.
		grid: 1D array-like, strictly increasing uniform linspace grid.
		q_pdf: 1D array-like, density values q(z) evaluated on grid.
		eps: float, small value added for zero-probability smoothing.
		normalize_q: bool, if True, integrates q_pdf over the grid to form a valid PMF.

	Returns:
		float: D_KL(P || Q) in nats.
	"""
	samples = np.asarray(samples, dtype=float)
	grid = np.asarray(grid, dtype=float)
	q_pdf = np.asarray(q_pdf, dtype=float)

	a, b = grid[0], grid[-1]
	num = len(grid)
	step = (b - a) / (num - 1)

	# 1. Filter samples that fall within grid domain [a, b]
	valid_samples = samples[(samples >= a) & (samples <= b)]
	if len(valid_samples) < len(samples):
		print(
			f"Warning: {len(samples) - len(valid_samples)} samples out of grid bounds [{a}, {b}]."
		)

	# 2. Bin samples onto the grid using O(1) linear index mapping
	# Compute bin indices for each sample
	bin_indices = np.clip(
		np.round((valid_samples - a) / step).astype(int), 0, num - 1
	)

	# Count empirical frequency for distribution P
	hist_p = np.bincount(bin_indices, minlength=num).astype(float)

	# 3. Convert q_pdf into discrete probabilities (PMF) over the grid intervals
	if normalize_q:
		q_pmf = q_pdf * step
	else:
		q_pmf = q_pdf.copy()

	# 4. Apply epsilon smoothing and normalize P and Q to sum to 1.0
	p_pmf = hist_p + eps
	q_pmf = q_pmf + eps

	p_pmf /= np.sum(p_pmf)
	q_pmf /= np.sum(q_pmf)

	# 5. Compute KL divergence using SciPy
	return entropy(p_pmf, q_pmf)


def kl_divergence_grid_vs_grid(grid, p_pdf, q_pdf, eps=1e-12):
	"""Computes D_KL(P || Q) between two continuous distributions given as density

	values evaluated on the exact same uniform grid.

	Parameters:
		grid: 1D array-like, strictly increasing uniform linspace grid
		p_pdf: 1D array-like, density values p(z) evaluated on grid (target P)
		q_pdf: 1D array-like, density values q(z) evaluated on grid (reference Q)
		eps: float, small smoothing value added to ensure numerical stability

	Returns:
		float: D_KL(P || Q) in nats
	"""
	grid = np.asarray(grid, dtype=float)
	p_pdf = np.asarray(p_pdf, dtype=float)
	q_pdf = np.asarray(q_pdf, dtype=float)

	# 1. Compute grid step size (dz)
	dz = (grid[-1] - grid[0]) / (len(grid) - 1)

	# 2. Convert continuous densities to discrete probability mass functions (PMFs)
	# P_i = p(z_i) * dz
	p_pmf = p_pdf * dz
	q_pmf = q_pdf * dz

	# 3. Apply epsilon smoothing for numerical stability
	p_pmf = p_pmf + eps
	q_pmf = q_pmf + eps

	# 4. Re-normalize to ensure valid probability distributions summing to 1.0
	p_pmf /= np.sum(p_pmf)
	q_pmf /= np.sum(q_pmf)

	# 5. Compute KL divergence using SciPy
	return entropy(p_pmf, q_pmf)

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

def nr_of_peaks(pdf_on_a_grid):
	pass

class StrangeAlphaException(Exception):
	def __init__(self, message, alpha_value):
		super().__init__(message)
		self.alpha_value = alpha_value
		print(message)
