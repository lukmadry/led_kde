from numbers import Number

import numpy as np
from scipy import integrate
from scipy.stats import entropy
from scipy.signal import fftconvolve

# TODO speed up convolutions by discarding values that are essentially zero, may need more specific variant of the code

try:
	import git
except ImportError:
	print("could not import git package")
def exact_gaussian_1d(x, var):
	sddev = np.sqrt(var)
	return 1/(2.0*np.pi*sddev) * np.exp(-np.square(x)/(2.0*var))

def exact_gaussian_2d(x, var):
	return 1/(2.0*np.pi*var) * np.exp(-np.sum(np.square(x))/(2.0*var))

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

# AI use : some (most) of these functions have been written with Gemini Flash
# multidim version of it is coming later, so far we're not doing this
def match_x_to_grid_and_val_1d(x, grid, values_grid,
                            raise_error_if_outside=False, interpolation=True):
    """
    it is crucial that grid has evenly-spaced points
    """
    grid_step = grid[1] - grid[0]
    grid_min = grid[0]
    grid_max = grid[-1]

    sm = x < grid_min
    bg = x >= grid_max
    if raise_error_if_outside and (sm or bg):
        raise ValueError("x is outside of grid")
    if sm:
        return values_grid[0]
    if bg:
        return values_grid[-1]

    idx = int( (x - grid_min) / grid_step )
    
    if interpolation:
        dright = grid[idx+1] - x # no problem with it because if we are here then x is strictly smaller than the final value
        weight = dright / grid_step
        return values_grid[idx] * weight + values_grid[idx+1] * (1-weight)
        
    return values_grid[idx]

def match_x_to_grid_and_val_2d(x, grid, values_grid,
                               raise_error_if_outside=False, interpolation=True):
	"""
	grid :
	"""
	grid, x_linspace, y_linspace = _interpret_two_dim_grid(grid)

	grid_stepx = x_linspace[1] - x_linspace[0]
	grid_stepy = y_linspace[1] - y_linspace[0]
	grid_minx = x_linspace[0]
	grid_miny = y_linspace[0]
	grid_maxx = x_linspace[-1]
	grid_maxy = y_linspace[-1]
	nx = len(x_linspace)
	ny = len(y_linspace)

	sm_x = x[0] < grid_minx
	sm_y = x[1] < grid_miny
	bg_x = x[0] >= grid_maxx
	bg_y = x[1] >= grid_maxy

	if raise_error_if_outside and (sm_x or sm_y or bg_x or bg_y):
		raise ValueError("x is not allowed outside the grid")

	if sm_x:
		idx_x = 0
	elif bg_x:
		idx_x = nx-1
	else:
		idx_x = int((x[0] - grid_minx) / grid_stepx)
	if sm_y:
		idx_y = 0
	elif bg_y:
		idx_y = ny-1
	else:
		idx_y = int( (x[1] - grid_miny) / grid_stepy )

	# a heuristic, not sure what is optimal here. Hopefully it doesn't matter all that much
	# I guess there exist a better variant where we use the diagonal as well
	# we should also interpolate other edges
	if interpolation:
		dright_x = values_grid[idx_x+1, idx_y] - x[0]
		dright_y = values_grid[idx_x, idx_y+1] - x[1]
		weight_x = dright_x / grid_stepx
		weight_y = dright_y / grid_stepy
		vx = values_grid[idx_x, idx_y] * weight_x + values_grid[idx_x+1, idx_y] * (1-weight_x)
		vy = values_grid[idx_x, idx_y] * weight_y + values_grid[idx_x, idx_y+1] * (1-weight_y)
		return vx /2.0 + vy/2.0

	return values_grid[idx_x, idx_y]

def match_x_to_grid_and_val(x, grid, values_grid, interpolation):
	if isinstance(x, Number):
		return match_x_to_grid_and_val_1d(x, grid, values_grid, raise_error_if_outside=False, interpolation=interpolation)
	elif len(x) == 2:
		return match_x_to_grid_and_val_2d(x, grid, values_grid, interpolation=interpolation, raise_error_if_outside=False)
	else:
		raise NotImplementedError("x has to be one or two-dimensional")

# remove ?
def convolve_continuous_pdf(f_pdf, g_pdf, x, support_g=(-np.inf, np.inf)):
	"""Computes u(x) = integral f(x - y) g(y) dy for a single scalar value x."""
	integrand = lambda y: f_pdf(x - y) * g_pdf(y)

	# Integrate across the support of g
	result, _ = integrate.quad(integrand, support_g[0], support_g[1])
	return result

# TODO to be sped up for the kernel with small support
# TODO check if you can use scipy.ndimage instead, it's probably optimised beyond what you can obtain here
def convolve_continuous_grid_1d_fft(f_pdf, g_pdf_grid, x_grid, y_bounds=None):
	# y_bounds is only there for argument compatibility with general implementation
	N = len(x_grid)
	dx = x_grid[1] - x_grid[0]
	diffs = np.arange(-(N-1)*dx, (N-1)*dx + 0.5*dx, step=dx)
	f_grid_conv = f_pdf(diffs)
	return fftconvolve(f_grid_conv, g_pdf_grid, mode='valid') * dx


def convolve_continuous_grid_2d_fft(f_pdf, g_pdf_grid, x_grid, y_bounds=None):
	ns_y = g_pdf_grid.shape
	x_grid, x1_1d, x2_1d = _interpret_two_dim_grid(x_grid)

	if y_bounds is None:
		y1_1d = x1_1d
		y2_1d = x2_1d
	else:
		y1_1d = np.linspace(y_bounds[0][0], y_bounds[0][1], ns_y[0])
		y2_1d = np.linspace(y_bounds[1][0], y_bounds[1][1], ns_y[1])

	dx1 = x1_1d[1] - x1_1d[0]
	dx2 = x2_1d[1] - x2_1d[0]
	dy1 = y1_1d[1] - y1_1d[0]
	dy2 = y2_1d[1] - y2_1d[0]

	d1 = np.arange(
		-(len(y1_1d)-1)*dy1, (len(x1_1d)-1)*dx1 + 0.5*dx1, step=dx1
	)
	d2 = np.arange(
		-(len(y2_1d)-1)*dy2, (len(x2_1d)-1)*dx2 + 0.5*dx2, step=dx2
	)
	D1, D2 = np.meshgrid(d1, d2)
	diff_stack = np.dstack([D1, D2])
	f_grid = f_pdf(diff_stack)
	conv_result = fftconvolve(f_grid, g_pdf_grid, mode="valid") * dy1 * dy2
	return conv_result

def convolve_continuous_grid(f_pdf, g_pdf_grid, x_grid, y_bounds=None):
	# will redirect to a matching implementation of convolve_... depending on the dimensions of x_grid
	if len(x_grid.shape) == 1:
		return convolve_continuous_grid_1d_fft(f_pdf=f_pdf, g_pdf_grid=g_pdf_grid, x_grid=x_grid, y_bounds=y_bounds)
	elif isinstance(x_grid, (tuple, list)) and len(x_grid) == 2 or len(x_grid.shape) == 3:
		return convolve_continuous_grid_2d_fft(f_pdf=f_pdf, g_pdf_grid=g_pdf_grid, x_grid=x_grid)
	else:
		raise NotImplementedError("convolution on more than two dimensions is not implemented and likely won't be")

def sample_via_importance_resampling(
	target_pdf,
	proposal_pdf,
	proposal_sampler,
	n_candidates=100_000,
	n_samples=10_000,
):
	"""
	Draws unweighted samples from target_pdf using Sampling-Importance Resampling (SIR).

	Parameters:
		target_pdf: callable, p(x) returning density values for shape (N, D) or (N,)
		proposal_pdf: callable, q(x) returning density values for shape (N, D) or (N,)
		proposal_sampler: callable, draws N samples from q, returning shape (N, D) or (N,)
		n_candidates: int, number of initial draws from proposal q
		n_samples: int, final number of unweighted samples desired from p

	Returns:
		unweighted_samples: array of shape (n_samples, D)
		ess: float, Effective Sample Size before resampling
	"""
	# 1. Sample candidate points from proposal q
	candidates = proposal_sampler(n_candidates)

	# 2. Evaluate likelihood ratios p(x) / q(x)
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

# OLD FUNCTIONS
def convolve_continuous_grid_1d(f_pdf, g_pdf_grid, x_grid, y_bounds=None):
	"""Evaluates continuous convolution u(x) over an array x_grid using vectorised quadrature.
	Parameters:
		f_pdf: continuous callable
		g_pdf_grid: 1D array of density values for g evaluated on y
		x_grid: 1D array of evaluation points x
		y_bounds: tuple (y_min, y_max) covering the support where g(y) > 0
	"""
	n_y = len(g_pdf_grid)
	if y_bounds is None:
		y_bounds = (x_grid.min(), x_grid.max())
	y = np.linspace(y_bounds[0], y_bounds[1], n_y)
	# 2D broadcasting: (M, 1) - (1, N) -> shape (M, N)
	diffs = x_grid[:, None] - y
	# print("about to compute integrand matrix")
	# Evaluate integrand matrix: f(x - y) * g(y)
	integrand_matrix = f_pdf(diffs) * g_pdf_grid
	# print("integrand matrix computed")
	# Integrate along the y axis (axis=1) using trapezoidal rule
	return np.trapezoid(integrand_matrix, y, axis=1)


def convolve_continuous_grid_2dim(f_pdf, g_pdf_grid, x_grid, y_bounds=None):
	"""

	Parameters:
		f_pdf: continuous callable
		g_pdf_grid: mD array of density values for g evaluated on y
		x_grid: 1) 2 arrays of shape (N, N), typically obtained through np.dstack applied to np.mgrid
				OR 2) 2 1D grids

	"""
	ns_y = g_pdf_grid.shape
	x_grid, x1_1d, x2_1d = _interpret_two_dim_grid(x_grid)

	if y_bounds is None:
		y_grid = x_grid.copy()
		y1_1d = x1_1d
		y2_1d = x2_1d
	else:
		y1_1d = np.linspace(y_bounds[0][0], y_bounds[0][1], ns_y[0])
		y2_1d = np.linspace(y_bounds[1][0], y_bounds[1][1], ns_y[1])
		y_grid = np.dstack((y1_1d, y2_1d))
	# TODO improve efficiency, takes an awful lot of memory if the grid is dense, you can use FFT
    # you can do it exploiting translation invariance of the first one
	diffs = x_grid[:, :, None, None] - y_grid[None, None, :, :]
	integrand_matrix = f_pdf(diffs) * g_pdf_grid[None, None, :, :]
	integrand_matrix = np.trapezoid(integrand_matrix, y2_1d, axis=3)
	integrand_matrix = np.trapezoid(integrand_matrix, y1_1d, axis=2)
	return integrand_matrix
