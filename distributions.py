import numpy as np
from scipy.stats import norm, multivariate_normal, uniform, rv_continuous
from numbers import Number

# base class that will allow us to easily add distributions together if needed
# TODO overload + operator to add distributions
class Distribution:
	def __init__(self, centers, weights=None, random_state=None):
		if isinstance(centers, Number):
			centers = [centers]

		self.centers = np.array(centers)
		self.N = len(self.centers)
		self.seed = random_state

		self.distributions = []
		if weights is not None:
			if len(weights) != self.N:
				raise ValueError(f"length of weights parameter should be equal to centers. Centers are: {self.centers}, weights: {weights}")
			self.weights = weights
		else:
			self.weights = [1]*self.N

		self.multivar = False # by default, we're dealing with one-dim objects

	def pdf(self, grid):
		"""
		grid: if multivar=False then just a 1D array. if multivar=True then an array in shape (N, 2)
		returns an array in the same shape as input
		"""
		weights = self.weights/np.sum(self.weights)
		if not self.multivar:
			final_density = np.zeros_like(grid)
			for i in range(self.N):
				final_density += weights[i]*self.distributions[i].pdf(grid)
			return final_density
		else:
			final_density = np.zeros(grid.shape[:-1]) # maybe we can parametrize it or leave more choice, idk now
			for i in range(self.N):
				final_density += weights[i]*self.distributions[i].pdf(grid)
			return final_density

	def rvs(self, no_samples):

		weights = self.weights/np.sum(self.weights)
		idx = np.random.choice(a=self.N, size=no_samples, p=weights)
		samples = []
		if not self.multivar:
			for i in idx:
				samples.append(self.distributions[i].rvs(random_state=self.seed))
		else:
			for i in idx:
				samples.append(self.distributions[i].rvs(random_state=self.seed).flatten())
		return np.array(samples)

class GaussianMixture(Distribution):
	def __init__(self, centers, scales, weights=None, scale_is_variance=False, random_state=None):
		super().__init__(centers, weights, random_state=random_state)
		if isinstance(scales, Number):
			scales = [scales]

		self.scales = np.array(scales)
		self.scale_is_variance = scale_is_variance

		if not ((len(self.scales) == self.N) or len(self.scales) == 1):
			raise ValueError("scales argument should be either equal to length of centers or take only one element")

		if len(self.scales) == 1:
			self.scales = [self.scales[0]] * self.N # not really optimal, TODO change

		for s in self.scales:
			if s < 0:
				raise ValueError("scales parameter cannot be negative or contain negative values if a list/array")

		self.multivar = False
		if not isinstance(self.centers[0], Number):
			self.multivar = True

		for c, s in zip(self.centers, self.scales):
			if scale_is_variance:
				s = np.sqrt(s)
			if self.multivar:
				dist = multivariate_normal(mean=c, cov=s)
			else:
				dist = norm(loc=c, scale=s)
			self.distributions.append(dist)

		if weights is not None:
			self.weights = weights
		else:
			self.weights = [1]*self.N
		self.weights = np.array(self.weights)

class UniformMixture(Distribution):
	def __init__(self, centers, widths, weights=None):
		super().__init__(centers, weights)
		if isinstance(widths, Number):
			widths = [widths]

		self.widths = np.array(widths)

		self.distributions = []

		if not ((len(self.widths) == self.N) or len(self.widths) == 1):
			raise ValueError("widths argument should be either equal to length of centers or take only one element")

		if len(self.widths) != self.N:
			self.widths = [self.widths[0]] * self.N

		for c, w in zip(self.centers, self.widths):
			dist = uniform(loc=c-w/2, scale=w)
			self.distributions.append(dist)

class UniformBox:
	def __init__(self, corner, shapes):
		self.corner = np.array(corner)
		self.shapes = np.array(shapes)

		self.dist_x = uniform(corner[0] - shapes[0]/2, scale=shapes[0])
		self.dist_y = uniform(corner[1] - shapes[1]/2, scale=shapes[1])

	def pdf(self, grid):
		"""
		grid: np.array with shape (N, K,  2)

		returns an array with shape (N, K)
		"""
		flat_grid = grid.reshape(-1, 2)
		pdf_x = self.dist_x.pdf(flat_grid[:, 0])
		pdf_y = self.dist_y.pdf(flat_grid[:, 1])
		pdf = pdf_x * pdf_y # since boxes are independent uniforms
		return pdf.reshape(grid.shape[:-1])

	def rvs(self, no_samples=None, random_state=None):
		"""
		no_samples : int
		"""
		if no_samples is None:
			no_samples = 1
		samples_x = self.dist_x.rvs(no_samples, random_state=random_state)
		samples_y = self.dist_y.rvs(no_samples, random_state=random_state)
		return np.stack([samples_x, samples_y], axis=1)

class UniformBoxMixture(Distribution):
	def __init__(self, centers, shapes, weights=None, random_state=None):
		super().__init__(centers, weights, random_state=random_state)

		self.shapes = np.array(shapes)
		if self.centers.ndim != self.shapes.ndim and self.centers.shape[0] != self.shapes.shape[0]:
			raise ValueError("centers and shapes should have the same dimensions")

		for (cx, cy), (wx, wy) in zip(self.centers, self.shapes):
			self.distributions.append(UniformBox(corner=(cx-wx/2, cy-wy/2), shapes=(wx, wy)))

		self.multivar = True

# will encapsulate kde for pdf and is for rvs
class CustomDistribution:
	def __init__(self, init_samples, sigma=0.02):
		self.init_samples = init_samples
		self.sigma = sigma

	def pdf(self, grid):
		pass


def sample_spiral(n_samples, low=np.pi, high=4*np.pi, radius_multiplier=1):
	thet = np.random.uniform(low=low, high=high, size=n_samples)
	x = thet * np.sin(thet)
	y = thet * np.cos(thet)
	if np.abs(radius_multiplier - 1) > 1e-3:
		x *= radius_multiplier
		y *= radius_multiplier
	return x, y
