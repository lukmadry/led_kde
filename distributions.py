import numpy as np
from scipy.stats import norm, multivariate_normal, uniform
from numbers import Number

from tools import convert_planar_to_xy

# base class that will allow us to easily add distributions together if needed
# TODO overload + operator to add distributions
class Distribution:
	def __init__(self, centers, weights=None):
		if isinstance(centers, float):
			centers = [centers]

		self.centers = np.array(centers)
		self.N = len(self.centers)

		if weights is not None:
			if len(weights) != self.N:
				raise ValueError("length of weights parameter should be equal to centers")
			self.weights = weights
		else:
			self.weights = [1]*self.N

		self.multivar = False # by default, we're dealing with one-dim objects

	def pdf(self, grid):
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
		for i in idx:
			samples.append(self.distributions[i].rvs())
		return np.array(samples)

class GaussianMixture(Distribution):
	def __init__(self, centers, scales, weights=None, scale_is_variance=False):
		super().__init__(centers, weights)
		if isinstance(scales, float):
			scales = [scales]

		self.scales = np.array(scales)
		self.scale_is_variance = scale_is_variance

		if not ((len(self.scales) == self.N) or len(self.scales) == 1):
			raise ValueError("scales argument should be either equal to length of centers or take only one element")

		if len(self.scales) == 1:
			self.scales = [self.scales[0]] * self.N # not really optimal, TODO change

		self.distributions = []

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
