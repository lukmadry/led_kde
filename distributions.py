import numpy as np
from scipy.stats import norm, multivariate_normal
from numbers import Number

from tools import convert_planar_to_xy

class GaussianCombination:
	def __init__(self, centers, scales, weights=None, scale_is_variance=False):
		if isinstance(centers, float):
			centers = [centers]
		if isinstance(scales, float):
			scales = [scales]

		self.centers = np.array(centers)
		self.scales = np.array(scales)
		self.scale_is_variance = scale_is_variance

		self.N = len(self.centers)
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

# this does not work very well for now, it doesn't crash but the result is not what I'd expect
# TODO check some other implementation of a spiral curve
def generate_spiral_comb(no_pts, starting_radius=1, starting_angle=0.5, delta_angle_rad=0.3,
						 radius_decay=0.99, scale=0.3, scale_is_variance=False):
	
	if scale_is_variance:
		scale = np.sqrt(scale)
	cov = np.array([[scale, 0], [0, scale]])

	radius = starting_radius
	angle = starting_angle

	centers = []
	for k in range(no_pts):
		x, y = convert_planar_to_xy(radius, angle)
		centers.append([x,y])

		radius *= radius_decay
		angle += delta_angle_rad

	combination = GaussianCombination(centers=centers, scales=[cov], scale_is_variance=False)
	return combination
