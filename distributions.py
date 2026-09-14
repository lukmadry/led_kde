import numpy as np
from scipy.stats import norm, multivariate_normal

from tools import convert_planar_to_xy

# TODO rewrite it so that the functions below can be put inside this class
class GaussianCombination:
	def __init__(self, centers, scales, weights, scale_is_variance=False):
		if isinstance(centers, float):
			centers = [centers]
		if isinstance(scales, float):
			scales = [scales]

		self.centers = np.array(centers)
		self.scales = np.array(scales)
		self.weights = np.array(weights)
		self.scale_is_variance = scale_is_variance

		self.N = len(self.centers)
		self.distributions = []

		self.multivar = False
		if len(self.centers[0]) > 1:
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
		final_density = np.zeros_like(grid)

		for i in range(self.N):
			final_density += weights[i]*self.distributions[i].pdf(grid)
		return final_density

	def rvs(self, no_samples):

		weights = self.weights/np.sum(self.weights)
		idx = np.random.choice(self.N, no_samples, weights)
		samples = []
		for i in idx:
			samples.append(self.distributions[i].rvs())
		return np.array(samples)

def generate_pdf_univariate_gaussian_peaks(grid, centers, scales, weights=None, scale_is_variance=False):
	N = len(centers)
	if N != len(scales):
		raise ValueError

	if len(grid.shape) > 1:
		raise ValueError("argument grid should be a 1D array")

	if N == 1:
		if len(centers) > 1 or len(scales) > 1:
			raise ValueError
		return norm.pdf(grid, loc=centers[0], scale=scales[0])

	densities = []

	for c, s in zip(centers, scales):
		if scale_is_variance:
			s = np.sqrt(s)
		pdf = norm.pdf(grid, loc=c, scale=s)
		densities.append(pdf)

	final_density = np.zeros_like(grid)
	if weights is not None:

		if len(weights) != N:
			raise ValueError

		sum_weights = sum(weights)
		weights = [x/sum_weights for x in weights]
	else:
		weights = [1]*N
		
	for i in range(N):
		final_density += densities[i] * weights[i]

	return final_density

def sample_univariate_gaussian_peaks(no_samples, centers, scales, weights=None, scale_is_variance=True):
	N = len(centers)
	if N != len(scales):
		raise ValueError

	if N == 1:
		if len(centers) > 1 or len(scales) > 1:
			raise ValueError
		return norm(loc=centers[0], scale=scales[0]).rvs(no_samples)

	samplers = []

	for c, s in zip(centers, scales):
		if scale_is_variance:
			s = np.sqrt(s)
		rvs = norm(loc=c, scale=s).rvs
		samplers.append(rvs)

	if weights is not None:
		weights = np.asarray(weights, float)
		weights /= np.sum(weights)
	else:
		weights = np.ones(N)*1/float(N)

	idx = np.random.choice(a=np.arange(N), size=no_samples, p=weights)
	return np.asarray([samplers[i]() for i in idx])


def generate_pdf_multivariate_gaussian_peaks(grid, centers, scales, weights=None, scale_is_variance=False):
	pass

# TODO create a class Sampler or something like this
def generate_spiral(grid, no_pts, starting_radius=1, starting_angle=0.5, delta_angle_rad=0.3, radius_decay=0.99, scale=0.3, scale_is_variance=False):
	
	if scale_is_variance:
		scale = np.sqrt(scale)
	cov = np.array([[scale, 0], [0, scale]])
	densities = []

	radius = starting_radius
	angle = starting_angle
	for k in range(no_pts):
		x, y = convert_planar_to_xy(radius, angle)

		pdf_eval = multivariate_normal.pdf(grid, mean=[x,y], cov=cov)
		densities.append(pdf_eval)

		radius *= radius_decay
		angle += delta_angle_rad

	final_density = np.zeros_like(grid)
	for d in densities:
		final_density += d / no_pts

	return final_density

def sample_pts_on_a_spiral(grid, no_pts, no_observations, starting_radius=1, starting_angle=0.5, delta_angle_rad=0.3, radius_decay=0.99, scale=0.3, scale_is_variance=False):

	if scale_is_variance:
		scale = np.sqrt(scale)
	cov = np.array([[scale, 0], [0, scale]])
	dists = []

	radius = starting_radius
	angle = starting_angle

	for k in range(no_pts):
		x, y = convert_planar_to_xy(radius, angle)

		dmv = multivariate_normal(mean=[x,y], cov=cov)
		dists.append(dmv)

		radius *= radius_decay
		angle += delta_angle_rad

	dist_indices = np.random.randint(no_pts, no_observations)
	sampled_obs = []
	for idx in dist_indices:
		new_pt = dists[idx].rvs()
		sampled_obs.append(new_pt)

	return sampled_obs