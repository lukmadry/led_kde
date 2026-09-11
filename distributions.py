import numpy as np
from scipy.stats import norm

def generate_pdf_univariate_gaussian_peaks(grid, centers, scales, weights=None, scale_is_variance=False):
	N = len(centers)
	if N != len(scales):
		raise ValueError

	if len(grid.shape) > 1:
		raise ValueError("argument grid should be a 1D array")

	if N == 1:
		if len(centers) > 1 or len(scales) > 1:
			raise ValueError
		return norm.pdf(grid, loc=centers, scale=scales)

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

def generate_pdf_multivariate_gaussian_peaks(grid, centers, scales, weights=None, scale_is_variance=False):
	pass

def generate_spiral():
	pass