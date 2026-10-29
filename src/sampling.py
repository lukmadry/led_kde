import numpy as np
from scipy.stats import norm, uniform, multivariate_normal

from src.alpha import alpha_func, _alpha_func_direct_1d
from src.tools import StrangeAlphaException, match_x_to_grid_and_val_1d, match_x_to_grid_and_val_2d, \
	NegativeProbabilityError


def sample_led_kde(n_samples, observations, grid, t_scale, k_scale,
				   verbose= False, display_every=20, kernel='gaussian',
				   alpha_filter_denom=True, interpolation=True, alpha_log_eps=1e-10,
                   alpha_denom_eps=1e-3, debug=False, alpha_grid=None):
	"""
	Samples observations with the LED-KDE method. If a parameter shares the name with one in alpha_func,
	it is identical to the description there and we will not repeat it here.

	n_samples : number of samples
	verbose : bool, True for printing some progress
	display_every : int will tell you that we sampled k values
	alpha_grid : np.ndarray, you can pass already computed alpha grid, this way computation here will be skipped
	interpolation : bool, whether the values samples will be interpolated to alpha or taken to be equal to values on the grid
				scheme tends to perform better if True
	sampling_method : str, 'standard' samples alpha function with special LED-KDE algorithm,
			'is' samples with importance sampling based on the density computation from alpha
	"""
	N = len(observations)
	ndim = len(observations.shape)
	if debug:
		print(f"Mean: {np.mean(observations):.2f}, std: {np.std(observations):.2f}")
		print(f"Grid shape: {grid.shape}")
	# precompute \alpha here (maybe you can factor it out, it was separated for testing and profiling)
	if alpha_grid is None:
		if alpha_filter_denom:
			alpha_grid, where_pos = alpha_func(grid, observations, t_scale, k_scale,
			                                   filter_denom=alpha_filter_denom)
			grid_pos = grid[where_pos]
		else:
			alpha_grid = alpha_func(grid, observations, t_scale, k_scale, filter_denom=alpha_filter_denom,
			                        denom_eps=alpha_denom_eps, log_eps=alpha_log_eps)
			grid_pos = grid
	else:
		grid_pos = grid
	if np.max(alpha_grid) > 1.01:
		raise StrangeAlphaException(f"something strange happened with alpha_grid, maximum value is {np.max(alpha_grid)}",
									np.max(alpha_grid))
	matcher = match_x_to_grid_and_val_1d if ndim == 1 else match_x_to_grid_and_val_2d
	# start = time.perf_counter()
	if ndim == 1 and kernel == 'gaussian':
		k_sampler = norm(loc=0, scale=np.sqrt(k_scale)).rvs
		l_sampler = norm(loc=0, scale=np.sqrt(t_scale)).rvs
		vals = np.zeros(n_samples)
	elif ndim == 1 and kernel == 'uniform':
		k_sampler = uniform(scale=k_scale).rvs
		l_sampler = uniform(scale=t_scale).rvs
		vals = np.zeros(n_samples)
	elif ndim == 2:
		k_sampler = multivariate_normal(mean=[0,0], cov=np.eye(2)*k_scale).rvs
		l_sampler = multivariate_normal(mean=[0,0], cov=np.eye(2)*t_scale).rvs
		vals = np.zeros((n_samples, 2))
	else:
		raise ValueError("ndim, inferred from shape of observations, must be 1 or 2")
	# end = time.perf_counter()
	# print(f"creating samplers take: {1000*(end-start):.3f} ms")

	# start = time.perf_counter()
	for i in range(n_samples):

		reject = True
		while reject:
			I = np.random.randint(N)
			u = l_sampler()
			w = k_sampler()
			R = np.random.uniform()
			X = observations[I] - u - w

			alph_val = matcher(X, grid_pos, alpha_grid, interpolation=interpolation) # technically it works with 2D but not super well
			if R < alph_val:
				vals[i] = X
				reject = False
		if verbose and (1+i) % display_every == 0:
			print(f"Sampled {1+i}/{n_samples} samples.", end="\r")
	# end = time.perf_counter()
	# print(f"Sampling takes: {1000*(end-start):.3f}ms (this part has to be repeated)")
	return vals


def sample_led_kde_direct_raw(n_samples, observations, grid, t_scale,
                              t_sampler, k_kernel_arr, k_sampler, dx,
                              alpha_log_eps=1e-10, alpha_denom_eps=1e-3,
                              interpolation=True, alpha_func_obj=_alpha_func_direct_1d,
                              matcher_fn=match_x_to_grid_and_val_1d, ndim=1, *args, **kwargs):
	alpha_grid = alpha_func_obj(grid=grid, observations=observations,
                               t_scale=t_scale, k_kernel_arr=k_kernel_arr,
                               log_eps=alpha_log_eps, denom_eps=alpha_denom_eps, dx=dx)
	if np.max(alpha_grid) > 1.01:
		raise StrangeAlphaException(f"something strange happened with alpha_grid, maximum value is {np.max(alpha_grid)}",
									np.max(alpha_grid))
	vals = np.zeros(n_samples) if ndim == 1 else np.zeros((n_samples, 2))
	N = len(observations)
	for i in range(n_samples):
		reject = True
		while reject:
			I = np.random.randint(N)
			u = t_sampler()
			w = k_sampler()
			R = np.random.uniform()
			X = observations[I] - u - w

			alph_val = matcher_fn(X, grid, alpha_grid,
			                                interpolation=interpolation)  # technically it works with 2D but not super well
			if R < alph_val:
				vals[i] = X
				reject = False

	return vals


def sample_led_kde_direct_raw_fast_is(n_samples, observations, grid, t_scale,
                                      t_sampler, k_kernel_arr, k_sampler, dx,
                                      alpha_log_eps=1e-10, alpha_denom_eps=1e-3, cands_multiplier=10,
                                      alpha_func_obj=_alpha_func_direct_1d, alpha_grid=None,
                                      matcher_fn=match_x_to_grid_and_val_1d, ndim=1, seed=None, debug=False):
	if alpha_grid is None:
		alpha_grid = alpha_func_obj(grid=grid, observations=observations,
	                               t_scale=t_scale, k_kernel_arr=k_kernel_arr,
	                               log_eps=alpha_log_eps, denom_eps=alpha_denom_eps, dx=dx)
	# if debug:
	# 	print("alpha control stats: ", alpha_grid.mean(), alpha_grid.std())
	if np.max(alpha_grid) > 1.01:
		raise StrangeAlphaException(f"something strange happened with alpha_grid, maximum value is {np.max(alpha_grid)}",
									np.max(alpha_grid))
	N = len(observations)
	n_candidates = cands_multiplier * n_samples
	u = t_sampler(size=n_candidates)
	w = k_sampler(size=n_candidates)
	if seed is None:
		seed = np.random
		idx = seed.randint(low=0, high=N, size=n_candidates) # TODO complain about it on a numpy forum or whatnot
	else:
		idx = seed.integers(low=0,high=N, size=n_candidates)
	# if debug:
	# 	print("first indices sampled: ", idx[:5])
	X = observations[idx] - u - w
	# if debug:
	# 	print("first values sampled: ", X[:5])
	alph_weights = matcher_fn(X, grid, alpha_grid, interpolation=True)
	if np.any(alph_weights > -1e-6): #sufficiently small that we can consider it an approximation error
		alph_weights = np.maximum(alph_weights, 0)
	else:
		raise NegativeProbabilityError(f"Observed negative probability of {np.min(alph_weights)}. "
		                               f"Stopping the run, you may consider rerunning the same "
		                               f"trial with linear interpolation", np.min(alph_weights))
	alph_weights /= np.sum(alph_weights)
	resampled_indices = seed.choice(
		n_candidates, size=n_samples, replace=True, p=alph_weights
	)
	if debug:
		print("first resampled indices: ", resampled_indices[:5])
	return X[resampled_indices]
