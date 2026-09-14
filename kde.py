import numpy as np
from functools import partial
from numbers import Number

from scipy.stats import norm
from scipy.signal import convolve

from tools import convolve_continuous_grid, match_x_to_grid_and_val

def compute_kde(x, observations, norm_scale, scale_is_variance=True):
	N = len(observations)
	val = 0
	if scale_is_variance:
		norm_scale = np.sqrt(norm_scale)
	for i in range(N):
		val += norm.pdf(x-observations[i], scale=norm_scale)
	return val / N

def compute_kde_scott(x, observations):
    N = len(observations)
    h = N**(-1/5.)
    val = 0
    for i in range(N):
        val += norm.pdf(x - observations[i], scale=h)
    return val/N


def kde_sampled_pdf_wrapper(x, grid, kde):
    if isinstance(x, Number):
        x = [x]
    vals = []
    for v in x:
        vals.append(match_x_to_grid_and_val(v, grid, kde))
    return vals

def kde_sampled_pdf_func(grid, kde):
	return partial(kde_sampled_pdf_wrapper, grid=grid, kde=kde)


def alpha_func(grid, observations, t_scale, k_pdf, eps=1e-8):
    q_L_approx = compute_kde(x=grid, observations=observations, norm_scale=t_scale, scale_is_variance=True)
    k_ql_conv = convolve_continuous_grid(f_pdf=k_pdf, g_pdf_grid=q_L_approx, x_grid=grid)
    k_log_ql_conv = convolve_continuous_grid(f_pdf=k_pdf, g_pdf_grid=np.log(eps + q_L_approx), x_grid=grid)
    return np.exp(k_log_ql_conv)/(k_ql_conv)


def compute_led_kde(grid, observations, t_scale, k_pdf, eps=1e-8, scale_is_variance=True):
    if scale_is_variance:
        t_scale = np.sqrt(t_scale)
    q_L_approx = compute_kde(x=grid, observations=observations, kernel=norm.pdf, scale=t_scale)
    k_log_ql_conv = convolve_continuous_grid(k_pdf, np.log(eps+q_L_approx), grid)
    expklog = np.exp(k_log_ql_conv)
    Z = np.sum(expklog) * (grid[1] - grid[0])
    return expklog / Z


def sample_led_kde(n_samples, observations, grid, t_scale, k_pdf,
                   k_sampler, l_sampler, verbose=True, display_every=20):
    N = len(observations)

    # precompute \alpha here
    alpha_grid = alpha_func(grid, observations, t_scale, k_pdf)
    vals = np.zeros(n_samples)

    # this can be presumably parallelized, although not by that much on my computer without GPU, some far away TODO would be maybe to try and write a kernel for this
    for i in range(n_samples):
        
        reject = True
        
        while reject:
            I = np.random.randint(N)
            u = l_sampler()
            w = k_sampler()
            R = np.random.uniform()
            X = observations[I] - u - w

            alph_val = match_x_to_grid_and_val(X, grid, alpha_grid) # this part does not work with more than one dimension yet
            if R < alph_val:
                vals[i] = X
                reject = False

        if verbose and (1+i) % display_every == 0:
            print(f"Sampled {1+i}/{n_samples} samples.", end="\r")

    return vals