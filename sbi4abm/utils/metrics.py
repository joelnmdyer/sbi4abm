from numba import njit
import numpy as np
import ot
import torch

METRIC="euclidean"

def wass(samples_a, samples_b):

	M = ot.dist(samples_a, samples_b, metric=METRIC)
	gamma, log = ot.emd([], [], M, log=True)
	return log["cost"]

def median_heuristic(samples):

	samples = torch.from_numpy(samples)
	distances = torch.cdist(samples, samples)
	return 2*np.median(distances.numpy())

@njit
def _gauss_rbf(xi, xj, c):
    diff = xi-xj
    dot_diff = np.sum(diff**2)
    return np.exp(-dot_diff/c)

@njit
def mmd(x, y, c):
    """
    Function for estimating the MMD between samples x and y using Gaussian RBF
    with scale c.

    Args:
        x (np.ndarray): (n_samples, n_dims) samples from first distribution.
        y (np.ndarray): (n_samples, n_dims) samples from second distribution.
    Returns:
        float: The mmd estimate."""

    n_x = x.shape[0]
    n_y = y.shape[0]

    factor1 = 0.
    for i in range(n_x):
        for j in range(n_x):
            if (j == i): continue
            factor1 += _gauss_rbf(x[i:i+1], x[j:j+1], c)
    factor1 /= (n_x*(n_x-1))

    factor2 = 0.
    for i in range(n_y):
        for j in range(n_y):
            if (j == i): continue
            factor2 += _gauss_rbf(y[i:i+1], y[j:j+1], c)
    factor2 /= (n_y*(n_y-1))

    factor3 = 0.
    for i in range(n_x):
        for j in range(n_y):
            factor3 += _gauss_rbf(x[i:i+1], y[j:j+1], c)
    factor3 *= 2/(n_x*n_y)

    return factor1 + factor2 - factor3

