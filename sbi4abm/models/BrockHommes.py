from numba import njit
import numpy as np
from scipy import stats

@njit
def _get_utility(g, b, x, R, i):
	Rx1 = R * x[i + 1]
	return (x[i + 2] - Rx1) * (g * x[i] + b - Rx1)

@njit
def _get_mean(n_h, g, b, x, R, i):
	return (n_h * (g * x[i + 2] + b)).sum() / R 

@njit
def _get_numbers(U_h, beta):

	exponents = beta * U_h
	exponents -= exponents.max()
	n_h = np.exp(exponents)
	return n_h / n_h.sum()

@njit
def bh(g, b, R=1.01, beta=120., sigma=0.04, T=100, seed=None):
	'''
	Input:

	- g:		np.array of floats, trend-following parameters
	- b:		np.array of floats, bias parameters
	- r:		float, determines interest rate
	- beta:		float, parameter in utility computations
	- sigma:	float, standard deviation of Gaussian noise
	- T:		int, total number of steps to simulate
	- seed:		int, seed for the numpy RNG
		
	Output:

	- np.array for sequence of prices
	'''
	
	# Set RNG
	if not seed is None:
		np.random.seed(seed)
	
	# Pre-generate random shocks
	#epsilon = np.random.normal(size=T) * sigma / R
	
	# This will be an array with the deviations from fundamental price
	x = np.zeros(T)	
 
	# Simulate
	for i in range(T - 3):
		
		# Utilities
		U_h = _get_utility(g, b, x, R, i)
		# Proportion of number of agents following each strategy
		n_h = _get_numbers(U_h, beta)
		# Get price deviations
		x[i + 3] = np.random.normal() * sigma / R + _get_mean(n_h, g, b, x, R, i)
	
	return x


class Model:
	def __init__(self, beta=120., sigma=0.04, r=0.01):
		self.beta = beta
		self.sigma = sigma
		self.r = r
		self.R = 1. + self.r
		self.g1 = 0.
		self.b1 = 0.
		self.g4 = 1.01
		self.b4 = 0.

	def log_likelihood(self, y, pars):
		assert len(y.shape) == 2, "Reshape so that y of shape (T, 1)"
		# Parameter values
		g2, b2, g3, b3 = [float(pars[i]) for i in range(4)]
		# Always observe 0s in the first three time steps
		y_ = np.zeros((3, 1))
		scale = self.sigma / self.R
		b = np.array([self.b1, b2, b3, self.b4])
		g = np.array([self.g1, g2, g3, self.g4])
		ll = 0.
		for t in range(y.shape[0]):
			U_h = _get_utility(g, b, y_, self.R, 0, 0)
			n_h = _get_numbers(U_h, self.beta)
			mean = _get_mean(n_h, g, b, y_, self.R, 0, 0)
			ll += stats.norm.logpdf(y[t], loc=mean, scale=scale)
			y_[:2] = y_[-2:]
			y_[-1] = y[t]
		return ll

	def simulate(self, pars=None, T=100, seed=None):
		g2, b2, g3, b3 = [float(pars[i]) for i in range(4)]
		g = np.array([self.g1, g2, g3, self.g4])
		b = np.array([self.b1, b2, b3, self.b4])

		x = bh(g, b, self.R, self.beta, sigma=self.sigma, T=T, seed=seed)
		return np.expand_dims(x[3:], axis=-1)
