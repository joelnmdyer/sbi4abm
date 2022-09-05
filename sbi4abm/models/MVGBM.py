import numpy as np
import scipy.stats

def _simulate(dtb_, x, es, seed=None):

	for t in range(1, x.shape[0]):
		x[t, :] = x[t-1, :] + dtb_ + es[t-1]
	return np.exp(x)

class Model:

	def __init__(self, x0=10.):

		self.x0 = x0
		self.dt = None
		self.sigma = np.array([[0.5, 0.1, 0.0],
							   [0.0, 0.1, 0.3],
							   [0.0, 0.0, 0.2]])
		self.const = 0.5 * np.sum(self.sigma**2, axis=0)
		self.ssT = self.sigma.dot(self.sigma.T)

	def simulate(self, pars=None, T=100, seed=None):

		self.dt = 1./(T - 1)
		self.cov = self.dt * self.ssT

		if pars is not None:
			b = np.array([float(pars[i]) for i in range(3)])

		if seed is not None:
			np.random.seed(seed)

		x = np.zeros((T + 1, 3))
		x[0, :] = np.log(self.x0)
		es = scipy.stats.multivariate_normal.rvs(size=T,
												 mean=np.zeros(3),
												 cov=self.cov)

		dtb_ = self.dt * (b - self.const)
		_simulate(dtb_, x, es, seed)
		return np.exp(x[1:])

	def log_likelihood(self, y, pars):

		# Assumes first point removed
		y = np.log(y)
		b = np.array([float(pars[i]) for i in range(3)])
		dtb_ = self.dt * (b - self.const)
		cov = self.dt * self.ssT

		norm_logpdf = scipy.stats.multivariate_normal.logpdf
		ll = norm_logpdf(y[0], dtb_ + np.log(self.x0), cov)
		for t in range(1, len(y)):
			mean = y[t-1] + dtb_
			ll += norm_logpdf(y[t], mean, cov)
		return ll

