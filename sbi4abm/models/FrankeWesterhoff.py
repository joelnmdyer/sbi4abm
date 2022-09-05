from numba import njit
import numpy as np
from scipy import stats

@njit
def _wp(mu, beta, phi, chi, alpha0, alphaw, eta, p_star, esf, esc, wf, wc, a, p, df, dc, seed):

	"""
	This implements the Wealth & Predisposition variant of F&W
	"""
	
	if not (seed is None):
		np.random.seed(seed)

	# Initial conditions:
	gf = 0.
	gc = 0.
	# Assume the population is split evenly to begin with
	nf = 0.5
	# Note that log price is taken to be zero for first two steps
	
	# Simulate
	for t in range(2, p.shape[0]-1):
		
		gf = (np.exp(p[t]) - np.exp(p[t - 1])) * df[t - 2]
		gc = (np.exp(p[t]) - np.exp(p[t - 1])) * dc[t - 2]
	
		# Wealth updates
		wf[t] = eta * wf[t - 1] + (1 - eta) * gf
		wc[t] = eta * wc[t - 1] + (1 - eta) * gc

		a[t] = alphaw * (wf[t] - wc[t]) + alpha0
		
		# New proportion of agents following strategy f
		nf = 1 / (1 + np.exp(-beta * a[t - 1]))
		
		# Demand updates
		dc[t] = chi * (p[t] - p[t - 1]) + esc[t] 
		df[t] = phi * (p_star - p[t]) + esf[t]
		
		# Price update
		p[t + 1] = p[t] + mu * (nf * df[t] + (1 - nf) * dc[t])	
	
	# We're calibrating against the time-series of log-returns
	return np.diff(p[1:])[1:]

@njit
def _hpm_mean(mu, nf, phi, chi, pt1, pt2):

	return ( ( 1 - mu * ( nf * phi - ( 1 - nf ) * chi) ) * pt1 - 
			 ( mu * (1 - nf) * chi ) * pt2 )

@njit
def _hpm_std(mu, nf, sigmaf, sigmac):

	return mu * np.sqrt( ( nf * sigmaf ) ** 2 + ( (1 - nf) * sigmac ) ** 2 )

@njit
def _hpm_a(alphan, nf, alpha0, alphap, p):

	return alphan * ( 2*nf - 1 ) + alpha0 + alphap * (p**2)

@njit
def _hpm(mu, beta, phi, chi, alpha0, alphan, alphap, sigmaf, sigmac, T, seed):	

	# Assume p_star is 0

	if not (seed is None):
		np.random.seed(seed)
	p = np.zeros(T+1)
	a = alpha0
	es = np.random.randn(T)

	for t in range(1, T):
		nf = 1. / (1. + np.exp( - beta * a ))
		p[t] = _hpm_mean(mu, nf, phi, chi, p[t-1], p[t-2]) + \
				_hpm_std(mu, nf, sigmaf, sigmac) * es[t-1]
		a = _hpm_a(alphan, nf, alpha0, alphap, p[t-1])
	return p[1:-1]

class Model:
	def __init__(self, flavour="hpm"):
		self.flavour = flavour
		self.mu = 0.01
		self.beta = 1
		if self.flavour == "hpm":
			self.phi = 0.12
			self.chi = 1.5
			self.sigmaf = 0.758
		elif self.flavour == "wp":
			self.phi = 1
			self.chi = 0.9
			self.alpha0 = 2.1
			self.sigmaf = 0.752
			self.scale = [15000., 1., 5.]

	def log_likelihood(self, y, pars):
		if self.flavour == "wp":
			raise RuntimeError("F&W flavour 'wp' does not permit exact likelihood evaluations")
		alpha0, alphan, alphap, sigmac = [float(pars[i]) for i in range(4)]
		y_ = np.zeros(2)
		y_[-1] = y[0]
		a = alpha0
		ll = 0.
		for t in range(1, len(y)):
			nf = 1. / (1. + np.exp( - self.beta * a ))
			mean = _hpm_mean(self.mu, nf, self.phi, self.chi, y_[-1], y_[-2])
			scale = _hpm_std(self.mu, nf, self.sigmaf, sigmac)	
			ll += stats.norm.logpdf(y[t], loc=mean, scale=scale)
			a = _hpm_a(alphan, nf, alpha0, alphap, y[t-1])
			y_[0] = y_[-1]
			y_[-1] = y[t]
		return ll

	def simulate(self, pars, T=50, seed=None):

		if self.flavour == "hpm":
			alpha0, alphan, alphap, sigmac = [float(pars[i]) for i in range(4)]
			return _hpm(self.mu, self.beta, self.phi, self.chi, alpha0, alphan,
						  alphap, self.sigmaf, sigmac, T, seed)
		elif self.flavour == "wp":
			alphaw, eta, sigmac = [float(pars[i])*self.scale[i] for i in range(3)]
			esf, esc = self.sigmaf * np.random.normal(size=T), sigmac * np.random.normal(size=T)
			wf = np.zeros(T)
			wc = np.zeros(T)
			a = np.zeros(T)
			p = np.zeros(T + 1)
			df = np.zeros(T)
			dc = np.zeros(T)
			return _wp(self.mu, self.beta, self.phi, self.chi, self.alpha0,
						  alphaw, eta, 0., esf, esc, wf, wc, a, p, df, dc, seed)
