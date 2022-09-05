import numpy as np
from sbi4abm.utils import sampling
from scipy import stats
import torch

class KDE:

	def __init__(self, simulator, prior, observation, bandwidth="silverman", R=1):

		"""
		Input:

		- simulator:		function, consumes parameter list and returns
							simulation
		- observation:		shold be of the same format as output from
							simulator, and is the observed data
		- bandwidth:		str, method for setting bandwidth for KDE

		WARNING: scipy.stats.gaussian_kde accepts data in the opposite
		convention to torch – torch generally takes (n_samples, n_dims), but
		scipy takes (n_dims, n_samples). Ensure your data is transposed as
		required
		"""

		self.simulator = simulator
		self.prior = prior
		self.y = observation
		self.bandwidth = bandwidth
		#self.kd_estimator = lambda x: stats.gaussian_kde(x, bandwidth)#self._create_kd_estimator(bandwidth)
		self.R = R

	def kd_estimator(self, x):

		return stats.gaussian_kde(x, self.bandwidth)

	#def _create_kd_estimator(self, bandwidth):

	#	def estimator(x):
	#		# This accepts x as shape (n_dims, n_samples)
	#		return stats.gaussian_kde(x, bandwidth)

	#	return estimator

	def loglike(self, theta):

		"""
		Evaluate the log-likelihood of observation at parameter theta
		"""

		# Generate simulation at parameter theta
		if self.R == 1:
			x = self.simulator(theta).T
		else:
			x = []
			for r in range(self.R):
				y = self.simulator(theta).T
				x.append(y)
			x = np.concatenate(x, axis=-1)
		# Obtain kernel density estimate for x
		kde = self.kd_estimator(x)
		# Return log-likelihood of observation y under kde
		return kde.logpdf(self.y).sum()
 
	#def set_prior_log_prob(self, plp):

		#self._plp = plp

	def log_prob(self, pars):

		pars = list(pars.reshape(-1))
		plp = float(self.prior.log_prob(torch.tensor(pars)))
		if plp == -float("inf"):
			return plp
		ll = self.loglike(pars)
		return plp + ll

	def _sample(self, start, scale, n_samples, cov):

		return sampling.mh(self.log_prob, start=start, scale=scale, cov=cov,
						   n_samples=n_samples)

	def estimate_covariance(self, tup_n_samples, x, start=None, scale=None):

		self.y = x.T
		trial_samples = self._sample(start,
									 scale,
									 tup_n_samples[0],
									 None)
		self._cov = np.cov(trial_samples.T)
		self._new_start = start

	def sample(self, tup_n_samples, x, start=None, scale=None):

		self.y = x.T
		if not (start is None):
			self._new_start = start
		samples = self._sample(self._new_start,
							   scale=scale,
							   n_samples=tup_n_samples[0],
							   cov=self._cov)
		self._new_start = samples[-1, :]
		return samples


def initialise_kde(simulator, prior, y, R):

	# TODO: verify that this produces results observed so far
	#def prior_log_prob(pars):
	#	"""
	#	Assumes only one parameter evaluated each function call. Also assumes
	#	the prior object is from the sbi package: expects torch.tensor and
	#	has log_prob method
	#	"""
	#	pars = torch.tensor(pars)
	#	return float(prior.log_prob(pars))

	grazz = KDE(simulator, prior, y, R=R)
	#grazz.set_prior_log_prob(prior_log_prob)
	return grazz

def kde_training(simulator, prior, y, start=None, scale=1.,
				 n_sims=[50_000, 100_000], R=1):

	"""
	Script for sampling from kde-mcmc posteriors
	"""

	grazz = initialise_kde(simulator, prior, y, R)

	if len(n_sims) == 1:
		n_sims = [50_000, n_sims[0]]
	if start is None:
		start = prior.sample().numpy().reshape(-1)

	# Should be able to replace the next three calls with
	# grazz.estimate_covariance
	# grazz.sample
	if n_sims[0] > 0:
		trial_samples = sampling.mh(grazz.log_prob, start=start,
									scale=scale, n_samples=n_sims[0])
		cov = np.cov(trial_samples.T)
		samples = sampling.mh(grazz.log_prob, start=start, scale=None, cov=cov,
							  n_samples=n_sims[1])
	else:
		samples = None

	# Return None first for consistency with format of neural posterior methods
	return None, samples
