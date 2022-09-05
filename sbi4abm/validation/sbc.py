import arviz as az
import logging
import multiprocessing as mp
import numpy as np
import torch
from tqdm import trange
import warnings

warnings.filterwarnings('ignore')

from sbi4abm.inference import kde
from sbi4abm.utils import sampling

def start_process():

	pass#print('Starting ', mp.current_process().name)

def _compute_ess_across_dims(post_samples, d, numpy=False):

	min_ess = np.inf
	for dim in range(d):
		if not numpy:
			ess = az.ess(post_samples[:, dim].numpy())
		else:
			ess = az.ess(post_samples[:, dim])
		if ess < min_ess:
			min_ess = ess
	return min_ess

def npe_sbc_script(post_sampler,
		simulator, 
		prior, 
		n_rank_computations, 
		n_hist_bins, 
		n_samples_per_iter,
		seed=None):

	"""
	Performs simulation-based calibration as described in Algorithm 2 of 
	Talts et al. (2018).

	Input:
	- post_sampler:			object with .sample((n_samples,), x=x) method,
							where n_samples is an int that determines the
							number of samples to draw from the posterior by
							the method implemented in post_sampler, and x
							is an observation. Assumes samples are returned
							with shape (n_samples, dimension_of_parameter)
	- simulator:			function, consumes parameter (output of 
							prior.sample()) and returns a random output
	- prior:				prior distribution, see sbi priors for required
							spec. Importantly the shape of the sample is 
							(sample_size, dimension_of_parameter)
	- n_rank_computations:	int, this is the number of times you desire the
							algorithm to compute a rank computation throughout
							the entire procedure, i.e. the final number of
							counts in the entire SBC histogram
	- n_hist_bins:			int, the number of bins in the histogram
	- n_samples_per_iter:	int, the initial number of samples to draw from the
							posterior. The algorithm may require more samples
							to be drawn in the event that these samples are
							insufficiently uncorrelated
	- seed:					int or None, if int it sets the random seed for
							numpy and torch
	"""

	if not (seed is None):
		np.random.seed(seed)
		torch.manual_seed(seed)

	d = int(prior.sample().size(-1))
	L = n_hist_bins - 1
	# Each iteration of the procedure generates a rank statistic. Store these
	# in an array such that one can call np.bincount on each of the d dims
	ranks = np.zeros((n_rank_computations, d), dtype=int)

	for i in trange(n_rank_computations):
		theta = prior.sample()
		x = np.expand_dims(simulator(theta), axis=0)
		#try:
		#	# Do a trial run to estimate the proposal covariance matrix, if
		#	# the method is kde-mcmc. Specifying start makes it remember to
		#	# start from this value when .sample is called below, but only the
		#	# first time, after this it'll start from the previous value
		#	post_sampler.estimate_covariance((50_000,), x=x, start=theta.numpy())
		#except AttributeError:
		#	pass
		post_samples = post_sampler.sample((n_samples_per_iter,), x=x, show_progress_bars=False)
		# Find the smallest effective sample size across all parameter dims,
		# and generate further samples if this is too small
		min_ess = _compute_ess_across_dims(post_samples, d)
		# Need effective sample size to be at least as large as n_hist_bins - 1
		if min_ess < L:
			sample_size = int(L * n_samples_per_iter / min_ess)
			more_samples = post_sampler.sample((sample_size,), x=x, show_progress_bars=False)
			post_samples = torch.cat((post_samples, more_samples), dim=0)
		#print(post_samples.size(0))

		# Out of interest, compute ESS for this extended posterior sample too
		min_ess = _compute_ess_across_dims(post_samples, d)
		# Log both the new ESS and the number of samples drawn
		for_logging = "Final ESS = {0}; number of samples drawn = {1}"
		logging.info(for_logging.format(min_ess, post_samples.size(0)))

		# Thin and truncate
		step = int(post_samples.size(0)/L)
		post_samples = post_samples[::step, :][:L, :].numpy()
		ranks[i, :] = (post_samples < theta.numpy()).sum(axis=0)
	return ranks

def p_nre_sbc_script(
		posterior,
		simulator, 
		prior, 
		n_rank_computations, 
		n_hist_bins, 
		n_samples_per_iter,
		scale,
		n_jobs=10
	):

	params = []
	rounds_per_job = n_rank_computations // n_jobs
	extra_last = n_rank_computations % n_jobs
	for n in range(n_jobs):
		if n == n_jobs - 1:
			rounds_per_job += extra_last
		params.append(
			[posterior,
			 simulator,
			 prior,
			 rounds_per_job,
			 n_hist_bins,
			 n_samples_per_iter,
			 scale,
			 n]
		)
	pool = mp.get_context('spawn').Pool(processes=n_jobs,
										initializer=start_process)
	ranks = pool.map(nre_sbc_script_head, params)
	pool.close()
	pool.join()
	return np.concatenate(ranks, axis=0)

def nre_sbc_script_head(args):

	return nre_sbc_script(*args)

def _compute_sir_ess(weights):

	return torch.sum(weights)**2 / torch.sum(weights**2)

def nre_sbc_script(posterior, 
		simulator,
		prior, 
		n_rank_computations, 
		n_hist_bins, 
		n_samples_per_iter,
		scale,
		seed=None):

	"""
	Performs simulation-based calibration as described in Algorithm 2 of 
	Talts et al. (2018).

	Input:
	- simulator:			function, consumes parameter (output of 
							prior.sample()) and returns a random output
	- prior:				prior distribution, see sbi priors for required
							spec. Importantly the shape of the sample is 
							(sample_size, dimension_of_parameter)
	- n_rank_computations:	int, this is the number of times you desire the
							algorithm to compute a rank computation throughout
							the entire procedure, i.e. the final number of
							counts in the entire SBC histogram
	- n_hist_bins:			int, the number of bins in the histogram
	- n_samples_per_iter:	int, the initial number of samples to draw from the
							posterior. The algorithm may require more samples
							to be drawn in the event that these samples are
							insufficiently uncorrelated
	- seed:					int or None, if int it sets the random seed for
							numpy and torch
	"""

	if not (seed is None):
		np.random.seed(seed)
		torch.manual_seed(seed)

	d = prior.sample().size(-1)
	if scale is None:
		scale = 2./np.sqrt(d)
	L = n_hist_bins - 1
	# Each iteration of the procedure generates a rank statistic. Store these
	# in an array such that one can call np.bincount on each of the d dims
	ranks = np.zeros((n_rank_computations, d), dtype=int)

	# For MH:
	# N_TRIAL_SAMPLES = 1_000

	for i in trange(n_rank_computations):
		theta = prior.sample()
		x = np.expand_dims(simulator(theta), axis=0)
		posterior.set_default_x(x)
		log_prob = posterior.log_prob

		#######################
		# METROPOLIS-HASTINGS #
		#######################
		# Do a trial run to estimate the proposal covariance matrix
		#trial_samples = sampling.mh(log_prob, start=theta.numpy(), scale=scale,
		#							n_samples=N_TRIAL_SAMPLES, to_torch=True,
		#							show_progress_bar=False)
		#cov = np.cov(trial_samples.T)
		#post_samples = sampling.mh(log_prob, start=trial_samples[-1], scale=scale,
		#						   cov=cov, n_samples=n_samples_per_iter, to_torch=True,
		#						   show_progress_bar=False)
		# Find the smallest effective sample size across all parameter dims,
		# and generate further samples if this is too small
		#min_ess = _compute_ess_across_dims(post_samples, d, numpy=True)
		# Need effective sample size to be at least as large as n_hist_bins - 1
		#if min_ess < L:
		#	sample_size = int(L * n_samples_per_iter / min_ess)
		#	more_samples = sampling.mh(log_prob, start=post_samples[-1], scale=scale,
		#							   cov=cov, n_samples=sample_size, to_torch=True,
		#							   show_progress_bar=False)
		#	post_samples = np.concatenate((post_samples, more_samples), axis=0)
		# Out of interest, compute ESS for this extended posterior sample too
		#min_ess = _compute_ess_across_dims(post_samples, d, numpy=True)
		# Thin and truncate
		#step = int(post_samples.shape[0]/L)
		#post_samples = post_samples[::step, :][:L, :]

		#######
		# SIR #
		#######
		# Generate some prior samples
		prior_samples = prior.sample((n_samples_per_iter,))
		# Compute their weights as torch.exp(posterior.log_prob - prior.log_prob)
		weights = torch.exp(posterior.log_prob(prior_samples) 
							- prior.log_prob(prior_samples))
		# Compute ESS
		min_ess = _compute_sir_ess(weights)
		# Generate more samples if necessary
		if min_ess < L:
			sample_size = int(L * n_samples_per_iter / min_ess)
			more_prior_samples = prior.sample((sample_size,))
			prior_samples = torch.cat((prior_samples, more_prior_samples), dim=0)
			weights = torch.cat((weights, 
								 torch.exp(posterior.log_prob(more_prior_samples)
										   - prior.log_prob(more_prior_samples))))

		min_ess = _compute_sir_ess(weights)
		# Resample with replacement and prob proportional to weights
		weights = weights.numpy().reshape(-1)
		wisnan = np.isnan(weights)
		if wisnan.any():
			print("Warning: detected nans in weights for SIR sampling - setting to 0")
			weights[wisnan] = 0.
		probs = weights/np.sum(weights)
		pisnan = np.isnan(probs)
		if pisnan.any():
			print("Warning: detected nans in probabilities for SIR sampling - setting to 0")
			probs[pisnan] = 0.
		idx = np.random.choice(np.arange(weights.shape[0]), 
							   p=probs,
							   size=L)
		# There's your sample
		post_samples = prior_samples.numpy()[list(idx)]

		#############
		# LOG RANKS #
		#############
		# Log both the new ESS and the number of samples drawn
		for_logging = "Final ESS = {0}; number of samples drawn = {1}"
		to_log = for_logging.format(min_ess, post_samples.shape[0])
		logging.info(to_log)
		ranks[i, :] = (post_samples < theta.numpy()).sum(axis=0)
	return ranks

def p_kde_sbc_script(simulator, 
		prior, 
		n_rank_computations, 
		n_hist_bins, 
		n_samples_per_iter,
		scale,
		R,
		n_jobs=10
	):

	params = []
	rounds_per_job = n_rank_computations // n_jobs
	extra_last = n_rank_computations % n_jobs
	for n in range(n_jobs):
		if n == n_jobs - 1:
			rounds_per_job += extra_last
		params.append(	
			 [simulator,
			 prior,
			 rounds_per_job,
			 n_hist_bins,
			 n_samples_per_iter,
			 scale,
			 R,
			 n]
		)
	pool = mp.get_context('spawn').Pool(processes=n_jobs,
										initializer=start_process)
	ranks = pool.map(kde_sbc_script_head, params)
	pool.close()
	pool.join()
	return np.concatenate(ranks, axis=0)

def kde_sbc_script_head(args):

	return kde_sbc_script(*args)

# TODO: make this use multiprocessing to run multiple jobs at once, else it'll
# take days
def kde_sbc_script(simulator, 
		prior, 
		n_rank_computations, 
		n_hist_bins, 
		n_samples_per_iter,
		scale,
		R,
		seed=None):

	"""
	Performs simulation-based calibration as described in Algorithm 2 of 
	Talts et al. (2018).

	Input:
	- simulator:			function, consumes parameter (output of 
							prior.sample()) and returns a random output
	- prior:				prior distribution, see sbi priors for required
							spec. Importantly the shape of the sample is 
							(sample_size, dimension_of_parameter)
	- n_rank_computations:	int, this is the number of times you desire the
							algorithm to compute a rank computation throughout
							the entire procedure, i.e. the final number of
							counts in the entire SBC histogram
	- n_hist_bins:			int, the number of bins in the histogram
	- n_samples_per_iter:	int, the initial number of samples to draw from the
							posterior. The algorithm may require more samples
							to be drawn in the event that these samples are
							insufficiently uncorrelated
	- seed:					int or None, if int it sets the random seed for
							numpy and torch
	"""

	if not (seed is None):
		np.random.seed(seed)
		torch.manual_seed(seed)

	def gen_prior_sample():
		return prior.sample().numpy()

	d = gen_prior_sample().shape[-1]
	L = n_hist_bins - 1
	# Each iteration of the procedure generates a rank statistic. Store these
	# in an array such that one can call np.bincount on each of the d dims
	ranks = np.zeros((n_rank_computations, d), dtype=int)

	for i in trange(n_rank_computations):
		theta = gen_prior_sample()
		x = simulator(theta)
		grazz = kde.initialise_kde(simulator, prior, x.T, R)
		# Do a trial run to estimate the proposal covariance matrix
		trial_samples = sampling.mh(grazz.log_prob, start=theta, scale=scale,
									n_samples=10_000, show_progress_bar=False)
		cov = np.cov(trial_samples.T)
		post_samples = sampling.mh(grazz.log_prob, start=trial_samples[-1], scale=scale,
								   cov=cov, n_samples=n_samples_per_iter, show_progress_bar=False)
		# Find the smallest effective sample size across all parameter dims,
		# and generate further samples if this is too small
		min_ess = _compute_ess_across_dims(post_samples, d, numpy=True)
		# Need effective sample size to be at least as large as n_hist_bins - 1
		if min_ess < L:
			sample_size = int(L * n_samples_per_iter / min_ess)
			more_samples = sampling.mh(grazz.log_prob, start=post_samples[-1], scale=scale,
									   cov=cov, n_samples=sample_size, show_progress_bar=False)
			post_samples = np.concatenate((post_samples, more_samples), axis=0)

		# Out of interest, compute ESS for this extended posterior sample too
		min_ess = _compute_ess_across_dims(post_samples, d, numpy=True)
		# Log both the new ESS and the number of samples drawn
		for_logging = "Final ESS = {0}; number of samples drawn = {1}"
		logging.info(for_logging.format(min_ess, post_samples.shape[0]))

		# Thin and truncate
		step = int(post_samples.shape[0]/L)
		post_samples = post_samples[::step, :][:L, :]
		ranks[i, :] = (post_samples < theta).sum(axis=0)
	return ranks
