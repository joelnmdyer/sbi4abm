import numpy as np
from scipy import stats
import torch
from tqdm import tqdm

def mh(log_prob, start, cov=None, scale=None, n_samples=100_000, seed=None, to_torch=False,
	   show_progress_bar=True):

	"""
	Input:

	- log_prob:			function taking in theta of shape (1,d) or (d,) and
						returning estimate of posterior density evaluated at
						the observation and theta
	- start:			np.array of shape (1,d) or (d,), indicating the
						initial value of the random walk
	"""

	# Parameter dimension
	d = start.size
	if cov is None:
		cov = np.eye(d)
	if scale is None:
		scale = 2/np.sqrt(d)
	cov = cov*scale
	if not (seed is None):
		np.random.seed(seed)

	# Innovations for random walk
	q = stats.multivariate_normal
	es = stats.multivariate_normal.rvs(np.zeros(d), cov, size=n_samples)
	ps = np.random.random(n_samples)

	samples = np.empty((n_samples, d))
	samples[0,:] = start
	th0 = start

	_log_prob = log_prob
	if to_torch:
		_log_prob = lambda x: log_prob(torch.from_numpy(x).float())
	lp0 = _log_prob(th0)

	if show_progress_bar:
		iterator = tqdm(range(n_samples), position=0)
	else:
		iterator = range(n_samples)
	n_test, n_acc = 0, 0

	for t in iterator:

		# Propose new sample
		th1 = th0 + es[t]
		if (th1.size != d):
			raise ValueError("Parameter of wrong dimension")

		################################
		# Find acceptance probability: #
		################################

		# Evaluate log-probability density of posterior density at proposed
		lp1 = _log_prob(th1)	
		if lp1 == -float("inf"):
			n_test += 1
			samples[t,:] = th0
			continue
		else:
			d_log_probs = lp1 - lp0
		# Compute log of acceptance probability
		loga = min([0, d_log_probs + 
					   q.logpdf(th0, mean=th1.reshape(-1), cov=cov) -
					   q.logpdf(th1, mean=th0.reshape(-1), cov=cov)])

		#################
		# Accept/reject #
		#################

		# Determine whether to accept or reject
		if loga == -float("inf"):
			n_test += 1
			samples[t,:] = th0
		else:
			if np.log(ps[t]) >= loga:
				# Reject th1
				samples[t,:] = th0
			else:
				# Accept th1
				samples[t,:] = th1
				n_acc += 1
				th0 = th1
				lp0 = lp1
		if show_progress_bar:
			iterator.set_postfix({"Acc. rate":n_acc/(t+1), "Test":n_test/(t+1),
								  "Loc.":["{:2f}".format(s) for s in th0.tolist()],
								  "Lp":float(lp0), "Lp1":float(lp1)})
			iterator.update()

	return samples
