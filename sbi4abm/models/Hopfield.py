import numpy as np
import scipy.stats
import torch
from tqdm import trange

def _simulate(rho, eps, lmbda, s, w, seed):

	if not (seed is None):
		np.random.seed(seed)

	N = w.shape[1]
	Nmin1 = N - 1
	oneminlmbda = 1. - lmbda
	aN = np.arange(N)
	NK = s[0].shape
	K = s.shape[-1]

	for t in range(w.shape[0] - 1):
		p = w[t].dot(s[t]) / Nmin1
		pi = 1./(1 + np.exp( - rho * p ))
		s[t+1] = 2 * ( pi > 0.5 + eps * ( np.random.random(NK) - .5 ) ) - 1
		w[t+1] = w[t] * oneminlmbda + lmbda * s[t+1].dot(s[t+1].T) / K
		np.fill_diagonal(w[t+1], 0.)
			
	return w, s

class Model:

	def __init__(self, N=100, K=2, initial_state=None):

		self._N = N
		self._K = K
		assert not (initial_state is None), "Must specify an initial state"
		self._is = initial_state

	def simulate(self, pars=None, T=None, seed=None):

		if not (pars is None):
			rho, eps, lmbda = [float(p) for p in pars]
		else:
			rho, eps, lmbda = 1., 0.8, 0.5
		if not (T is None):
			assert isinstance(T, int) and (T > 0), "T must be positive int"
		else:
			T = 50
		s = np.ones((T+1, self._N, self._K))
		s[0, :] = self._is[:, -2:]
		w = np.zeros((T+1, self._N, self._N))
		w[0] = self._is[:, :-2] #np.random.random((self._N, self._N))
		#np.fill_diagonal(w[0], 0.)
		w, s = _simulate(rho, eps, lmbda, s, w, seed)
		y = torch.from_numpy(np.concatenate((w, s), axis=-1)).float()
		return y

