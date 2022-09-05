import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import os
import pickle
from sbi4abm.inference import kde, neural
from sbi4abm.utils import io
from sbi4abm.validation import sbc
import torch
import warnings


def _parse_nsims(nsims):

	"""
	Allows nsims in either format of list, in which case interpreted as list
	of simulations at each round, or of string 
	"""

	if len(nsims) == 1:
		n = nsims[0].split("x")
		if len(n) == 1:
			return list(map(int, n))
		elif len(n) == 2:
			n, r = n[0], n[1]
			return [int(n) for i in range(int(r))]
	else:
		return list(map(int, nsims))

def _de2class(de_name):

	"""
	Infers posterior estimation technique from densit estimator name
	"""

	if de_name in ["maf", "nsf", "made", "mdn"]:
		return "SNPE"
	else:
		return "SNRE"

def _acf(x, lag):
	"""
	Computes autocorrelation at specified lag for 1D time series
	"""
	return np.dot(x[:-lag], x[lag:]) / (x.shape[0] - 1)

class Summariser1D:

	def __init__(self, simulator):

		self.simulator = simulator

	def __call__(self, pars):

		"""
		Summarises 1D time series x with: quantiles 0, 25, 50, 75, 100; mean;
		variance; autocorrelations at lags 1, 2, 3
		"""

		x = self.simulator(pars)
		return self.summarise(x)

	def summarise(self, x):

		x = x.reshape(-1)
		sx = np.array([
			np.min(x),
			np.quantile(x, 0.25),
			np.median(x),
			np.quantile(x, 0.75),
			np.max(x),
			np.mean(x),
			np.var(x),
			_acf(x, 1),
			_acf(x, 2),
			_acf(x, 3)
		])
		return sx

#def _summary_wrap_simulator(simulator):
#	"""
#	Wrap raw simulator in a summarising function to summarise output
#	"""
#	def _simulator(pars):
#		x = simulator(pars)
#		return _summariser(x)
#
#	return _simulator

def _kde_prepare_simulator_observation(simulator, args, y):

	_simulator = simulator
	# Should we do this for neural methods too?
	if (args.task == "mvgbm"):
		y = np.diff(y, axis=0)
		_simulator = lambda pars: np.diff(simulator(pars), axis=0)

	# KDE needs data in shape (n_dims, n_points) i.e. the opposite of sbi
	# So finally transpose the simulated data
	#def __simulator(pars):
	#	return _simulator(pars).T
	# DO THE ABOVE TWO LINES INTERNALLY WITHIN THE KDE CLASS

	return _simulator, y.T

def _neural_prepare_estimator_observation(args, y, simulator):

	big = False
	small = False
	naive = False
	network = "gru"
	# See which summary network to use
	de_name = args.method.split("_")
	if len(de_name) == 2:
		de_name, network = de_name
		naive = network == "s"
		graph = network == "gcn"
	elif len(de_name) == 3:
		de_name, network, _ = de_name
		naive = network == "s"
		big = _ == "big"
		small = _ == "small"
	else:
		de_name = de_name[0]
	# Hand-crafted
	if naive:
		sim_pp = lambda x: x
		embedding_kwargs = None
		simulator = Summariser1D(simulator)
		y = simulator.summarise(y)
	# Learned
	else:
		sim_pp = lambda x: x.unsqueeze(-1)

		if args.task == "fw_hpm":
			sim_pp = lambda x: (x[:, 1:] - x[:, :-1]).unsqueeze(-1)

		if args.task in ["mvgbm"]:
			_ = simulator(prior.sample())
			N = -1
			sim_pp = lambda x: x.reshape(-1, _.shape[0], _.shape[1])

		NO = True

		if args.task == "hop":
			_ = simulator(prior.sample())
			N = _.size(1)
			# The + 1 makes it non-negative, which made it work \_o_/
			sim_pp = lambda x: x.reshape(-1, _.size(0), N, N+2) + 1
			y = sim_pp(torch.from_numpy(y))
			NO = False

		if NO:
			if len(y.shape) == 1:
				IDIM = 1
				y = sim_pp(torch.from_numpy(y).unsqueeze(0))
			elif len(y.shape) == 2:
				IDIM = y.shape[-1]
				y = torch.from_numpy(y).unsqueeze(0)

		if big:
			HDIM = 32
			N_LAYERS = 2
			ODIM = [32, 16, 16]
		elif small:
			HDIM = 32
			N_LAYERS = 1
			ODIM = [32, 16]
		else:
			HDIM = 32
			N_LAYERS = 2
			ODIM = 16
		HOUT_DIM = -1

		if network == "gcn":
			IDIM = -1
			HDIM = 32
			ODIM = [32, 16, 16]
			N_LAYERS = -1
			HOUT_DIM = 16
		if network == "rgcn":
			IDIM = 2
			HDIM = 64
			ODIM = [32, 16, 16]
			N_LAYERS = -1
			HOUT_DIM = 16
		embedding_kwargs = {"input_dim":IDIM,
							"hidden_dim":HDIM,
							"hidden_out_dim":HOUT_DIM,
							"num_layers":N_LAYERS,
							"mlp_dims":ODIM,
							"flavour":network,
							"N":N}

	# Create the density estimator
	density_estimator = neural.create_density_estimator(embedding_kwargs,
														de_name)
	sbi_method = _de2class(de_name)
	return density_estimator, sbi_method, y, simulator, sim_pp

def _get_postprocessor(args):
	
	sim_pp = lambda x: x.unsqueeze(-1)
	if args.task == "fw_hpm":
		sim_pp = lambda x: (x[:, 1:] - x[:, :-1]).unsqueeze(-1)
	if args.task in ["mvgbm"]:
		_ = simulator(prior.sample())
		sim_pp = lambda x: x.reshape(-1, _.shape[0], _.shape[1])
	return sim_pp

def _raise_kde_warnings(args, nsims):

	if args.method[-2:] == "_s":
		warnings.warn("Method kde incompatible with summary statistics; ignoring")
	if len(nsims) > 2:
		warning = ("Multi-round inference not compatible with " +
				   "method kde; treating first entry as trial run and second" +
				   " as desired simulation budget")
		warnings.warn(warning)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Job script for sbi4abm",
									 formatter_class=RawTextHelpFormatter)
	# TODO: create list of tasks to be run and include in description
	parser.add_argument('--task', type=str,
						help=("Name of inference task to run. " + 
							  "Current choices are:\n- bh_noisy\n- bh_smooth"+
							  "\n- fw_hpm\n- hop\n- mvgbm"))
	# TODO: create list of inference methods that can be called and include in
	# description. Maybe any of the density estimation methods compatible with
	# the sbi package, meaning that method with a GRU embedding net, or
	# additionally followed by _s if they are to be used with fixed, naive 
	# summary statistics instead
	parser.add_argument('--method', type=str,
						help=("Name of density estimator/classifier. Suffix " + 
							  "'_s' indicates that no embedding net is " + 
							  "to be used."))
	parser.add_argument('--nsims', type=str, nargs='+',
						help=("Number of simulations to run, e.g. 1000 1000" +
							  " 2000 indicates 3 rounds with 1000 in first " +
							  "two and 2000 in third. Alternatively, write " +
							  "<nsim>x<nround> to ask for nround rounds of " +
							  "nsim simulations, e.g. 1000x10 asks for 10 " + 
							  "rounds of 1000 simulation each."))
	parser.add_argument('--outloc', type=str,
						help="Location to dump inference results.")
	parser.add_argument('--scale', type=float,
						help="Scale (> 0) for MCMC proposal covariance.")
	parser.add_argument('--R', type=int, nargs="?", default=1,
						help="Number of simulations per likelihood evaluation for KDE.")
	parser.add_argument('--save_sim_prior', action='store_true', default=False,
						help="If included, the simulator and prior are saved to outloc")
	parser.add_argument('--sbc', action='store_true', default=False,
						help="If included, simulation-based calibration is performed")
	parser.add_argument('--sampler', type=str, default='mh',
						help=("Which sampling method to use for (S)NRE for the " + 
							  "final posterior samples. ['sir', 'mh']"))
	parser.add_argument('--load_post', type=str, nargs="?", default="",
						help="Location to load an pretrained posterior from")
	parser.add_argument('--nw', type=int, nargs="?", default=15,
						help="Number of workers to use in simulating for sbi")
	args = parser.parse_args()

	# Prepare the outloc
	outloc = io.prep_outloc(args)
	# Figure out how many simulations were asked for
	nsims = _parse_nsims(args.nsims)
	# Load the objects needed for the task
	simulator, prior, y, true_theta = io.load_task(args.task)

	# For simulation-based calibration
	N_RANK_COMPUTATIONS = 5_000
	N_BINS = 10#{5_000:512, 10_000:1024}[N_RANK_COMPUTATIONS]#int(N_RANK_COMPUTATIONS / 200.)
	ranks = None

	# On the basis of the value of args.method, decide how to wrap <simulator>
	# and how to then adjust output of prepare_for_sbi

	#################
	###### KDE ######
	#################
	if args.method[:3] == "kde":
		_raise_kde_warnings(args, nsims)

		simulator, y = _kde_prepare_simulator_observation(simulator, args, y)

		posteriors, samples = kde.kde_training(simulator,
											   prior,
											   y,
											   start=true_theta,
											   scale=args.scale,
											   n_sims=nsims,
											   R=args.R)
		if args.sbc:
			# TODO: write this script
			N_SAMPLES_PER_ITER = 10*N_BINS
			ranks = sbc.p_kde_sbc_script(simulator,
										 prior,
										 N_RANK_COMPUTATIONS,
										 N_BINS,
										 N_SAMPLES_PER_ITER,
										 scale=args.scale,
										 R=args.R,
										 n_jobs=60)

	##################
	# NEURAL METHODS #
	##################
	else:
		if args.sbc and isinstance(nsims, list):
			if len(nsims) > 1:
				error_msg = ("Currently no implementation of SBC with " + 
							 "non-amortised density (ratio) estimators " + 
							 "exists; consider resubmitting this job with " +
							 "one training round")
				raise NotImplemented(error_msg)

		#sim_pp = _get_postprocessor(args)

		# Determine whether to use hand-crafted or learned summary statistics
		density_estimator, sbi_method, y, simulator, sim_pp = _neural_prepare_estimator_observation(args, y, simulator)

		# Sample
		if sbi_method == "SNPE":
			N_SAMPLES = 10_000
			sampler = None
		elif sbi_method == "SNRE":
			if args.sampler == "sir":
				N_SAMPLES = 10_000
			else:
				N_SAMPLES = 100_000
			sampler = args.sampler
		if len(args.load_post) == 0:
			posteriors, samples = neural.sbi_training(simulator,
													  prior,
													  y,
													  sbi_method,
													  density_estimator, 
													  n_samples=N_SAMPLES,
													  n_sims=nsims,
													  sim_postprocess=sim_pp,
													  start=true_theta,
													  scale=args.scale,
													  sampler=sampler,
													  num_workers=args.nw,
													  z_score_x=args.task not in ["hop"],
													  outloc=outloc)
			io.save_output(posteriors, samples, ranks, outloc)
		else:
			with open(args.load_post, "rb") as fh:
				posteriors = pickle.load(fh)
			samples = None
		if args.sbc:
			if sbi_method == "SNPE":
				N_SAMPLES_PER_ITER = 2*(N_BINS - 1) 
				ranks = sbc.npe_sbc_script(posteriors[-1],
										   simulator,
										   prior,
										   N_RANK_COMPUTATIONS,
										   N_BINS,
										   N_SAMPLES_PER_ITER)
			elif sbi_method == "SNRE": 
				N_SAMPLES_PER_ITER = 100*N_BINS
				ranks = sbc.p_nre_sbc_script(posteriors[-1],
										   simulator,
										   prior,
										   N_RANK_COMPUTATIONS,
										   N_BINS,
										   N_SAMPLES_PER_ITER,
										   scale=args.scale,
										   n_jobs=2)
	
	io.save_output(posteriors, samples, ranks, outloc)
