import numpy as np
from sbi4abm.sbi import inference, utils
from sbi4abm.sbi.inference import SNPE, SNRE, SNRE_A, simulate_for_sbi, prepare_for_sbi
from torch import nn

from sbi4abm.networks import graph, recurrent_graphs, time_series
from sbi4abm.utils import io, sampling


N_TRIAL_SAMPLES = 50_000

def create_density_estimator(embedding_kwargs, density_estimator):

	"""
	Input:

	- embedding_kwargs:			dict or None, if None no embedding net is used,
								else the dict is unpacked and creates the corresponding
								network
	- density_estimator:		str, in ["maf", "nsf", "made", "mdn"] for SNPE,
								["linear", "resnet", "mlp"] for SNRE
	"""

	if embedding_kwargs is None:
		embedding_net = nn.Identity()
	else:
		flavour = embedding_kwargs['flavour']
		if flavour == "gcn":
			# DON'T HARD-CODE THIS
			N_FEATURES = 1
			embedding_kwargs['input_dim'] = N_FEATURES
			embedding_net = graph.GCNEncoder(**embedding_kwargs)
			z_score_x = False
		elif flavour == "rgcn":
			embedding_net = recurrent_graphs.GConvGRUEmbedding(**embedding_kwargs)
			z_score_x = False
		else:
			embedding_net = time_series.RNN(**embedding_kwargs)
			z_score_x = True
	num_pars = sum(p.numel() for p in embedding_net.parameters() if p.requires_grad)
	print("Embedding net has {0} trainable parameters".format(num_pars))
	# If it's a conditional density estimator, assume it's for posterior estimation.
	# Then just use default settings (since these are from benchmarking paper)
	if density_estimator in ["maf", "nsf", "made", "mdn"]:
		density_estimator = utils.posterior_nn(model=density_estimator,
											   embedding_net=embedding_net,
											   z_score_x=z_score_x)
	# Else it's a ratio estimator (classifier), so just use default settings again
	elif density_estimator in ["linear", "mlp", "resnet"]:
		density_estimator = utils.get_nn_models.classifier_nn(density_estimator,
															  embedding_net_x=embedding_net,
															  z_score_x=z_score_x)
	return density_estimator, z_score_x


def sbi_training(simulator,
				 prior,
				 y,
				 method,
				 density_estimator,
				 n_samples=10_000,
				 n_sims=[10_000], 
				 sim_postprocess=lambda x: x,
				 sampler=None,
				 start=None,
				 scale=None,
				 num_workers=15,
				 z_score_x=True,
				 outloc=None):

	sbi_simulator, sbi_prior = prepare_for_sbi(simulator, prior)
	posteriors = []
	proposal = sbi_prior
	if method == "SNPE":
		inference = SNPE(prior=sbi_prior, density_estimator=density_estimator)
	elif method == "SNRE":
		inference = SNRE(prior=sbi_prior, classifier=density_estimator)

	for sim_count in n_sims:
		theta, x = simulate_for_sbi(sbi_simulator, proposal, num_simulations=sim_count,
									num_workers=num_workers)
		# This is usually for reshaping for the embedding net
		x = sim_postprocess(x)
		print("Shape of simulated batch of data", x.size())
		if method == "SNPE":
			density_estimator = inference.append_simulations(theta, x, proposal=proposal)
			print("Train")
			density_estimator = density_estimator.train(z_score_x=z_score_x)
		elif method == "SNRE":
			density_estimator = inference.append_simulations(theta, x)
			print("Train")
			density_estimator = density_estimator.train(z_score_x=z_score_x)
		posterior = inference.build_posterior(density_estimator)
		posteriors.append(posterior)
		proposal = posterior.set_default_x(y)

	if not outloc is None:
		io.save_output(posteriors, None, None, outloc)

	if n_samples > 0:
		if sampler is None:
			samples = proposal.sample((n_samples,))
		elif sampler == "mh":
			_start = start
			if _start is None:
				_start = prior.sample().numpy()
			if scale is None:
				scale = 2/np.sqrt(_start.size)
			trial_samples = sampling.mh(proposal.log_prob, _start, n_samples=N_TRIAL_SAMPLES,
								  scale=scale, to_torch=True)
			cov = np.cov(trial_samples.T)
			if start is None:
				start = np.mean(trial_samples[::50], axis=0)
			samples = sampling.mh(proposal.log_prob, start, n_samples=n_samples,
								  cov=cov, scale=scale, to_torch=True)
		elif sampler == "sir":
			prior_samples = prior.sample((20*n_samples,))
			weights = np.exp(proposal.log_prob(prior_samples).numpy()
							 - prior.log_prob(prior_samples).numpy())
			weights = weights.reshape(-1)
			idx = np.random.choice(np.arange(weights.shape[0]), p=weights/np.sum(weights),
								   size=n_samples)
			samples = prior_samples.numpy()[list(idx)]
		else:
			samples = proposal.sample((n_samples,), mcmc_method=sampler)
	else:
		samples = None
	return posteriors, samples
