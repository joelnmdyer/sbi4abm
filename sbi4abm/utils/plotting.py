import matplotlib.pyplot as plt
import scipy.stats

def plot_sbc_hist(hist, labels=[], limits=[], perc=.99):

	num_prior_samples = hist.sum()/hist.shape[1]
	d = hist.shape[1]
	if len(labels) == 0:
		labels = ["" for i in range(d)]
	nbins = hist.shape[0]
	min_perc = (1. - perc)/2.
	max_perc = min_perc + perc
	low_perc = scipy.stats.binom.ppf(min_perc, num_prior_samples, 1./nbins)
	high_perc = scipy.stats.binom.ppf(max_perc, num_prior_samples, 1./nbins)
	fig, axes = plt.subplots(1, d, figsize=(15,4))
	for j, ax in enumerate(axes):
		ax.bar(range(nbins), hist[:, j], width=1., edgecolor='black')
		if len(limits) > 0:
			ax.set_ylim(limits[j])
		ax.axhspan(low_perc, high_perc, facecolor='grey', alpha=0.3)
		ax.axhline(num_prior_samples/nbins, color='black')
		ax.set_xlabel(labels[j], fontsize=15)
		ax.set_yticks([])
	plt.show(block=False)
