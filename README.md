# sbi4abm: simulation-based inference for agent-based modelling

Calibrating agent-based model (ABM) parameters is a challenging task: ABMs are expensive, stochastic, dynamical models that can generate complex, non-equilibrium, and high-dimensional time-series data as output. Furthermore, ABMs typically do not have tractable likelihood functions. Procedures for _simulation-based inference_ are thus indispensable in this setting, but many of these are not well-suited to ABMs, often requiring many hundreds of thousands of calls to the simulator and/or inappropriate assumptions on the form of the data generated by the ABM. This limits their applicability to arbitrary ABMs.

In this repository, we provide code that builds on the [`sbi` package](https://github.com/mackelab/sbi) to allow agent-based modellers to perform **black-box Bayesian estimation** of an ABM's parameters using powerful simulation-based inference procedures based on neural networks. These procedures -- known as neural posterior estimation (`NPE`) and neural density ratio estimation (`NRE`) -- have been seen to generate remarkably accurate parameter posteriors, capturing the uncertainty in parameter estimates using orders of magnitude fewer simulation runs than more conventional inference methods. Recent research (see **Papers** section below) has shown that such approaches are highly-suited to the particular challenges faced when estimating ABM parameters, and can flexibly and automatically handle high-dimensional and non-linear data of various types that would arise in ABM settings without the need to impose inappropriate assumptions on the form of the ABM or on the data it generates.

## Papers
The code in this repository has been used in the following two papers:

[**Black-box Bayesian inference for agent-based models**](https://arxiv.org/abs/2202.00625)\
_Dyer, J.; Cannon, P.; Farmer, J. D.; Schmon, S. M._\
Forthcoming in the Journal of Economic Dynamics and Control (2024)\
[45-minute talk on this paper on INET Oxford YouTube](https://www.youtube.com/watch?v=yVNE8focE30)

and

[**Calibrating agent-based models to microdata with graph neural networks**](https://openreview.net/pdf?id=ZWyHGTUcgJD)\
_Dyer, J.; Cannon, P.; Farmer, J. D.; Schmon, S. M._\
Soptlight Paper and Best Short Paper at the ICML 2022 Workshop on AI for Agent-based Modelling (2022)\
[5-minute video from the ICML 2022 AI4ABM Workshop](https://slideslive.com/38985937)

These papers can be cited using the following citation info:

```
@article{dyer2022black,
  title={{Black-box Bayesian inference for economic agent-based models}},
  author={Dyer, Joel and Cannon, Patrick and Farmer, J Doyne and Schmon, Sebastian},
  journal={arXiv preprint arXiv:2202.00625},
  year={2022}
}
```
and
```
@article{dyer2022calibrating,
         title={{Calibrating Agent-based Models to Microdata with Graph Neural Networks}},
         author={Dyer, Joel and Cannon, Patrick and Farmer, J Doyne and Schmon, Sebastian M},
         journal={arXiv preprint arXiv:2206.07570},
         year={2022}
}
```

This package makes use of the [`sbi` package](https://github.com/mackelab/sbi), and citation info on the relevant papers can be found in the documentation for this package. I changed a few lines in the source code for `sbi` to allow $z$-scoring of data to be specified as a boolean argument for `SNPE` and `SNRE`, and I've included the modified package in this repository for that reason.

## Example usage
If you want to get a parameter posterior out for the Hopfield model using a recurrent graph neural network and masked autoregressive flow with a budget of 1000 simulations from the ABM to train the density estimator, you can navigate to the `sbi4abm/utils` folder and run
```
python job_script.py --task hop --method maf_rgcn --outloc <location_to_save_output> --nsims 1000
```

## Applying these methods to a new model
Applying these methods to a new model not contained in this respository can be done like so:
* Add the model to the `models` folder, wrapping the model up in the same format as in the examples contained in that folder
* Add details on the prior density, model instantiation, and "true" data in the `utils/io.py` file
* Ensure the density (ratio) estimator is equipped with an appropriate embedding network in the `utils/job_script.py` and `inference/neural.py` files
* Training the density (ratio) estimator using the same command as in the **Example usage** section above
