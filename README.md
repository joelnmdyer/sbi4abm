# sbi4abm
This is the code used for two papers:

**Black-box Bayesian inference for economic agent-based models**\
_Dyer, J; Cannon, P; Farmer, J. D.; Schmon, S. M._\
arxiv 2202:00625 (2022)

and

**Calibrating agent-based models to microdata with graph neural networks**\
_Dyer, J; Cannon, P; Farmer, J. D.; Schmon, S. M._\
Soptlight Paper at the ICML 2022 Workshop on AI for Agent-based Modelling (2022)

Citation info below:

```
@article{dyer2022black,
  title={Black-box Bayesian inference for economic agent-based models},
  author={Dyer, Joel and Cannon, Patrick and Farmer, J Doyne and Schmon, Sebastian},
  journal={arXiv preprint arXiv:2202.00625},
  year={2022}
}
```
and
```
@article{dyer2022calibrating,
         title={Calibrating Agent-based Models to Microdata with Graph Neural Networks},
         author={Dyer, Joel and Cannon, Patrick and Farmer, J Doyne and Schmon, Sebastian M},
         journal={arXiv preprint arXiv:2206.07570},
         year={2022}
}
```

This package makes use of the excellent `sbi` package (https://github.com/mackelab/sbi). I changed a few lines in the source code for `sbi` to allow $z$-scoring of data to be specified as a boolean argument for `SNPE` and `SNRE`, and I've included the modified package in this repository for that reason.

## Example usage
If you want to get a parameter posterior out for the Hopfield model using a recurrent graph neural network and masked autoregressive flow with a budget of 1000 simulations from the ABM to train the density estimator, you can navigate to the `sbi4abm/utils` folder and run
```
python job_script.py --task hop --method maf_rgcn --outloc <location_to_save_output> --nsims 1000
```
