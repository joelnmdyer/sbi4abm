from setuptools import find_packages, setup

REQUIRED = [
    "arviz",
    "joblib>=1.0.0",
    "matplotlib",
    "numba",
    "numpy",
    "pillow",
    "POT",
    "pyknos>=0.14.2",
    "pyro-ppl>=1.3.1",
    "scikit-learn",
    "scipy",
    "tensorboard",
    "torch>=1.8.0",
    "torch_geometric==2.3",
    "torch-geometric-temporal",
    "tqdm",
]

setup(name='sbi4abm',
      version='0.1',
      description='Black-box Bayesian inference for agent-based models in the social sciences',
      url='http://github.com/joelnmdyer/sbi4abm',
      author='Joel Dyer',
      author_email='joel.dyer@maths.ox.ac.uk',
      license='AGPLv3',
      packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
      install_requires=REQUIRED
)
