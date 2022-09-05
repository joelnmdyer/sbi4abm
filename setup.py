from setuptools import find_packages, setup

setup(name='sbi4abm',
      version='0.1',
      description='Black-box Bayesian inference for agent-based models in the social sciences',
      url='http://github.com/joelnmdyer/sbi4abm',
      author='Joel Dyer',
      author_email='joel.dyer@maths.ox.ac.uk',
      license='AGPLv3',
      packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"])
)
