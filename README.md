# EUV/Xray Analytic Model

This program is a Bayesian hierarchical model for constraining the underlying planetary distribution of close-in exoplanets. This is achieved by exploiting the photoevaporation valley uncovered from the California Kepler Survey (CKS) (Fulton et al. 2017) and the analytic model proposed by Owen and Wu (2017).

# Versions

- Version 1.0 (stable)
  * This is parallelised using an the python library 'joblib'
- Version 2.0 (stable)
  * This is parallelised using MPI
  * Priors restrict core mass and initial envelope mass functions to be log-normal
- Version 3.0 (under construction)
  * MPI parallelisation with arbitrary core mass and initial envelope mass functions using 6th order Bernstein polynomials
  * Likelihood function is constructed by calculating KDE of model for a set of parameters and evaluating likelihoods summing CKS values
- Version 4.0 (under construction)
  * MPI parallelisation with arbitrary core mass and initial envelope mass functions using 6th order Bernstein polynomials
  * This is an Approximate Bayesian Calculation (ABC) using a Kullback-Leibler (KL) divergence as statistic for likelihood proxy.

# Files

- main_MCMC.py
  * run this for constraining the 1D radius space histogram.

- main_MCMC2D.py
  * run this for constraining the 2D period-radius space histogram.

# Data and Analysis

- All MCMC walker positions are saved to a "./RESULTS" directory. All analysis can be done from there.
