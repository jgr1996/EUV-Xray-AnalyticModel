# EUV/Xray Analytic Model

This program is a Bayesian hierarchical model for constraining the underlying planetary distribution of close-in exoplanets. This is achieved by exploiting the photoevaporation valley uncovered from the California Kepler Survey (CKS) (Fulton et al. 2017) and the analytic model proposed by Owen and Wu (2017).

# Versions

- Version 1.0 (Decapricated)
  * This is parallelised using an the python library 'joblib'
- Version 2.0 (Decapricated)
  * This is parallelised using MPI
  * Priors restrict core mass and initial envelope mass functions to be log-normal
- Version 3.0 (Decapricated)
  * MPI parallelisation with arbitrary core mass and initial envelope mass functions using 6th order Bernstein polynomials
  * Likelihood function is constructed by calculating KDE of model for a set of parameters and evaluating likelihoods summing CKS values
  - Version 4.0 (under construction)
  * Cythonised and MPI'ed for speedup requirements
  * Likelihood function modified to avoid unnecessary planet calculation
  * Can work with log-normal distirbution or Bernstein polynomials
 

# Files

- main_MCMC.py
  * runs the hierarchical model using emcee
- likelihood_function.py
  * Calculates KDE's required for likelihood function
- evolve_population.py
  * Draws and evolves an ensemble of planets through photoevaporation
- mass_fraction_evolver.pyx
  * Cythonised python file to evolve a single planet through photoevaporation according to analytic model from Owen and Wu 2017
  


# Data and Analysis

- All MCMC walker positions are saved to a "./RESULTS" directory. All analysis can be done from there.
