# EUV/Xray Analytic Model

This program is a Bayesian hierarchical model for constraining the underlying planetary distribution of close-in exoplanets. This is achieved by exploiting the photoevaporation valley uncovered from the California Kepler Survey (CKS) (Fulton et al. 2017) and the analytic model proposed by Owen and Wu (2017). 

Script is parallelised using MPI and optimised using cython.

# Versions

 - Gaussian 
   * Fits the core composition between -1 (pure ice) and +1 (pure iron), as well as the core mass distribution and initial envelope mass fraction distributions as log-Gaussians.
 - Bernstein 
   * Fits the core composition between -1 (pure ice) and +1 (pure iron), as well as the core mass distribution and initial envelope mass fraction distributions using Bernstein polynomials.
 

 

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
