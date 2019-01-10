import os
import sys
import datetime
import time
import multiprocessing
import emcee
import numpy as np
from numpy import random as rand
from joblib import Parallel, delayed
import likelihood_function
import evolve_population



if __name__ == '__main__':

    # use time as label for output directory
    current_time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y_%H.%M.%S')

    # get CKS data radius bins
    data_bins, data_histogram = likelihood_function.make_CKS_histograms()
    N = np.sum(data_histogram)

    # initial guess [X_mean, X_stdev, M_core_mean, M_core_stdev, period_power, period_cutoff]
    # theta = [0.1, 0.05, 2.0, 0.4, 1.9, 7.6]
    theta = [2.0, 0.4]
    ndim = len(theta)
    nwalkers = 4

    theta_guesses = []
    for i in range(nwalkers):
        theta_guesses.append([x + rand.uniform(0, 1e-3*x) for x in theta])


    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    likelihood_function.likelihood_R_space,
                                    args=(N, current_time_string, data_histogram))

    # sampler.run_mcmc(theta_guesses, 2)
    # results = sampler.chain
    # print results

    for result in sampler.sample(theta_guesses, iterations=2, storechain=False):
        position = result[0]
        np.savetxt('./RESULTS/{}/chain.csv'.format(current_time_string), position, delimiter=',')


    # theta1 = [0.1, 0.05, 2.0, 0.4, 1.9, 7.6]
    # theta2 = [0.1, 0.05, 6.0, 0.1, 1.9, 7.6]
    #
    # L1 = likelihood_function.likelihood_R_space(theta1, N, current_time_string, data_histogram)
    # L2 = likelihood_function.likelihood_R_space(theta2, N, current_time_string, data_histogram)
    #
    # print 'L1 =', L1
    # print 'L2 =', L2
