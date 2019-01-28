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
    theta = [0.1, 0.05, 2.0, 0.4]
    ndim = len(theta)
    nwalkers = 8

    theta_guesses = []
    for i in range(nwalkers):
        theta_guesses.append([x + rand.uniform(0, 1e-2*x) for x in theta])


    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    likelihood_function.likelihood_R_space,
                                    args=(N, current_time_string, data_histogram))


    iteration = 0
    for result in sampler.sample(theta_guesses, iterations=1000, storechain=False):
        position = result[0]
        np.savetxt('./RESULTS/{0}/position_{1}.csv'.format(current_time_string, iteration), position, delimiter=',')
        iteration = iteration + 1


    # chain = []
    # for i in range(iteration):
    #     position_i = np.loadtxt("./RESULTS/{0}/position_{1}.csv".format(current_time_string, i), delimiter=',')
    #     chain.append(position_i)
    #
    # np.savetxt("chain_final.csv", chain, delimiter=",")
