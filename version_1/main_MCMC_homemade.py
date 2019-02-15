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
from metropolis_alg import *



if __name__ == '__main__':

    # use time as label for output directory
    current_time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y_%H.%M.%S')

    newpath = './RESULTS/{0}'.format(current_time_string)
    if not os.path.exists(newpath):
        os.makedirs(newpath)


    # get CKS data radius bins
    data_histogram = likelihood_function.make_CKS_histograms()
    N = np.sum(data_histogram)

    # initial guess [X_mean, X_stdev, M_core_mean, M_core_stdev, period_power, period_cutoff]
    # theta = [0.1, 0.05, 2.0, 0.4, 1.9, 7.6]
    # theta = [0.026, 0.37, 7.7, 0.29]
    theta = [3.0, 0.5]
    ndim = len(theta)
    n_iterations = 1000


    file = open("./RESULTS/{0}/simulation_details.txt".format(current_time_string), "w")
    file.write("----------------- MCMC Simulation ------------------\n")
    # file.write("Parameters estimated: [X_mean, X_stdev, M_mean, M_stdev]\n")
    file.write("Parameters estimated: [M_mean, M_stdev]\n")
    file.write("Initial guess: {}\n".format(theta))
    file.write("Number of iterations: {}\n".format(n_iterations))
    file.write("----------------------------------------------------\n")
    file.write("Extra info:\n")
    file.close()


    parameter_samples, L_samples, R = metropolis_alg(likelihood_function.likelihood_R_space, (N, current_time_string, data_histogram), theta, n_iterations, 0.01)

    np.savetxt('./RESULTS/{}/positions.csv'.format(current_time_string), parameter_samples, delimiter=',')
    np.savetxt('./RESULTS/{}/likelihoods.csv'.format(current_time_string), L_samples, delimiter=',')
