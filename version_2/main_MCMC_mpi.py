import os
import sys
import datetime
import time
import emcee
import numpy as np
from numpy import random as rand
from mpi4py import MPI
import likelihood_function
import evolve_population



if __name__ == '__main__':

    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank        # rank of this process
    rand.seed(12345)

    # initial guess [X_mean, X_stdev, M_core_mean, M_core_stdev, period_power, period_cutoff]
    theta = [0.06, 0.42, 7.72, 1.86]
    ndim = len(theta)
    n_walkers = 100
    n_iterations = 10000

    theta_guesses = []
    for i in range(n_walkers):
        theta_guesses.append([x + rand.uniform(0, 1e-2*x) for x in theta])

    # get CKS data radius bins
    data_histogram = likelihood_function.make_CKS_histograms()
    N = np.sum(data_histogram)

    if rank == 0:
        # use time as label for output directory
        current_time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y_%H.%M.%S')

        newpath = './RESULTS/{0}'.format(current_time_string)
        if not os.path.exists(newpath):
            os.makedirs(newpath)


        file = open("./RESULTS/{0}/simulation_details.txt".format(current_time_string), "w")
        file.write("----------------- MCMC Simulation ------------------\n")
        file.write("Parameters estimated: [X_mean, X_stdev, M_mean, M_stdev]\n")
        file.write("Initial guess localised to: {}\n".format(theta))
        file.write("Number of Walkers: {}\n".format(n_walkers))
        file.write("Number of iterations: {}\n".format(n_iterations))
        file.write("Number of cores: 24\n")
        file.write("----------------------------------------------------\n")
        file.write("Notes: Please Work...\n")
        file.close()


    sampler = emcee.EnsembleSampler(n_walkers,
                                    ndim,
                                    likelihood_function.likelihood_R_space,
                                    args=(N, data_histogram))


    iteration = 0
    for result in sampler.sample(theta_guesses, iterations=n_iterations, storechain=False):
        if rank == 0:
            position = result[0]
            np.savetxt('./RESULTS/{0}/position_{1}.csv'.format(current_time_string, iteration), position, delimiter=',')

        iteration = iteration + 1
