import os
import sys
import datetime
import time
import emcee
import numpy as np
from numpy import random as rand
from mpi4py import MPI
import multiprocessing
import likelihood_function
import evolve_population



if __name__ == '__main__':

    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank        # rank of this process

    # initial guess [X_poly_coeffs, M_poly_coeffs, density_mean]
    theta = [0.03, 0.21, 0.47, 0.25, 0.41, 0.95, 0.04, 0.09, 0.41, 0.55, 0.40, 1.36, 6.0]


    ndim = len(theta)
    n_walkers = 100
    n_iterations = 500

    theta_guesses = []
    for i in range(n_walkers):
        theta_guesses.append([x + rand.uniform(0, 1e-1*x) for x in theta])
    # theta_guesses = np.loadtxt("./RESULTS/09.03.2019_04.20.20/position_80.csv", delimiter=',')

    N = 5000

    if rank == 0:
        # use time as label for output directory
        current_time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y_%H.%M.%S')

        newpath = './RESULTS/{0}'.format(current_time_string)
        if not os.path.exists(newpath):
            os.makedirs(newpath)


        file = open("./RESULTS/{0}/simulation_details.txt".format(current_time_string), "w")
        file.write("----------------- MCMC Simulation ------------------\n")
        file.write("Parameters estimated: [Initial X Bernstein Polynomial Coefficients, Core Mass Bernstein Polynomials Coefficients, Mean Core Density]\n")
        file.write("Initial guess localised to: {}\n".format(theta))
        file.write("Number of Walkers: {}\n".format(n_walkers))
        file.write("Number of iterations: {}\n".format(n_iterations))
        file.write("Number of cores: {}\n".format(multiprocessing.cpu_count()))
        file.write("----------------------------------------------------\n")
        file.write("Notes: Please Work...\n")
        file.close()

    CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
    sampler = emcee.EnsembleSampler(n_walkers,
                                    ndim,
                                    likelihood_function.likelihood_PR_space,
                                    args=(N, CKS_array))


    iteration = 0
    for result in sampler.sample(theta_guesses, iterations=n_iterations, storechain=False):
        if rank == 0:
            position = result[0]
            np.savetxt('./RESULTS/{0}/position_{1}.csv'.format(current_time_string, iteration), position, delimiter=',')

        iteration = iteration + 1
