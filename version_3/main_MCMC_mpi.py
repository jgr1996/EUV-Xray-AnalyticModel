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

    density_guess = [5.0]          # core density
    # period_guess = [1.9,0.01,7.9]  # period powers and break
    X_guess = [0.026,2.34]         # env mass fraction mean and std
    M_guess = [7.7,1.95]           # core mass mean and std

    theta = density_guess + X_guess + M_guess

    ndim = len(theta)
    n_walkers = 100
    n_iterations = 10000

    theta_guesses = []
    for i in range(n_walkers):
        theta_guesses.append([x + rand.uniform(0, 5e-2*x) for x in theta])

    N = 1000

    if rank == 0:
        # use time as label for output directory
        current_time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y_%H.%M.%S')

        newpath = './RESULTS/{0}'.format(current_time_string)
        if not os.path.exists(newpath):
            os.makedirs(newpath)


        file = open("./RESULTS/{0}/simulation_details.txt".format(current_time_string), "w")
        file.write("----------------- MCMC Simulation ------------------\n")
        file.write("Parameters estimated: [density, period_params, X_params, M_params]\n")
        file.write("Initial guess localised to: {}\n".format(theta))
        file.write("Number of Walkers: {}\n".format(n_walkers))
        file.write("Number of iterations: {}\n".format(n_iterations))
        file.write("----------------------------------------------------\n")
        file.write("Notes: Just Work...\n")
        file.close()

    CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
    CKS_completeness_array = np.loadtxt("survey_comp.csv", delimiter=',')
    # step_size = emcee.moves.GaussianMove(cov=step_size)
    sampler = emcee.EnsembleSampler(n_walkers,
                                    ndim,
                                    likelihood_function.likelihood,
                                    args=(N, CKS_array, CKS_completeness_array))


    iteration = 0
    for result in sampler.sample(theta_guesses, iterations=n_iterations, store=True):
        if rank == 0:

            # save current chain status
            np.savetxt('./RESULTS/{0}/position_{1}.csv'.format(current_time_string, iteration), result.coords, delimiter=',')
            iteration = iteration + 1

            # calculate autocorreclation time every 100 iterations
            if sampler.iteration % 100:
                continue
            tau = sampler.get_autocorr_time(tol=0)
            file = open("./RESULTS/{0}/Autocorrelation_{1}.txt".format(current_time_string,iteration), "w")
            file.write("The autocorreclation time is {0} for iteration {1} \n".format(tau, sampler.iteration))
            file.close()
