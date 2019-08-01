from __future__ import division
import os
import sys
import datetime
import time
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy import stats
from scipy import interpolate
from mpi4py import MPI
import evolve_population
from evolve_population import bernstein_poly


def likelihood(theta, N, data_array, completeness_array):

    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank        # rank of this process

    # bernstein coeffs cannot be less than 0.0
    if any(n < 0.0 for n in theta[2:]):
        return -np.inf

    # max for bernstein coeffs
    if any(n > 10.0 for n in theta[2:0]):
        return -np.inf

    # density cannot be negative
    if theta[0] > 10.0 or theta[0] < 0.0:
        return -np.inf


    # # ensure Bernstein CDFs are monotonically increasing
    # a_range = np.arange(0.0, 1.0001, 0.0001)
    #
    # initial_X_coeffs = [theta[i] for i in range(6)]
    # X_min, X_max = -3.0, 0.0
    # X_poly_min, X_poly_max = bernstein_poly(0, 5, initial_X_coeffs), bernstein_poly(1, 5, initial_X_coeffs)
    # X_poly_norm = X_poly_max - X_poly_min
    # X_norm = X_max - X_min
    # CDF_X = [(((X_norm/X_poly_norm) * ((bernstein_poly(i, 5, initial_X_coeffs)) - X_poly_min)) + X_min) for i in a_range]
    # for i in range(1,len(CDF_X)):
    #     if CDF_X[i-1] > CDF_X[i]:
    #         print "Not increasing CDF for X"
    #         return -np.inf
    #
    # core_mass_coeffs = [theta[i+6] for i in range(6)]
    # M_min, M_max = 0.5, 10.0
    # M_poly_min, M_poly_max = bernstein_poly(0, 5, core_mass_coeffs), bernstein_poly(1, 5, core_mass_coeffs)
    # M_poly_norm = M_poly_max - M_poly_min
    # M_norm = M_max - M_min
    # CDF_M = [(((M_norm/M_poly_norm) * ((bernstein_poly(i, 5, core_mass_coeffs)) - M_poly_min)) + M_min) for i in a_range]
    # for i in range(1,len(CDF_M)):
    #     if CDF_M[i-1] > CDF_M[i]:
    #         print "Not increasing CDF for M"
    #         return -np.inf


    # calculate population
    # time1 = time.time()
    R, P, M, X, R_core, M_star = evolve_population.CKS_synthetic_observation(N, theta)
    # time2 = time.time()

    if rank == 0:
        # print "It took {0} seconds for {1} planets".format(time2-time1, len(R))
        if len(R) <= 0.5 * N:
            return -np.inf

        x = np.logspace(-1.0,2.5,150)
        y = np.logspace(-1.0,1.5,150)
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([np.log(X.ravel()), np.log(Y.ravel())])
        data = np.vstack([np.log(P),np.log(R)])
        kernel = stats.gaussian_kde(data, bw_method=0.2)
        Z = np.reshape(kernel(positions).T, X.shape)
        Z_biased = Z * completeness_array.T
        KDE_interp = interpolate.RectBivariateSpline(y,x,Z_biased)


        R_data = []
        P_data = []
        for i in range(len(data_array[3,:])):
            if data_array[2,i] >= 0.8 and data_array[2,i] <= 6.0:
                R_data.append(data_array[2,i])
                P_data.append(data_array[3,i])

        logL = 0
        for i in range(len(P_data)):
            # MUST SWAP ORDER OF P AND R!!!
            logL_i = KDE_interp(R_data[i], P_data[i])[0,0]
            logL = logL + np.log10(abs(logL_i))

        np.savetxt("../R_test.csv", R, delimiter=',')
        np.savetxt("../Mstar_test.csv", M_star, delimiter=',')
        np.savetxt("../P_test.csv", P, delimiter=',')
        plt.figure(1)
        plt.contourf(x,y,Z, cmap='Oranges')
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(P, R, s=0.3, color='black', alpha=0.3)
        plt.xlim([1,100])
        plt.ylim([1,12])
        plt.savefig("../Figures/biased.pdf")

        plt.figure(3)
        plt.contourf(x,y,Z_biased, cmap='Oranges')
        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(P_data, R_data, s=0.5, color='black', alpha=0.3)
        plt.xlim([1,100])
        plt.ylim([1,12])
        plt.savefig("../Figures/unbiased.pdf")
        # plt.show()

        return logL
    else:
        return 0




# ///////////////////////////////// TEST ///////////////////////////////////// #

comm = MPI.COMM_WORLD   # get MPI communicator object
rank = comm.rank        # rank of this process
CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
CKS_completeness_array = np.loadtxt("survey_completeness.txt", delimiter=',')
# thetas = [[0.04,0.55,6.52,1.39,3.62]]#,[0.10, 0.43, 7.79, 1.55, 5.50]]
thetas = [[5.25,0.76,0.02,5.54,0.07,1.54,6.63,2.58]]

for i in range(len(thetas)):
    L = likelihood(thetas[i], 5000, CKS_array, CKS_completeness_array)
    if rank == 0:
        print "Theta = {}".format(thetas[i])
        print "logL = {}".format(L)
