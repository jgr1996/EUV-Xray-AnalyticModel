from __future__ import division
import os
import sys
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
from mpi4py import MPI
import evolve_population
from evolve_population import bernstein_poly


def likelihood(theta, N, data_array, completeness_array):

    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank        # rank of this process

    if any(n < 0.0 for n in theta):
        return -np.inf

    if any(n > 10.0 for n in theta):
        return -np.inf

    # ensure Bernstein CDFs are monotonically increasing
    a_range = np.arange(0.0, 1.0001, 0.0001)

    initial_X_coeffs = [theta[i] for i in range(6)]
    X_min, X_max = -3.0, 0.0
    X_poly_min, X_poly_max = bernstein_poly(0, 5, initial_X_coeffs), bernstein_poly(1, 5, initial_X_coeffs)
    X_poly_norm = X_poly_max - X_poly_min
    X_norm = X_max - X_min
    CDF_X = [(((X_norm/X_poly_norm) * ((bernstein_poly(i, 5, initial_X_coeffs)) - X_poly_min)) + X_min) for i in a_range]
    for i in range(1,len(CDF_X)):
        if CDF_X[i-1] > CDF_X[i]:
            # print "Not increasing CDF for X"
            return -np.inf

    core_mass_coeffs = [theta[i+6] for i in range(6)]
    M_min, M_max = 0.5, 10.0
    M_poly_min, M_poly_max = bernstein_poly(0, 5, core_mass_coeffs), bernstein_poly(1, 5, core_mass_coeffs)
    M_poly_norm = M_poly_max - M_poly_min
    M_norm = M_max - M_min
    CDF_M = [(((M_norm/M_poly_norm) * ((bernstein_poly(i, 5, core_mass_coeffs)) - M_poly_min)) + M_min) for i in a_range]
    for i in range(1,len(CDF_M)):
        if CDF_M[i-1] > CDF_M[i]:
            # print "Not increasing CDF for M"
            return -np.inf


    # calculate population
    # time1 = time.time()
    R, P, M, X, R_core = evolve_population.CKS_synthetic_observation(N, theta)
    # time2 = time.time()


    if rank == 0:
        # print "It took {0} seconds for {1} planets".format(time2-time1, len(R))
        if len(R) <= 0.5 * N:
            return -np.inf

        x = np.logspace(-1,2,150)
        y = np.logspace(-1,1.5,150)
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([np.log(X.ravel()), np.log(Y.ravel())])
        data = np.vstack([np.log(P),np.log(R)])
        kernel = stats.gaussian_kde(data)
        Z = np.reshape(kernel(positions).T, X.shape)
        Z_biased = Z * completeness_array.T
        KDE_interp = interpolate.RectBivariateSpline(y,x,Z_biased)

        P_data = data_array[3,:]
        R_data = data_array[2,:]

        logL = 0
        for i in range(len(P_data)):
            # MUST SWAP ORDER OF P AND R!!!
            logL_i = KDE_interp(R_data[i], P_data[i])[0,0]
            logL = logL + np.log10(abs(logL_i))

        # plt.figure(1)
        # plt.contourf(x,y,Z, cmap='Oranges')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.scatter(P, R, s=0.5, color='black')
        #
        # plt.figure(2)
        # plt.contourf(x,y,Z_biased, cmap='Oranges')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.scatter(P_data, R_data, s=0.5, color='black')
        # plt.show()

        return logL
    else:
        return 0




# ///////////////////////////////// TEST ///////////////////////////////////// #

# comm = MPI.COMM_WORLD   # get MPI communicator object
# rank = comm.rank        # rank of this process
# CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
# CKS_completeness_array = np.loadtxt("survey_completeness.txt", delimiter=',')
# for i in range(1):
#     L = likelihood([0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 6.0], 1000, CKS_array, CKS_completeness_array)
#
#     if rank == 0:
#         print L
