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


def likelihood(theta, N, data_array):

    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank        # rank of this process

    if any(n < 0.0 for n in theta):
        return -np.inf

    if any(n > 10.0 for n in theta):
        return -np.inf

    time1 = time.time()
    R, P, M, X, R_core, R_rejected, P_rejected, M_rejected, X_rejected, R_core_rejected = evolve_population.CKS_synthetic_observation(N, theta)
    time2 = time.time()
    print "It took {0} seconds for {1} planets".format(time2-time1, len(R))

    if rank == 0:

        if len(R) <= 0.5 * N:
            print "Only {0} planets for {1}".format(len(R), theta)
            return -np.inf

        x = np.logspace(-1,2,150)
        y = np.logspace(-1,1.5,150)
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([np.log(X.ravel()), np.log(Y.ravel())])
        data = np.vstack([np.log(P),np.log(R)])
        kernel = stats.gaussian_kde(data)
        Z = np.reshape(kernel(positions).T, X.shape)
        KDE_interp = interpolate.RectBivariateSpline(y,x,Z)

        P_data = data_array[3,:]
        R_data = data_array[2,:]

        logL = 0
        for i in range(len(P_data)):
            # MUST SWAP ORDER OF P AND R!!!
            logL_i = KDE_interp(R_data[i], P_data[i])[0,0]
            logL = logL + np.log10(abs(logL_i))

        # plt.contourf(x,y,KDE_interp(y,x), cmap='gnuplot')
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.scatter(P_data, R_data, s=0.5, color='black')
        # plt.show()

        return logL / len(P)
    else:
        return 0




# ///////////////////////////////// TEST ///////////////////////////////////// #

# comm = MPI.COMM_WORLD   # get MPI communicator object
# rank = comm.rank        # rank of this process
# CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
# for i in range(1):
#     L = likelihood([0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 6.5], 1000, CKS_array)
#
#     if rank == 0:
#         print L
