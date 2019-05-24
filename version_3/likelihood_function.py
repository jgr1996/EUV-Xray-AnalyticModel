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
    if rank == 0:

        if any(n < 0.0 for n in theta):
            return -np.inf

    R, P, M, X, R_core, R_rejected, P_rejected, M_rejected, X_rejected, R_core_rejected = evolve_population.CKS_synthetic_observation(N, theta)

    if rank == 0:

        if len(R) <= 0.5 * N:
            return -np.inf

        x = np.logspace(-1,2,150)
        y = np.logspace(-1,1.5,150)
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([np.log(X.ravel()), np.log(Y.ravel())])
        data = np.vstack([np.log(P),np.log(R)])
        kernel = stats.gaussian_kde(data)
        Z = np.reshape(kernel(positions).T, X.shape)
        Z_norm = 1 / np.sum(Z)
        Z = Z_norm * Z

        KDE_interp = interpolate.RectBivariateSpline(y,x,Z)

        P_data = data_array[3,:]
        R_data = data_array[2,:]

        logL = 0
        for i in range(len(P_data)):
            # MUST SWAP ORDER OF P AND R!!!
            logL_i = KDE_interp(R_data[i], P_data[i])[0,0]
            if logL_i <= 0.0:
                logL = logL - 300
            else:
                logL = logL + np.log(abs(logL_i))

        return logL

    else:
        return 0.0



# ///////////////////////////////// TEST ///////////////////////////////////// #

# comm = MPI.COMM_WORLD   # get MPI communicator object
# rank = comm.rank        # rank of this process
# CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
# for i in range(5):
#     L = likelihood_PR_space([0.03, 0.21, 0.47, 0.25, 0.41, 0.95, 0.04, 0.09, 0.41, 0.55, 0.40, 1.36, 5.0],
#                             1000,
#                             CKS_array)
#
#     if rank == 0:
#         print L
