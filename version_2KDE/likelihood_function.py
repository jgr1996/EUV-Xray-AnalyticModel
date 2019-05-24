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


def make_model_histogram(R, P, N, nD=False):

    if nD==False:
        R_bins = np.logspace(-1.0,1.0, 50)
        hist, bins = np.histogram(R, bins=R_bins)
        return (946/N)*hist
    else:
        R_bins = np.logspace(-1.0,1.0, 50)
        P_bins = np.logspace(0.0,2.0, 50)
        H, xbins, ybins = np.histogram2d(P, R, bins=[P_bins, R_bins])
        H = H.flatten()
        H = [float(i) for i in H]
        return (904/N)*H


def make_CKS_histograms(nD=False):

    # read filtered CKS data
    CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')

    if nD==False:
        R_bins = np.logspace(-1.0,1.0, 50)
        hist, bins = np.histogram(CKS_array[2,:], bins=R_bins)
        return hist
    else:
        R_bins = np.logspace(-1.0,1.0, 50)
        P_bins = np.logspace(0.0,2.0, 50)
        H, xbins, ybins = np.histogram2d(CKS_array[3,:], CKS_array[2,:], bins=[P_bins, R_bins])
        H = H.flatten()
        H = [float(i) for i in H]
        return H

# //////////////////////////////////////////////////////////////////////////// #
def KS_2sample_test(hist1, hist2):

    empirical_dist1 = np.array(empirical_dist(hist1))
    empirical_dist2 = np.array(empirical_dist(hist2))
    statistic = np.max(np.absolute(empirical_dist1 - empirical_dist2))
    return statistic


def empirical_dist(hist):

    N = np.sum(hist)
    distribution = []
    running_total = 0
    for i in range(len(hist)):
        if hist[i] == 0:
            distribution.append(float(running_total))
        else:
            running_total = running_total + (hist[i]/N)
            distribution.append(float(running_total))

    return distribution

# //////////////////////////////////////////////////////////////////////////// #

def likelihood_R_space(theta, N, data_histogram):

    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank        # rank of this process

    if rank == 0:
        [X_mean, X_stdev, M_core_mean, M_core_stdev, density_mean] = theta
        if any(n <= 0 for n in theta):
            return -np.inf
        if X_mean >= 0.4:
            return -np.inf
        if X_stdev >= 0.5:
            return -np.inf
        if M_core_mean >= 12.0:
            return -np.inf
        if M_core_mean <= 1.0:
            return -np.inf
        if M_core_stdev >= 2.5:
            return -np.inf
        if density_mean <= 1.5:
            return -np.inf
        if density_mean >= 12.0:
            return -np.inf


    R, P = evolve_population.CKS_synthetic_observation(N, theta)

    if rank == 0:
        model_histogram = make_model_histogram(R,P,N)

        # ///////////////////// VERSION 1 ///////////////////////// #
        model_hist_temp = model_histogram + np.ones(len(model_histogram))
        data_hist_temp = data_histogram + np.ones(len(data_histogram))

        L = (data_hist_temp - model_hist_temp) \
          + data_hist_temp * np.log(model_hist_temp / data_hist_temp)

        L = np.sum(L)

        return L

    else:
        return 0.0


# //////////////////////////////////////////////////////////////////////////// #
# ---------------------------------------------------------------------------- #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


def likelihood_PR_space(theta, N, data_array):

    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank        # rank of this process
    if rank == 0:
        [X_mean, X_stdev, M_core_mean, M_core_stdev, density_mean] = theta
        if any(n <= 0 for n in theta):
            return -np.inf

        if X_mean >= 0.4:
            return -np.inf
        if X_stdev >= 0.7:
            return -np.inf
        if M_core_mean >= 12.0:
            return -np.inf
        if M_core_mean <= 0.3:
            return -np.inf
        if M_core_stdev >= 2.5:
            return -np.inf

        if density_mean <= 2.5:
            return -np.inf
        if density_mean >= 12.0:
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
# data_histogram = make_CKS_histograms()
# L = likelihood_R_space([0.06, 0.42, 7.72, 1.86], 500, data_histogram)
#
# comm = MPI.COMM_WORLD   # get MPI communicator object
# rank = comm.rank        # rank of this process
#
# if rank == 0:
#     print

# H = make_CKS_histograms(nD=True)
# print sum(H)
