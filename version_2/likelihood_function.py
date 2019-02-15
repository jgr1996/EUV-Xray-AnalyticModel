from __future__ import division
import os
import sys
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.misc import factorial
import evolve_population
import ndtest


def make_model_histogram(R, P, nD=False):

    if nD==False:
        R_bins = np.logspace(-1.0,1.0, 50)
        hist, bins = np.histogram(R, bins=R_bins)
        return hist
    else:
        R_bins = np.logspace(-1.0,1.0, 50)
        P_bins = np.logspace(0.0,2.0, 50)
        H, xbins, ybins = np.histogram2d(P, R, bins=[P_bins, R_bins])
        return H


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

def likelihood_R_space(theta, N, current_time_string, data_histogram):


    # [M_core_mean, M_core_stdev] = theta
    # if any(n <= 0 for n in theta):
    #     return -np.inf

    [X_mean, X_stdev, M_core_mean, M_core_stdev] = theta
    if any(n <= 0 for n in theta):
        return -np.inf
    if X_mean >= 0.4:
        return -np.inf
    if X_stdev >= 0.5:
        return -np.inf
    if M_core_mean >= 12.0:
        return -np.inf
    if M_core_mean <= 0.5:
        return -np.inf
    if M_core_stdev >= 2.5:
        return -np.inf



    R, P = evolve_population.run_single_population(N, theta, current_time_string)
    # R, P = evolve_population.run_population(theta,
    #                                         N,
    #                                         current_time_string,
    #                                         period_bias=False,
    #                                         pipeline_recovery=False)

    model_histogram = make_model_histogram(R,P)

    # ///////////////////// VERSION 1 ///////////////////////// #
    model_hist_temp = model_histogram + np.ones(len(model_histogram))
    data_hist_temp = data_histogram + np.ones(len(data_histogram))

    L = data_hist_temp * np.log(model_hist_temp) \
      - model_hist_temp \
      - data_hist_temp * np.log(data_hist_temp) - data_hist_temp

    L = np.sum(L)

    return L


    # ///////////////////// VERSION 2 ///////////////////////// #
    # L = KS_2sample_test(model_histogram, data_histogram)
    # return -L

    # # ///////////////////// VERSION 3 ///////////////////////// #
    # L = stats.anderson_ksamp([model_histogram, data_histogram])
    # return -L[0]


# //////////////////////////////////////////////////////////////////////////// #
# ---------------------------------------------------------------------------- #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


def likelihood_P_R_space(theta, N, current_time_string, data_histogram):

    [X_mean, X_stdev, M_core_mean, M_core_stdev] = theta
    if any(n <= 0 for n in theta):
        return -np.inf
    if X_mean >= 0.4:
        return -np.inf
    if X_stdev >= 0.5:
        return -np.inf
    if M_core_mean >= 12.0:
        return -np.inf
    if M_core_mean <= 0.5:
        return -np.inf
    if M_core_stdev >= 2.5:
        return -np.inf


    R, P = evolve_population.run_single_population(N, theta, current_time_string)
    model_histogram = make_model_histogram(R, P, nD=True)

    # ///////////////////// VERSION 2 ///////////////////////// #
    L = ndtest.ks2d2s(model_histogram, data_histogram)
    return L






# R_bins = np.logspace(-1.0,1.0, 30)
# CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
# plt.hist(CKS_array[2,:], bins=R_bins)
#
# N = len(CKS_array[2,:])
# data_bins, data_histogram = make_CKS_histograms()
# L, R_model = likelihood_R_space([0.1, 0.05, 3.0, 0.5], N, "test_output", data_histogram)
# plt.hist(R_model, bins=R_bins)
# plt.show()
# print L


# CKS_hist = make_CKS_histograms()
# print CKS_hist
# empirical_dist(CKS_hist)