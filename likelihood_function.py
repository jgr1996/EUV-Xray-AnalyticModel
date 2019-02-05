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
        R_bins = np.logspace(-1.0,1.0, 30)
        hist, bins = np.histogram(R, bins=R_bins)
        return hist
    else:
        R_bins = np.logspace(-1.0,1.0, 30)
        P_bins = np.logspace(0.0,2.0, 30)
        H, xbins, ybins = np.histogram2d(P, R, bins=[P_bins, R_bins])
        return H


def make_CKS_histograms(nD=False):

    # read filtered CKS data
    CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')

    if nD==False:
        R_bins = np.logspace(-1.0,1.0, 30)
        hist, bins = np.histogram(CKS_array[2,:], bins=R_bins)
        return hist
    else:
        R_bins = np.logspace(-1.0,1.0, 30)
        P_bins = np.logspace(0.0,2.0, 30)
        H, xbins, ybins = np.histogram2d(CKS_array[3,:], CKS_array[2,:], bins=[P_bins, R_bins])
        return H


def likelihood_R_space(theta, N, current_time_string, data_histogram):

    [X_mean, X_stdev, M_core_mean, M_core_stdev] = theta
    if any(n <= 0 for n in theta):
        return -np.inf
    if X_mean >= 1.0:
        return -np.inf
    if X_stdev >= 1.0:
        return -np.inf
    if M_core_mean >= 10.0:
        return -np.inf
    if M_core_stdev >= 0.8:
        return -np.inf


    R, P = evolve_population.run_single_population(N, theta, current_time_string)
    model_histogram = make_model_histogram(R,P)

    # # ///////////////////// VERSION 1 ///////////////////////// #
    # model_hist_temp = model_histogram + np.ones(len(model_histogram))
    # data_hist_temp = data_histogram + np.ones(len(data_histogram))
    #
    # L = data_hist_temp * np.log(model_hist_temp) \
    #   - model_hist_temp \
    #   - data_hist_temp * np.log(data_hist_temp) - data_hist_temp
    #
    # L = np.sum(L)
    #
    # return L


    # ///////////////////// VERSION 2 ///////////////////////// #
    L = stats.ks_2samp(model_histogram, data_histogram)
    return L[1]

    # # ///////////////////// VERSION 3 ///////////////////////// #
    # L = np.absolute(model_histogram - data_histogram)
    # return -np.sum(L)


# //////////////////////////////////////////////////////////////////////////// #
# ---------------------------------------------------------------------------- #
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #


def likelihood_P_R_space(theta, N, current_time_string, data_histogram):

    [X_mean, X_stdev, M_core_mean, M_core_stdev] = theta
    if any(n <= 0 for n in theta):
        return -np.inf
    if X_mean >= 1.0:
        return -np.inf
    if X_stdev >= 1.0:
        return -np.inf
    if M_core_mean >= 12.0:
        return -np.inf
    if M_core_stdev >= 1.0:
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
