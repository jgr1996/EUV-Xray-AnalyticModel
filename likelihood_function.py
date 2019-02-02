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


def make_model_histogram(R,P):

    R_bins = np.logspace(-1.0,1.0, 30)
    hist, bins = np.histogram(R, bins=R_bins)

    return R_bins, hist


def make_CKS_histograms():

    # read filtered CKS data
    CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')

    R_bins = np.logspace(-1.0,1.0, 30)
    hist, bins = np.histogram(CKS_array[2,:], bins=R_bins)

    return R_bins, hist


def likelihood_R_space(theta, N, current_time_string, data_histogram):

    [X_mean, X_stdev, M_core_mean, M_core_stdev] = theta
    if any(n <= 0 for n in theta):
        return -np.inf
    if X_mean >= 0.4:
        return -np.inf
    if X_stdev >= 0.4:
        return -np.inf
    if M_core_mean >= 6.0:
        return -np.inf
    if M_core_stdev >= 2.5:
        return -np.inf


    R, P = evolve_population.run_single_population(N, theta, current_time_string)
    model_bins, model_histogram = make_model_histogram(R,P)

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


    # # ///////////////////// VERSION 2 ///////////////////////// #
    # L = stats.ks_2samp(model_histogram, data_histogram)
    # return L[1]

    # ///////////////////// VERSION 3 ///////////////////////// #

    L = np.absolute(model_histogram - data_histogram)
    return -np.sum(L)



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
