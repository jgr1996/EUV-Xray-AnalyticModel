from __future__ import division
import os
import sys
import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
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
    hist, bins = np.histogram(CKS_array[1,:], bins=R_bins)

    return R_bins, hist


def likelihood_R_space(theta, N, current_time_string, data_histogram):

    R, P = evolve_population.run_single_population(N, theta, current_time_string)
    model_bins, model_histogram = make_model_histogram(R,P)

    model_hist_temp = model_histogram + np.ones(len(model_histogram))
    data_hist_temp = data_histogram + np.ones(len(data_histogram))

    L = data_hist_temp * np.log(model_hist_temp) \
      - model_hist_temp \
      - data_hist_temp * np.log(data_hist_temp) - data_hist_temp

    L = np.sum(L)

    return L
