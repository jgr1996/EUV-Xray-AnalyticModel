import os
import sys
import datetime
import time
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
import likelihood_function
import evolve_population



if __name__ == '__main__':

    # use time as label for output directory
    current_time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y_%H.%M.%S')

    # initial guess (X, core_density, M_core, Period, t_KH)
    theta = ((0.1, 0.5), (5.5, 0.0), (2.0, 0.4), (1.9, 7.6), (100, 0.0))

    # get CKS data radius bins
    data_bins, data_histogram = likelihood_function.make_CKS_histograms()
    N = np.sum(data_histogram)

    L = likelihood_function.likelihood_R_space(theta, N, current_time_string, data_histogram)
    print L
