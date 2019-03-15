from __future__ import division
import numpy as np
from numpy import random as rand


def metropolis_alg(f, f_params, theta_0, K, covariance):
    """
    f is the likelihood function
    theta_0 is the initial guess in parameter space
    K is the sample number
    covariance is the covariance matrix for the proposal distribution
    """

    N_obs, current_time_string, data_histogram = f_params

    dimensionality = len(theta_0)
    theta_i = theta_0
    f_i = f(theta_0, N_obs, current_time_string, data_histogram)
    param_output = []
    f_output = []

    acceptance_counter = 0
    for i in range(K):

        # propose now position
        step = rand.normal(0, covariance, dimensionality)
        theta_proposed = theta_i + step
        f_proposed = f(theta_proposed, N_obs, current_time_string, data_histogram)

        # accept new position?
        p = rand.uniform()
        if np.log(p) <= f_proposed - f_i:
            theta_i = theta_proposed
            f_i = f_proposed
            acceptance_counter += 1

        # add to list
        param_output.append(list(theta_i))
        f_output.append(f_i)

    return np.array(param_output), f_output, acceptance_counter/K
