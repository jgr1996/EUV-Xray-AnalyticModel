from __future__ import division
import os
import numpy as np
from numpy import random as rand
import time
from scipy import stats
from scipy import interpolate
from scipy import special
from mpi4py import MPI
import matplotlib.pyplot as plt
from constants import *
import mass_fraction_evolver

"""
Author: Rogers, J. G
Date: 01/12/2018

This file simulates the evolution of random populations of small, close-in planetes.
The main purpose of this code is the constrain the underlying distributions of
exoplanets by evolving an initial ensemble through EUV/Xray photoevaporation.
"""


def enum(*sequential, **named):
    """
    Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


def bernstein_poly(x, order, coefficients):

    coefficients = np.array(coefficients)
    poly_array = np.array([special.binom(order, i)*(x**i)*((1-x)**(order-i)) for i in range(order+1)])
    B = np.dot(coefficients, poly_array)

    return B



def make_planet(initial_X_coeffs, core_mass_coeffs, period_params):

    """
    This function takes parameters for planet distributions in order to randomly
    generate a planet. It returns the initial envelope mass-fraction, core
    density, core mass, period and stellar mass. Note that it also considers the
    transit probability and pipeline efficiency in order to make a synthetic
    observation.
    """

    # random initial mass fraction according to log-normal distribution
    X_normalisation = 0.4 / bernstein_poly(1.0, 5, initial_X_coeffs)
    U_X = rand.uniform()
    X_initial = bernstein_poly(U_X, 5, initial_X_coeffs) * X_normalisation

    # random core mass according to Rayleigh
    M_normalisation = 12.0 / bernstein_poly(1.0, 5, core_mass_coeffs)
    U_M = rand.uniform()
    core_mass = bernstein_poly(U_M, 5, core_mass_coeffs) * M_normalisation

    # random period according to CKS data fit
    (power, period_cutoff) = period_params
    period_bias_power = -2/3
    U_P = rand.random()
    if U_P <= 0.33:  #0.14
        U_P = rand.random()
        power = power + period_bias_power
        c = period_cutoff**power / (power)
        P = (power * U_P * c)**(1/power)
    else:
        U_P = rand.random()
        power = period_bias_power
        c = period_cutoff**power / (power)
        P = (power * U_P * c)**(1/power)

    return X_initial, core_mass, P



def CKS_synthetic_observation(N, distribution_parameters):

    R = []
    P = []

    # Define MPI message tags
    tags = enum('READY', 'DONE', 'EXIT', 'START')
    rand.seed(12345)

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    if rank == 0:
        # start = time.time()

        # Master process executes code below

        CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')

        X_initial_list = []
        core_density_list = []
        M_core_list = []
        period_list = []
        M_star_list = []
        KH_timescale_cutoff_list = []

        initial_X_coeffs = [distribution_parameters[i] for i in range(6)]
        core_mass_coeffs = [distribution_parameters[i+6] for i in range(6)]
        density_mean = 5.0
        KH_timescale = 100

        transit_number = 0
        while transit_number < N:
            X_initial, M_core, period= make_planet(initial_X_coeffs=initial_X_coeffs,
                                                   core_mass_coeffs=core_mass_coeffs,
                                                   period_params=(1.9, 7.6))

            # mass of planet and envelope mass fraction cannot be negative
            if M_core <= 1.0:
                continue
            if X_initial < 0.0:
                continue
            # model does not cover self-gravitating planets
            if X_initial >= 0.9:
                continue
            if M_core >= 12.0:
                continue
            # model breaks down for very small periods
            if period <= 0.5:
                continue
            # CKS data does not include P > 100
            if period > 100.0:
                continue


            X_initial_list.append(X_initial)
            core_density_list.append(density_mean)
            M_core_list.append(M_core)
            period_list.append(period)
            CKS_index = transit_number%(946)
            M_star_list.append(CKS_array[1,CKS_index])
            KH_timescale_cutoff_list.append(KH_timescale)
            transit_number = transit_number + 1

        # end0 = time.time()
        # print 'serial = ', end0 - start
        tasks = range(N)
        task_index = 0
        num_workers = size - 1
        closed_workers = 0
        while closed_workers < num_workers:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == tags.READY:
                # Worker is ready, so send it a task
                if task_index < len(tasks):
                    params = [k[task_index] for k in [X_initial_list, core_density_list, M_core_list, period_list, M_star_list, KH_timescale_cutoff_list]]
                    comm.send(params, dest=source, tag=tags.START)
                    task_index += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)
            elif tag == tags.DONE:
                results = data
                if np.isnan(results).any():
                    continue
                if results[0] < 1.0: # can't observe planet smaller than earth!
                    continue
                if results[0] == None: # this will be due to a Brentq error
                    continue
                else:
                    R.append(float(results[0]))
                    P.append(float(results[1]))
            elif tag == tags.EXIT:
                closed_workers += 1

        # end1 = time.time()
        # print 'total time elapsed = ',end1 - start
        return R, P


    else:
        # Worker processes execute code below
        name = MPI.Get_processor_name()
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            params = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:
                R_ph, P = mass_fraction_evolver.RK45_driver(1, 3000, 0.01, 1e-5, params)
                if np.isnan(R_ph):
                    print 'R is NaN for ',params
                comm.send((R_ph,P), dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)

    return None, None


# comm = MPI.COMM_WORLD   # get MPI communicator object
# size = comm.size        # total number of processes
# rank = comm.rank        # rank of this process
# status = MPI.Status()   # get MPI status object
#
#
# N_range = [10000]
# for i in N_range:
#
#     start = time.time()
#     R, P = CKS_synthetic_observation(i, [0.03, 0.21, 0.47, 0.25, 0.42, 0.95, 0.04, 0.09, 0.41, 0.55, 0.40, 1.36, 5.0])
#     finish = time.time()
#
#     if rank == 0:
#         x = np.logspace(-1,2,150)
#         y = np.logspace(-1,1.5,150)
#         X, Y = np.meshgrid(x, y)
#         positions = np.vstack([np.log(X.ravel()), np.log(Y.ravel())])
#         data = np.vstack([np.log(P),np.log(R)])
#         kernel = stats.gaussian_kde(data)
#         Z = np.reshape(kernel(positions).T, X.shape)
#         Z_norm = 1 / np.sum(Z)
#         Z = Z_norm * Z
#
#         KDE_interp = interpolate.RectBivariateSpline(x,y,Z)
#
#         CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
#
#         if rank == 0:
#             x = np.logspace(-1,2,150)
#             y = np.logspace(-1,1.5,150)
#             X, Y = np.meshgrid(x, y)
#             positions = np.vstack([np.log(X.ravel()), np.log(Y.ravel())])
#
#             data = np.vstack([np.log(P),np.log(R)])
#             kernel = stats.gaussian_kde(data)
#             Z = np.reshape(kernel(positions).T, X.shape)
#             Z_norm = 1 / np.sum(Z)
#             Z_model = Z_norm * Z
#
#             P_data = CKS_array[3,:]
#             R_data = CKS_array[2,:]
#             CKS_data = np.vstack([np.log(P_data),np.log(R_data)])
#             kernel = stats.gaussian_kde(CKS_data)
#             Z = np.reshape(kernel(positions).T, X.shape)
#             Z_norm = 1 / np.sum(Z)
#             Z_data = Z_norm * Z
#
#             KL_stat = stats.entropy(Z_model.flatten(), Z_data.flatten())
#
#
#         newpath = './RESULTS/convergence_test'
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#
#         np.savetxt("{0}/KDE_lowdens_2.csv".format(newpath, i), Z_model, delimiter=',')
#

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #

# X_range = []
# core_density_range = []
# core_mass_range = []
# P_range = []
# stellar_mass_range = []
#
# for i in range(100000):
#     X_initial, core_density, core_mass, P, stellar_mass, KH_timescale, CKS_index = make_planet(initial_X_params=(1, 0.37),
#                                                                                                core_density_params=None,
#                                                                                                core_mass_params=(7.0, 0.29),
#                                                                                                period_params=(1.9,7.6),
#                                                                                                KH_timescale_params=None)
#
#     X_range.append(X_initial)
#     core_density_range.append(core_density)
#     core_mass_range.append(core_mass)
#     P_range.append(P)
#     stellar_mass_range.append(stellar_mass)
#
# plt.figure(1)
# plt.hist(X_range, bins=np.logspace(-3,1))
# plt.xlabel('X')
# plt.xscale('log')
#
# plt.figure(2)
# plt.hist(core_density_range, bins=50)
# plt.xlabel('core density g/cm^3')

# plt.figure(3)
# plt.hist(core_mass_range, bins=np.logspace(-1,2))
# plt.xlabel('core mass')
# plt.xscale('log')


# plt.figure(4)
# plt.hist(P_range, bins=np.logspace(0,2,30), histtype='step', color='black', linewidth=2)
# plt.xlabel('Period')
# plt.xscale('log')
#
# CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
# P_data = CKS_array[3,:]
# plt.hist(P_data, bins=np.logspace(0,2,30), color='r', histtype='step')


# plt.figure(5)
# plt.hist(stellar_mass_range, bins=50)
# plt.xlabel('stellar mass')

# plt.show()

# //////////////////////////// RUN SINGLE POPULATION ///////////////////////// #

# current_time_string = "test"
# newpath = './RESULTS/{0}'.format(current_time_string)
# if not os.path.exists(newpath):
#     os.makedirs(newpath)
#
# R_bins = np.logspace(-1.0,1.0, 30)
# hist, bins = np.histogram(CKS_array[2,:], bins=R_bins)
#
# # get CKS data radius bins
# N = np.sum(hist)
#
# distribution_parameters = [0.06, 0.42, 7.72, 1.86]
# R, P = run_single_population(5*N, distribution_parameters, current_time_string)
# np.savetxt('{0}/R.csv'.format(newpath), R, delimiter=',')
# np.savetxt('{0}/P.csv'.format(newpath), P, delimiter=',')
