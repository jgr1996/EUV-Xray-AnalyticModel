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
import mass_fraction_evolver_cython

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



def make_planet(initial_X_coeffs, core_mass_coeffs, period_cdf):

    """
    This function takes parameters for planet distributions in order to randomly
    generate a planet. It returns the initial envelope mass-fraction, core
    density, core mass, period and stellar mass. Note that it also considers the
    transit probability and pipeline efficiency in order to make a synthetic
    observation.
    """

    # random period according to double power law
    P = period_cdf(rand.uniform())

    (X_mean, X_stdev) = initial_X_coeffs
    X_initial = rand.lognormal(np.log10(X_mean),X_stdev)

    # random core mass according to Rayleigh
    (core_mass_mean, core_mass_stdev) = core_mass_coeffs
    core_mass = rand.lognormal(np.log10(core_mass_mean), core_mass_stdev)

    # random stellar mass from CKS-I fit
    M_star = rand.normal(loc=1.04, scale=0.15)


    return X_initial, core_mass, P, M_star


def period_distribution_CDF(power1, power2, cutoff):

    P_range = np.logspace(0,2,100)
    pdf = []

    # create pdf using double power law
    for i in range(len(P_range)):
        if P_range[i] <= cutoff:
            pdf_i = P_range[i] ** power1
            pdf.append(pdf_i)
        else:
            pdf_i = (cutoff ** (power1-power2)) * (P_range[i] ** power2)
            pdf.append(pdf_i)

    # normalise pdf, calculate cdf
    pdf = pdf / np.sum(pdf)
    cdf = np.cumsum(pdf)
    cdf[0] = 0.0
    cdf[-1] = 1.0

    # remove repeats
    cdf, mask = np.unique(cdf, return_index=True)
    P_mask = P_range[mask]

    # interpolate
    cdf_interp = interpolate.interp1d(cdf, P_range)

    return cdf_interp

# //////////////////////////////////////////////////////////////////////////// #


def CKS_synthetic_observation(N, distribution_parameters):

    R_results = []
    P_results = []
    M_results = []
    X_results = []
    R_core_results = []
    M_star_results = []

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

        density_mean = distribution_parameters[0]
        # period_coeffs = [distribution_parameters[i+1] for i in range(3)]
        period_coeffs = [1.9,0.01,7.9]
        initial_X_coeffs = (distribution_parameters[1],distribution_parameters[2])
        core_mass_coeffs = (distribution_parameters[3],distribution_parameters[4])

        KH_timescale = 100
        period_cdf = period_distribution_CDF(period_coeffs[0],
                                             period_coeffs[1],
                                             period_coeffs[2])

        start_time = time.time()
        transit_number = 0

        while transit_number < N:

            if time.time() - start_time >= 300:
                print "5 minutes exceeded, returning whatever has been done so far"
                return R_results, P_results, M_results, X_results, R_core_results, M_star_results

            X_initial, M_core, period, M_star = make_planet(initial_X_coeffs=initial_X_coeffs,
                                                            core_mass_coeffs=core_mass_coeffs,
                                                            period_cdf=period_cdf)

            # envelope mass fraction cannot be negative
            if X_initial < 1e-4:
                continue
            # model does not cover self-gravitating planets
            if X_initial >= 1.0:
                continue
            if M_core >= 15.0:
                continue
            # model does not consider dwarf planets
            if M_core <= 0.75:
                continue
            # model breaks down for very small periods
            if period <= 1.0:
                continue
            # CKS data does not include P > 100
            if period > 100.0:
                continue


            X_initial_list.append(X_initial)
            core_density_list.append(density_mean)
            M_core_list.append(M_core)
            period_list.append(period)
            M_star_list.append(M_star)
            KH_timescale_cutoff_list.append(KH_timescale)
            transit_number = transit_number + 1



        tasks = range(N)
        task_index = 0
        num_workers = size - 1
        closed_workers = 0
        start_time = time.time()
        while closed_workers < num_workers:

            if time.time() - start_time >= 300:
                comm.send(None, dest=source, tag=tags.EXIT)

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
                if np.all(results) == 0.0: # this will be due to a Brentq error
                    continue
                if np.isnan(results).any():
                    continue
                if results[0] < 0.1: # can't observe planet smaller than earth!
                    continue
                else:
                    R_results.append(float(results[0]))
                    P_results.append(float(results[1]))
                    M_results.append(float(results[2]))
                    X_results.append(float(results[3]))
                    R_core_results.append(float(results[4]))
                    M_star_results.append(float(results[5]))
            elif tag == tags.EXIT:
                closed_workers += 1

        return R_results, P_results, M_results, X_results, R_core_results, M_star_results


    else:
        # Worker processes execute code below
        name = MPI.Get_processor_name()
        while True:

            comm.send(None, dest=0, tag=tags.READY)
            params = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == tags.START:
                [initial_X, core_density, M_core, period, M_star, KH_timescale_cutoff] = params
                R_ph, P, M, X, R_core = mass_fraction_evolver_cython.RK45_driver(1, 3000, 0.01, 1e-5, initial_X,
                                                                                         core_density, M_core, period,
                                                                                         M_star, KH_timescale_cutoff)
                comm.send((R_ph, P, M, X, R_core, M_star), dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                break

        comm.send(None, dest=0, tag=tags.EXIT)

    return None, None, None, None, None, None#, None, None, None, None, None


# comm = MPI.COMM_WORLD   # get MPI communicator object
# size = comm.size        # total number of processes
# rank = comm.rank        # rank of this process
# status = MPI.Status()   # get MPI status object
#
#
# N_range = [5000]
# params = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 0.01, 0.2, 0.2, 0.3, 0.8, 1.0, 6.5]
#  # CHECK FILE NUMBER BEFORE RUNNING!!!
#
# for i in N_range:
#
#     start = time.time()
#     R, P, M, X, R_core, R_rejected, P_rejected, M_rejected, X_rejected, R_core_rejected = CKS_synthetic_observation(i, params)
#     finish = time.time()
#
#     if rank == 0:
#
#         results_accepted = np.array([R, P, M, X, R_core])
#         results_rejected = np.array([R_rejected, P_rejected, M_rejected, X_rejected, R_core_rejected])
#
#         print "We have detected {0} out of a total of {1}".format(len(R), N_range[0])
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
#         newpath = './RESULTS/likelihood_test/'
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#
#         file = 0
#         np.savetxt("{0}/paramaters_{1}.csv".format(newpath,file), params, delimiter=',')
#         np.savetxt("{0}/results_accepted_{1}.csv".format(newpath,file), results_accepted, delimiter=',')
#         np.savetxt("{0}/results_rejected_{1}.csv".format(newpath,file), results_rejected, delimiter=',')
#         np.savetxt("{0}/KDE_{1}.csv".format(newpath,file), Z, delimiter=',')

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #

# X_range = []
# core_density_range = []
# core_mass_range = []
# P_range = []
# stellar_mass_range = []
# period_cdf = period_distribution_CDF(1.7,-1.0,7.6)
#
# for i in range(10000):
#     X_initial, core_mass, P = make_planet(initial_X_coeffs=[0.0,0.2,0.4,0.6,0.8,1.0],
#                                           core_mass_coeffs=[0.0,0.2,0.4,0.6,0.8,1.0],
#                                           period_cdf=period_cdf)
#
#     # X_range.append(X_initial)
#     # # core_density_range.append(core_density)
#     # if core_mass <= 10.0 and core_mass >= 0.75:
#     #     core_mass_range.append(core_mass)
#     P_range.append(P)
#     # stellar_mass_range.append(stellar_mass)
#
# # plt.figure(1)
# # plt.hist(X_range, bins=np.logspace(-3,1))
# # plt.xlabel('X',fontsize=12)
# # plt.xscale('log')
# # plt.xticks([0.01,0.1,1.0],[0.01,0.1,1.0],fontsize=12, fontname = "Courier New")
#
# #
# # plt.figure(2)
# # plt.hist(core_density_range, bins=50)
# # plt.xlabel('core density g/cm^3')
# #
# # plt.figure(3)
# # plt.hist(core_mass_range, bins=np.logspace(-1,2))
# # plt.xlabel('core mass',fontsize=12)
# # plt.xscale('log')
# # plt.xtick_ranges([0.1,1.0,10.0],[0.1,1.0,10.0],fontsize=12, fontname = "Courier New")
#
# plt.figure(4)
# plt.hist(P_range, bins=np.logspace(0,2,30), histtype='step', color='black', linewidth=2)
# # plt.plot(np.logspace(0,np.log10(7.6),100),[10*x**1.9 for x in np.logspace(0,np.log10(7.6),100)])
# plt.xlabel('Period')
# plt.xscale('log')
#
# # CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
# # P_data = CKS_array[3,:]
# # plt.hist(P_data, bins=np.logspace(0,2,30), color='r', histtype='step')
#
#
# # plt.figure(5)
# # plt.hist(stellar_mass_range, bins=50)
# # plt.xlabel('stellar mass')
#
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
