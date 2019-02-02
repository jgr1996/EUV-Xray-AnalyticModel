from __future__ import division
import os
import numpy as np
from numpy import random as rand
from scipy import stats
import multiprocessing
from joblib import Parallel, delayed
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

CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')




def make_planet(initial_X_params, core_density_params, core_mass_params,
                period_params, KH_timescale_params):

    """
    This function takes parameters for planet distributions in order to randomly
    generate a planet. It returns the initial envelope mass-fraction, core
    density, core mass, period and stellar mass. Note that it also considers the
    transit probability and pipeline efficiency in order to make a synthetic
    observation.
    """

    # random initial mass fraction according to log-normal distribution
    (X_mean, X_stdev) = initial_X_params
    X_initial = rand.normal(X_mean, X_stdev)


    # random core_density according to gaussian
    #(density_mean, density_stdev) = core_density_params
    #core_density = rand.normal(density_mean, density_stdev)
    core_density = 5.5

    # random core mass according to Rayleigh
    (core_mass_mean, core_mass_stdev) = core_mass_params
    core_mass = rand.normal(core_mass_mean, core_mass_stdev)
    #
    # #random period according to CKS data fit
    # (power, period_cutoff) = period_params
    # U_P = rand.random()
    # if U_P <= 0.14:  #0.14
    #     U_P = rand.random()
    #     c = period_cutoff**power / (power)
    #     P = (power * U_P * c)**(1/power)
    # else:
    #     U_P = rand.random()
    #     k_P = np.log(100/period_cutoff)
    #     P = period_cutoff * np.exp(k_P * U_P)

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





    # random stellar mass from CKS data
    CKS_index = rand.randint(0,946)
    stellar_mass = CKS_array[1,CKS_index]

    #(KH_timescale_mean, KH_timescale_stdev) = KH_timescale_params
    #KH_timescale = rand.normal(KH_timescale_mean, KH_timescale_stdev)
    KH_timescale = 100

    return X_initial, core_density, core_mass, P, stellar_mass, KH_timescale, CKS_index


def run_population(distribution_parameters, N, output_directory_name, period_bias=False, pipeline_recovery=False, job_number=0):

    """
    This function evolves an ensemble of N planets randomly generated using the
    make_planet function through the mass_fraction_evolver file (in particular
    the "RK45_driver" function).
    """

    R_planet_pop = []
    period_pop = []
    transit_number = 0

    initial_X_params = (distribution_parameters[0],distribution_parameters[1])
    #core_density_params = (distribution_parameters[2],distribution_parameters[3])
    core_mass_params = (distribution_parameters[2],distribution_parameters[3])
    #period_params = (distribution_parameters[4],distribution_parameters[5])
    #KH_timescale_params = (distribution_parameters[8],distribution_parameters[9])

    while transit_number < N:
        # generate planet parameters
        X_initial, core_density, M_core, period, M_star, KH_timescale_cutoff, CKS_index = make_planet(initial_X_params=initial_X_params,
                                                                                                      core_density_params=None,
                                                                                                      core_mass_params=core_mass_params,
                                                                                                      period_params=(1.9, 7.6),
                                                                                                      KH_timescale_params=None)

        # mass of star/planet and envelope mass fraction cannot be negative
        if M_star <= 0:
            continue
        if M_core <= 0:
            continue
        if X_initial < 0:
            continue

        # model does not cover self-gravitating planets
        if X_initial >= 1.0:
            continue
        # model does not consider dwarf planets
        if M_core <=0.1:
            continue
        # model breaks down for very small periods
        if period <= 0.5:
            continue

        # calculate probability of transit using P = b(R_pl + R_*) / a
        # if period_bias == True:
        #
        #     a_meters = ((period * 24 * 60 * 60)**2 * G * M_star * M_sun / (4 * pi * pi))**(1/3)
        #     R_star_meters = R_sun * CKS_array[0, CKS_index]
        #     prob_of_transit = b_cutoff * R_star_meters / a_meters
        #
        #     # for a random inclination, reject planet if not transiting
        #     random_number_transit = rand.random()
        #     if random_number_transit > prob_of_transit:
        #         continue
        #
        # # now consider injection recovery using pipeline efficiency
        # if pipeline_recovery == True:
        #     # random signal to noise, m
        #     m = rand.randint(0,1000)
        #     # Gamma CDF function for probability of injection recovery - (a, scale, loc) taken from Fulton et al. 2017
        #     prob_of_injection_recovery = stats.gamma.cdf(m, a=17.56, loc=1.0, scale=0.49)
        #
        #     # determine whether injection is recovered
        #     random_number_inj_recovery = rand.random()
        #     if random_number_inj_recovery > prob_of_injection_recovery:
        #         continue


        # print '//////// {0} TRANSITS \'OBSERVED\' for job {1}///////////'.format(transit_number, job_number)
        R_core, t, X, R_ph = mass_fraction_evolver.RK45_driver(1, 3000, 0.01, 1e-5,
                                                               X_initial, core_density, M_core,
                                                               period, M_star, KH_timescale_cutoff)


        # add random error to planetary radius from gaussian distribution
        true_planet_radius = R_ph[-1]
        observed_planet_radius = true_planet_radius + rand.normal(true_planet_radius, 0.1*true_planet_radius)

        R_planet_pop.append(observed_planet_radius)
        period_pop.append(period)
        transit_number = transit_number + 1

    newpath = './RESULTS/{0}'.format(output_directory_name)

    np.savetxt('{0}/R_array_{1}.csv'.format(newpath, job_number), R_planet_pop, delimiter=',')
    np.savetxt('{0}/P_array_{1}.csv'.format(newpath, job_number), period_pop, delimiter=',')

    return None



def run_single_population(N, distribution_parameters, current_time_string):

    """
    This function takes a set of distribution parameters and runs an synthetic
    observation for N planets. This is of particular use when splitting the
    total population over several cores.
    """

    # get number of cores
    number_of_cores = int(multiprocessing.cpu_count())

    observations_per_core = int(N / number_of_cores)
    job_list = np.arange(number_of_cores)

    Parallel(n_jobs=32)(delayed(run_population)(distribution_parameters,
                                                observations_per_core,
                                                current_time_string,
                                                period_bias=True,
                                                pipeline_recovery=True,
                                                job_number=i) for i in job_list)

    # ensure there are equal numbers of planets as in CKS population
    top_up_observations = N - observations_per_core*number_of_cores
    run_population(distribution_parameters,
                   top_up_observations,
                   current_time_string,
                   period_bias=True,
                   pipeline_recovery=True,
                   job_number=job_list[-1] + 1)

    R = []
    P = []
    job_list = np.append(job_list, job_list[-1] + 1)
    for i in job_list:
        R_i = np.loadtxt("./RESULTS/{0}/R_array_{1}.csv".format(current_time_string, i), delimiter=',')
        P_i = np.loadtxt("./RESULTS/{0}/P_array_{1}.csv".format(current_time_string, i), delimiter=',')

        R = np.append(R, R_i)
        P = np.append(P, P_i)

        os.remove("./RESULTS/{0}/R_array_{1}.csv".format(current_time_string, i))
        os.remove("./RESULTS/{0}/P_array_{1}.csv".format(current_time_string, i))

    return R, P

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #

# def period_distribution_test(N, period_bias=False):
#
#     period_pop = []
#     transit_number = 0
#
#     while transit_number < N:
#         # generate planet parameters
#         X_initial, core_density, M_core, period, M_star, KH_timescale_cutoff, CKS_index = make_planet(initial_X_params=(0.1,0.01),
#                                                                                                       core_density_params=None,
#                                                                                                       core_mass_params=(3, 0.5),
#                                                                                                       period_params=(1.9, 7.6),
#                                                                                                       KH_timescale_params=None)
#
#         # mass of star cannot be negative
#         if M_star <= 0:
#             continue
#         if M_core <= 0:
#             continue
#         if X_initial < 0:
#             continue
#
#
#         # calculate probability of transit using P = b(R_pl + R_*) / a
#         if period_bias == True:
#
#             a_meters = ((period * 24 * 60 * 60)**2 * G * M_star * M_sun / (4 * pi * pi))**(1/3)
#             R_star_meters = R_sun * CKS_array[0, CKS_index]
#             prob_of_transit = b_cutoff * R_star_meters / a_meters
#
#             # for a random inclination, reject planet if not transiting
#             random_number_transit = rand.random()
#             if random_number_transit > prob_of_transit:
#                 continue
#
#
#             # random signal to noise, m
#             m = rand.randint(0,1000)
#             # Gamma CDF function for probability of injection recovery - (a, scale, loc) taken from Fulton et al. 2017
#             prob_of_injection_recovery = stats.gamma.cdf(m, a=17.56, loc=1.0, scale=0.49)
#
#             # determine whether injection is recovered
#             random_number_inj_recovery = rand.random()
#             if random_number_inj_recovery > prob_of_injection_recovery:
#                 continue
#
#             period_pop.append(period)
#             transit_number = transit_number + 1
#             print transit_number
#
#         else:
#             period_pop.append(period)
#             transit_number = transit_number + 1
#             print transit_number
#
#     return period_pop

# R, P = run_population(10, period_bias=True, pipeline_recovery=True)
# plt.style.use('classic')
# plt.scatter(P, R)
# plt.xlim([1,100])
# plt.ylim([1,10])
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Period [days]')
# plt.ylabel(r'Planet Radius $[R_\oplus]$')
# plt.show()

# X_range = []
# core_density_range = []
# core_mass_range = []
# P_range = []
# stellar_mass_range = []
#
# for i in range(10000):
#     X_initial, core_density, core_mass, P, stellar_mass, KH_timescale, CKS_index = make_planet(initial_X_params=(0.1,0.01),
#                                                                                                core_density_params=None,
#                                                                                                core_mass_params=(3, 0.5),
#                                                                                                period_params=(1.9,7.6),
#                                                                                                KH_timescale_params=None)
#
#     X_range.append(X_initial)
#     core_density_range.append(core_density)
#     core_mass_range.append(core_mass)
#     P_range.append(P)
#     stellar_mass_range.append(stellar_mass)

# plt.figure(1)
# plt.hist(X_range, bins=np.logspace(-2,0))
# plt.xlabel('X')
# plt.xscale('log')
#
# plt.figure(2)
# plt.hist(core_density_range, bins=50)
# plt.xlabel('core density g/cm^3')

# plt.figure(3)
# plt.hist(core_mass_range, bins=50)
# plt.xlabel('core mass')

# plt.figure(4)
# plt.hist(P_range, bins=np.logspace(0,2))
# plt.xlabel('Period')
# plt.xscale('log')

# plt.figure(5)
# plt.hist(stellar_mass_range, bins=50)
# plt.xlabel('stellar mass')

# plt.show()

# N = 10000
# P = period_distribution_test(N, period_bias=False)
# P_bins = np.logspace(0,2,100)
# plt.hist(P, bins=P_bins)
#
# P_bins1 = np.logspace(0,0.881,50)
# P_bins2 = np.logspace(0.881,2,50)
#
#
# plt.plot(P_bins1, [15*i**(1.233) for i in P_bins1], color='black')
# plt.plot(P_bins2, [750*i**(-2/3) for i in P_bins2], color='black')
#
#
# plt.xlabel('Period')
# plt.xscale('log')
# plt.show()

# //////////////////////////////////////////////////////////////////////////// #

# current_time_string = "test3"
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
# distribution_parameters = [0.05, 0.1, 1.5, 1.0]
# R, P = run_single_population(N, distribution_parameters, current_time_string)
# np.savetxt('{0}/R.csv'.format(newpath), R, delimiter=',')
