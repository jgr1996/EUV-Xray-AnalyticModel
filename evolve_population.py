from __future__ import division
import numpy as np
from numba import jit
from numpy import random as rand
from scipy import stats
import matplotlib.pyplot as plt
from constants import *
import mass_fraction_evolver

"""
Author: Rogers, J. G
Date: 01/12/2018

This file simulates the evolution of random populations of small, close-in planetes.
The main purpose of this code is the contrain the underlying distributions of
exoplanets by evolving an initial ensemble through EUV/Xray photoevaporation.
"""



def make_planet(initial_X_params, core_density_params, core_mass_params, period_params, stellar_mass_params):

    """
    This planet takes parameters for planet distributions in order to randomly
    generate a planet. It returns the initial envelope mass-fraction, core
    density, core mass, period and stellar mass. Note that it also considers the
    transit probability and pipeline efficiency in order to make a synthetic
    observation.
    """

    # random initial mass fraction according to log-normal distribution
    (X_min, X_max) = initial_X_params
    U_X = rand.random()
    k_X = np.log(X_max/X_min)
    X_initial = X_min * np.exp(k_X*U_X)


    # random core_density according to gaussian
    (density_mean, density_stdev) = core_density_params
    core_density = rand.normal(density_mean, density_stdev)

    # random core mass according to Rayleigh
    (core_mass_dev) = core_mass_params
    core_mass = rand.rayleigh(core_mass_dev)

    # random period according to CKS data fit
    (power, period_cutoff) = period_params
    U_P = rand.random()
    if U_P <= 0.14:  #0.17
        U_P = rand.random()
        c = period_cutoff**power / (power)
        P = (power * U_P * c)**(1/power)
    else:
        U_P = rand.random()
        k_P = np.log(100/period_cutoff)
        P = period_cutoff * np.exp(k_P * U_P)

    # random stellar mass according to gaussian
    (stellar_mass_mean, stellar_mass_stdev) = stellar_mass_params
    stellar_mass = rand.normal(stellar_mass_mean, stellar_mass_stdev)

    return X_initial, core_density, core_mass, P, stellar_mass


def run_population(N, period_bias=False, pipeline_recovery=False):

    """
    This function evolves an ensemble of N planets randomly generated using the
    make_planet function through the mass_fraction_evolver file (in particular
    the "RK45_driver" function).
    """

    R_planet_pop = []
    period_pop = []
    transit_number = 0
    while transit_number <= N:
        # generate planet parameters
        X_initial, core_density, M_core, period, M_star = make_planet(initial_X_params=(0.01,0.3),
                                                                      core_density_params=(5.5,0.0),
                                                                      core_mass_params=(3),
                                                                      period_params=(1.9,7.6),
                                                                      stellar_mass_params=(1.3,0.3))


        # mass of star cannot be negative
        if M_star <= 0:
            continue


        # calculate probability of transit using P = b(R_pl + R_*) / a
        if period_bias == True:
            core_density_SI= core_density * 1000
            R_core_meters = (3 * M_core * M_earth / (4 * pi * core_density_SI))**(1/3)
            a_meters = ((period * 24 * 60 * 60)**2 * G * M_star * M_sun / (4 * pi * pi))**(1/3)
            R_star_meters = R_sun * (M_star)**(3/7)
            prob_of_transit = b_cutoff * (R_star_meters + R_core_meters) / a_meters

            # for a random inclination, reject planet if not transiting
            random_number_transit = rand.random()
            if random_number_transit > prob_of_transit:
                continue

        # now consider injection recovery using pipeline efficiency
        if pipeline_recovery == True:
            # random signal to noise, m
            m = rand.randint(0,1000)
            # Gamma CDF function for probability of injection recovery - (a, scale, loc) taken from Fulton et al. 2017
            prob_of_injection_recovery = stats.gamma.cdf(m, a=17.56, loc=1.0, scale=0.49)

            # determine whether injection is recovered
            random_number_inj_recovery = rand.random()
            if random_number_inj_recovery > prob_of_injection_recovery:
                continue


        print '///////////// {0} TRANSITS \'OBSERVED\' ////////////////'.format(transit_number)
        R_core, t, X, R_ph = mass_fraction_evolver.RK45_driver(1, 3000, 0.01, 1e-8,
                                                               X_initial, core_density, M_core, period, M_star)


        transit_number = transit_number + 1
        R_planet_pop.append(R_ph[-1])
        period_pop.append(period)

    return R_planet_pop, period_pop

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
# for i in range(100000):
#     X_initial, core_density, core_mass, P, stellar_mass = make_planet(initial_X_params=(0.01,0.3),
#                                                                               core_density_params=(5.5,0),
#                                                                               core_mass_params=(3),
#                                                                               period_params=(1.9,7.6),
#                                                                               stellar_mass_params=(1.3,0.3))
#     X_range.append(X_initial)
#     core_density_range.append(core_density)
#     core_mass_range.append(core_mass)
#     P_range.append(P)
#     stellar_mass_range.append(stellar_mass)
#
# plt.figure(1)
# plt.hist(X_range, bins=np.logspace(-2,0))
# plt.xlabel('X')
# plt.xscale('log')
#
# plt.figure(2)
# plt.hist(core_density_range, bins=50)
# plt.xlabel('core density g/cm^3')
#
# plt.figure(3)
# plt.hist(core_mass_range, bins=50)
# plt.xlabel('core mass')
#
# plt.figure(4)
# plt.hist(P_range, bins=np.logspace(0,2))
# plt.plot([7.6,7.6],[0,7000])
# plt.xlabel('Period')
# plt.xscale('log')
#
#
#
# plt.show()
