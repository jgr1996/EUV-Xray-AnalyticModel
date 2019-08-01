from __future__ import division
import numpy as np
from numpy import random as rand_py
from scipy import stats

from libc.stdlib cimport rand

from constants import *
import R_photosphere_cython
import RKF45

"""
Author: Rogers, J. G
Date: 12/11/2018

This file carries out analytic calculations presented in Owen & Wu (2017) that
forward integrate the atmospheric mass fraction of a small, close-in planet.
"""
cdef double pi = 3.141598
cdef double stefan = 5.67e-8
cdef double kappa_0 = 2.294e-8
cdef double G = 6.674e-11
cdef double gamma = 5/3
cdef double Delta_ab = (gamma-1)/gamma
cdef double mu = 2.35
cdef double m_H = 1.67e-27
cdef double k_B = 1.381e-23
cdef double alpha = 0.68
cdef double beta = 0.45
cdef double M_sun = 1.989e30
cdef double L_sun = 3.828e26
cdef double AU = 1.496e11
cdef double T_eq_earth = (L_sun / (16 * stefan * pi * AU * AU))**0.25
cdef double M_earth = 5.927e24
cdef double R_earth = 6.378e6
cdef double eta_0 = 0.17
cdef double Myr = 1e6 * 365 * 24 * 60 * 60
cdef double t_sat = 100 * Myr
cdef double a0 = 0.5
cdef double b_cutoff = 0.7
cdef double R_sun = 6.957e8


# ///////////////////////// CALCULATE MASS-LOSS TIMESCALE //////////////////// #
cdef double calculate_tX(double t, double X, double M_core, double M_star, double a, double R_core, double KH_timescale_cutoff, double R_guess):

    """
    This function calculated the mass-loss timescale of the atmosphere. See Owen
    and Wu (2017) for details.
    """

    # convert to SI units
    cdef double M_core_kg = M_core * M_earth
    cdef double a_meters = a * AU
    cdef double t_seconds = t * Myr
    cdef double R_core_meters = R_core * R_earth

    # Calculate envelope mass (kg)
    cdef double M_env_kg = X * M_core_kg

    # calculate saturation luminosity for photoevaporation
    cdef double L_sat = 10**(-3.5) * L_sun * M_star
    cdef double L_HE
    if t_seconds < t_sat:
        L_HE = L_sat
    else:
        L_HE = L_sat * (t_seconds/t_sat)**(-1-a0)

    # Calulate photospheric radius
    cdef double R_ph = R_photosphere_cython.calculate_R_photosphere(t, M_star, a, M_core, R_core, X, KH_timescale_cutoff, R_guess)

    if R_ph == 0.0:
        return 0.0

    cdef double escape_velocity = np.sqrt(2*G*M_core_kg / R_core_meters) * 0.001
    cdef double eta = eta_0 * (escape_velocity / 23)**(-0.42)

    # Calculate mass loss rate due to photoevaporation
    cdef double M_env_dot = eta * R_ph**3 * L_HE / (4 * a_meters * a_meters * G * M_core_kg)

    # Calculate mass-loss timescale
    cdef double tX = (M_env_kg / M_env_dot) / Myr

    return tX

# /////////////////////////// MASS FRACTION EVOLUTION ODE //////////////////// #

cdef double dXdt_ODE(double t, double X, parameters):

    """
    This function presents the ODE for atmospheric mass-loss dX/dt = - X / tX
    where X = M_atmosphere / M_core and tX is the mass-loss timeascale.
    """
    cdef double M_core, M_star, a, R_core, KH_timescale_cutoff, R_guess
    M_core = parameters[0]
    M_star = parameters[1]
    a = parameters[2]
    R_core = parameters[3]
    KH_timescale_cutoff = parameters[4]
    R_guess = parameters[5]
    #M_core, M_star, a, R_core, KH_timescale_cutoff, R_guess = parameters

    cdef double tX = calculate_tX(t, X, M_core, M_star, a, R_core, KH_timescale_cutoff, R_guess)

    if tX == 0.0:
        return 0.0

    cdef double dXdt = - X / tX

    return dXdt


# /////////////////// EVOLVE MASS FRACTION ACCORDING TO ODE ////////////////// #

cpdef RK45_driver(double t_start, double t_stop, double dt_try, double accuracy,
                  double initial_X, double core_density, double M_core,
                  double period, double M_star, double R_star,
                  double KH_timescale_cutoff):

    """
    This function controls the integration of the mass-loss ODE. It calls upon
    the "calculate_R_photosphere" function from the R_photosphere file as well
    as the "step_control" function in the RKF45 file to calculate the evolution
    of the mass fraction X.
    """

    # core density to core radius (measure in Earth radii)
    cdef double density_norm = 1.7589 * core_density**(-0.3329)
    cdef double R_core = density_norm * ((M_core)**0.25)

    # orbital period to semi-major axis measured in AU
    cdef double a = (((period * 24 * 60 * 60)**2 * G * M_star * M_sun / (4 * pi * pi))**(1/3)) / AU

    #calculate initial photospheric radius
    cdef double R_ph_init = R_photosphere_cython.calculate_R_photosphere(t_start, M_star, a, M_core,R_core, initial_X, KH_timescale_cutoff,R_guess=0.0) / R_earth

    # define initial variables
    cdef double X = initial_X
    cdef double t = t_start
    cdef double R_ph = R_ph_init
    cdef double dt = dt_try
    cdef double R_ph_new, observed_radius, rad_err, X_new, t_new, dt_next

    # setup loop
    while t <= t_stop:
        #perform an adaptive RK45 step
        [t_new, X_new, dt_next] = RKF45.step_control(t, X, dt, dXdt_ODE, accuracy, parameters=[M_core, M_star, a, R_core, KH_timescale_cutoff, R_ph])

        if [t_new, X_new, dt_next] == [0.0, 0.0, 0.0]:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # calculate new R_ph
        R_ph_new = R_photosphere_cython.calculate_R_photosphere(t_new, M_star, a, M_core,R_core, X_new, KH_timescale_cutoff, 0.0) / R_earth


        # if X becomes very small, we can assume all atmosphere is eroded
        if X_new <= 1e-4 or R_ph_new <= R_core:
                rad_err = rand() % (0.18 - 0.08) + 0.08 * R_core
                observed_radius = R_core + rand_py.choice([-1.0,1.0])*rad_err
                return observed_radius, period, M_core, initial_X, R_core, R_star

        # update step size and t according to step-control
        else:
            t = t_new
            dt = dt_next
            X = X_new
            R_ph = R_ph_new

    rad_err = rand() % (0.18 - 0.08) + 0.08 * R_ph
    observed_radius = R_ph + rand_py.choice([-1.0,1.0])*rad_err
    return observed_radius, period, M_core, initial_X, R_core, R_star

#////////////////////////////////// X vs t PLOT ////////////////////////////// #
# def X_2(t, period, M_star, rho_core, M_core):
#
#     if t < 100:
#         KH_timescale = 100
#     else:
#         KH_timescale = t
#
#     X_2 = 0.0027 * (period / 10)**0.08 * (M_star)**-0.15 * (KH_timescale / 100)**0.37 * \
#           (rho_core / 5.5)**-0.82 * (M_core / 5)**0.17
#
#     return X_2
#
# t_range = np.arange(1,3300)
# X_2_range = [X_2(t=i,period=10,M_star=1.0,rho_core=5.5,M_core=5.0) for i in t_range]
# X_3over2_range = [i/10 for i in X_2_range]
#
# X_range = np.logspace(-3.3,0.0, 20)
# plt.style.use('classic')
# for i in X_range:
#     print 'X = ',i
#     params = [i, 5.5, 5.0, 10, 1.0, 1.0, 100]
#     observed_radius, period, M_core, X, R_core, R_star, t = RK45_driver(1, 3000, 0.01, 1e-6, params)
#
#     plt.loglog([i*1e6 for i in t],X, color='black', linewidth=1.0)
#
# plt.loglog([i*1e6 for i in t_range], X_2_range, linewidth=1.7, linestyle='--', color='blue')
# plt.loglog([i*1e6 for i in t_range], X_3over2_range, linewidth=1.7, linestyle='--', color='green')
# hfont = {'fontname':'Courier New'}
# plt.ylabel(r'Envelope Mass Fraction (X)', fontsize=16, **hfont)
# plt.xlabel(r'Time [yrs]', fontsize=16, **hfont)
# plt.ylim([1e-4,1])
# plt.text(3.5e9, 1e-2, s=r'$2 R_c$', color='blue', fontsize=15, **hfont)
# plt.text(3.5e9, 1.3e-3, s=r'$1.5 R_c$', color='green', fontsize=15, **hfont)
# plt.xticks(fontsize=16, fontname = "Courier New")
# plt.yticks(fontsize=16, fontname = "Courier New")
# plt.tick_params(which='major', length=8)
# plt.tick_params(which='minor', length=4)
# plt.show()

#///////////////////////////////// X vs t_X plot ///////////////////////////// #
# plt.rcParams["font.family"] = "Courier New"
# hfont = {'fontname':'Courier New'}
# params = {
#    'xtick.labelsize': 12,
#    'ytick.labelsize': 12,}
# plt.rcParams.update(params)
# # plt.tick_params("both", length=6, which="major", width="1", direction="in")
# # plt.tick_params("both", length=3, which="minor", width="1", direction="in")
# plt.style.use('classic')
#
#
# X_range = np.logspace(-4,-0.5,100)
# tX_range_1 = []
# tX_range_2 = []
#
# for i in X_range:
#     tX1 = calculate_tX(t=1, X=i, M_core=5, M_star=1, a=0.1, R_core=1.6, KH_timescale_cutoff=100, R_guess=None)
#
#     tX_range_1.append(tX1)
#
# plt.loglog(X_range,tX_range_1,color='black')
# plt.xscale('log')
# plt.xlim([5e-4,1])
# plt.xticks([0.001,0.01,0.1,1],[0.001,0.01,0.1,1],fontsize=16, fontname = "Courier New")
# plt.yticks([1,10,100], [1,10,100], fontsize=19, fontname = "Courier New")
# plt.xlabel(r"Envelope Mass Fraction",fontsize=19, **hfont)
# plt.ylabel(r'Mass-loss Timescale [Myrs]',fontsize=19, **hfont)
# plt.tick_params(which='major', length=8)
# plt.tick_params(which='minor', length=4)
#
# plt.tight_layout()
# plt.savefig('../Figures/MassLossTimescale.pdf', format='pdf', dpi=2000)
#
# plt.show()

# //////////////////////////////////////////////////////////////////////////// #


# R_core, t, X, R_ph = RK45_driver(t_start=1, t_stop=3000, dt_try=0.01, accuracy=1e-5,
#                                  initial_X=0.0122, core_density=5.5, M_core=0.2, period=1.0, M_star=1.058, KH_timescale_cutoff=100)
#
# print "{0} --> {1}".format(X[0], X[-1])
# print R_core

# print snr_recovery(1.0, 1.0, 1.0)

# params = [0.1, 5.0, 2.0, 10, 1.0, 1.0, 100]
# for i in range(100):
#     observed_radius, period, M_core, initial_X, R_core, R_star = RK45_driver(1, 3000, 0.01, 1e-5, params)
