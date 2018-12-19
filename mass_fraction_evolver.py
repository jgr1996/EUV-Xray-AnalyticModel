from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from constants import *
import R_photosphere
import RKF45

# ///////////////////////// CALCULATE MASS-LOSS TIMESCALE //////////////////// #
def calculate_tX(t, X, M_core, M_star, a, R_core):

    # convert to SI units
    M_core_kg = M_core * M_earth
    a_meters = a * AU
    t_seconds = t * Myr

    # Calculate envelope mass (kg)
    M_env_kg = X * M_core_kg

    # calculate saturation luminosity for photoevaporation
    L_sat = 10**(-3.5) * L_sun * M_star
    if t_seconds < t_sat:
        L_HE = L_sat
    else:
        L_HE = L_sat * (t_seconds/t_sat)**(-1-a0)

    # Calulate photospheric radius
    R_ph = R_photosphere.calculate_R_photosphere(t, M_star, a,
                                                 M_core, R_core, X)

    # Calculate mass loss rate due to photoevaporation
    M_env_dot = eta * R_ph**3 * L_HE / (4 * a_meters * a_meters * G * M_core_kg)

    # Calculate mass-loss timescale
    tX = (M_env_kg / M_env_dot) / Myr

    return tX

# /////////////////////////// MASS FRACTION EVOLUTION ODE //////////////////// #

def dXdt_ODE(t, X, parameters):

    M_core, M_star, a, R_core = parameters

    tX = calculate_tX(t, X, M_core, M_star, a, R_core)

    dXdt = - X / tX

    return dXdt

# /////////////////// EVOLVE MASS FRACTION ACCORDING TO ODE ////////////////// #

def RK45_driver(t_start, t_stop, dt_try, accuracy,
                initial_X, core_density, M_core, period, M_star):


    # convert variables to be constrained to algorithm variables:

    # core density to core radius (measure in Earth radii)
    core_density_SI= core_density * 1000
    R_core = ((3 * M_core * M_earth / (4 * pi * core_density_SI))**(1/3)) / R_earth

    # orbital period to semi-major axis measured in AU
    a = (((period * 24 * 60 * 60)**2 * G * M_star * M_sun / (4 * pi * pi))**(1/3)) / AU

    #calculate initial photospheric radius
    R_ph_init = R_photosphere.calculate_R_photosphere(t_start, M_star, a,
                                                      M_core, R_core, initial_X)

    # define arrays for storing t steps and variables
    X_array = np.array([initial_X])
    t_array = np.array([t_start])
    R_ph_array = np.array([R_ph_init/R_earth])

    # define first time and time step
    t = t_array[0]
    dt = dt_try

    # setup loop
    while t<= t_stop:
        #perform an adaptive RK45 step
        (t_new, X_new, dt_next) = RKF45.step_control(t, X_array[-1], dt, dXdt_ODE, accuracy,
                                                     parameters=(M_core, M_star, a, R_core))


        # calculate new R_ph
        R_ph_new = R_photosphere.calculate_R_photosphere(t_new, M_star, a,
                                                         M_core, R_core, X_new)



        if R_ph_new/R_earth <= R_core:
                # update new variables
                X_array = np.append(X_array, 0.0)
                # update new time
                t_array = np.append(t_array, t_stop+1e-7)
                # update new R_ph
                R_ph_array = np.append(R_ph_array, R_core)
                return (R_core, t_array, X_array, R_ph_array)
        else:
            # update new variables
            X_array = np.append(X_array, X_new)
            # update new time
            t_array = np.append(t_array, t_new)
            # update new R_ph
            R_ph_array = np.append(R_ph_array, R_ph_new/R_earth)

        # update step size and t according to step-control
        t = t_array[-1]
        dt = dt_next

    return R_core, t_array, X_array, R_ph_array

#//////////////////////////////////// TEST 1 //////////////////////////////// #
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
# X_range = np.logspace(-3.3,-0.2, 20)
# plt.style.use('classic')
# for i in X_range:
#     print 'X = ',i
#     R_core, t, X, R_ph = RK45_driver(t_start=1, t_stop=3000, dt_try=0.01, accuracy=1e-8,
#                                      initial_X=i, core_density=5.5, M_core=5.0, period=10, M_star=1.0)
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

#//////////////////////////////////// TEST 2 //////////////////////////////// #

# X_range = np.logspace(-3,0,100)
# tX_range_1 = []
# tX_range_2 = []
#
# for i in X_range:
#     tX1 = calculate_tX(t=1, X=i, M_core=5.0, M_star=1.0, a=0.1, R_core=1.7)
#     tX2 = calculate_tX(t=1, X=i, M_core=5.0, M_star=1.0, a=0.1, R_core=1.8)
#
#     tX_range_1.append(tX1)
#     tX_range_2.append(tX2)
#
# plt.loglog(X_range,tX_range_1)
# plt.loglog(X_range,tX_range_2)

#//////////////////////////////////// TEST 3 //////////////////////////////// #
# R_core, t, X, R_ph = RK45_driver(t_start=1, t_stop=3000, dt_try=0.01, accuracy=1e-8,
#                                  initial_X=0.1, core_density=5.5, M_core=5.0, period=10, M_star=1.0)
# print X
# print R_ph
#
# print 'planet radius is {0} earth radii'.format(R_ph[-1] + R_core)
