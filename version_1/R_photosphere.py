from __future__ import division
import numpy as np
from scipy.optimize import brentq, fsolve
import math
from constants import *

"""
Author: Rogers, J. G
Date: 07/11/2018

This file carries out analytic calculations presented in Owen & Wu (2017) that
are used to determine the photospheric radius of a planet undergoing EUV-Xray
photo-evaporation.
"""

# /////////////////////////// IMPORT TABULATED INTEGRALS ///////////////////// #

dR_Rc_array = np.loadtxt("dR_Rc_array.csv", delimiter=',')
I2_array = np.loadtxt("I2.csv", delimiter=',')
I2_I1_array = np.loadtxt("I2_I1.csv", delimiter=',')

# /////////////////////////// INTERPOLATION OF INTEGRALS ///////////////////// #

def I2_interpolate(dR_Rc):
    """
    interpolates from tabulated integral values of I2:
    dR_Rc = dR / R_c
    """

    I2_value = np.interp(dR_Rc, dR_Rc_array, I2_array)
    return I2_value


def I2_I1_interpolate(dR_Rc):
    """
    interpolates from tabulated integral values of I2/I1:
    dR_Rc = dR / R_c
    """

    I2_I1_value = np.interp(dR_Rc, dR_Rc_array, I2_I1_array)
    return I2_I1_value

# //////////////////////////// CALCULATE STATE VARIABLES ///////////////////// #

def calculate_T_eq(M_star, a):
    """
    Calculates the equilibrium temperature of the planet:
    M_star: Mass of host star (solar masses)
    a = semi-major axis of orbit (AU)
    """

    T_eq = T_eq_earth * (1/a)**0.5 * (M_star)**0.8
    return T_eq


def calculate_sound_speed_squared(T_eq):
    """
    Calculates the isothermal sound speed of atmosphere:
    T_eq = Equilibrium temperature of planet (K)
    """

    c_s_squared = (k_B * T_eq) / (mu * m_H)
    return c_s_squared

# ///////////////////// ANALYTIC MODEL EQUATIONS FOR R_rcb /////////////////// #

def R_rcb_equation(R_rcb, T_eq, c_s_squared, KH_timescale_seconds,
                   M_core_kg, R_core_meters, X):
    """
    Returns equation for which solution is R_Rcb:

    R_rcb: Radiative-convective boundary radius
    a = semi-major axis of orbit (AU)
    KH_timescale = Kelvin-Helmholtz (cooling) timescale for atmosphere (years)
    M_core = Mass of planet core (Earth masses)
    R_core = Radius of planet core (Earth Radius)
    X = Envelope mass fraction M_env / M_core
    """

    # collecting physical constants and terms which aren't a function of R_rcb
    c = (4 * pi * mu * m_H / (M_core_kg * k_B)) * \
        ((Delta_ab * G * M_core_kg / c_s_squared)**(1/(gamma-1))) * \
        ((64 * pi * stefan * (T_eq**(3-alpha-beta)) * KH_timescale_seconds / \
        (3 * kappa_0 * M_core_kg * X))**(1/(alpha+1)))

    # full equation
    equation = c * (R_rcb**3) * \
               I2_interpolate((R_rcb/R_core_meters)-1) * ((1/R_rcb)**(1/(gamma-1))) * \
               ((I2_I1_interpolate((R_rcb/R_core_meters)-1) * R_rcb)**(1/(alpha+1))) - X

    return equation


# /////////////////////////////// SOLVE FOR R_rcb //////////////////////////// #

def solve_Rho_rcb_and_R_rcb(T_eq, c_s_squared, KH_timescale_seconds,
                            M_core_kg, R_core_meters, X, R_guess):
    """
    Calculates solution of R_rcb_equation using Newton-Raphson/secant method. Then
    finds Rho_rcb using the solution R_rcb:

    T_eq: Equilibrium temperature of planet (K)
    c_s_squared: isothermal sound speed (m^2 s^-2)
    a: semi-major axis of orbit (AU)
    KH_timescale_seconds: Kelvin-Helmholtz (cooling) timescale for atmosphere (seconds)
    M_core_kg: Mass of planet core (kg)
    R_core_meters: Radius of planet core (m)
    X: Envelope mass fraction M_env / M_core
    """


    if R_guess == None:
        R_rcb = brentq(R_rcb_equation, 0.001, 500*R_earth, args=(T_eq, c_s_squared, KH_timescale_seconds,
                       M_core_kg, R_core_meters, X))
    else:
        R_rcb = brentq(R_rcb_equation, 0.001, R_earth*(1.0+R_guess), args=(T_eq, c_s_squared, KH_timescale_seconds,
                       M_core_kg, R_core_meters, X))


    Rho_rcb = (mu * m_H / k_B) * ((I2_I1_interpolate((R_rcb/R_core_meters)-1) *  \
              64 * pi * stefan * (T_eq**(3-alpha-beta)) * KH_timescale_seconds * R_rcb / \
              (3 * kappa_0 * M_core_kg * X))**(1/(alpha+1)))


    return R_rcb, Rho_rcb



# ////////////////////////// SOLVE FOR R_photosphere ///////////////////////// #

def calculate_R_photosphere(t, M_star, a, M_core, R_core, X, KH_timescale_cutoff, R_guess):
    """
    Returns the photospheric radius of the planet (in meters):

    M_star: Mass of host star (solar masses)
    a: semi-major axis of orbit (AU)
    KH_timescale = Kelvin-Helmholtz (cooling) timescale for atmosphere (years)
    M_core = Mass of planet core (Earth masses)
    R_core = Radius of planet core (Earth Radius)
    X = Envelope mass fraction M_env / M_core

    """

    # calculate temperature and sound speed for system
    T_eq = calculate_T_eq(M_star, a)
    c_s_squared = calculate_sound_speed_squared(T_eq)

    if t < KH_timescale_cutoff:
        KH_timescale_seconds = KH_timescale_cutoff*Myr
    else:
        KH_timescale_seconds = t*Myr

    # convert to SI
    M_core_kg = M_core * M_earth
    R_core_meters = R_core * R_earth

    # solve simultaneous equations for radiative-convective boundary radius and density
    R_rcb, Rho_rcb = solve_Rho_rcb_and_R_rcb(T_eq, c_s_squared, KH_timescale_seconds,
                                             M_core_kg, R_core_meters, X, R_guess)

    # calculate gravitational constant (assumed constant)
    g = G * M_core_kg / (R_rcb*R_rcb)

    # calcualte scale height
    H = (k_B * T_eq) / (mu * m_H * g)

    # locate photosphere by finding pressure at which P=2/3 * g/kappa
    P_photosphere = (2 * g / (3 * kappa_0 * T_eq**beta))**(1/(alpha+1))

    # calculate photospheric density
    Rho_photosphere = P_photosphere * mu * m_H / (k_B * T_eq)

    # calculate photospheric radius
    R_photosphere = R_rcb + H * np.log(Rho_rcb / Rho_photosphere)

    return R_photosphere

#test script
# R_photosphere = calculate_R_photosphere(t=3000, M_star=1, a=0.1, M_core=5.0, R_core=2.0, X=0.05, KH_timescale_cutoff=3000, R_guess=None)
# print R_photosphere/R_earth
