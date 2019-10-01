from __future__ import division
import numpy as np
import math
from scipy.optimize import brentq
from constants import *

from libc.math cimport exp, log10, log, fabs
from cpython cimport bool

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

cpdef double I2(double dR_Rc):
    cdef double log_dR_Rc = log10(dR_Rc)
    cdef double log_I2 = log(1.2598238012325735 + exp(-2.4535812359864062*1.0798633578446974*log_dR_Rc)) / -1.0798633578446974 - 4.8814961274562990e-01
    cdef double I2 = 10**log_I2
    return I2


def I2_I1_interpolate(dR_Rc):
    """
    interpolates from tabulated integral values of I2/I1:
    dR_Rc = dR / R_c
    """

    I2_I1_value = np.interp(dR_Rc, dR_Rc_array, I2_I1_array)
    return I2_I1_value

cpdef I2_I1(dR_Rc):
    cdef double log_dR_Rc = log10(dR_Rc)
    cdef double log_I2_I1 = -7.6521840215423576e-01 / (1.0 + exp(-(log_dR_Rc-5.7429970641208375e-02)/6.4338705851296174e-01))**1.8214254374336605E+00
    cdef double I2_I1 = 10**log_I2_I1
    return I2_I1

# ///////////////////////////////// BRENT METHOD ///////////////////////////// #

cpdef double brent(double lower, double upper, double tol, unsigned int max_iter,
                  double T_eq, double c_s_squared, double KH_timescale_seconds,
                  double M_core_kg, double R_core_meters, double X):

    cdef double a = lower
    cdef double b = upper
    cdef double fa = R_rcb_equation(a, T_eq, c_s_squared,
                                    KH_timescale_seconds, M_core_kg,
                                    R_core_meters, X)

    cdef double fb = R_rcb_equation(b, T_eq, c_s_squared,
                                    KH_timescale_seconds, M_core_kg,
                                    R_core_meters, X)
    cdef double fs = 0

    if fa * fb >= 0:
        print "f(a) same sign as f(b)"
        return 0.0

    if fabs(fa) < fabs(fb):
        a = a + b
        b = a - b
        a = a - b
        fa = fa + fb
        fb = fa - fb
        fa = fa - fb

    cdef double c = a
    cdef double fc = fa
    cdef bool mflag = True
    cdef double s = 0
    cdef double d = 0
    cdef unsigned int i = 1

    for i in range(max_iter):
        if fabs(b-a) < tol:
            return s

        if fa != fc and fb != fc:
            s = ( a * fb * fc / ((fa - fb) * (fa - fc)) ) \
              + ( b * fa * fc / ((fb - fa) * (fb - fc)) ) \
              + ( c * fa * fb / ((fc - fa) * (fc - fb)) )
        else:
            s = b - fb * (b - a) / (fb - fa)

        if ((s < (3 * a + b) * 0.25) or (s > b)) or \
           ( mflag and (fabs(s-b) >= (fabs(b-c) * 0.5)) ) or \
           ( not mflag and (fabs(s-b) >= (fabs(c-d) * 0.5)) ) or \
           ( mflag and (fabs(b-c) < tol) ) or \
           ( not mflag and (fabs(c-d) < tol)):

            s = (a+b)*0.5
            mflag = True

        else:
            mflag = False;

        fs = R_rcb_equation(s, T_eq, c_s_squared,
                            KH_timescale_seconds, M_core_kg,
                            R_core_meters, X)
        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs

        else:
            a = s
            fa = fs

        if fabs(fa) < fabs(fb):
            a = a + b
            b = a - b
            a = a - b

            fa = fa + fb
            fb = fa - fb
            fa = fa - fb

    print "The solution does not converge or iterations are not sufficient"
    return 0.0


# //////////////////////////// CALCULATE STATE VARIABLES ///////////////////// #

cdef double calculate_T_eq(double M_star, double a):
    """
    Calculates the equilibrium temperature of the planet:
    M_star: Mass of host star (solar masses)
    a = semi-major axis of orbit (AU)
    """

    cdef double T_eq = T_eq_earth * (1/a)**0.5 * (M_star)**0.8
    return T_eq


cdef double calculate_sound_speed_squared(double T_eq):
    """
    Calculates the isothermal sound speed of atmosphere:
    T_eq = Equilibrium temperature of planet (K)
    """

    cdef double c_s_squared = (k_B * T_eq) / (mu * m_H)
    return c_s_squared

# ///////////////////// ANALYTIC MODEL EQUATIONS FOR R_rcb /////////////////// #


cdef double R_rcb_equation(double R_rcb, double T_eq, double c_s_squared, double KH_timescale_seconds, double M_core_kg, double R_core_meters, double X):


    # collecting physical constants and terms which aren't a function of R_rcb
    cdef double c1 = (4 * pi * mu * m_H / (M_core_kg * k_B))
    cdef double c2 = ((Delta_ab * G * M_core_kg / c_s_squared)**(1/(gamma-1)))
    cdef double c3 = ((64 * pi * stefan * (T_eq**(3-alpha-beta)) * KH_timescale_seconds / (3 * kappa_0 * M_core_kg * X))**(1/(alpha+1)))

    cdef double c = c1 * c2 * c3

    # full equation
    cdef double equation = c * (R_rcb**3) * I2((R_rcb/R_core_meters)-1) * ((1/R_rcb)**(1/(gamma-1))) * ((I2_I1((R_rcb/R_core_meters)-1) * R_rcb)**(1/(alpha+1))) - X

    return equation



# /////////////////////////////// SOLVE FOR R_rcb //////////////////////////// #

cdef solve_Rho_rcb_and_R_rcb(double T_eq, double c_s_squared, double KH_timescale_seconds, double M_core_kg, double R_core_meters, double X, double R_guess):
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


    if R_guess == 0.0:
        sign_test1 = np.sign(R_rcb_equation(R_core_meters*1.0001, T_eq, c_s_squared, KH_timescale_seconds, M_core_kg, R_core_meters, X))
        sign_test2 = np.sign(R_rcb_equation(500*R_core_meters, T_eq, c_s_squared, KH_timescale_seconds,M_core_kg, R_core_meters, X))
        if sign_test1 == sign_test2:
            print "ERROR WITH BRENTQ SOLVER: f(a) and f(b) have same sign"
            print "PARAMS"
            print T_eq, c_s_squared, KH_timescale_seconds,M_core_kg, R_core_meters, X
            return 0.0, 0.0
#        R_rcb = brent(R_core_meters*1.0001, 500*R_earth, 1e-5, 100, T_eq, c_s_squared, KH_timescale_seconds,M_core_kg, R_core_meters, X)
        R_rcb = brentq(R_rcb_equation, R_core_meters*1.0001, 500*R_earth, args=(T_eq, c_s_squared, KH_timescale_seconds,
                       M_core_kg, R_core_meters, X), disp=False)

    else:
        sign_test1 = np.sign(R_rcb_equation(R_core_meters*1.0001, T_eq, c_s_squared, KH_timescale_seconds,M_core_kg, R_core_meters, X))
        sign_test2 = np.sign(R_rcb_equation(R_earth*(1.0+R_guess), T_eq, c_s_squared, KH_timescale_seconds,M_core_kg, R_core_meters, X))
        if sign_test1 == sign_test2:
            print "ERROR WITH BRENTQ SOLVER: f(a) and f(b) have same sign"
            print "PARAMS"
            print T_eq, c_s_squared, KH_timescale_seconds,M_core_kg, R_core_meters, X
            return 0.0, 0.0
#        R_rcb = brent(R_core_meters*1.0001, R_earth*(1.0+R_guess), 1e-5, 100, T_eq, c_s_squared, KH_timescale_seconds,M_core_kg, R_core_meters, X)
        R_rcb = brentq(R_rcb_equation, R_core_meters*1.0001, R_earth*(1.0+R_guess), args=(T_eq, c_s_squared, KH_timescale_seconds,
                       M_core_kg, R_core_meters, X), disp=False)



    cdef double Rho_rcb_1 = (2.35 * 1.67e-27 / 1.381e-23)
    cdef double Rho_rcb_2 = ((I2_I1((R_rcb/R_core_meters)-1) * 64.0 * 3.14159 * 5.67e-8 * (T_eq**(3-0.68-0.45)) * KH_timescale_seconds * R_rcb / (3 * 2.294e-8 * M_core_kg * X))**(1/(0.68+1)))
    cdef double Rho_rcb = Rho_rcb_1 * Rho_rcb_2


    return R_rcb, Rho_rcb



# ////////////////////////// SOLVE FOR R_photosphere ///////////////////////// #

cpdef double calculate_R_photosphere(double t, double M_star, double a, double M_core, double R_core, double X, double KH_timescale_cutoff, double R_guess):
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
    cdef double T_eq = calculate_T_eq(M_star, a)
    cdef double c_s_squared = calculate_sound_speed_squared(T_eq)

    cdef double KH_timescale_seconds
    if t < KH_timescale_cutoff:
        KH_timescale_seconds = KH_timescale_cutoff*Myr
    else:
        KH_timescale_seconds = t*Myr

    # convert to SI
    cdef double M_core_kg = M_core * M_earth
    cdef double R_core_meters = R_core * R_earth

    # solve simultaneous equations for radiative-convective boundary radius and density
    cdef double R_rcb, Rho_rcb
    R_rcb, Rho_rcb = solve_Rho_rcb_and_R_rcb(T_eq, c_s_squared, KH_timescale_seconds, M_core_kg, R_core_meters, X, R_guess)

    if (R_rcb, Rho_rcb) == (0.0, 0.0):
        return 0.0

    # calculate gravitational constant (assumed constant)
    cdef double g = G * M_core_kg / (R_rcb*R_rcb)

    # calcualte scale height
    cdef double H = (k_B * T_eq) / (mu * m_H * g)

    # locate photosphere by finding pressure at which P=2/3 * g/kappa
    cdef double P_photosphere = (2 * g / (3 * kappa_0 * T_eq**beta))**(1/(alpha+1))

    # calculate photospheric density
    cdef double Rho_photosphere = P_photosphere * mu * m_H / (k_B * T_eq)

    # calculate photospheric radius
    cdef double R_photosphere = R_rcb + H * log(Rho_rcb / Rho_photosphere)

    return R_photosphere

#test script
# R_photosphere = calculate_R_photosphere(t=3000, M_star=1, a=0.1, M_core=5.0, R_core=2.0, X=0.05, KH_timescale_cutoff=3000, R_guess=None)
# print R_photosphere/R_earth
