from __future__ import division
import time
import numpy as np
import matplotlib.pyplot as plt
import mass_fraction_evolver_cython
import R_photosphere_cython
import evolve_population


if __name__ == '__main__':
    # time_1 = time.time()
    # for i in range(1):
    #     print R_photosphere_cython.I2_I1_interpolate(1.0)
    # time_2 = time.time()
    # print time_2-time_1
    #
    # time_1 = time.time()
    # for i in range(1):
    #     print R_photosphere_cython.I2_I1(1.0)
    # time_2 = time.time()
    # print time_2-time_1

    time_1 = time.time()

    mean = 0.0
    for i in range(100):
        observed_radius, period, M_core, initial_X, R_core, R_star = mass_fraction_evolver_cython.RK45_driver(1, 3000, 0.01, 1e-5, 0.1, 5.0, 2.0, 10, 1.0, 1.0, 100)
        mean += observed_radius

    print mean/100
    time_2 = time.time()

    print "time elapsed: {}s".format(time_2 - time_1)




    # X_range = np.logspace(-3.0,0.0, 20)
    # plt.style.use('classic')
    # for i in X_range:
    #     print 'X = ',i
    #     params = [i, 5.5, 11.0, 10, 1.0, 1.0, 100]
    #     observed_radius, period, M_core, X, R_core, R_star, t = mass_fraction_evolver_cython.RK45_driver(1, 3000, 0.01, 1e-6, params)
    #
    #     plt.loglog([i*1e6 for i in t],X, color='black', linewidth=1.0)
    #
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

#     X_initial_list = []
#     core_density_list = []
#     M_core_list = []
#     period_list = []
#     M_star_list = []
#     R_star_list = []
#     KH_timescale_cutoff_list = []
#
#     initial_X_coeffs = [0.01516316, 0.37426462, 0.21944966, 0.32256333, 0.61806997, 0.46451604]
#     core_mass_coeffs = [0.0094975, 0.35338382, 0.40983246, 0.358155, 0.57118452, 0.77865198]
#     density_mean = 5.5
#     KH_timescale = 100
#
#     transit_number = 0
#     while transit_number < 10:
#
#
#         X_initial, M_core, period= evolve_population.make_planet(initial_X_coeffs=initial_X_coeffs,
#                                                core_mass_coeffs=core_mass_coeffs,
#                                                period_params=(1.9, 7.6))
#
#         # model breaks down for very small periods
#         if period <= 0.5:
#             continue
#         # CKS data does not include P > 100
#         if period > 100.0:
#             continue
#
#
#         X_initial_list.append(X_initial)
#         core_density_list.append(density_mean)
#         M_core_list.append(M_core)
#         period_list.append(period)
#         transit_number = transit_number + 1
#
#
# print X_initial_list
