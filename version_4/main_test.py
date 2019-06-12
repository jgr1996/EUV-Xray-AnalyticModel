from __future__ import division
import time
import mass_fraction_evolver_cython


if __name__ == '__main__':

    time_1 = time.time()

    params = [0.1, 5.0, 2.0, 10, 1.0, 1.0, 100]
    mean = 0.0
    for i in range(100):
        observed_radius, period, M_core, initial_X, R_core, R_star = mass_fraction_evolver_cython.RK45_driver(1, 3000, 0.01, 1e-5, params)
        mean += observed_radius

    print mean/100
    time_2 = time.time()

    print "time elapsed: {}s".format(time_2 - time_1)
