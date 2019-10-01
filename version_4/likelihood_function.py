from __future__ import division
import os
import sys
import datetime
import time
import numpy as np
from numpy import log10
import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
from scipy import interpolate
from mpi4py import MPI
import evolve_population
from evolve_population import bernstein_poly


def likelihood(theta, N, data_array, completeness_array):



    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank        # rank of this process

    # composition must be between -1 and 1
    if theta[0] < -1 or theta[0] > 1:
        return -np.inf

    # bernstein coeffs cannot be less than 0.0
    if any(n < 0.0 for n in theta[1:]):
        return -np.inf

    # # max for bernstein coeffs
    # if any(n > 50.0 for n in theta[1:]):
    #     return -np.inf


    # ensure Bernstein CDFs are monotonically increasing
    a_range = np.arange(0.0, 1.0001, 0.0001)

    initial_X_coeffs = theta[1:7]
    order_X = len(initial_X_coeffs) - 1
    X_min, X_max = -4.0, 0.0
    X_poly_min, X_poly_max = bernstein_poly(0, order_X, initial_X_coeffs), bernstein_poly(1, order_X, initial_X_coeffs)
    X_poly_norm = X_poly_max - X_poly_min
    X_norm = X_max - X_min
    CDF_X = [(((X_norm/X_poly_norm) * ((bernstein_poly(i, order_X, initial_X_coeffs)) - X_poly_min)) + X_min) for i in a_range]
    for i in range(1,len(CDF_X)):
        if CDF_X[i-1] > CDF_X[i]:
            print "Not increasing CDF for X"
            return -np.inf

    core_mass_coeffs = theta[7:]
    order_M = len(core_mass_coeffs) - 1
    M_min, M_max = 0.8, 15.0
    M_poly_min, M_poly_max = bernstein_poly(0, order_M, core_mass_coeffs), bernstein_poly(1, order_M, core_mass_coeffs)
    M_poly_norm = M_poly_max - M_poly_min
    M_norm = M_max - M_min
    CDF_M = [(((M_norm/M_poly_norm) * ((bernstein_poly(i, order_M, core_mass_coeffs)) - M_poly_min)) + M_min) for i in a_range]
    for i in range(1,len(CDF_M)):
        if CDF_M[i-1] > CDF_M[i]:
            print "Not increasing CDF for M"
            return -np.inf


    # calculate population
    R, P, M, X, R_core, M_star = evolve_population.CKS_synthetic_observation(N, theta)


    if rank == 0:

        R_min, R_max = 0.75, 8.0
        P_min, P_max = 1.0, 100.0

        if len(R) == 0:
            return -np.inf

        x = np.linspace(log10(0.1),log10(300),150)
        y = np.linspace(log10(0.1),log10(30),150)
        x = x[:-1]
        y = y[:-1]
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([X.ravel(), Y.ravel()])
        data = np.vstack([log10(P),log10(R)])
        kernel = stats.gaussian_kde(data, bw_method=0.25)
        Z = np.reshape(kernel(positions).T, X.shape)
        Z[Z==0] = 1e-299
        Z_biased = Z * completeness_array.T

        norm = 0
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] >= log10(P_min) and x[i] <= log10(P_max):
                    if y[j] >= log10(R_min) and y[j] <= log10(R_max):
                        norm = norm + Z_biased[i,j]
        norm = 1.0 / ( log10(P_max/P_min) * log10(R_max/R_min) * norm )
        Z_biased = norm * Z_biased

        KDE_interp = interpolate.RectBivariateSpline(y,x,Z_biased)

        data_CKS = np.vstack([log10(data_array[3,:]),log10(data_array[2,:])])
        kernel_CKS = stats.gaussian_kde(data_CKS, bw_method=0.25)
        Z_CKS = np.reshape(kernel_CKS(positions).T, X.shape)

        # np.savetxt("../../../Computing/IDL plots/Wu2018/comparison_bernstein/Z_synth.csv", Z_biased, delimiter=',')
        # np.savetxt("../../../Computing/IDL plots/Wu2018/comparison_bernstein/Z.csv", Z_CKS, delimiter=',')
        # np.savetxt("../../../Computing/IDL plots/Wu2018/comparison_bernstein/X.csv", X, delimiter=',')
        # np.savetxt("../../../Computing/IDL plots/Wu2018/comparison_bernstein/Y.csv", Y, delimiter=',')


        P_data = []
        R_data = []
        for i in range(len(data_array[3,:])):
            if data_array[2,i] >= R_min and data_array[2,i] <= R_max:
                if data_array[3,i] >= P_min and data_array[3,i] <= P_max:
                    R_data.append(log10(data_array[2,i]))
                    P_data.append(log10(data_array[3,i]))


        logL = 0
        for i in range(len(P_data)):
            # MUST SWAP ORDER OF P AND R!!!
            logL_i = log10(abs(KDE_interp(R_data[i], P_data[i])[0,0]))
            # print 10**P_data[i], 10**R_data[i], logL_i
            logL = logL + logL_i

        # plt.figure(1)
        # plt.contourf(x,y,Z, cmap='Oranges')
        # plt.colorbar()
        # plt.scatter(log10(P), log10(R), s=0.3, color='black', alpha=0.3)
        # plt.xlim([0,2])
        # plt.ylim([-0.3,1.1])
        # plt.xticks(log10([1,2,4,10,30,100]),[1,2,4,10,30,100])
        # plt.yticks(log10([0.5,1,2,4,10]),[0.5,1,2,4,10])
        #
        # plt.figure(3)
        # plt.contourf(x,y,Z_biased, cmap='Oranges')
        # plt.scatter(P_data, R_data, s=0.5, color='black', alpha=0.3)
        # plt.xlim([0,2])
        # plt.ylim([0,1.1])
        # plt.xticks(log10([1,2,4,10,30,100]),[1,2,4,10,30,100])
        # plt.yticks(log10([0.5,1,2,4,10]),[0.5,1,2,4,10])

        # Z_integrated = np.sum(Z_biased, axis=1)
        # Z_integrated = Z_integrated / Z_integrated.max()
        # Z_CKS_integrated = np.sum(Z_CKS, axis=1)
        # Z_CKS_integrated = Z_CKS_integrated / Z_CKS_integrated.max()
        #
        # plt.figure(4)
        # plt.plot(10**y,200*Z_integrated, label='Rogers')
        # # plt.plot(10**y,Z_CKS_integrated, label='CKS')
        # plt.hist(data_array[2,:], bins=np.logspace(log10(0.5),log10(10),20))
        # plt.xscale('log')0
        # plt.legend()
        #
        # plt.figure(3)
        # hist1, bins1 = np.histogram(R, bins=np.logspace(np.log10(0.5),np.log10(10),20))
        # hist2, bins2 = np.histogram(R_data, bins=np.logspace(np.log10(0.5),np.log10(10),20))
        #
        # hist1 = (hist1 / max(hist1)) * 150
        # hist2 = (hist2 / max(hist2)) * 150
        # print hist2
        # np.savetxt("../../../Computing/IDL plots/Wu2018/radius_gaps/bins.csv", bins1[:-1], delimiter=',')
        # np.savetxt("../../../Computing/IDL plots/Wu2018/radius_gaps/histRog.csv", hist1, delimiter=',')
        # np.savetxt("../../../Computing/IDL plots/Wu2018/radius_gaps/histCKS.csv", hist2, delimiter=',')
        # np.savetxt("../../../Computing/IDL plots/Wu2018/radius_gaps/RogErr.csv", [np.sqrt(k) for k in hist1], delimiter=',')
        # np.savetxt("../../../Computing/IDL plots/Wu2018/radius_gaps/CKSErr.csv", [np.sqrt(k) for k in hist2], delimiter=',')

        # plt.xscale('log')


        return logL
    else:
        return 0




# ///////////////////////////////// TEST ///////////////////////////////////// #

# comm = MPI.COMM_WORLD   # get MPI communicator object
# rank = comm.rank        # rank of this process
# CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
# CKS_completeness_array = np.loadtxt("survey_comp.csv", delimiter=',')
# thetas = [[-0.2,0.0,2.3,8.3,2.0,4.54,10.0,0.0,0.24,0.54,0.0,3.6,10.0]]
# # thetas = [[0.71,0.0,13.0,0.0,15.79,14.5,20.63,35.0,0.0,0.0,2.37,43.0,32.15,50.0]]
#
# for i in range(len(thetas)):
#     L = likelihood(thetas[i], 1000, CKS_array, CKS_completeness_array)
#     if rank == 0:
#         print "Theta = {}".format(thetas[i])
#         print "logL = {}".format(L)
#         plt.show()
