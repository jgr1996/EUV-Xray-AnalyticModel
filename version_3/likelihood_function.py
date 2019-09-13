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


def likelihood(theta, N, data_array, completeness_array):

    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank        # rank of this process

    # bernstein coeffs cannot be less than 0.0
    if any(n < 0.0 for n in theta[:]):
        return -np.inf

    # max for bernstein coeffs
    if any(n > 10.0 for n in theta[:]):
        return -np.inf

    # # density cannot be negative
    # if theta[0] > 10.0 or theta[0] < 0.0:
    #     return -np.inf


    # calculate population
    # time1 = time.time()
    R, P, M, X, R_core, M_star = evolve_population.CKS_synthetic_observation(N, theta)
    # time2 = time.time()

    # P = [3,3,3,3,3,60]
    # R = [2,2,2,4,4,4]

    if rank == 0:

        if len(R) == 0:
            return -np.inf

        x = np.linspace(log10(0.1),log10(300),150)
        y = np.linspace(log10(0.1),log10(30),150)
        x = x[:-1]
        y = y[:-1]
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([X.ravel(), Y.ravel()])
        data = np.vstack([log10(P),log10(R)])
        kernel = stats.gaussian_kde(data)#, bw_method=0.2)
        Z = np.reshape(kernel(positions).T, X.shape)
        Z[Z==0] = 1e-299
        Z_biased = Z * completeness_array.T
        KDE_interp = interpolate.RectBivariateSpline(y,x,Z_biased)

        data_CKS = np.vstack([log10(data_array[3,:]),log10(data_array[2,:])])
        kernel_CKS = stats.gaussian_kde(data_CKS)#, bw_method=0.2)
        Z_CKS = np.reshape(kernel_CKS(positions).T, X.shape)

        np.savetxt("../../../Computing/IDL plots/Wu2018/comparison_gaussian/Z_synth.csv", Z_biased, delimiter=',')
        np.savetxt("../../../Computing/IDL plots/Wu2018/comparison_gaussian/Z.csv", Z_CKS, delimiter=',')
        np.savetxt("../../../Computing/IDL plots/Wu2018/comparison_gaussian/X.csv", X, delimiter=',')
        np.savetxt("../../../Computing/IDL plots/Wu2018/comparison_gaussian/Y.csv", Y, delimiter=',')

        print len(X)


        P_data = []
        R_data = []
        for i in range(len(data_array[3,:])):
            if data_array[2,i] >= 0.7 and data_array[2,i] <= 6.0:
                R_data.append(log10(data_array[2,i]))
                P_data.append(log10(data_array[3,i]))


        logL = 0
        for i in range(len(P_data)):
            # MUST SWAP ORDER OF P AND R!!!
            logL_i = log10(abs(KDE_interp(R_data[i], P_data[i])[0,0]))
            # print 10**P_data[i], 10**R_data[i], logL_i
            logL = logL + logL_i

        plt.figure(1)
        plt.contourf(x,y,Z, cmap='Oranges')
        plt.colorbar()
        plt.scatter(log10(P), log10(R), s=0.3, color='black', alpha=0.3)
        plt.xlim([0,2])
        plt.ylim([0,1.1])
        plt.xticks(log10([1,2,4,10,30,100]),[1,2,4,10,30,100])
        plt.yticks(log10([1,2,4,10]),[1,2,4,10])


        plt.figure(3)
        plt.contourf(x,y,Z_biased, cmap='Oranges')
        plt.scatter(P_data, R_data, s=0.5, color='black', alpha=0.3)
        plt.xlim([0,2])
        plt.ylim([0,1.1])
        plt.xticks(log10([1,2,4,10,30,100]),[1,2,4,10,30,100])
        plt.yticks(log10([1,2,4,10]),[1,2,4,10])

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

comm = MPI.COMM_WORLD   # get MPI communicator object
rank = comm.rank        # rank of this process
CKS_array = np.loadtxt("CKS_filtered.csv", delimiter=',')
CKS_completeness_array = np.loadtxt("survey_comp.csv", delimiter=',')
thetas = [[6.23,0.01,1.55,9.0,2.13]]
# thetas = [[5.25,0.76,0.02,5.54,0.07,1.54,6.63,2.58]]

for i in range(len(thetas)):
    L = likelihood(thetas[i], 1000, CKS_array, CKS_completeness_array)
    if rank == 0:
        print "Theta = {}".format(thetas[i])
        print "logL = {}".format(L)
        plt.show()
