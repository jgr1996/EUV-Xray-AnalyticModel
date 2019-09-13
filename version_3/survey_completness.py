from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from constants import *
from matplotlib.colors import LogNorm

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ SIMPLE PROBLEM \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #
P_array = np.logspace(-1.0,2.5,150)
Rpl_array = np.logspace(-1.0,1.5,150)
P,R = np.meshgrid(P_array,Rpl_array)

prob_of_detection = np.ndarray((len(P_array), len(Rpl_array)))
prob_of_transit = np.ndarray((len(P_array), len(Rpl_array)))
total_completness = np.ndarray((len(P_array), len(Rpl_array)))
mean = 0

for i in range(prob_of_detection.shape[0]):
    for j in range(prob_of_detection.shape[1]):

        P_i = P_array[i]
        R_j = Rpl_array[j]

        a_i = (((P_i * 24 * 60 * 60)**2 * G * 1.1 * M_sun / (4 * pi * pi))**(1/3))
        prob_of_transit[i,j] = 0.7 * ((1.25*R_sun) / a_i)


        m_i = 1.4e5 * ((R_j*R_earth) / (1.1 * R_sun))**2 * np.sqrt((4*365)/P_i) * (1/60)
        prob_of_detection[i,j] = stats.gamma.cdf(m_i,17.56,scale=0.49)
        mean =+ prob_of_detection[i,j]

        total_completness[i,j] = prob_of_transit[i,j]#*prob_of_detection[i,j]

np.savetxt("survey_completeness.txt", total_completness, delimiter=',')


print total_completness.max()
plt.figure(1)
plt.xscale("log")
plt.yscale("log")
plt.contourf(P,R,prob_of_transit.T, cmap="Blues_r", linewidth=1)
plt.title(r'p$_{transit}$', fontsize=16)
plt.xticks([1,3,10,30,100,300],[1,3,10,30,100,300])
plt.yticks([0.3,0.5,1,2,4,10],[0.3,0.5,1,2,4,10])
plt.xlim([1,300])
plt.ylim([0.3,10])
plt.colorbar()

plt.figure(2)
plt.xscale("log")
plt.yscale("log")
plt.contourf(P,R,prob_of_detection.T, cmap="Blues_r", linewidth=1)
plt.title(r'p$_{detection}$', fontsize=16)
plt.xticks([1,3,10,30,100,300],[1,3,10,30,100,300])
plt.yticks([0.3,0.5,1,2,4,10],[0.3,0.5,1,2,4,10])
plt.xlim([1,300])
plt.ylim([0.3,10])
plt.colorbar()

plt.figure(3)
plt.xscale("log")
plt.yscale("log")
plt.contourf(P,R,total_completness.T, levels=[0.0, 1e-3, 1e-2, 2e-2, 5e-2, 0.1] , cmap="Blues_r")
plt.title(r'p$_{transit}$ * p$_{detection}$', fontsize=16)
plt.colorbar()
plt.xticks([1,3,10,30,100,300],[1,3,10,30,100,300])
plt.yticks([0.3,0.5,1,2,4,10],[0.3,0.5,1,2,4,10])
plt.xlim([1,300])
plt.ylim([0.3,10])
plt.show()

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ 5D PROBLEM \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ #

# P_array = np.logspace(0,2,50)
# X_array = np.logspace(-4,-0.3,50)
# rho_array = np.linspace(3.0, 8.0,50)
# M_core_array= np.logspace(-1,1,50)
#
# prob_of_detection = np.ndarray((len(P_array), len(X_array), len(rho_array), len(M_core_array)))
# prob_of_transit = np.ndarray((len(P_array), len(X_array), len(rho_array), len(M_core_array)))
# total_completness = np.ndarray((len(P_array), len(X_array), len(rho_array), len(M_core_array)))
#
#
# for a in range(prob_of_detection.shape[0]):
#     print "{}%...".format(int(100*a/len(P_array))+1)
#     for b in range(prob_of_detection.shape[1]):
#         for c in range(prob_of_detection.shape[2]):
#             for d in range(prob_of_detection.shape[3]):
#
#
#                 P_i = P_array[a]
#                 a_i = (((P_i * 24 * 60 * 60)**2 * G * 1.05 * M_sun / (4 * pi * pi))**(1/3))
#                 prob_of_transit_i = 0.7 * ((1.2*R_sun) / a_i)
#                 prob_of_transit[a,b,c,d] = prob_of_transit_i
#
#                 density_norm_i = 1.7589 * rho_array[c]**(-0.3329)
#                 R_core_i = density_norm_i * ((M_core_array[d])**0.25)
#                 R_i = R_photosphere_cython.calculate_R_photosphere(1.0, 1.1, a_i, M_core_array[d], R_core_i, X_array[b], 100.0, 0.0) / R_earth
#
#                 m_i = 3e5 * ((R_i*R_earth) / (1.05 * R_sun))**2 * np.sqrt((4*365)/P_i) * (1/60)
#                 prob_of_detection_i = stats.gamma.cdf(m_i,17.56,scale=0.49)
#                 prob_of_transit[a,b,c,d] = prob_of_detection_i
#
#                 total_completness[a,b,c,d] = prob_of_transit_i*prob_of_detection_i
#
# np.save("./completeness_array.npy", total_completness)
