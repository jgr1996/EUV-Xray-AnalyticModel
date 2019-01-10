from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas

"""
This file takes the CKS data (available at https://california-planet-search.github.io/cks-website/)
and applies the filters applied in Fulton et al. (2017) to form the CKS III sample of Kepler planets.
Upon running the script the file is saved as 'CKS_filtered.csv'. Functions are also suppled to allow
1D and 2D histograms/KDEs.
"""

full_CKS = pandas.read_csv("CKS.csv")
full_CKS = np.array(full_CKS)

rows_to_delete = []
for i in range(full_CKS.shape[0]):

    # remove 'not dispositioned' data
    if np.isnan(full_CKS[i, 109]):
        rows_to_delete.append(i)

    # remove false positives
    if full_CKS[i,5] in ["FALSE POSITIVE"]:
        rows_to_delete.append(i)

    # remove Kp > 14.2
    if full_CKS[i,68] >= 14.2:
        rows_to_delete.append(i)

    # remove impact parameter b > 0.7
    if full_CKS[i,12] >= 0.7:
        rows_to_delete.append(i)

    # remove period > 100 days
    if full_CKS[i,6] >= 100:
        rows_to_delete.append(i)

    # remove giant stars
    condition = 10**((0.00025*(full_CKS[i,85] - 5500)) + 0.20)
    if full_CKS[i,94] >= condition:
        rows_to_delete.append(i)

    # temperature range for high precision spectroscopy
    if full_CKS[i,85] >= 6500:
        rows_to_delete.append(i)
    if full_CKS[i,85] <= 4700:
        rows_to_delete.append(i)

rows_to_delete = list(set(rows_to_delete))
filtered_CKS = np.delete(full_CKS, rows_to_delete, 0)

# new array of [stellar mass, planet radius, planet period]
useful_CKS = np.array([filtered_CKS[:,97], filtered_CKS[:,109], filtered_CKS[:,6]])
np.savetxt('CKS_filtered.csv', useful_CKS, delimiter=',')
print useful_CKS.shape


def plot_Period_Radius():

    R = list(useful_CKS[1,:])
    P = list(useful_CKS[2,:])


    plt.style.use('classic')
    plt.xlim([1,100])
    plt.ylim([0.5,10])
    plt.xscale('log')
    plt.yscale('log')

    x = np.logspace(0,2,100)
    y = np.logspace(-0.301,1,100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([np.log(X.ravel()), np.log(Y.ravel())])
    data = np.vstack([np.log(P),np.log(R)])
    kernel = stats.gaussian_kde(data)
    Z = np.reshape(kernel(positions).T, X.shape)
    Z_max = Z.max()
    Z = Z / Z_max

    plt.rcParams["font.family"] = "Courier New"
    hfont = {'fontname':'Courier New'}
    plt.contourf(X,Y,Z, cmap='Oranges')
    cba = plt.colorbar(boundaries=np.linspace(0,0.95,5))
    cba.set_label('Normalised Planet Density', fontsize=16, **hfont)
    plt.scatter(P, R, alpha=0.1, s=0.3, color='black')

    plt.title("CKS Data", fontsize=22, **hfont)
    plt.xlabel('Period [days]', fontsize=16, **hfont)
    plt.ylabel(r'Planet Radius [R$_\oplus]$', fontsize=16, **hfont)
    plt.xticks([1,10,100],[1,10,100],fontsize=16, fontname = "Courier New")
    plt.yticks([0.5,1,2,3,4,6,10], [0.5,1,2,3,4,6,10], fontsize=16, fontname = "Courier New")
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4)
    plt.show()

def plot_R_histogram():

    R = list(useful_CKS[1,:])
    P = list(useful_CKS[2,:])

    plt.style.use('classic')
    R_bins = np.logspace(-0.301,0.6, 20)
    plt.hist(R, density=True, bins=R_bins, histtype='step', color='black', linewidth=2)
    plt.xscale('log')
    plt.xlim([0.5,6])
    plt.rcParams["font.family"] = "Courier New"
    hfont = {'fontname':'Courier New'}
    plt.title("CKS Data", fontsize=22, **hfont)
    plt.xlabel(r'Planet Radius [R$_\oplus]$', fontsize=16, **hfont)
    plt.ylabel(r'dN/dlog R', fontsize=16, **hfont)
    plt.xticks([0.5,1,2,3,4,5,6],[0.5,1,2,3,4,5,6], fontsize=16, fontname = "Courier New")
    plt.yticks(fontsize=16, fontname = "Courier New")
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4)
    plt.show()

def plot_P_histogram():

    R = list(useful_CKS[1,:])
    P = list(useful_CKS[2,:])

    plt.style.use('classic')
    P_bins = np.logspace(0,2,40)
    plt.hist(P, density=True, bins=P_bins, histtype='step', color='black', linewidth=2)
    plt.xscale('log')
    plt.rcParams["font.family"] = "Courier New"
    hfont = {'fontname':'Courier New'}
    plt.title("CKS Data", fontsize=22, **hfont)
    plt.xlabel(r'Orbital Period [days]', fontsize=16, **hfont)
    plt.ylabel(r'dN/dlog P', fontsize=16, **hfont)
    plt.xticks([1,10,100],[1,10,100], fontsize=16, fontname = "Courier New")
    plt.yticks(fontsize=16, fontname = "Courier New")
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4)
    plt.show()
plot_P_histogram()
