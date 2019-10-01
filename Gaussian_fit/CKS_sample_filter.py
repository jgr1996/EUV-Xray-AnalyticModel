from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas
from scipy.stats import norm, lognorm, entropy
from numpy import random as rand
import emcee
import corner

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

# new array of [stellar radius, stellar mass, planet radius, planet period]
useful_CKS = np.array([filtered_CKS[:,94], filtered_CKS[:,97], filtered_CKS[:,109], filtered_CKS[:,6]])
np.savetxt('CKS_filtered.csv', useful_CKS, delimiter=',')
print useful_CKS.shape


def plot_Period_Radius():

    R = list(useful_CKS[2,:])
    P = list(useful_CKS[3,:])


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
    cba.set_label('Normalised Planet Density', fontsize=16, args=(x, y, yerr)*hfont)
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

    R = list(useful_CKS[2,:])
    P = list(useful_CKS[3,:])

    plt.style.use('classic')
    R_bins = np.logspace(-0.301,0.6, 20)
    plt.hist(R, bins=R_bins, histtype='step', color='black', linewidth=2)
    plt.xscale('log')
    plt.xlim([0.5,6])
    plt.rcParams["font.family"] = "Courier New"
    hfont = {'fontname':'Courier New'}
    plt.title("CKS Data", fontsize=22, **hfont)
    plt.xlabel(r'Planet Radius [R$_\oplus]$', fontsize=16, **hfont)
    plt.ylabel(r'N', fontsize=16, **hfont)
    plt.xticks([0.5,1,2,3,4,5,6],[0.5,1,2,3,4,5,6], fontsize=16, fontname = "Courier New")
    plt.yticks([20,40,60,80,100,120], [20,40,60,80,100,120], fontsize=16, fontname = "Courier New")
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4)
    plt.show()

def plot_P_histogram():

    R = list(useful_CKS[2,:])
    P = list(useful_CKS[3,:])

    plt.style.use('classic')
    P_bins = np.logspace(0,2,40)
    plt.hist(P, bins=P_bins, histtype='step', color='black', linewidth=2)
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

def planet_radius_err():

    radii = filtered_CKS[:,109]
    err1 = filtered_CKS[:,110]
    err2 = filtered_CKS[:,111]
    print radii[0]
    print err1[0]
    print err2[0]

    mean_err = []
    for i in range(len(err1)):
        mean_i = (err1[i] - err2[i]) / 2
        ratio_i = mean_i / radii[i]
        mean_err.append(ratio_i)

    plt.hist(mean_err, bins=20)
    plt.show()

def stellar_mass():
    M_star = list(useful_CKS[1,:])
    plt.hist(M_star, bins=10, histtype='step', color='black', linewidth=2)
    plt.show()

def stellar_radius():
    M_star = list(useful_CKS[0,:])
    plt.hist(M_star, bins=50, histtype='step', color='black', linewidth=2)
    plt.show()

def fit_smass_err():

    """
    best for CKS-I is { 's':0.8705, 'loc':0.0301, 'scale':0.01072 }
    """

    smass = filtered_CKS[:,97]
    err1 = filtered_CKS[:,98]
    err2 = filtered_CKS[:,99]
    print smass[0]
    print err1[0]
    print err2[0]

    mean_err = []
    for i in range(len(err1)):
        mean_i = (err1[i] - err2[i]) / 2
        ratio_i = mean_i / smass[i]
        mean_err.append(ratio_i)

    plt.hist(mean_err, bins=20, density=True, alpha=0.6, color='g')

    s, loc, scale = lognorm.fit(mean_err)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = lognorm.pdf(x, s, loc=loc, scale=scale)

    plt.plot(x, p, 'k', linewidth=2)

    print s, loc, scale
    plt.show()



def smass_likelihood(theta):

    [mu, std] = theta
    if any(n < 0.0 for n in theta):
        return -np.inf
    if any(n > 2.0 for n in theta):
        return -np.inf

    smass = []
    for i in range(946):
        smass_i = rand.normal(loc=mu, scale=std)
        smass_i = smass_i * (1 + rand.choice([-1.0,1.0])*lognorm.rvs(0.8705, loc=0.0301, scale=0.01072))
        smass.append(smass_i)

    smass_data = list(useful_CKS[1,:])
    bins = np.linspace(min(smass_data),max(smass_data),20)

    hist, _bins = np.histogram(smass, bins=bins)
    hist_data, _bins_data = np.histogram(list(useful_CKS[1,:]), bins=bins)

    L = np.sum(-(hist - hist_data)*(hist - hist_data) / 0.1)

    if np.isnan(L):
        return -np.inf
    else:
        return L





def fit_stellar_mass():

    """
    best for CKS-I is {'mu':1.04, 'std':0.15}
    """

    theta = [1.04, 0.15]

    ndim = len(theta)
    n_walkers = 400

    theta_guesses = []
    for i in range(n_walkers):
        theta_guesses.append([x + rand.uniform(0, 5e-2*x) for x in theta])

    sampler = emcee.EnsembleSampler(n_walkers, ndim, smass_likelihood)
    sampler.run_mcmc(theta_guesses, 5000, progress=True)
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    np.savetxt('../smass_chain.csv', flat_samples, delimiter=',')

    labels = [r'M$_{\odot,mean}$',r'$\sigma_{M_\odot}$']
    fig = corner.corner(flat_samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('../CKS-smass-corner.pdf', format='pdf', dpi=1000)


# fit_stellar_mass()
