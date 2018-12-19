from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


R = np.loadtxt("Period_and_injection_trial1/R_array.csv", delimiter=',')
P = np.loadtxt("Period_and_injection_trial1/P_array.csv", delimiter=',')


plt.style.use('classic')
plt.xlim([1,100])
plt.ylim([1,10])
plt.xscale('log')
plt.yscale('log')

x = np.logspace(0,2,100)
y = np.logspace(0,1,100)
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

plt.xlabel('Period [days]', fontsize=16, **hfont)
plt.ylabel(r'Planet Radius [R$_\oplus]$', fontsize=16, **hfont)
plt.xticks([1,10,100],[1,10,100],fontsize=16, fontname = "Courier New")
plt.yticks([1,2,3,4,6,10], [1,2,3,4,6,10], fontsize=16, fontname = "Courier New")
plt.tick_params(which='major', length=8)
plt.tick_params(which='minor', length=4)
plt.show()

# plt.style.use('classic')
# R_bins = np.logspace(0,0.6, 20)
# plt.hist(R, density=True, bins=R_bins, histtype='step', color='black', linewidth=2)
# plt.xscale('log')
# plt.xlim([1,6])
# plt.rcParams["font.family"] = "Courier New"
# hfont = {'fontname':'Courier New'}
# plt.xlabel(r'Planet Radius [R$_\oplus]$', fontsize=16, **hfont)
# plt.ylabel(r'dN/dlog R', fontsize=16, **hfont)
# plt.xticks([1,2,3,4,5,6],[1,2,3,4,5,6], fontsize=16, fontname = "Courier New")
# plt.yticks(fontsize=16, fontname = "Courier New")
# plt.tick_params(which='major', length=8)
# plt.tick_params(which='minor', length=4)
# plt.show()
