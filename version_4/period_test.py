from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rand
from scipy import interpolate


power1, power2, cutoff = 1.8, -0.2, 10
P_range = np.logspace(0,2,100)
pdf = []
cdf = []

for i in range(len(P_range)):

    if P_range[i] <= cutoff:
        pdf_i = P_range[i] ** power1
        pdf.append(pdf_i)
    else:
        pdf_i = (cutoff ** (power1-power2)) * (P_range[i] ** power2)
        pdf.append(pdf_i)



pdf = pdf / np.sum(pdf)
cdf = np.cumsum(pdf)

plt.figure(1)
plt.plot(P_range,pdf)
plt.xscale("log")

plt.figure(2)
plt.plot(P_range,cdf)
plt.xscale("log")

cdf, mask = np.unique(cdf, return_index=True)
P_mask = P_range[mask]

random_numbers = rand.uniform(low=min(cdf), high=max(cdf), size=10000)
f = interpolate.interp1d(cdf, P_range)
draw = f(random_numbers)


plt.figure(3)
plt.hist(draw, bins=np.logspace(0,2,50), histtype='step', color='black', linewidth=2)
plt.xscale("log")

plt.show()
