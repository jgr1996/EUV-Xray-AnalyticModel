from __future__ import division
import numpy as np
from scipy import special
from numpy import random as rand
import matplotlib.pyplot as plt



def B(x, order, coefficients):

    coefficients = np.array(coefficients)
    poly_array = np.array([special.binom(order, i)*(x**i)*((1-x)**(order-i)) for i in range(order+1)])
    B = np.dot(coefficients, poly_array)

    return B

order = 5
x_range = np.arange(0.0, 1.01, 0.01)

# plt.figure(1)
# for i in range(order+1):
#     coefficients = np.zeros(order+1)
#     coefficients[i] = 1.0
#     plt.plot(x_range, [B(j, order, coefficients) for j in x_range], label='i = {}'.format(i))
# plt.legend()

M_coeffs = [0.04, 0.09, 0.41, 0.55, 0.40, 1.36]
M_norm = 12 / B(1.0, order, M_coeffs)

X_coeffs = [0.03, 0.21, 0.47, 0.25, 0.41, 0.95]
X_norm = 0.4 / B(1.0, order, X_coeffs)


plt.style.use('classic')
plt.rcParams["font.family"] = "Courier New"
hfont = {'fontname':'Courier New'}

plt.figure(1)
plt.plot(x_range, [M_norm*B(i, order, M_coeffs) for i in x_range], linewidth=3)
plt.ylabel(r"CDF$_M$", fontsize=18, **hfont)
plt.xticks(fontsize=16, fontname = "Courier New")
plt.yticks(fontsize=16, fontname = "Courier New")


plt.figure(2)
N_range = np.arange(12000)
measurements = []
for i in range(len(N_range)):
    U = rand.uniform()
    measurements.append(M_norm*B(U, order, M_coeffs))
plt.hist(measurements, bins=50, histtype='step', linewidth=3)
plt.ylabel(r"PDF$_M$", fontsize=18, **hfont)
plt.xlabel(r"M", fontsize=18, **hfont)
plt.xticks(fontsize=16, fontname = "Courier New")
plt.yticks(fontsize=16, fontname = "Courier New")




plt.figure(3)
plt.plot(x_range, [X_norm*B(i, order, X_coeffs) for i in x_range], linewidth=3)
plt.ylabel(r"CDF$_X$", fontsize=18, **hfont)
plt.xticks(fontsize=16, fontname = "Courier New")
plt.yticks(fontsize=16, fontname = "Courier New")


plt.figure(4)
N_range = np.arange(12000)
measurements = []
for i in range(len(N_range)):
    U = rand.uniform()
    measurements.append(X_norm*B(U, order, X_coeffs))
plt.hist(measurements, bins=50, histtype='step', linewidth=3)
plt.ylabel(r"PDF$_X$", fontsize=18, **hfont)
plt.xlabel(r"X", fontsize=18, **hfont)
plt.xticks(fontsize=16, fontname = "Courier New")
plt.yticks(fontsize=16, fontname = "Courier New")

plt.show()
