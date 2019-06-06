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
x_range = np.arange(0.0, 1.0001, 0.0001)

# plt.figure(1)
# for i in range(order+1):
#     coefficients = np.zeros(order+1)
#     coefficients[i] = 1.0
#     plt.plot(x_range, [B(j, order, coefficients) for j in x_range], label='i = {}'.format(i))
# plt.legend()
M_coeffs = [0.05, 0.25, 0.2, 0.3, 0.8, 1.0]
M_min, M_max = 0.5, 10.0
M_poly_min, M_poly_max = B(0, order, M_coeffs), B(1, order, M_coeffs)
M_poly_norm = M_poly_max - M_poly_min
M_norm = M_max - M_min

X_coeffs = [0.1, 0.2, 0.4, 0.3, 1.0, 1.36]
X_min, X_max = -4.0, 0.0
X_poly_min, X_poly_max = B(0, order, X_coeffs), B(1, order, X_coeffs)
X_poly_norm = X_poly_max - X_poly_min
X_norm = X_max - X_min


plt.style.use('classic')
plt.rcParams["font.family"] = "Courier New"
hfont = {'fontname':'Courier New'}

plt.figure(1)
plt.plot(x_range, [(((M_norm/M_poly_norm) * ((B(i, order, M_coeffs)) - M_poly_min)) + M_min) for i in x_range], linewidth=3)
plt.ylabel(r"CDF$_M$", fontsize=18, **hfont)
plt.xticks(fontsize=16, fontname = "Courier New")
plt.yticks(fontsize=16, fontname = "Courier New")


plt.figure(2)
N_range = np.arange(12000)
measurements = []
for i in range(len(N_range)):
    U = rand.uniform()
    measurements.append(((M_norm/M_poly_norm) * ((B(U, order, M_coeffs)) - M_poly_min)) + M_min)
plt.hist(measurements, bins=50, histtype='step', linewidth=3)
plt.ylabel(r"PDF$_M$", fontsize=18, **hfont)
plt.xlabel(r"M", fontsize=18, **hfont)
plt.xticks(fontsize=16, fontname = "Courier New")
plt.yticks(fontsize=16, fontname = "Courier New")




# plt.figure(3)
# plt.plot(x_range, [(((X_norm/X_poly_norm) * ((B(i, order, X_coeffs)) - X_poly_min)) + X_min) for i in x_range], linewidth=3)
# plt.ylabel(r"log( CDF$_X$ )", fontsize=18, **hfont)
# plt.xticks(fontsize=16, fontname = "Courier New")
# plt.yticks(fontsize=16, fontname = "Courier New")
#
#
# plt.figure(4)
# N_range = np.arange(12000)
# measurements = []
# for i in range(len(N_range)):
#     U = rand.uniform()
#     measurements.append(10**(((X_norm/X_poly_norm) * ((B(U, order, X_coeffs)) - X_poly_min)) + X_min))
# plt.hist(measurements, bins=np.logspace(-4,0,50), histtype='step', linewidth=3)
# plt.xscale('log')
# plt.ylabel(r"PDF$_X$", fontsize=18, **hfont)
# plt.xlabel(r"X", fontsize=18, **hfont)
# plt.xticks(fontsize=16, fontname = "Courier New")
# plt.yticks(fontsize=16, fontname = "Courier New")

plt.show()
