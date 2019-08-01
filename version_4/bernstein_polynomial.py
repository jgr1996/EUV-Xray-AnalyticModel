from __future__ import division
import numpy as np
from scipy import special
from numpy import random as rand
import matplotlib.pyplot as plt
from scipy import interpolate
import seaborn as sns



def B(x, order, coefficients):

    coefficients = np.array(coefficients)
    poly_array = np.array([special.binom(order, i)*(x**i)*((1-x)**(order-i)) for i in range(order+1)])
    B = np.dot(coefficients, poly_array)

    return B

order = 5
x_range = np.arange(0.0, 1.0001, 0.0001)

# coeffs = [0.01,0.2,0.3,0.6,0.8,1.0,0.01,0.2,0.3,0.6,0.8,1.0,4.0]
coeffs = [0.0,4.91,0.88,5.85,2.04,10.0,0.0,1.21,1.0,1.95,8.0,10.0,4.11]

# coeffs = [0.0,4.91,0.88,5.85,2.04,10.0,0.00,0.45,0.63,0.77,0.89,1.00,4.0]
# coeffs = [0.01,0.21,0.54,0.66,0.9,1.0,0.0,0.0,0.18,0.71,0.69,0.74,8.95]
print coeffs[0:6]
print coeffs[6:12]

M_coeffs = coeffs[6:12]
M_min, M_max = 0.75, 10.0
M_poly_min, M_poly_max = B(0, order, M_coeffs), B(1, order, M_coeffs)
M_poly_norm = M_poly_max - M_poly_min
M_norm = M_max - M_min

X_coeffs = coeffs[0:6]
X_min, X_max = -3.0, 0.0
X_poly_min, X_poly_max = B(0, order, X_coeffs), B(1, order, X_coeffs)
X_poly_norm = X_poly_max - X_poly_min
X_norm = X_max - X_min


plt.style.use('classic')
plt.rcParams["font.family"] = "Courier New"
hfont = {'fontname':'Courier New'}

# plt.figure(1)
# plt.plot(x_range, [(((M_norm/M_poly_norm) * ((B(i, order, M_coeffs)) - M_poly_min)) + M_min) for i in x_range], linewidth=3)
# plt.ylabel(r"CDF$_M$", fontsize=18, **hfont)
# plt.xticks(fontsize=16, fontname = "Courier New")
# plt.yticks(fontsize=16, fontname = "Courier New")


plt.figure(2)
for i in range(1):
    N_range = np.arange(10000)
    measurements = []
    for i in range(len(N_range)):
        U = rand.uniform()
        measurements.append(((M_norm/M_poly_norm) * ((B(U, order, M_coeffs)) - M_poly_min)) + M_min)
    sns.distplot(measurements, hist = False, kde = True, kde_kws={'color':'black', 'linewidth':4, 'alpha':1.0})

# np.savetxt("bernstein_mass_function.txt", hist, delimiter='')
# np.savetxt("bernstein_mass_function_bins.txt", bins, delimiter='')
plt.ylabel(r"Normalised PDF", fontsize=20, **hfont)
plt.xlabel(r"Core Mass [M$_\oplus$]", fontsize=20, **hfont)
plt.xticks(fontsize=18, fontname = "Courier New")
plt.yticks(fontsize=18, fontname = "Courier New")
plt.xlim([0.8,10.0])
plt.savefig("../Figures/realizations/Bernstein_1/TESS_talk_1.pdf", dpi=1000)




# plt.figure(3)
# plt.plot(x_range, [(((X_norm/X_poly_norm) * ((B(i, order, X_coeffs)) - X_poly_min)) + X_min) for i in x_range], linewidth=3)
# plt.ylabel(r"log( CDF$_X$ )", fontsize=18, **hfont)
# plt.xticks(fontsize=16, fontname = "Courier New")
# plt.yticks(fontsize=16, fontname = "Courier New")


# plt.figure(4)
# N_range = np.arange(12000)
# measurements = []
# for i in range(len(N_range)):
#     U = rand.uniform()
#     measurements.append(10**(((X_norm/X_poly_norm) * ((B(U, order, X_coeffs)) - X_poly_min)) + X_min))
# plt.hist(measurements, bins=np.logspace(-4,1,50), histtype='step', linewidth=3)
# plt.xscale('log')
# plt.ylabel(r"PDF$_X$", fontsize=18, **hfont)
# plt.xlabel(r"X", fontsize=18, **hfont)
# plt.xticks(fontsize=16, fontname = "Courier New")
# plt.yticks(fontsize=16, fontname = "Courier New")

plt.show()
