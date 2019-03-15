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


coefficients = [0.0, 0.2, 0.4, 0.4, 0.2, 1.0]
normalisation = 12 / B(1.0, order, coefficients)

plt.figure(2)
plt.plot(x_range, [normalisation*B(i, order, coefficients) for i in x_range])

plt.figure(3)
N_range = np.arange(12000)
measurements = []

for i in range(len(N_range)):
    U = rand.uniform()
    measurements.append(normalisation*B(U, order, coefficients))

plt.hist(measurements, bins=100)
plt.show()
