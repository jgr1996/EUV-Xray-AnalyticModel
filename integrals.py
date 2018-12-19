from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from constants import *

# /////////////////////////////  I1  AND I2 INTEGRANDS /////////////////////// #

def I1(x):
    integrand = x * ((1/x) - 1)**(1/(gamma-1))
    return integrand

def I2(x):
    integrand = x*x * ((1/x) - 1)**(1/(gamma-1))
    return integrand

# ///////////////////  WRITE FILES WITH TABULATED INTEGRALS ////////////////// #

dR_Rc_range = np.logspace(-2, 3, 100)

I2_array = []
I2_I1_array = []

for i in dR_Rc_range:
    Rc_Rp = 1/(i+1)
    I1_i = integrate.quad(I1, Rc_Rp, 1.0)[0]
    I2_i = integrate.quad(I2, Rc_Rp, 1.0)[0]
    I2_I1_i = I2_i / I1_i

    I2_array.append(I2_i)
    I2_I1_array.append(I2_I1_i)

np.savetxt('dR_Rc_array.csv', dR_Rc_range, delimiter=',')
np.savetxt('I2.csv', I2_array, delimiter=',')
np.savetxt('I2_I1.csv', I2_I1_array, delimiter=',')

# //////////////////////////////////////////////////////////////////////////// #

# plt.loglog(dR_Rc_range, I2_array)
# plt.show()
