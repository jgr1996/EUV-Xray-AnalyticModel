from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from constants import *

"""
See Owen & Wu (2017) for references to these integrals (in particular the appendix).
All variables follow the system used in the paper.
"""

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

# //////////////////////////// CREATE POLYNOMIAL FITS //////////////////////// #

I2_fit = np.polyfit(dR_Rc_range, I2_array, 100)

I2_I1_fit = np.polyfit(dR_Rc_range, I2_I1_array, 100)

# //////////////////////////////////////////////////////////////////////////// #

# plt.figure(1)
# plt.loglog(dR_Rc_range, I2_array)
# plt.loglog(dR_Rc_range, np.polyval(I2_fit,dR_Rc_range))
#
# plt.figure(2)
# plt.loglog(dR_Rc_range, [1/i for i in I2_I1_array])
#
# plt.show()
