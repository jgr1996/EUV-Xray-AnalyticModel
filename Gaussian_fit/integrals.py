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

I2_fitting_array = np.column_stack((np.log10(dR_Rc_range),np.log10(I2_array)))
I2_I1_fitting_array = np.column_stack((np.log10(dR_Rc_range),np.log10(I2_I1_array)))

np.savetxt("I2_fitting_array.csv", I2_fitting_array, delimiter=',')
np.savetxt("I2_I1_fitting_array.csv", I2_I1_fitting_array, delimiter=',')

# //////////////////////////// CREATE POLYNOMIAL FITS //////////////////////// #
def I2(log_Rc_Rp):
    log_I1 = np.log(1.2598238012325735 + np.exp(-2.4535812359864062*1.0798633578446974*log_Rc_Rp)) / -1.0798633578446974 - 4.8814961274562990e-01
    return log_I1

def I2_I1(log_Rc_Rp):
    log_I2_I1 = -7.6521840215423576e-01 / (1.0 + np.exp(-(log_Rc_Rp-5.7429970641208375e-02)/6.4338705851296174e-01))**1.8214254374336605E+00
    return log_I2_I1



# //////////////////////////////////////////////////////////////////////////// #

plt.figure(1)
plt.plot(np.log10(dR_Rc_range), np.log10(I2_array))
plt.plot(np.log10(dR_Rc_range), I2(np.log10(dR_Rc_range)))

plt.figure(2)
plt.plot(np.log10(dR_Rc_range), np.log10(I2_I1_array))
plt.plot(np.log10(dR_Rc_range), I2_I1(np.log10(dR_Rc_range)))

plt.show()
