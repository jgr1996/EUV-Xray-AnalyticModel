import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

radii = [1.55,1.36,1.08,1.0,0.95,0.77]
densities = [1.46,2.17,4.33,5.45,6.36,11.95]
def f(x, a, b):

    return a * (x)**(-b)

p_opt, p_cov = optimize.curve_fit(f, densities, radii)
print p_opt

plt.plot(densities, radii, '+', linestyle='None')
plt.plot(np.arange(0.5,12,0.01), [f(i,p_opt[0],p_opt[1]) for i in np.arange(0.5,12,0.01)])
plt.show()
