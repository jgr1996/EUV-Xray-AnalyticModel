from __future__ import division
import numpy as np
from numpy import log10

def Rcore_imf(Mcore, imf):

    _R1 = (0.0912*imf + 0.1603) * (log10(Mcore)*log10(Mcore))
    _R2 = (0.3330*imf + 0.7387) *  log10(Mcore)
    _R3 = (0.4639*imf + 1.1193)
    return _R1 + _R2 + _R3

def Rcore_rmf(Mcore, rmf):
    _R1 = (0.0592*rmf + 0.0975) * (log10(Mcore)*log10(Mcore))
    _R2 = (0.2337*rmf + 0.4938) *  log10(Mcore)
    _R3 = (0.3102*rmf + 0.7932)
    return _R1 + _R2 + _R3


M = 0.8
print Rcore_imf(M, 1.0)
print Rcore_imf(M, 0.0)
print Rcore_rmf(M, 0.0)
