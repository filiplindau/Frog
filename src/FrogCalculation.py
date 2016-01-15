'''
Created on 15 Jan 2016

@author: Filip Lindau
'''

import numpy as np
import sys
import SimulatePulse as sp
from SimulatePulse import SimulatedPulse
from SimulatePulse import SimulatedSHGFrogTrace

class FrogCalculation(object):
    def __init__(self):
        pass

if __name__ == '__main__':
    tspan = 0.2e-12
    Nt = 128
    l0 = 380e-9
    lspan = 60e-9
    Nl = 100
    frog = sp.SimulatedSHGFrogTrace(16384, tau = 50e-15, l0 = 800e-9, tspan=1e-12)
    frog.pulse.generateGaussianCubicPhase(0.001e30, 0.00002e45)
    Ifrog = frog.generateSHGTrace(tspan, Nt, l0, lspan, Nl)
    tauVec = np.linspace(-tspan/2.0, tspan/2.0, Nt)
    lVec = l0 + np.linspace(-lspan/2.0, lspan/2.0, Nl)
    X, Y = np.meshgrid(lVec, tauVec)