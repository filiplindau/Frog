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
        self.Esignal_w = None
        self.Esignal_t = None
    
    def initSignalField(self, l0, dl, l_start, l_stop, n_l, t_start, t_stop, n_t):
        """ Initiate signal field with parameters:
        l0: central wavelength
        dl: spectral width, FWHM
        l_start: starting wavelength in spectrum
        l_stop: final wavelength in spectrum
        n_l: number of points in spectrum        
        """
        c = 299792458.0
        w0 = 2*np.pi*c/l0
        dw = w0*dl/l0
        dt_fwhm = 0.441/dw
        
        l_spectrum = np.linspace(l_start, l_stop, n_l)
        w_spectrum = 2*np.pi*c/l_spectrum
        
        E_sig_w = np.exp(-4*np.log(2)*(w_spectrum-w0)**2/dw**2)
        
        t = np.linspace(t_start, t_stop, n_t)
        ph = 0.0
        self.Et = np.exp(-4*np.log(2)*t**2/dt_fwhm**2 + 1j*w0*t + ph)

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