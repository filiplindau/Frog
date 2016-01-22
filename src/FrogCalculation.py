'''
Created on 15 Jan 2016

@author: Filip Lindau
'''

import numpy as np
import sys
import SimulatePulse as sp
from SimulatePulse import SimulatedPulse
from SimulatePulse import SimulatedSHGFrogTrace
import logging
import logging.handlers

root = logging.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
root.addHandler(fh)
root.setLevel(logging.INFO)
    
class FrogCalculation(object):
    def __init__(self):
        self.Esignal_w = None
        self.Esignal_t = None
    
    def initPulseField(self, l0, dl, l_start, l_stop, n_l, tau_start, tau_stop, n_tau):
        """ Initiate signal field with parameters:
        l0: central wavelength
        dl: spectral width, FWHM
        l_start: starting wavelength in spectrum
        l_stop: final wavelength in spectrum
        n_l: number of points in spectrum        
        """
        c = 299792458.0
        w0 = 2*np.pi*c/l0
        dw = w0*dl/l0       # Spectral bandwidth
        dt_fwhm = 0.441/dw  # Pulse duration
        
        self.l_vec = np.linspace(l_start, l_stop, n_l)
        w_spectrum = 2*np.pi*c/l_vec
        
        w_res = np.abs(w_spectrum[1]-w_spectrum[0])     # Spectrometer resolution
        t_span = 2*np.pi/w_res
        t_res = np.abs((tau_stop-tau_start)/n_tau)
        n_t = np.int(t_span/t_res)
        
        self.tau_vec = np.linspace(-tau_start, tau_stop, n_tau)
    
        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))
        E_sig_w = np.exp(-4*np.log(2)*(w_spectrum-w0)**2/dw**2)
        
        self.t = np.linspace(-t_span/2, t_span/2, n_t)
        ph = 0.0
        self.Et = np.exp(-4*np.log(2)*self.t**2/dt_fwhm**2 + 1j*w0*self.t + ph)
        
    def generateEsig_t_tau(self, tauVec):
        Et_mat = np.tile(self.Et, (tauVec.shape[0],1))
        Et_mat_tau = Et_mat[-1,:]       # Address each element inversely with tau stagger
        
    def generateEsig_w_tau(self):
        self.Esig_w_tau = np.fft.fft(self.Esig_t_tau, axis=0)

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
    
    frogCalc = FrogCalculation()
    frogCalc.initPulseField(800e-9, 10e-9, 700e-9, 900e-9, 200, -500e-15, 500e-15, 2000)