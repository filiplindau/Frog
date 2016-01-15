'''
Created on 14 Jan 2016

@author: Filip Lindau
'''

import numpy as np
from scipy.interpolate import interp1d

class SimulatedPulse(object):
    def __init__(self, Nt = 2048, tau = 50e-15, l0 = 800e-9, tspan=1e-12):
        self.Et = np.zeros(Nt, dtype=np.complex)
        self.Nt = Nt
        self.tau = tau
        self.l0 = l0
        self.tspan = tspan
        
        self.generateGaussian()
        
    def generateGaussian(self):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.Nt)
        w0 = 2*np.pi*299792458.0/self.l0
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2 + 1j*w0*t + ph)
        self.setEt(Eenv, t)
        
    def generateGaussianQuadraticPhase(self, b = 0):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.Nt)
        w0 = 2*np.pi*299792458.0/self.l0
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2 + 1j*(w0*t + b*t**2) + ph)
        self.setEt(Eenv, t)

    def generateGaussianCubicPhase(self, b = 0, c = 0):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.Nt)
        w0 = 2*np.pi*299792458.0/self.l0
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2 + 1j*(w0*t + b*t**2 + c*t**3) + ph)
        self.setEt(Eenv, t)
        
    def getFreqSpectrum(self, Nw = 2048):
        Ew = np.fft.fft(self.Et, Nw)        
#        w0 = 0*2*np.pi*299792458.0/self.l0
        w = 2*np.pi*np.fft.fftfreq(Nw, d = self.tspan/self.Nt)
        return w, Ew
    
    def getSpectrogram(self, Nl = 2048):
        w, Ew = self.getFreqSpectrum(Nl)
        l = 2*np.pi*299792458.0/w
        Il = Ew*Ew.conjugate()
        return l, Il
    
    def setEt(self, Et, t = None):
        if t != None:
            self.t = t
        self.Et = Et
        self.Nt = Et.shape[0]
        self.Et_int = interp1d(self.t, self.Et, kind='linear', bounds_error=False, fill_value=0.0)
        
    def getInterpolatedEt(self, ti):
        Eti = self.Et_int(ti)
        return Eti
    
class SimulatedSHGFrogTrace(object):
    def __init__(self, Nt = 2048, tau = 50e-15, l0 = 800e-9, tspan=1e-12):
        self.pulse = SimulatedPulse(Nt, tau, l0, tspan)
        
    def generateSHGTrace(self, tspan, Nt, l0, lspan, Nl):
        signalPulse = SimulatedPulse()
        tauVec = np.linspace(-tspan/2.0, tspan/2.0, Nt)
        t = self.pulse.t
        lVec = l0 + np.linspace(-lspan/2.0, lspan/2.0, Nl)
        Et = self.pulse.getInterpolatedEt(t)
        Ifrog = []
        
        for tau in tauVec:
            signalPulse.setEt(Et*self.pulse.getInterpolatedEt(t-tau), t)
            l, Il = signalPulse.getSpectrogram(t.shape[0])
            try:
                Il_int = interp1d(l, Il, kind='linear', bounds_error=False, fill_value=0.0)
                Ifrog.append(Il_int(lVec))
            except Exception, e:
                print "Error for tau=", tau, ": ", str(e)
                Il_int = np.zeros(Nl)
                Ifrog.append(Il_int)
            
            
        return np.array(Ifrog).real
    
if __name__ == '__main__':
    p = SimulatedPulse(2048, tau = 50e-15, l0 = 800e-9, tspan=0.5e-12)
    l, Il = p.getSpectrogram(2048)
    
    tspan = 0.2e-12
    Nt = 128
    l0 = 380e-9
    lspan = 60e-9
    Nl = 100
    frog = SimulatedSHGFrogTrace(16384, tau = 50e-15, l0 = 800e-9, tspan=1e-12)
    frog.pulse.generateGaussianCubicPhase(0.001e30, 0.00002e45)
    Ifrog = frog.generateSHGTrace(tspan, Nt, l0, lspan, Nl)
    tauVec = np.linspace(-tspan/2.0, tspan/2.0, Nt)
    lVec = l0 + np.linspace(-lspan/2.0, lspan/2.0, Nl)
    X, Y = np.meshgrid(lVec, tauVec)