'''
Created on 3 Feb 2016

@author: Filip Lindau

Calculate FROG with carrier frequency removed
'''

import numpy as np

class SimulatedPulse(object):
    def __init__(self, N = 128, dt = 10e-15, l0 = 800e-9, tau = 50e-15):
        self.Et = np.zeros(N, dtype=np.complex)
        self.N = N
        self.tau = tau
        self.l0 = l0
        self.tspan = N*dt
        self.dt = dt
        
        self.generateGaussian(tau)
        
    def generateGaussian(self, tau):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        w0 = 2*np.pi*299792458.0/self.l0
        self.w0 = w0
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2 + ph)
        self.setEt(Eenv, t)
        
    def generateGaussianQuadraticPhase(self, b = 0):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        w0 = 2*np.pi*299792458.0/self.l0
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2 + 1j*(b*t**2) + ph)
        self.setEt(Eenv, t)

    def generateGaussianCubicPhase(self, b = 0, c = 0):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        w0 = 2*np.pi*299792458.0/self.l0
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2 + 1j*(b*t**2 + c*t**3) + ph)
        self.setEt(Eenv, t)
        
    def getFreqSpectrum(self, Nw = None):
        if Nw is None:
            Nw = self.t.shape[0]
        Ew = np.fft.fft(self.Et, Nw)        
#        w0 = 0*2*np.pi*299792458.0/self.l0
        w = self.w0 + 2*np.pi*np.fft.fftfreq(Nw, d = self.dt)
        return w, Ew

    def getSpectrogram(self, Nl = None):
        if Nl is None:
            Nl = self.t.shape[0]
        w, Ew = self.getFreqSpectrum(Nl)
        l = 2*np.pi*299792458.0/w
        Il = Ew*Ew.conjugate()
#        Il = Il/max(Il)
        return l, Il

    
    def setEt(self, Et, t = None):
        if t is not None:
            self.t = t
            self.tspan = np.abs(t.max()-t.min())
            self.N = t.shape[0]
            self.dt = self.tspan/self.N
        self.Et = Et
        
    def getShiftedEt(self, shift):
        Ets = np.zeros_like(self.Et)
        if shift < 0:
            Ets[0:self.N+shift] = self.Et[-shift:]
        else:
            Ets[shift:] = self.Et[0:self.N-shift]
        return Ets

class SimulatedSHGFrogTrace(object):
    def __init__(self, N = 128, dt = 10e-15, l0 = 800e-9, tau = 50e-15):
        self.pulse = SimulatedPulse(N, dt, l0, tau)
        self.l_vec = None
        self.tau_vec = None
        
    def generateSHGTraceDt(self, N, dt, l0):
        signalPulse = SimulatedPulse(self.pulse.N, self.pulse.dt, self.pulse.l0/2, self.pulse.tau)
        tspan = N*dt
        self.tau_vec = np.linspace(-tspan/2.0, tspan/2.0, N)
        t = self.pulse.t
        Et = self.pulse.Et
        Ifrog = []
        
        signalPulse.setEt(Et*Et, t)
        l, Il = signalPulse.getSpectrogram(t.shape[0])
        l_shift = l0 - (l.max()+l.min())/2
        nl_shift = np.int(l_shift/np.abs(l[1]-l[0]))
        self.l_vec = np.fft.fftshift(l + l_shift)
        
        shiftVec = np.arange(N) - N/2
        
        for sh in shiftVec:
            Ils = np.zeros_like(Il)
            signalPulse.setEt(Et*self.pulse.getShiftedEt(sh), t)
            l, Iln = signalPulse.getSpectrogram(t.shape[0])
            if nl_shift < 0:
                Ils[0:N+nl_shift] = np.array(np.fft.fftshift(Iln))[-nl_shift:]
            else:
                Ils[nl_shift:] = np.array(np.fft.fftshift(Iln))[0:N-nl_shift]
            Ifrog.append(Ils)
            
        Ifrog = np.array(Ifrog).real
        self.Ifrog = Ifrog/Ifrog.max()
        
        return self.Ifrog
    
    def getWavelengths(self):
        return self.l_vec
    
    def getTimedelays(self):
        return self.tau_vec
    
if __name__ == '__main__':
    N = 128
    dt = 8e-15
    tau = 100e-15 / (2*np.sqrt(np.log(2)))
    l0 = 800e-9
    p = SimulatedPulse(N, dt, l0, tau)
    p.generateGaussianCubicPhase(5e24, 1e40)
    gt = SimulatedSHGFrogTrace()
    gt.pulse = p
    Ifrog = gt.generateSHGTraceDt(N, dt, 390e-9)
    l = gt.getWavelengths()
    t = gt.getTimedelays()