'''
Created on 15 Jan 2016

@author: Filip Lindau
'''

import numpy as np
from scipy.interpolate import interp2d
import scipy.interpolate as si
import sys
import time
import SimulatePulse as sp
reload(sp)
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
root.setLevel(logging.DEBUG)
    
class FrogCalculation(object):
    def __init__(self):
        self.Esignal_w = None
        self.Esignal_t = None
    
    def initPulseFieldOld(self, l0, dl, l_start, l_stop, n_l, tau_start, tau_stop, n_tau):
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
        w_spectrum = 2*np.pi*c/self.l_vec
        
        w_res = np.abs(w_spectrum[1]-w_spectrum[0])     # Spectrometer resolution
        t_span = 2*np.pi/w_res
        t_res = np.abs((tau_stop-tau_start)/n_tau)
        f_max = (w_spectrum[-1]-w_spectrum[0])/(2*np.pi)
        t_res = 1/(2*f_max)
        n_t = np.int(t_span/t_res)+1
        
        self.tau_vec = np.linspace(tau_start, tau_stop, n_tau)
    
        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))
        E_sig_w = np.exp(-4*np.log(2)*(w_spectrum-w0)**2/dw**2)
        
        self.t = np.linspace(-t_span/2, t_span/2, n_t)
        ph = 0.0
        self.Et = np.exp(-4*np.log(2)*self.t**2/dt_fwhm**2 + 1j*w0*self.t + ph)

    def initPulseFieldOld2(self, l0, dl, l_start, l_stop, n_l, tau_start, tau_stop, n_tau):
        """ Initiate signal field with parameters:
        l0: central wavelength of the pulse
        dl: spectral width of the pulse, FWHM
        l_start: starting wavelength in spectrum
        l_stop: final wavelength in spectrum
        n_l: number of points in spectrum
        tau_start: starting time delay
        tau_stop: final time delay        
        
        Creates the following variables:
        self.l_vec
        self.w
        self.dw
        self.t
        self.dt
        self.tau
        self.Et        
        """
        c = 299792458.0
        w0 = 2*np.pi*c/l0
        dw = w0*dl/l0       # Pulse spectral bandwidth
        dt_fwhm = 0.441/dw  # Pulse duration
        
        # Start by converting the wavelengths to a frequency scale
        self.l_vec = np.linspace(l_start, l_stop, n_l)
        w_range = np.array([2*np.pi*c/l_start, 2*np.pi*c/l_stop])
        n_w = n_l       # Frequency resolution is not linear for the spectrometer, but we neglect that here
        w_res = (w_range.max()-w_range.min())/n_w     # Frequency resolution resolution
        w_spectrum = np.arange(w_range.min(), w_range.max(), w_res)
        self.dw = w_res 
        self.w = w_spectrum

        # Now we calculate the time span supported by the
        # frequency resolution
        t_span = 2*np.pi/w_res
        
        # Time resolution is given by the frequency span
        f_max = np.abs(w_range[1]-w_range[0])/(2*np.pi)/2
        t_res = 1/(2*f_max)
        n_t = np.int(t_span/t_res)        
        # Create time vector
        self.dt = t_res
        self.t = np.linspace(-t_span/2, t_span/2, n_t)
        self.t = np.arange(-t_span/2, t_span/2, t_res)
        
        # Vector of delay positions
        # We also store the index of the tau_start and tau_stop 
        tau_start_t = self.t[np.argmin(np.abs(self.t-tau_start))]
        tau_stop_t = self.t[np.argmin(np.abs(self.t-tau_stop))]
#        self.tau_start_ind = np.argmin(np.abs(self.t-tau_start))
#        self.tau_stop_ind = np.argmin(np.abs(self.t-tau_stop))
        self.tau_start_ind = np.int(np.round(tau_start_t/self.dt))
        self.tau_stop_ind = np.int(np.round(tau_stop_t/self.dt))
        self.tau = self.t[np.argmin(np.abs(self.t-tau_start)) : np.argmin(np.abs(self.t-tau_stop))]
    
        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))
        E_w = np.exp(-4*np.log(2)*(w_spectrum-w0)**2/dw**2)
        
        # Finally calculate a gaussian E-field from the 
        ph = 0.0
        self.Et = np.exp(-4*np.log(2)*self.t**2/dt_fwhm**2 + 1j*w0*self.t + ph)

    def initPulseField(self, l0, dl, t_res, t_span, tau_start, tau_stop):
        """ Initiate signal field with parameters:
        l0: central wavelength of the pulse
        dl: spectral width of the pulse, FWHM
        t_res: time resolution of the reconstruction
        t_span: time span covered by the reconstruction        
        tau_start: starting time delay
        tau_stop: final time delay        
        
        Creates the following variables:
        self.w
        self.dw
        self.t
        self.dt
        self.tau
        self.Et        
        """
        # Setting up the initial pulse
        c = 299792458.0
        w0 = 2*np.pi*c/l0
        dw = w0*dl/l0       # Pulse spectral bandwidth
        dt_fwhm = 2*np.pi*0.441/dw  # Pulse duration
        
        N = np.int(t_span/t_res)
        
        # Now we calculate the frequency resolution required by the
        # time span
        w_res = 2*np.pi/t_span

        # Frequency span is given by the time resolution          
        f_max = 1/(2*t_res)
        w_span = f_max*2*2*np.pi
#        w_spectrum = np.linspace(w0-w_span/2, w0+w_span/2, n_t)
        w_spectrum = np.arange(-w_span/2, -w_span/2+w_res*N, w_res)
        self.dw = w_res 
        self.w = w_spectrum
        
                
        # Create time vector
        self.dt = t_res
        self.t = np.arange(-t_span/2, -t_span/2+N*t_res, t_res)
        
        # Vector of delay positions
        # We also store the index of the tau_start and tau_stop
        # snapped to available time values 
        tau_start_t = self.t[np.argmin(np.abs(self.t-tau_start))]
        tau_stop_t = self.t[np.argmin(np.abs(self.t-tau_stop))]
        n_tau = (tau_stop_t-tau_start_t)/t_res
#        self.tau_start_ind = np.argmin(np.abs(self.t-tau_start))
#        self.tau_stop_ind = np.argmin(np.abs(self.t-tau_stop))
        self.tau_start_ind = np.int(np.round(tau_start_t/self.dt))
        self.tau_stop_ind = np.int(np.round(tau_stop_t/self.dt))
        self.tau_stop_ind = np.int(self.tau_start_ind + n_tau)
        self.tau = self.t[np.argmin(np.abs(self.t-tau_start)) : np.argmin(np.abs(self.t-tau_start))+n_tau]
    
        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))
        E_w = np.exp(-4*np.log(2)*(w_spectrum-w0)**2/dw**2)
        
        # Finally calculate a gaussian E-field from the 
        ph = 0.0
        self.Et = np.exp(-4*np.log(2)*self.t**2/dt_fwhm**2 + 1j*w0*self.t + ph)
        root.info('Finished')

    def initPulseFieldRandom(self, t_res, t_span, tau_start, tau_stop):
        """ Initiate signal field with parameters:
        t_res: time resolution of the reconstruction
        t_span: time span covered by the reconstruction
        tau_start: starting time delay
        tau_stop: final time delay        
        
        Creates the following variables:
        self.w
        self.dw
        self.t
        self.dt
        self.tau
        self.Et        
        """        
        N = np.int(t_span/t_res)
        
        # Now we calculate the frequency resolution required by the
        # time span
        w_res = 2*np.pi/t_span

        # Frequency span is given by the time resolution          
        f_max = 1/(2*t_res)
        w_span = f_max*2*2*np.pi
#        w_spectrum = np.linspace(w0-w_span/2, w0+w_span/2, n_t)
        w_spectrum = np.arange(-w_span/2, -w_span/2+w_res*N, w_res)
        self.dw = w_res 
        self.w = w_spectrum
        
                
        # Create time vector
        self.dt = t_res
        self.t = np.arange(-t_span/2, -t_span/2+N*t_res, t_res)
        
        # Vector of delay positions
        # We also store the index of the tau_start and tau_stop
        # snapped to available time values 
        tau_start_t = self.t[np.argmin(np.abs(self.t-tau_start))]
        tau_stop_t = self.t[np.argmin(np.abs(self.t-tau_stop))]
        n_tau = (tau_stop_t-tau_start_t)/t_res
#        self.tau_start_ind = np.argmin(np.abs(self.t-tau_start))
#        self.tau_stop_ind = np.argmin(np.abs(self.t-tau_stop))
        self.tau_start_ind = np.int(np.round(tau_start_t/self.dt))
        self.tau_stop_ind = np.int(np.round(tau_stop_t/self.dt))
        self.tau_stop_ind = np.int(self.tau_start_ind + n_tau)
        self.tau = self.t[np.argmin(np.abs(self.t-tau_start)) : np.argmin(np.abs(self.t-tau_start))+n_tau]
    
        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))
        
        # Finally calculate a gaussian E-field from the 
        self.Et = np.exp(1j*np.pi/2*np.random.randn(self.t.shape[0]))
        root.info('Finished')
                        
    def initPulseFieldPerfect(self, t_res, t_span, tau_start, tau_stop):
        """ Initiate signal field with parameters:
        t_res: time resolution of the reconstruction
        t_span: time span covered by the reconstruction
        tau_start: starting time delay
        tau_stop: final time delay        
        
        Creates the following variables:
        self.w
        self.dw
        self.t
        self.dt
        self.tau
        self.Et        
        """        
        N = np.int(t_span/t_res)
        
        # Now we calculate the frequency resolution required by the
        # time span
        w_res = 2*np.pi/t_span

        # Frequency span is given by the time resolution          
        f_max = 1/(2*t_res)
        w_span = f_max*2*2*np.pi
#        w_spectrum = np.linspace(w0-w_span/2, w0+w_span/2, n_t)
        w_spectrum = np.arange(-w_span/2, -w_span/2+w_res*N, w_res)
        self.dw = w_res 
        self.w = w_spectrum
        
                
        # Create time vector
        self.dt = t_res
        self.t = np.arange(-t_span/2, -t_span/2+N*t_res, t_res)
        
        p = sp.SimulatedSHGFrogTrace(10000, tau = 100e-15, l0 = 800e-9, tspan=t_span)
        p.pulse.generateGaussian()
        tspan_frog = t_span
        Nt = N
        l0 = 390e-9
        lspan = 80e-9
        Nl = N
        Ifrog = p.generateSHGTrace(tspan_frog, Nt, l0, lspan, Nl)
        
        # Vector of delay positions
        # We also store the index of the tau_start and tau_stop
        # snapped to available time values 
        tau_start_t = self.t[np.argmin(np.abs(self.t-tau_start))]
        tau_stop_t = self.t[np.argmin(np.abs(self.t-tau_stop))]
        n_tau = (tau_stop_t-tau_start_t)/t_res
#        self.tau_start_ind = np.argmin(np.abs(self.t-tau_start))
#        self.tau_stop_ind = np.argmin(np.abs(self.t-tau_stop))
        self.tau_start_ind = np.int(np.round(tau_start_t/self.dt))
        self.tau_stop_ind = np.int(np.round(tau_stop_t/self.dt))
        self.tau_stop_ind = np.int(self.tau_start_ind + n_tau)
        self.tau = self.t[np.argmin(np.abs(self.t-tau_start)) : np.argmin(np.abs(self.t-tau_start))+n_tau]
    
        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))
        
        # Finally calculate a gaussian E-field from the 
        self.Et = p.pulse.Et
        self.t = p.pulse.t

        tauVec = np.linspace(-tspan_frog/2.0, tspan_frog/2.0, Nt)
        lVec = l0 + np.linspace(-lspan/2.0, lspan/2.0, Nl)
#        self.conditionFrogTrace(Ifrog, lVec[0], lVec[-1], tauVec[0], tauVec[-1])
        self.generateEsig_t_tau_SHG()
        self.I_w_tau = np.abs(np.fft.fft(self.Esig_t_tau, axis = 1))**2
        
        self.p = p
            
        root.info('Finished')
                             
    def initPulseFieldPerfectSmall(self, t_res, N):
        """ Initiate signal field with parameters:
        t_res: time resolution of the reconstruction
        t_span: time span covered by the reconstruction
        tau_start: starting time delay
        tau_stop: final time delay        
        
        Creates the following variables:
        self.w
        self.dw
        self.t
        self.dt
        self.tau
        self.Et        
        """        
        t_span = N*t_res
        
        # Now we calculate the frequency resolution required by the
        # time span
        w_res = 2*np.pi/t_span

        # Frequency span is given by the time resolution          
        f_max = 1/(2*t_res)
        w_span = f_max*2*2*np.pi
#        w_spectrum = np.linspace(w0-w_span/2, w0+w_span/2, n_t)
        w_spectrum = np.arange(-w_span/2, -w_span/2+w_res*N, w_res)
        self.dw = w_res 
        self.w = w_spectrum
        
                
        # Create time vector
        self.dt = t_res
        self.t = np.arange(-t_span/2, -t_span/2+N*t_res, t_res)
        
        p = sp.SimulatedSHGFrogTrace(N, tau = 100e-15, l0 = 800e-9, tspan=t_span)
        p.pulse.generateGaussian()
        tspan_frog = t_span
        Nt = N
        l0 = 390e-9
        lspan = 80e-9
        Nl = N
        Ifrog = p.generateSHGTraceSmall(tspan_frog, Nt, l0, lspan, Nl)
        
        # Vector of delay positions
        # We also store the index of the tau_start and tau_stop
        # snapped to available time values 
        tau_start_t = self.t[0]
        tau_stop_t = self.t[-1]
        n_tau = N
#        self.tau_start_ind = np.argmin(np.abs(self.t-tau_start))
#        self.tau_stop_ind = np.argmin(np.abs(self.t-tau_stop))
        self.tau_start_ind = 0
        self.tau_stop_ind = N-1
        self.tau = self.t
    
        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))
        
        # Finally calculate a gaussian E-field from the 
        self.Et = p.pulse.Et
        self.t = p.pulse.t

        tauVec = np.linspace(-tspan_frog/2.0, tspan_frog/2.0, Nt)
        lVec = l0 + np.linspace(-lspan/2.0, lspan/2.0, Nl)
#        self.conditionFrogTrace(Ifrog, lVec[0], lVec[-1], tauVec[0], tauVec[-1])
        self.generateEsig_t_tau_SHG()
        self.I_w_tau = np.abs(np.fft.fft(self.Esig_t_tau, axis = 1))**2
        
        self.p = p
            
        root.info('Finished')
        
    def generateEsig_t_tau_SHG(self):
        ''' Generate the time shifted E-field matrix for the SHG process.
        
        Output: 
        self.Esig_t_tau, a n_tau x n_t matrix where each row is Esig(t,tau)
        '''      
        root.debug('Generating new Esig_t_tau from SHG')  
        t0 = time.clock()

        Et_mat = np.tile(self.Et, (self.tau.shape[0],1))    # Repeat Et into a matrix
        Et_mat_tau = np.zeros_like(Et_mat)
        n_e = self.Et.shape[0]
        for ind, offs in enumerate(range(self.tau_start_ind, self.tau_stop_ind)):
#            root.info(''.join(('Index ', str(ind), ', offset ', str(offs))))
            if offs > 0:
                Et_mat_tau[ind, offs:] = self.Et[0:n_e-offs]
            else:
                # Offs is negative!
                Et_mat_tau[ind, 0:n_e+offs] = self.Et[-offs:]
        self.Esig_t_tau = Et_mat*Et_mat_tau
        self.Et_mat = Et_mat
        self.Et_mat_tau = Et_mat_tau
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))
        
    def generateEsig_w_tau(self):
        root.debug('Generating new Esig_w_tau with fft')
        t0 = time.clock()
        self.Esig_w_tau = np.fft.fft(self.Esig_t_tau, axis=1)
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))

    def applyIntesityData(self, I_w_tau=None):
        root.debug('Applying intensity data from experiment')
        t0 = time.clock()
        if I_w_tau==None:
            I_w_tau = self.I_w_tau
        eps = 0.01
        Esig_mag = np.abs(self.Esig_w_tau)
        self.Esig_w_tau_p = np.zeros_like(self.Esig_w_tau)
        good_ind = np.where(Esig_mag > eps)
        self.Esig_w_tau_p[good_ind[0], good_ind[1]] = np.sqrt(I_w_tau[good_ind[0], good_ind[1]])*self.Esig_w_tau[good_ind[0], good_ind[1]]/Esig_mag[good_ind[0], good_ind[1]]
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))
        
    def updateEt_vanilla(self):
        root.debug('Updating Et using vanilla algorithm')
        t0 = time.clock()
        Esig_t_tau_p = np.fft.ifft(self.Esig_w_tau_p, axis=1)
#        self.Et = np.trapz(Esig_t_tau_p, self.tau, axis=0)
        self.Et = Esig_t_tau_p.sum(axis=0)
        self.Et = self.Et/np.abs(self.Et).max()
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))
        
    def conditionFrogTrace(self, Idata, l_start, l_stop, tau_start, tau_stop):
        ''' Take the measured intensity data and interpolate it to the
        internal w, tau grid.
        
        Idata.shape[0] = number of tau points
        Idata.shape[1] = number of spectrum points
        '''
        tau_data = np.linspace(tau_start, tau_stop, Idata.shape[0])
        l_data = np.linspace(l_start, l_stop, Idata.shape[1])
        w_start = 2*np.pi*299792458.0/l_stop
        w_stop = 2*np.pi*299792458.0/l_start
        w_data = np.linspace(w_start, w_stop, Idata.shape[1])
#        w_data = 2*np.pi*299792458.0/l_data[::-1].copy()
        
        Idata_i = np.flipud(Idata).copy()
        Idata_i[0:2,:] = 0.0
        Idata_i[-2:,:] = 0.0
        Idata_i[:,0:2] = 0.0
        Idata_i[:,-2:] = 0.0
        Idata_i = Idata_i / Idata_i.max()
        root.info('Creating interpolator')
        
        t0 = time.clock()
        Idata_interp = si.RectBivariateSpline(tau_data, w_data, Idata_i)
#        Idata_interp = interp2d(tau_mat, w_mat, Idata, kind='linear', fill_value = 0.0, bounds_error = False)
        root.info(''.join(('Time spent: ', str(time.clock()-t0))))

        root.info(''.join(('Interpolating frog trace to ', str(self.tau.shape[0]), 'x', str(self.w.shape[0]))))
        t0 = time.clock()
        self.I_w_tau = np.fft.fftshift(np.maximum(Idata_interp(self.tau, self.w), 0.0), axes=1)
        root.info(''.join(('Time spent: ', str(time.clock()-t0))))
        
    def runCycleVanilla(self, cycles = 1):
        root.debug('Starting FROG reconstruction cycle using the vanilla algorithm')
        for c in range(cycles):
            root.debug(''.join(('Cycle ', str(c+1), '/', str(cycles))))
            self.generateEsig_t_tau_SHG()
            self.generateEsig_w_tau()
            self.applyIntesityData()
            self.updateEt_vanilla()            
            G = self.calcReconstructionError()
            root.debug('-------------------------------------------')
            root.debug(''.join(('Error G = ', str(G))))
            root.debug('-------------------------------------------')
            
    def calcReconstructionError(self):
        I_rec_w_tau = np.real(self.Esig_w_tau*np.conj(self.Esig_w_tau))
        my = 1/I_rec_w_tau.max()
        G = np.sqrt(((self.I_w_tau-my*I_rec_w_tau)**2).sum()/(I_rec_w_tau.shape[0]*I_rec_w_tau.shape[1]))        
        return G
        
if __name__ == '__main__':
    tspan = 0.4e-12
    Nt = 128
    l0 = 390e-9
    lspan = 80e-9
    Nl = 100
    frogTrace = sp.SimulatedSHGFrogTrace(8000, tau = 50e-15, l0 = 800e-9, tspan=2e-12)
#    frogTrace.pulse.generateGaussianCubicPhase(0.001e30, 0.00002e45)
#    frogTrace.pulse.generateGaussianQuadraticPhase(0.01e30)
#    frogTrace.pulse.generateGaussian()
#    Ifrog = frogTrace.generateSHGTrace(tspan, Nt, l0, lspan, Nl)
    tauVec = np.linspace(-tspan/2.0, tspan/2.0, Nt)
    lVec = l0 + np.linspace(-lspan/2.0, lspan/2.0, Nl)
    X, Y = np.meshgrid(lVec, tauVec)
    
    frog = FrogCalculation()
#    frog.initPulseField(800e-9, 10e-9, 1200e-9, 400e-9, 1024, -500e-15, 500e-15, 256)
#    frog.initPulseFieldRandom(0.25e-15, 2e-12, -200e-15, 200e-15)
    dt=800e-9/299792458.0/20
    dt=8e-15
    frog.initPulseFieldPerfectSmall(dt, 128)
    root.info('Calling condition frog trace')
#    frog.conditionFrogTrace(Ifrog, lVec[0], lVec[-1], tauVec[0], tauVec[-1])
    
#     frog.generateEsig_t_tau_SHG()
#     frog.generateEsig_w_tau()
#     frog.applyIntesityData()
    