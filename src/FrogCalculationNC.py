'''
Created on 3 Feb 2016

@author: Filip Lindau
'''

import numpy as np
from scipy.interpolate import interp2d
import scipy.interpolate as si
import sys
import time
import FrogSimulatePulseNC as sp
reload(sp)
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
        
    def initPulseFieldPerfect(self, N, t_res, l0):
        """ Initiate signal field with parameters:
        t_res: time resolution of the reconstruction
        N: number of points in time and wavelength axes
        l0: center wavelength
        
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
        
        c = 299792458.0
        w0 = 2*np.pi*c/l0
#        w_spectrum = np.linspace(w0-w_span/2, w0+w_span/2, n_t)
        w_spectrum = np.linspace(-w_span/2, -w_span/2+w_res*N, N)
        self.dw = w_res 
        self.w = w_spectrum
        self.w0 = w0
        
                
        # Create time vector
        self.dt = t_res
        self.t = np.linspace(-t_span/2, t_span/2, N)
        
        tau_pulse = 100e-15
        p = sp.SimulatedSHGFrogTrace(N, dt, tau = tau_pulse, l0 = l0)
        p.pulse.generateGaussian(tau_pulse)
        tspan_frog = t_span
        Nt = N
        l_shg = 390e-9
        lspan = 80e-9
        Nl = N
        Ifrog = p.generateSHGTraceDt(N, t_res, l0)
        
        # Vector of delay positions
        # We also store the index of the tau_start and tau_stop
        # snapped to available time values 
        tau_start_t = self.t.min()
        tau_stop_t = self.t.max()
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

#        tauVec = self.t
#        lVec = l0 + np.linspace(-lspan/2.0, lspan/2.0, Nl)
        tauVec = p.getTimedelays()
        lVec = p.getWavelengths()

        self.generateEsig_t_tau_SHG()
        self.I_w_tau = np.abs(np.fft.fft(self.Esig_t_tau, axis = 1))**2
        
        self.p = p
            
        root.info('Finished')
        
    def initPulseFieldGaussian(self, N, t_res, l0, tau_pulse):
        """ Initiate signal field with parameters:
        t_res: time resolution of the reconstruction
        N: number of points in time and wavelength axes
        l0: center wavelength
        
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
        
        c = 299792458.0
        w0 = 2*np.pi*c/l0
#        w_spectrum = np.linspace(w0-w_span/2, w0+w_span/2, n_t)
        w_spectrum = np.linspace(-w_span/2, -w_span/2+w_res*N, N)
        self.dw = w_res 
        self.w = w_spectrum
        self.w0 = w0
        
                
        # Create time vector
        self.dt = t_res
        self.t = np.linspace(-t_span/2, t_span/2, N)
        
        p = sp.SimulatedSHGFrogTrace(N, dt, tau = tau_pulse, l0 = l0)
        p.pulse.generateGaussian(tau_pulse)
        self.tau_start_ind = 0
        self.tau_stop_ind = N-1
        self.tau = self.t
    
        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))
        
        # Finally calculate a gaussian E-field from the 
        self.Et = np.abs(p.pulse.Et) * np.exp(1j*2*np.pi*np.random.rand(N))
        self.t = p.pulse.t

        
        self.p = p
            
        root.info('Finished')        
        
    def initPulseFieldRandom(self, N, t_res, l0):
        """ Initiate signal field with parameters:
        t_res: time resolution of the reconstruction
        N: number of points in time and wavelength axes
        l0: center wavelength
        
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
        
        c = 299792458.0
        w0 = 2*np.pi*c/l0
#        w_spectrum = np.linspace(w0-w_span/2, w0+w_span/2, n_t)
        w_spectrum = np.linspace(-w_span/2, -w_span/2+w_res*N, N)
        self.dw = w_res 
        self.w = w_spectrum
        self.w0 = w0
        
                
        # Create time vector
        self.dt = t_res
        self.t = np.linspace(-t_span/2, t_span/2, N)
        
        self.tau_start_ind = 0
        self.tau_stop_ind = N-1
        self.tau = self.t
    
        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))
        
        # Finally calculate a gaussian E-field from the 
        self.Et = np.exp(1j*2*np.pi*np.random.rand(N))
            
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
        N = self.Et.shape[0]
        
        shiftVec = np.arange(N) - N/2
        
        for ind, sh in enumerate(shiftVec):
        
            if sh < 0:
                Et_mat_tau[ind, 0:N+sh] = self.Et[-sh:] # Here should be a factor exp(-1j*w0*tau)
            else:
                Et_mat_tau[ind, sh:] = self.Et[0:N-sh] # Here should be a factor exp(-1j*w0*tau)
        
        self.Esig_t_tau = Et_mat*Et_mat_tau 
        self.Et_mat = Et_mat
        self.Et_mat_tau = Et_mat_tau
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))

    def generateEsig_t_tau_Outer(self):
        ''' Generate the time shifted E-field matrix for the SHG process.
        
        Output: 
        self.Esig_t_tau, a n_tau x n_t matrix where each row is Esig(t,tau)
        '''      
        root.debug('Generating new Esig_t_tau from SHG')  
        t0 = time.clock()

#         N = self.Et.shape[0]
        Et_mat = np.outer(self.P, self.G)    # Repeat Et into a matrix
#         i = np.arange(N*N)   
#         i2 = np.arange(N).repeat(N)
# #        shiftInds = (N/2-i-i2)%N + i2*N          
#         shiftInds = (i+i2-N/2)%N + i2*N
        
        self.Esig_t_tau = np.rot90(Et_mat.flatten()[self.shiftInds].reshape(N,N)) 
        self.Et_mat = Et_mat
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))

    def generateOuterProduct(self):
        ''' Generate the outer product from time shifted E-field matrix.
        
        Output: 
        self.Esig_outer, a n_tau x n_t matrix 
        '''      
        root.debug('Generating new Esig_outer')  
        t0 = time.clock()

#         N = self.Et.shape[0]
#         i = np.arange(N*N)   
#         i2 = np.arange(N).repeat(N)
# #        shiftInds = (N/2-i-i2)%N + i2*N          
#         shiftInds = (i+i2-N/2)%N + i2*N
        self.O = np.rot90(np.flipud(np.rot90(self.Esig_t_tau_p)).flatten()[self.shiftInds].reshape(N,N))
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
        self.Esig_t_tau_p = np.fft.ifft(self.Esig_w_tau_p, axis=1)
#        self.Et = np.trapz(Esig_t_tau_p, self.tau, axis=0)
        self.Et = self.Esig_t_tau_p.sum(axis=0)
        self.Et = self.Et/np.abs(self.Et).max()
        self.P = self.Et
        self.G = self.Et
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))

    def updateEt_SVD(self, it = 1):
        root.debug('Updating Et using SVD power algorithm')
        t0 = time.clock()
        self.Esig_t_tau_p = np.fft.ifft(self.Esig_w_tau_p, axis=1)
        self.generateOuterProduct()
        for i in range(it):
            self.P = np.dot(np.dot(self.O, np.transpose(self.O)), self.P)
            self.G = np.dot(np.dot(np.transpose(self.O), self.O), self.G)
#        self.Et = np.trapz(Esig_t_tau_p, self.tau, axis=0)
        self.P = self.P/np.abs(self.P).max()
        self.G = self.G/np.abs(self.G).max()
        self.Et = self.P
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))
        
    def conditionFrogTrace(self, Idata, l_start, l_stop, tau_start, tau_stop):
        ''' Take the measured intensity data and interpolate it to the
        internal w, tau grid.
        
        Idata.shape[0] = number of tau points
        Idata.shape[1] = number of spectrum points
        '''
        tau_data = np.linspace(tau_start, tau_stop, Idata.shape[0])
        l_data = np.linspace(l_start, l_stop, Idata.shape[1])
        if l_start > l_stop:
            w_start = 2*np.pi*299792458.0/l_start
            w_stop = 2*np.pi*299792458.0/l_stop
            w0 = 2*np.pi*299792458.0/((l_stop+l_start)/2)
            Idata_i = Idata.copy()
        else:
            w_start = 2*np.pi*299792458.0/l_stop
            w_stop = 2*np.pi*299792458.0/l_start
            w0 = 2*np.pi*299792458.0/((l_stop+l_start)/2)
            Idata_i = np.fliplr(Idata).copy()
            
        root.debug(''.join(('w_start: ', str(w_start))))
        root.debug(''.join(('w_stop: ', str(w_stop))))
        w_data = np.linspace(w_start, w_stop, Idata.shape[1])-w0
#        w_data = 2*np.pi*299792458.0/l_data[::-1].copy()
        
        Idata_i = np.flipud(Idata_i).copy()
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
#        self.I_w_tau = np.maximum(Idata_interp(self.tau, self.w), 0.0)
        root.info(''.join(('Time spent: ', str(time.clock()-t0))))
        
        return Idata_i, w_data, tau_data
        
    def centerPeakTime(self):
        ind = np.argmax(abs(self.Et))
        shift = self.Et.shape[0]/2 - ind
        self.Et = np.roll(self.Et, shift)
        
    def setupVanillaAlgorithm(self):
        self.P = self.Et
        self.G = self.Et

        N = self.Et.shape[0]
        i = np.arange(N*N)   
        i2 = np.arange(N).repeat(N)
#        shiftInds = (N/2-i-i2)%N + i2*N          
        self.shiftInds = (i+i2-N/2)%N + i2*N
        
        
    def runCycleVanilla(self, cycles = 1):
        root.debug('Starting FROG reconstruction cycle using the vanilla algorithm')
        t0 = time.clock()
        er = []
        self.setupVanillaAlgorithm()
        for c in range(cycles):
            root.debug(''.join(('Cycle ', str(c+1), '/', str(cycles))))
            self.generateEsig_t_tau_Outer()
            self.generateEsig_w_tau()
            self.applyIntesityData()
            self.updateEt_vanilla()
            self.centerPeakTime()            
            G = self.calcReconstructionError()
            root.debug('-------------------------------------------')
            root.debug(''.join(('Error G = ', str(G))))
            root.debug('-------------------------------------------')
            er.append(G)
        deltaT = time.clock() - t0
        root.debug(''.join(('Total runtime ', str(deltaT))))
        root.debug(''.join((str(cycles/deltaT), ' iterations/s')))
        return np.array(er)

    def setupSVDAlgorithm(self):
        self.P = self.Et
        self.G = self.Et
        
        N = self.Et.shape[0]
        i = np.arange(N*N)   
        i2 = np.arange(N).repeat(N)
#        shiftInds = (N/2-i-i2)%N + i2*N          
        self.shiftInds = (i+i2-N/2)%N + i2*N
        
            
    def runCycleSVD(self, cycles = 1):
        root.debug('Starting FROG reconstruction cycle using the SVD power algorithm')
        t0 = time.clock()
        er = []
        self.setupSVDAlgorithm()
        for c in range(cycles):
            root.debug(''.join(('Cycle ', str(c+1), '/', str(cycles))))
            self.generateEsig_t_tau_Outer()
            self.generateEsig_w_tau()
            self.applyIntesityData()
            self.updateEt_SVD(1)
            self.centerPeakTime()            
            G = self.calcReconstructionError()
            root.debug('-------------------------------------------')
            root.debug(''.join(('Error G = ', str(G))))
            root.debug('-------------------------------------------')
            er.append(G)
        deltaT = time.clock() - t0
        root.debug(''.join(('Total runtime ', str(deltaT))))
        root.debug(''.join((str(cycles/deltaT), ' iterations/s')))
            
        return np.array(er)
            
    def calcReconstructionError(self):
        I_rec_w_tau = np.real(self.Esig_w_tau*np.conj(self.Esig_w_tau))
        my = self.I_w_tau.max()/I_rec_w_tau.max()
        G = np.sqrt(((self.I_w_tau-my*I_rec_w_tau)**2).sum()/(I_rec_w_tau.shape[0]*I_rec_w_tau.shape[1]))        
#        G = np.sqrt(((self.I_w_tau-my*I_rec_w_tau)**2).sum()/(self.I_w_tau.sum()))
        return G
        
    def getTraceAbs(self):
        self.centerPeakTime()
        return np.abs(self.Et)
    
    def getTracePhase(self):
        self.centerPeakTime()
        ph0 = np.angle(self.Et[self.Et.shape[0]/2])
        return np.angle(self.Et)-ph0
    
    def getT(self):
        return self.t
    
if __name__ == '__main__':    
    N = 256
    dt = 8e-15
    l0 = 800e-9
    tau_pulse = 100e-15
    
    p = sp.SimulatedPulse(N, dt, l0, tau_pulse)
    p.generateGaussianCubicPhase(5e24, 3e39)
    gt = sp.SimulatedSHGFrogTrace(N, dt, l0)
    gt.pulse = p
    Ifrog = gt.generateSHGTraceDt(N, dt, 400e-9)
    l = gt.getWavelengths()
    t = gt.getTimedelays()
    
    frog = FrogCalculation()
    
#    frog.initPulseFieldPerfect(128, dt, 800e-9)
#    frog.initPulseFieldGaussian(N, dt, l0, 50e-15)
    frog.initPulseFieldRandom(N, dt, l0)
    frog.conditionFrogTrace(Ifrog, l[0], l[-1], t[0], t[-1])
    
        