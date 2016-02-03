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
        
    def initPulseFieldPerfect(self, t_res, N, l0):
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
        w_spectrum = w0+np.linspace(-w_span/2, -w_span/2+w_res*N, N)
        self.dw = w_res 
        self.w = w_spectrum
        
                
        # Create time vector
        self.dt = t_res
        self.t = np.linspace(-t_span/2, t_span/2, N)
        
        p = sp.SimulatedSHGFrogTrace(N, dt, tau = 100e-15, l0 = l0)
        p.pulse.generateGaussian()
        tspan_frog = t_span
        Nt = N
        l_shg = 390e-9
        lspan = 80e-9
        Nl = N
        Ifrog = p.generateSHGTrace(tspan_frog, Nt, t_res, l_shg)
        
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
    frog = FrogCalculation()
    dt=8e-15
    frog.initPulseFieldPerfect(dt, 128)
        