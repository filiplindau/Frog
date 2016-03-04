'''
Created on 9 Feb 2016

@author: Filip Lindau
'''
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
import matplotlib.pyplot as mpl
from gpyfft.fft import FFT
from scipy.interpolate import interp2d
from scipy.signal import medfilt2d
#from scipy.optimize import minimize_scalar as fmin
from scipy.optimize import fmin as fmin
import scipy.interpolate as si
import sys
import time
import FrogSimulatePulseNC as sp
reload(sp)
import logging
import logging.handlers
import FrogClKernels
reload(FrogClKernels)

root = logging.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
root.addHandler(fh)
root.setLevel(logging.CRITICAL)
    
class FrogCalculation(object):
    def __init__(self):
        self.useGPU = True
        self.rollFFT = True # There is a peculiarity in the fft calculation of a set of vectors
                        # so that the first fft end up in the end... Set this switch to roll
                        # back the last line of the Esig_w_tau fft matrix
        
        
        self.dtype_c = np.complex64
        self.dtype_r = np.float32        
        
        self.Esignal_w = None
        self.Esignal_t = None
                
        self.initCl(useCPU = False)
        
    def initCl(self, useCPU = False):
        root.debug('Initializing opencl')
        pl = cl.get_platforms()
        d = None
        v = None
        root.debug(''.join(('Found ', str(pl.__len__()), ' platforms')))
        vendorDict = {'amd': 3, 'nvidia': 2, 'intel': 1}
        if useCPU == False:
            for p in pl:
                root.debug(p.vendor.lower())
                if 'amd' in p.vendor.lower():
                    vTmp = 'amd'
                elif 'nvidia' in p.vendor.lower():
                    vTmp = 'nvidia'
                else:
                    vTmp = 'intel'
                
                if v == None:
                    d = p.get_devices()
                    v = vTmp
                else:
                    if vendorDict[vTmp] > vendorDict[v]:
                        d = p.get_devices()
                        v = vTmp
        else:
            for p in pl:
                d = p.get_devices(device_type=cl.device_type.CPU)
                if d != []:
                    break
        root.debug(''.join(('Using device ', str(d), ' from ', v)))
        self.ctx = cl.Context(devices = d)
        self.q = cl.CommandQueue(self.ctx)
        
        self.progs = FrogClKernels.FrogClKernels(self.ctx)
        
        
        
    def initClBuffers(self):
        mf = cl.mem_flags
#         self.Et_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.Et)
        self.Et_cla = cla.to_device(self.q, self.Et)
        
        self.Esig_t_tau = np.zeros((self.N, self.N), self.dtype_c)
#         self.Esig_t_tau_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.Esig_t_tau)
        self.Esig_t_tau_cla = cla.to_device(self.q, self.Esig_t_tau)
        
        self.Esig_w_tau = np.zeros((self.N, self.N), self.dtype_c)
        self.Esig_w_tau_cla = cla.to_device(self.q, self.Esig_w_tau)
        
        self.Esig_w_tau_fft = FFT(self.ctx, self.q, (self.Esig_t_tau_cla,) , (self.Esig_w_tau_cla,) , axes = [1])
        
        self.I_w_tau_cla = cla.to_device(self.q, self.I_w_tau)
        
        self.Esig_t_tau_p = np.zeros((self.N, self.N), self.dtype_c)
        self.Esig_t_tau_p_cla = cla.to_device(self.q, self.Esig_t_tau_p)
        
        self.Esig_t_tau_p_fft = FFT(self.ctx, self.q, (self.Esig_w_tau_cla,) , (self.Esig_t_tau_p_cla,) , axes = [1])
        
        self.initClBuffersGP()
    
    def initClBuffersGP(self):
        # Gradient vector for the functional distance in the generalized projection
        self.dZ_cla = cla.zeros(self.q, (self.N), self.dtype_c)
        
        # Vector for intermediate results for the error minimization calculation
        self.X0_cla = cla.zeros(self.q, (self.N), self.dtype_r)
        self.X1_cla = cla.zeros(self.q, (self.N), self.dtype_r)
        self.X2_cla = cla.zeros(self.q, (self.N), self.dtype_r)
        self.X3_cla = cla.zeros(self.q, (self.N), self.dtype_r)
        self.X4_cla = cla.zeros(self.q, (self.N), self.dtype_r)
        self.X5_cla = cla.zeros(self.q, (self.N), self.dtype_r)
        self.X6_cla = cla.zeros(self.q, (self.N), self.dtype_r)
        
        self.Esig_t_tau_norm = np.zeros((self.N, self.N), self.dtype_r)
        self.Esig_t_tau_norm_cla = cla.to_device(self.q, self.Esig_t_tau_norm)
        
    def initPulseFieldRandom(self, N, t_res, l0, seed = 0):
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
        np.random.seed(seed)
        self.N = np.int32(N)
        
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
        self.Et = (np.exp(1j*2*np.pi*np.random.rand(N))).astype(self.dtype_c)
                        
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
        self.N = np.int32(N)
        
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
        
        p = sp.SimulatedFrogTrace(N, self.dt, tau = tau_pulse, l0 = l0)
        p.pulse.generateGaussian(tau_pulse)
        self.tau_start_ind = 0
        self.tau_stop_ind = N-1
        self.tau = self.t
    
        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))
        
        # Finally calculate a gaussian E-field from the 
        self.Et = (np.abs(p.pulse.Et) * np.exp(1j*2*np.pi*np.random.rand(N))).astype(self.dtype_c)
        self.t = p.pulse.t

        self.p = p
        
        root.info('Finished')  
        
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
        self.I_w_tau = np.fft.fftshift(np.maximum(Idata_interp(self.tau, self.w), 0.0), axes=1).astype(self.dtype_r)
#        self.I_w_tau = np.maximum(Idata_interp(self.tau, self.w), 0.0)

        if self.rollFFT == True:
            self.I_w_tau = np.roll(self.I_w_tau, 1, axis=0) 

        root.info(''.join(('Time spent: ', str(time.clock()-t0))))
        
        return Idata_i, w_data, tau_data
    
    def filterFrogTrace(self, Idata, kernel=5, thr=0.1):
        Idata_f = medfilt2d(Idata, kernel)
    
    def generateEsig_t_tau_SHG(self):
        ''' Generate the time shifted E-field matrix for the SHG process.
        
        Output: 
        self.Esig_t_tau, a n_tau x n_t matrix where each row is Esig(t,tau)
        '''      
        root.debug('Generating new Esig_t_tau from SHG')  
        t0 = time.clock()
        krn = self.progs.progs['generateEsig_t_tau_SHG'].generateEsig_t_tau_SHG
        krn.set_scalar_arg_dtypes((None, None, np.int32))
        krn.set_args(self.Et_cla.data, self.Esig_t_tau_cla.data, self.N)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_t_tau.shape, None)
        ev.wait()
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))

    def generateEsig_t_tau_SD(self):
        ''' Generate the time shifted E-field matrix for the SD process.
        
        Output: 
        self.Esig_t_tau, a n_tau x n_t matrix where each row is Esig(t,tau)
        '''      
        root.debug('Generating new Esig_t_tau from SD')  
        t0 = time.clock()
        krn = self.progs.progs['generateEsig_t_tau_SD'].generateEsig_t_tau_SD
        krn.set_scalar_arg_dtypes((None, None, np.int32))
        krn.set_args(self.Et_cla.data, self.Esig_t_tau_cla.data, self.N)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_t_tau.shape, None)
        ev.wait()
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))
        
    def generateEsig_w_tau(self):
        ''' Generate the fft of the time shifted E(t)
        '''
        root.debug('Generating Esig_w_tau')
        rollFFT = False  # There is a peculiarity in the fft calculation of a set of vectors
                        # so that the first fft end up in the end... Set this switch to roll
                        # back the last line of the Esig_w_tau fft matrix
        tic = time.clock()
#         transform = FFT(self.ctx, self.q, (self.Esig_t_tau_cla,) , (self.Esig_w_tau_cla,) , axes = [1])        
#         events = transform.enqueue()
        if self.useGPU == True:
            events = self.Esig_w_tau_fft.enqueue()
            for e in events:
                e.wait()
#         if self.rollFFT == True:
#             krn = self.progs.progs['rollEsigWTau'].rollEsigWTau
#             krn.set_scalar_arg_dtypes((None, np.int32))
#             krn.set_args(self.Esig_w_tau_cla.data, self.N)
#             ev = cl.enqueue_nd_range_kernel(self.q, krn, [self.Esig_w_tau.shape[0]], None)
#             ev.wait()
        else:
            Esig_t_tau = self.Esig_t_tau_cla.get()
            if self.rollFFT == True:
                Esig_w_tau = np.roll(np.fft.fft(Esig_t_tau, axis=1).astype(self.dtype_c), 1, axis=0)
            else:
                Esig_w_tau = np.fft.fft(Esig_t_tau, axis=1).astype(self.dtype_c)
            self.Esig_w_tau_cla.set(Esig_w_tau.copy())
        toc = time.clock()
        root.debug(''.join(('Time spent: ', str(toc-tic))))
        
    def applyIntensityData(self, I_w_tau=None):        
        root.debug('Applying intensity data from experiment')        
        t0 = time.clock()

        krn = self.progs.progs['applyIntensityData'].applyIntensityData
        krn.set_scalar_arg_dtypes((None, None, np.int32))
        krn.set_args(self.Esig_w_tau_cla.data, self.I_w_tau_cla.data, self.N)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_w_tau.shape, None)
        ev.wait()

#         if self.useGPU == True:
#             krn = self.progs.progs['applyIntensityData'].applyIntensityData
#             krn.set_scalar_arg_dtypes((None, None, np.int32))
#             krn.set_args(self.Esig_w_tau_cla.data, self.I_w_tau_cla.data, self.N)
#             ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_w_tau.shape, None)
#             ev.wait()
#         else:
#             eps = 0.00
#             Esig_w_tau = self.Esig_w_tau_cla.get()
#             Esig_mag = np.abs(Esig_w_tau)
#             
#             Esig_w_tau_p = np.zeros_like(Esig_w_tau)
#             good_ind = np.where(Esig_mag > eps)
#             Esig_w_tau_p[good_ind[0], good_ind[1]] = np.sqrt(self.I_w_tau_cla.get()[good_ind[0], good_ind[1]])*Esig_w_tau[good_ind[0], good_ind[1]]/Esig_mag[good_ind[0], good_ind[1]]

        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))
        
    def updateEt_vanilla(self, algo='SHG'):
        root.debug('Updating Et using vanilla algorithm')
        t0 = time.clock()
#         transform = FFT(self.ctx, self.q, (self.Esig_w_tau_cla,) , (self.Esig_t_tau_p_cla,) , axes = [1])        
#         events = transform.enqueue(forward = False)
            
#         self.Esig_t_tau_p_cla.set(np.fft.ifft(self.Esig_w_tau_cla.get(), axis=1).astype(self.dtype_c).copy())
            
        if self.useGPU == True:
            events = self.Esig_t_tau_p_fft.enqueue(forward = False)
            for e in events:
                e.wait()
            if algo == 'SD':
                krn = self.progs.progs['updateEtVanillaSumSD'].updateEtVanillaSumSD
                krn.set_scalar_arg_dtypes((None, None, np.int32))
                krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.N)
                ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
                ev.wait()
                
                Et = self.Et_cla.get()
                self.Et_cla.set(-np.conj(Et).astype(self.dtype_c).copy())
                
#                 Esig_w_tau = self.Esig_w_tau_cla.get()
#                 Gm  = np.conj(Esig_w_tau.sum(axis=1))[::-1]
#                 self.Et_cla.set(Gm.copy())
                
            else:
                krn = self.progs.progs['updateEtVanillaSumSHG'].updateEtVanillaSumSHG
                krn.set_scalar_arg_dtypes((None, None, np.int32))
                krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.N)
                ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
                ev.wait()                                
    
            krn = self.progs.progs['updateEtVanillaNorm'].updateEtVanillaNorm
            krn.set_scalar_arg_dtypes((None, np.int32))
            krn.set_args(self.Et_cla.data, self.N)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, [1], None)
            ev.wait()
        else:
            self.Esig_t_tau_p_cla.set(np.fft.ifft(self.Esig_w_tau_cla.get(), axis=1).astype(self.dtype_c).copy())
            Esig_t_tau_p = self.Esig_t_tau_p_cla.get()
            if algo=='SD':
                Et = np.sqrt(Esig_t_tau_p.sum(axis=0))
#                 Et = (Esig_t_tau_p.sum(axis=0))
            else:
                Et = Esig_t_tau_p.sum(axis=0)
            Et = Et/np.abs(Et).max()
            self.Et_cla.set(Et)
            

        
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))
        
    def gradZSHG_naive(self):
        root.debug('Calculating dZ for SHG using for loops')
        Et = self.Et_cla.get()
        Esigp = self.Esig_t_tau_p_cla.get()
        dZ = np.zeros_like(Et)
        N = Esigp.shape[0]
        sz = N*N
        for t0 in range(N):
            T = 0.0 + 1j*0.0
            for tau in range(N):
                tp = t0 - (tau-N/2)
                if tp >=0 and tp < N:
                    T += (Et[t0]*Et[tp] - Esigp[tau, t0])*np.conj(Et[tp])
                tp = t0 + (tau-N/2)
                if tp >=0 and tp < N:
                    T += (Et[t0]*Et[tp] - Esigp[tau, tp])*np.conj(Et[tp])
            dZ[t0] = -T/sz
        self.dZ_cla.set(dZ.copy())

    def gradZSHG_gpu(self):
        root.debug('Calculating dZ for SHG using gpu')
        krn = self.progs.progs['gradZSHG'].gradZSHG
        krn.set_scalar_arg_dtypes((None, None, None, np.int32))
        krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.dZ_cla.data, self.N)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
        ev.wait()

    def gradZSD_naive(self):
        # Todo: fix this algorithm
        root.debug('Calculating dZ for SD using for loops')
        Et = self.Et_cla.get()
        Esigp = self.Esig_t_tau_p_cla.get()
        dZ = np.zeros_like(Et)
        N = Esigp.shape[0]
        sz = N*N
        for t0 in range(N):
            T = 0.0 + 1j*0.0
            for tau in range(N):
                tp = t0 - (tau-N/2)
                if tp >=0 and tp < N:
                    EtEtp = np.conj(Et[t0])*Et[tp]
                    T += 4*(Et[t0]*np.conj(EtEtp) - Esigp[tau, t0])*EtEtp
                tp = t0 + (tau-N/2)
                if tp >=0 and tp < N:
                    EtpEtp = Et[tp]*Et[tp]
                    T += 2*(Et[t0]*np.conj(EtpEtp) - np.conj(Esigp[tau, tp]))*EtpEtp
            dZ[t0] = -T/sz
        self.dZ_cla.set(dZ.copy())

    def gradZSD_gpu(self):
        root.debug('Calculating dZ for SD using gpu')
        krn = self.progs.progs['gradZSD'].gradZSD
        krn.set_scalar_arg_dtypes((None, None, None, np.int32))
        krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.dZ_cla.data, self.N)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
        ev.wait()
        
    def minZerrKernSHG_naive(self):
        Et0 = self.Et_cla.get()
        Esig = self.Esig_t_tau_p_cla.get()
        dZ = self.dZ_cla.get()
        N = Esig.shape[0]
        
        mx = 0.0
        X = np.zeros(5)
        
        for t in range(N):
            for tau in range(N):
                T = np.abs(Esig[tau, t])**2
                if mx < T:
                    mx = T
                tp = t-(tau-N/2)
                if tp >=0 and tp < N:
                    dZdZ = dZ[t]*dZ[tp]
                    dZE = dZ[t]*Et0[tp] + dZ[tp]*Et0[t]
                    DEsig = Et0[t]*Et0[tp] - Esig[tau, t]
                    
                    X[0] += np.abs(dZdZ)**2
                    X[1] += 2.0*np.real(dZE*np.conj(dZdZ))
                    X[2] += 2.0*np.real(DEsig*np.conj(dZdZ)) + np.abs(dZE)**2
                    X[3] += 2.0*np.real(DEsig*np.conj(dZE))
                    X[4] += np.abs(DEsig)**2
        T = N*N*mx
        X[0] = X[0]/T
        X[1] = X[1]/T
        X[2] = X[2]/T
        X[3] = X[3]/T
        X[4] = X[4]/T
        
        root.debug(''.join(('Esig_t_tau_p norm max: ', str(mx))))
        
        return X

    def minZerrKernSHG_gpu(self):
        krn = self.progs.progs['minZerrSHG'].minZerrSHG
        krn.set_scalar_arg_dtypes((None, None, None, None, None, None, None, None, np.int32))
        krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.dZ_cla.data, 
                     self.X0_cla.data, self.X1_cla.data, self.X2_cla.data, self.X3_cla.data, self.X4_cla.data, self.N)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
        ev.wait()
        
        krn = self.progs.progs['normEsig'].normEsig
        krn.set_scalar_arg_dtypes((None, None, np.int32))
        krn.set_args(self.Esig_t_tau_p_cla.data, self.Esig_t_tau_norm_cla.data, self.N)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_t_tau_p.shape, None)
        ev.wait()
        mx = cla.max(self.Esig_t_tau_norm_cla).get() * self.N*self.N

#         Esig_t_tau = self.Esig_t_tau_p_cla.get()
#         mx = ((Esig_t_tau*Esig_t_tau.conj()).real).max() * self.N*self.N
        
        X0 = cla.sum(self.X0_cla, queue=self.q).get() / mx
        X1 = cla.sum(self.X1_cla, queue=self.q).get() / mx
        X2 = cla.sum(self.X2_cla, queue=self.q).get() / mx
        X3 = cla.sum(self.X3_cla, queue=self.q).get() / mx
        X4 = cla.sum(self.X4_cla, queue=self.q).get() / mx
        
        root.debug(''.join(('X0=', str(X0), ', type ', str(type(X0)))))
        
        root.debug(''.join(('Poly: ', str(X4), ' x^4 + ', str(X3), ' x^3 + ', str(X2), ' x^2 + ', str(X1), ' x + ', str(X0))))
        # Polynomial in dZ (expansion of differential)
        X = np.array([X0, X1, X2, X3, X4]).astype(np.double)
        
        root.debug(''.join(('Esig_t_tau_p norm max: ', str(mx/(self.N*self.N)))))
        
        return X       
     
    def minZerrKernSD_naive(self):
        # Todo: fix this algorithm
        Et0 = self.Et_cla.get()
        Esig = self.Esig_t_tau_p_cla.get()
        dZ = self.dZ_cla.get()
        N = Esig.shape[0]
        
        mx = 0.0
        X = np.zeros(7)
        
        for t in range(N):
            for tau in range(N):
                T = np.abs(Esig[tau, t])**2
                if mx < T:
                    mx = T
                tp = t-(tau-N/2)
                if tp >=0 and tp < N:
                    a0 = Esig[tau, t] - Et0[t]*Et0[t]*np.conj(Et0[tp])
                    a1 = -(2*Et0[t]*dZ[t]*np.conj(Et0[tp]) + Et0[t]*Et0[t]*np.conj(dZ[tp]))
                    a2 = -(dZ[t]*dZ[t]*np.conj(Et0[tp]) + 2*Et0[t]*np.conj(dZ[tp])*dZ[t])
                    a3 = -dZ[t]*dZ[t]*np.conj(dZ[tp])
                    
                    X[0] += np.real(a3*np.conj(a3))
                    X[1] += np.real(a2*np.conj(a3) + a3*np.conj(a2))
                    X[2] += np.real(a1*np.conj(a3) + a3*np.conj(a1) + a2*np.conj(a2))
                    X[3] += np.real(a0*np.conj(a3) + a3*np.conj(a0) + a1*np.conj(a2) + a2*np.conj(a1))
                    X[4] += np.real(a0*np.conj(a2) + a2*np.conj(a0) + a1*np.conj(a1))
                    X[5] += np.real(a0*np.conj(a1) + a1*np.conj(a0))
                    X[6] += np.real(a0*np.conj(a0))
        T = N*N*mx
        X = X/T
        
        root.debug(''.join(('Esig_t_tau_p norm max: ', str(mx))))
        
        return X

    def minZerrKernSD_gpu(self):
        krn = self.progs.progs['minZerrSD'].minZerrSD
        krn.set_scalar_arg_dtypes((None, None, None, None, None, None, None, None, None, None, np.int32))
        krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.dZ_cla.data, 
                     self.X0_cla.data, self.X1_cla.data, self.X2_cla.data, self.X3_cla.data, 
                     self.X4_cla.data, self.X5_cla.data, self.X6_cla.data, self.N)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
        ev.wait()
        
        krn = self.progs.progs['normEsig'].normEsig
        krn.set_scalar_arg_dtypes((None, None, np.int32))
        krn.set_args(self.Esig_t_tau_p_cla.data, self.Esig_t_tau_norm_cla.data, self.N)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_t_tau_p.shape, None)
        ev.wait()
        mx = cla.max(self.Esig_t_tau_norm_cla).get() * self.N*self.N

#         Esig_t_tau = self.Esig_t_tau_p_cla.get()
#         mx = ((Esig_t_tau*Esig_t_tau.conj()).real).max() * self.N*self.N
        
        X0 = cla.sum(self.X0_cla, queue=self.q).get() / mx
        X1 = cla.sum(self.X1_cla, queue=self.q).get() / mx
        X2 = cla.sum(self.X2_cla, queue=self.q).get() / mx
        X3 = cla.sum(self.X3_cla, queue=self.q).get() / mx
        X4 = cla.sum(self.X4_cla, queue=self.q).get() / mx
        X5 = cla.sum(self.X5_cla, queue=self.q).get() / mx
        X6 = cla.sum(self.X6_cla, queue=self.q).get() / mx
        
        root.debug(''.join(('X0=', str(X0), ', type ', str(type(X0)))))
        
        root.debug(''.join(('Poly: ', str(X6), ' x^6 + ', str(X5), ' x^5 + ', str(X4), ' x^4 + ', str(X3), ' x^3 + ', str(X2), ' x^2 + ', str(X1), ' x + ', str(X0))))
        # Polynomial in dZ (expansion of differential)
        X = np.array([X0, X1, X2, X3, X4, X5, X6]).astype(np.double)
        
        root.debug(''.join(('Esig_t_tau_p norm max: ', str(mx/(self.N*self.N)))))
        
        return X       
         
    def updateEt_gp(self, algo = 'SHG'):
        root.debug('Updating Et using GP algorithm')
        tic = time.clock()
 
        events = self.Esig_t_tau_p_fft.enqueue(forward = False)
        for e in events:
            e.wait()
       
        if algo == 'SHG':
            # Calculate the gradient of the functional distance:
            if self.useGPU == True:
                self.gradZSHG_gpu()
            else:
                self.gradZSHG_naive()
            
            # Calculate error minimization polynomial
            if self.useGPU == True:
                p1 = self.minZerrKernSHG_gpu()
            else:
                p1 = self.minZerrKernSHG_naive()
            root.debug(''.join(('Poly: ', str(p1[4]), ' x^4 + ', str(p1[3]), ' x^3 + ', str(p1[2]), ' x^2 + ', str(p1[1]), ' x + ', str(p1[0]))))
        elif algo == 'SD':
            # Calculate the gradient of the functional distance:
            if self.useGPU == True:
                self.gradZSD_gpu()
            else:
                self.gradZSD_naive()
            
            # Calculate error minimization polynomial
            if self.useGPU == True:
                p1 = self.minZerrKernSD_gpu()
            else:
                p1 = self.minZerrKernSD_naive()
            root.debug(''.join(('Poly: ', str(p1[4]), ' x^4 + ', str(p1[3]), ' x^3 + ', str(p1[2]), ' x^2 + ', str(p1[1]), ' x + ', str(p1[0]))))
            
        # Root finding of the polynomial in the gradient expansion
        p = np.polyder(p1)
        r = np.roots(p)
        X = r[np.abs(r.imag)<1e-9].real
        root.debug(''.join(('Real roots: ', str(X))))
        
        Z1 = np.polyval(p1, X)
        minZInd = Z1.argmin() 
        Z = np.maximum(3e-16*X[-1], Z1[minZInd])

        Z = np.sqrt(Z)
        X = X[minZInd].astype(self.dtype_r)

        # Update Et
        if self.useGPU == True:
            krn = self.progs.progs['updateEtGP'].updateEtGP
            krn.set_scalar_arg_dtypes((None, None, self.dtype_r, np.int32))
            krn.set_args(self.Et_cla.data, self.dZ_cla.data, X, self.N)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
            ev.wait()
        else:
            root.debug(''.join(('Moving distance X=', str(X))))
            Et = self.Et_cla.get()
            Et_new = Et + X*self.dZ_cla.get()
            self.Et_cla.set(Et_new.copy())
        
        toc = time.clock()
        root.debug(''.join(('Time spent: ', str(toc-tic))))
        
        return Z
        
    def centerPeakTime(self):
        Et = self.Et_cla.get()
        ind = np.argmax(abs(Et))
        shift = Et.shape[0]/2 - ind
        Et = np.roll(Et, shift)
        self.Et_cla.set(Et)

    def calcReconstructionError(self):
        root.debug('Calculating reconstruction error')
        tic = time.clock()
        Esig_w_tau = self.Esig_w_tau_cla.get()
        I_rec_w_tau = np.real(Esig_w_tau*np.conj(Esig_w_tau))
        I_w_tau = self.I_w_tau_cla.get()
        my = I_w_tau.max()/I_rec_w_tau.max()
        root.debug(''.join(('My=', str(my))))
        G = np.sqrt(((I_w_tau-my*I_rec_w_tau)**2).sum()/(I_rec_w_tau.shape[0]*I_rec_w_tau.shape[1]))        
#        G = np.sqrt(((self.I_w_tau-my*I_rec_w_tau)**2).sum()/(self.I_w_tau.sum()))
        toc = time.clock()
        root.debug(''.join(('Time spent: ', str(toc-tic))))
        return G
            
    def getData(self):
        root.debug('Retrieving data from opencl buffers')
        tic=time.clock()
        self.Esig_t_tau_cla.get()
        self.Et_cla.get()
        self.Esig_t_tau_p_cla.get()
        self.Esig_w_tau_cla.get()
        toc=time.clock()
        root.debug(''.join(('Time spent: ', str(toc-tic))))
                  
    def getTraceAbs(self):
        self.centerPeakTime()
        return np.abs(self.Et_cla.get())
    
    def getTracePhase(self):
        self.centerPeakTime()
        Et = self.Et_cla.get()
        ph0 = np.angle(Et[Et.shape[0]/2])
        return np.angle(Et)-ph0
    
    def getT(self):
        return self.t
                  
    def setupVanillaAlgorithm(self):
        pass
                  
    def runCycleVanilla(self, cycles = 1, algo = 'SHG', useGPU = None):
        root.debug('Starting FROG reconstruction cycle using the vanilla algorithm')
        if useGPU is not None:
            self.useGPU = useGPU
            self.rollFFT = useGPU
        
        t0 = time.clock()
        er = []
        self.setupVanillaAlgorithm()
        for c in range(cycles):
            root.debug(''.join(('Cycle ', str(c+1), '/', str(cycles))))
            if algo=='SD':
                self.generateEsig_t_tau_SD()
            else:
                self.generateEsig_t_tau_SHG()
            self.generateEsig_w_tau()
            G = self.calcReconstructionError()
            self.applyIntensityData()
            self.updateEt_vanilla('SD')
#             self.centerPeakTime()                        
            root.debug('-------------------------------------------')
            root.debug(''.join(('Error G = ', str(G))))
            root.debug('-------------------------------------------')
            er.append(G)
        deltaT = time.clock() - t0
        root.debug(''.join(('Total runtime ', str(deltaT))))
        root.debug(''.join((str(cycles/deltaT), ' iterations/s')))
        print ''.join((str(cycles/deltaT), ' iterations/s'))
        return np.array(er)

    def setupGPAlgorithm(self):
        pass
                  
    def runCycleGP(self, cycles = 1, algo = 'SHG', useGPU = None):
        root.debug('Starting FROG reconstruction cycle using the GP algorithm')
        if useGPU is not None:
            self.useGPU = useGPU
            self.rollFFT = useGPU
        t0 = time.clock()
        er = []
        self.setupGPAlgorithm()
        for c in range(cycles):
            root.debug(''.join(('Cycle ', str(c+1), '/', str(cycles))))
            if algo=='SD':
                self.generateEsig_t_tau_SD()
            else:
                self.generateEsig_t_tau_SHG()
            self.generateEsig_w_tau()
            G = self.calcReconstructionError()
            self.applyIntensityData()
            self.updateEt_gp(algo)
#             self.centerPeakTime()                        
            root.debug('-------------------------------------------')
            root.debug(''.join(('Error G = ', str(G))))
            root.debug('-------------------------------------------')
            er.append(G)
        deltaT = time.clock() - t0
        root.debug(''.join(('Total runtime ', str(deltaT))))
        root.debug(''.join((str(cycles/deltaT), ' iterations/s')))
        print ''.join((str(cycles/deltaT), ' iterations/s'))
        return np.array(er)       
    
    def runComplete(self):
        tic = time.clock()
        er = self.runCycleVanilla(30)
        oldEr = np.min(er) 
        er = self.runCycleGP(30)        
        newEr = np.min(er)
        epochs = 0
        while oldEr-newEr > 1e-5 and epochs<20:
            oldEr = newEr
            er=self.runCycleGP(30)
            newEr = np.min(er)
            epochs += 1
            print "Epoch ", epochs, ", error ", newEr
        print "Epochs: ", epochs
        toc = time.clock()
        print "Total reconstruction time ", toc-tic, " s"
                    
if __name__ == '__main__':    
    N = 256
    dt = 8e-15
    l0 = 800e-9
    tau_pulse = 100e-15
    
    p = sp.SimulatedPulse(N, dt, l0, tau_pulse)
    p.generateGaussianCubicSpectralPhase(0, 1e-40)
#     p.generateGaussianCubicPhase(5e24, 3e39)
#     p.generateGaussian(tau_pulse)
#    p.generateDoublePulse(tau_pulse, deltaT=0.5e-12)
    gt = sp.SimulatedFrogTrace(N, dt, l0)
    gt.pulse = p
    IfrogSHG = gt.generateSHGTraceDt(N, dt, 400e-9)
    IfrogSD = gt.generateSDTraceDt(N, dt, 800e-9)
#     Ifrog = gt.addNoise(0.01, 0.1)
    l = gt.getWavelengths()
    t = gt.getTimedelays()
    
    frog = FrogCalculation()
    
#    frog.initPulseFieldPerfect(128, dt, 800e-9)
#    frog.initPulseFieldGaussian(N, dt, l0, 50e-15)

    frog.initPulseFieldRandom(N, dt, l0)
    frog.conditionFrogTrace(IfrogSD[::-1,:], l[0], l[-1], t[0], t[-1])
    frog.initClBuffers()
#     er=frog.runCycleVanilla(1)
#    frog.generateEsig_t_tau_SHG()
    
#     frog.getData()
