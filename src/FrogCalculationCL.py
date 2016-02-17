'''
Created on 9 Feb 2016

@author: Filip Lindau
'''
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from gpyfft.fft import FFT
from scipy.interpolate import interp2d
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
root.setLevel(logging.DEBUG)
    
class FrogCalculation(object):
    def __init__(self):
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
        root.info(''.join(('Time spent: ', str(time.clock()-t0))))
        
        return Idata_i, w_data, tau_data
    
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
        
    def generateEsig_w_tau(self):
        ''' Generate the fft of the time shifted E(t)
        '''
        root.debug('Generating Esig_w_tau')
        tic = time.clock()
#         transform = FFT(self.ctx, self.q, (self.Esig_t_tau_cla,) , (self.Esig_w_tau_cla,) , axes = [1])        
#         events = transform.enqueue()
        events = self.Esig_w_tau_fft.enqueue()
        for e in events:
            e.wait()
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
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))
        
    def updateEt_vanilla(self):
        root.debug('Updating Et using vanilla algorithm')
        useGPU = True
        t0 = time.clock()
#         transform = FFT(self.ctx, self.q, (self.Esig_w_tau_cla,) , (self.Esig_t_tau_p_cla,) , axes = [1])        
#         events = transform.enqueue(forward = False)
        events = self.Esig_t_tau_p_fft.enqueue()
        for e in events:
            e.wait()
            
        if useGPU == True:
            krn = self.progs.progs['updateEtVanillaSum'].updateEtVanillaSum
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
            Esig_t_tau_p = self.Esig_t_tau_p_cla.get()
            Et = Esig_t_tau_p.sum(axis=0)
            Et = Et/np.abs(Et).max()
            self.Et_cla.set(Et)
            

        
        root.debug(''.join(('Time spent: ', str(time.clock()-t0))))

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
        my = self.I_w_tau.max()/I_rec_w_tau.max()
        G = np.sqrt(((self.I_w_tau-my*I_rec_w_tau)**2).sum()/(I_rec_w_tau.shape[0]*I_rec_w_tau.shape[1]))        
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
                  
    def setupVanillaAlgorithm(self):
        pass
                  
    def runCycleVanilla(self, cycles = 1):
        root.debug('Starting FROG reconstruction cycle using the vanilla algorithm')
        t0 = time.clock()
        er = []
        self.setupVanillaAlgorithm()
        for c in range(cycles):
            root.debug(''.join(('Cycle ', str(c+1), '/', str(cycles))))
            self.generateEsig_t_tau_SHG()
            self.generateEsig_w_tau()
            self.applyIntensityData()
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
        print ''.join((str(cycles/deltaT), ' iterations/s'))
        return np.array(er)
                  
                    
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
    frog.initClBuffers()
#     er=frog.runCycleVanilla(1)
#    frog.generateEsig_t_tau_SHG()
    
#     frog.getData()
