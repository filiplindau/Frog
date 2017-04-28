"""
Created on 24 Apr 2017

@author: Filip Lindau
"""

"""
Created on 17 Feb 2017

@author: Filip Lindau
"""

import numpy as np
from scipy.misc import imread
from scipy.signal import medfilt2d
from scipy.interpolate import interp1d
from scipy.optimize import fmin as fmin
from scipy.optimize import fmin_cg, fmin_bfgs
import scipy.interpolate as si
import sys
import time
import FrogSimulatePulseNC as sp
if sys.version_info > (3, 0):
    from importlib import reload

import pyopencl as cl
import pyopencl.array as cla
from gpyfft.fft import FFT
import FrogClKernels

reload(sp)
reload(FrogClKernels)

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
    def __init__(self, use_gpu=False):

        self.dtype_c = np.complex64     # datatype for complex values
        self.dtype_r = np.float32       # datatype for real values

        self.dw = None      # Frequency resolution for frog trace
        self.w = None       # Frequency vector for frog trace
        self.w0 = None      # Central frequency for frog trace
        self.dt = None      # Time resolution for frog trace
        self.t = None       # Time vector for frog trace
        self.tau = None     # Time shift vector
        self.n_frog = None       # Size of the frog trace
        self.Et = None          # Electric field vs time vector
        self.I_w_tau = None     # Measured FROG image, intensity for frequency and time shift
        self.shiftinds = None       # Generated index vector for generating time shifted electric fields
        self.shiftinds_neg = None
        self.shiftinds_pos = None
        self.Esig_t_tau = None      # FROG signal electric field vs time and time shift tau
        self.Et_mat = None          # Matrix of repeated electric field
        self.Et_mat_tau = None      # Matrix of electric field vs time and time shift tau
        self.Esig_w_tau = None      # FROG signal electric field vs frequency and time shift tau
        self.Esig_w_tau_p = None    # FROG signal electric field vs frequency and time shift after intensity projection
        self.Esig_t_tau_p = None    # FROG signal electric field vs time and time shift after intensity projection
        self.dZ = None              # Gradient of the functional distance Z
        self.alpha = []             # List recording of the distance moved in the dZ direction each iteration
        self.G_hist = []            # List recording of the reconstruction error each iteration

        self.phase_thr = 0.1        # Threshold amplitude for phase information methods (get_phase etc.)

        self.use_gpu = use_gpu
        self.ctx = None             # OpenCL context
        self.q = None               # OpenCL queue
        self.progs = None           # OpenCL kernels
        self.Et_cla = None
        self.Esig_t_tau_cla = None
        self.Esig_w_tau_cla = None
        self.Esig_w_tau_p = None
        self.Esig_t_tau_p_cla = None
        self.Esig_t_tau_p_fft = None
        self.Esig_w_tau_fft = None
        self.I_w_tau_cla = None
        self.dZ_cla = None
        self.X0_cla = None
        self.X1_cla = None
        self.X2_cla = None
        self.X3_cla = None
        self.X4_cla = None
        self.X5_cla = None
        self.X6_cla = None

        self.init_cl(self.use_gpu)

    def init_cl(self, use_gpu=False):
        root.debug('Initializing opencl')
        pl = cl.get_platforms()
        d = None
        v = None
        root.debug(''.join(('Found ', str(pl.__len__()), ' platforms')))
        vendorDict = {'amd': 3, 'nvidia': 2, 'intel': 1}
        if use_gpu is True:
            for p in pl:
                root.debug(p.vendor.lower())
                if 'amd' in p.vendor.lower():
                    vTmp = 'amd'
                elif 'nvidia' in p.vendor.lower():
                    vTmp = 'nvidia'
                else:
                    vTmp = 'intel'

                if v is None:
                    d = p.get_devices()
                    v = vTmp
                else:
                    if vendorDict[vTmp] > vendorDict[v]:
                        d = p.get_devices()
                        v = vTmp
        else:
            for p in pl:
                if 'amd' in p.vendor.lower():
                    vTmp = 'amd'
                elif 'nvidia' in p.vendor.lower():
                    vTmp = 'nvidia'
                else:
                    vTmp = 'intel'
                d = p.get_devices(device_type=cl.device_type.CPU)
                if not d == []:
                    v = vTmp
                    break
        root.debug(''.join(('Using device ', str(d), ' from ', v)))
        self.ctx = cl.Context(devices=d)
        self.q = cl.CommandQueue(self.ctx)

        self.progs = FrogClKernels.FrogClKernels(self.ctx)

    def initClBuffers(self):
        self.Et_cla = cla.to_device(self.q, self.Et)

        self.Esig_t_tau = np.zeros((self.n_frog, self.n_frog), self.dtype_c)
        self.Esig_t_tau_cla = cla.to_device(self.q, self.Esig_t_tau)

        self.Esig_w_tau = np.zeros((self.n_frog, self.n_frog), self.dtype_c)
        self.Esig_w_tau_cla = cla.to_device(self.q, self.Esig_w_tau)

        self.Esig_w_tau_fft = FFT(self.ctx, self.q, (self.Esig_t_tau_cla,), (self.Esig_w_tau_cla,), axes=[1])

        if self.I_w_tau is None:
            self.I_w_tau = np.zeros((self.n_frog, self.n_frog), dtype=self.dtype_r)
        self.I_w_tau_cla = cla.to_device(self.q, self.I_w_tau)

        self.Esig_t_tau_p = np.zeros((self.n_frog, self.n_frog), self.dtype_c)
        self.Esig_t_tau_p_cla = cla.to_device(self.q, self.Esig_t_tau_p)

        self.Esig_t_tau_p_fft = FFT(self.ctx, self.q, (self.Esig_w_tau_cla,), (self.Esig_t_tau_p_cla,), axes=[1])

        self.initClBuffersGP()

    def initClBuffersGP(self):
        # Gradient vector for the functional distance in the generalized projection
        self.dZ_cla = cla.zeros(self.q, (self.n_frog), self.dtype_c)

        # Vector for intermediate results for the error minimization calculation
        self.X0_cla = cla.zeros(self.q, (self.n_frog), self.dtype_r)
        self.X1_cla = cla.zeros(self.q, (self.n_frog), self.dtype_r)
        self.X2_cla = cla.zeros(self.q, (self.n_frog), self.dtype_r)
        self.X3_cla = cla.zeros(self.q, (self.n_frog), self.dtype_r)
        self.X4_cla = cla.zeros(self.q, (self.n_frog), self.dtype_r)
        self.X5_cla = cla.zeros(self.q, (self.n_frog), self.dtype_r)
        self.X6_cla = cla.zeros(self.q, (self.n_frog), self.dtype_r)

        self.Esig_t_tau_norm = np.zeros((self.n_frog, self.n_frog), self.dtype_r)
        self.Esig_t_tau_norm_cla = cla.to_device(self.q, self.Esig_t_tau_norm)

    def init_pulsefield_random(self, n_t, t_res, l_center, tau_pulse=100e-15, seed=0):
        """ Initiate signal field with parameters:
        t_res: time resolution of the reconstruction
        n_t: number of points in time and wavelength axes
        l_center: center wavelength

        Creates the following variables:
        self.w
        self.dw
        self.t
        self.dt
        self.tau
        self.Et
        """
        np.random.seed(seed)
        t_span = n_t * t_res

        # Now we calculate the frequency resolution required by the
        # time span
        w_res = 2 * np.pi / t_span

        # Frequency span is given by the time resolution
        f_max = 1 / (2 * t_res)
        w_span = f_max * 2 * 2 * np.pi

        c = 299792458.0
        w0 = 2 * np.pi * c / l_center   # Keep this so that the frog trace is centered in the field of view
        # w0 = w0 / 2 * (1 + np.sqrt(1 + w_span**2 / 4 / (w0/2)**2))
        w_spectrum = np.linspace(-w_span / 2, -w_span / 2 + w_res * n_t, n_t)
        self.dw = w_res
        self.w = w_spectrum
        self.w0 = w0        # This is the central frequency of the frog trace, not the electric field

        # Create time vector
        self.dt = t_res
        self.t = np.linspace(-t_span / 2, t_span / 2, n_t)

        self.tau = self.t

        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))

        # self.Et = np.exp(-self.t ** 2 / tau_pulse ** 2 + 1j * 2 * np.pi * np.random.rand(n_t)).astype(self.dtype_c)
        self.Et = (np.random.rand(n_t) * np.exp(1j * 2 * np.pi * np.random.rand(n_t))).astype(self.dtype_c)

        self.n_frog = n_t
        self.initClBuffers()

        root.info('Finished')

    def init_pulsefield_perfect(self, n_t, t_res, l_center, tau_pulse=100e-15):
        """ Initiate signal field with parameters:
        t_res: time resolution of the reconstruction
        n_t: number of points in time and wavelength axes
        l_center: center wavelength

        Creates the following variables:
        self.w
        self.dw
        self.t
        self.dt
        self.tau
        self.Et
        """
        t_span = n_t * t_res

        # Now we calculate the frequency resolution required by the
        # time span
        w_res = 2 * np.pi / t_span

        # Frequency span is given by the time resolution
        f_max = 1 / (2 * t_res)
        w_span = f_max * 2 * 2 * np.pi

        c = 299792458.0
        w0 = 2 * np.pi * c / l_center
        #        w_spectrum = np.linspace(w0-w_span/2, w0+w_span/2, n_t)
        w_spectrum = np.linspace(-w_span / 2, -w_span / 2 + w_res * n_t, n_t)
        self.dw = w_res
        self.w = w_spectrum
        self.w0 = w0

        # Create time vector
        self.dt = t_res
        self.t = np.linspace(-t_span / 2, t_span / 2, n_t)

        p = sp.SimulatedFrogTrace(n_t, dt_frog, l0=l_center, tau=tau_pulse)
        p.pulse.generateGaussian(tau_pulse)

        self.tau = self.t

        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))

        # Finally calculate a gaussian E-field from the
        self.Et = p.pulse.Et
        self.t = p.pulse.t
        self.Et_cla = cla.to_device(self.q, self.Et)

        self.init_shiftind(n_t)

        self.n_frog = n_t

        root.info('Finished')

    def init_shiftind(self, n_t):
        """
        Generate shiftind matrixes for time shifting operations
        :param n_t:
        :return:
        """
        i = np.arange(n_t * n_t)
        i2 = np.arange(n_t).repeat(n_t)

        self.shiftinds = (i + i2 - n_t / 2) % n_t + i2 * n_t
        self.shiftinds_neg = (i + i2 - n_t / 2) % n_t + i2 * n_t
        self.shiftinds_pos = (-n_t / 2 + i - i2) % n_t + i2 * n_t

    def load_frog_trace2(self, filename, thr=0.0, l_start_pixel=0, l_stop_pixel=-1, t_start_pixel=0, t_stop_pixel=-1):
        """
        Load a frog trace image from file and condition it to the internal w, tau grid (by calling
        condition_frog_trace). The variables self.w and self.tau must be set up first (be e.g. calling
        one of the init_pulsefield functions). The method needs three files: the image png filename_image.png
        (with time delay as the first dimension), _timevector.txt (time delay for each row in the image), and
        _wavelengthvector.txt (wavelength for each column in the image)

        :param filename: Filename of the image file, ending in _image.png
        :param thr:
        :param l_start_pixel:
        :param l_stop_pixel:
        :param t_start_pixel:
        :param t_stop_pixel:
        :return:
        """
        f_name_root = '_'.join((filename.split('_')[:-1]))
        root.debug(f_name_root)
        t_data = np.loadtxt(''.join((f_name_root, '_timevector.txt')))
        t_data = t_data - t_data.mean()
        l_data = np.loadtxt(''.join((f_name_root, '_wavelengthvector.txt')))
        if l_data[0] > 1:
            l_data = l_data * 1e-9
        pic = np.float32(imread(''.join((f_name_root, '_image.png'))))
        pic_n = pic / pic.max()

        if t_stop_pixel == -1:
            t_stop_pixel = pic_n.shape[0] - 1
        if l_stop_pixel == -1:
            l_stop_pixel = pic_n.shape[1] - 1

        picF = self.filter_frog_trace(pic_n, 3, thr)

        self.condition_frog_trace2(picF[t_start_pixel:t_stop_pixel, l_start_pixel:l_stop_pixel],
                                   l_data[l_start_pixel], l_data[l_stop_pixel], t_data[t_start_pixel],
                                   t_data[t_stop_pixel], self.Et.shape[0], thr, False)

    def condition_frog_trace2(self, Idata, l_start, l_stop, tau_start, tau_stop, n_frog=256, thr=0.15, filter_img=True):
        """ Take the measured intensity data and interpolate it to the
        internal w, tau grid. The variables self.w and self.tau must be set up
        first (be e.g. calling one of the init_pulsefield functions).

        Idata.shape[0] = number of tau points
        Idata.shape[1] = number of spectrum points
        """
        if self.n_frog != n_frog:
            self.n_frog = n_frog
            l0 = 2 * np.pi * 299792458.0 / self.w0
            self.init_pulsefield_random(n_frog, self.dt, l0)
        # Setup trace frequency and time parameters
        tau_data = np.linspace(tau_start, tau_stop, Idata.shape[0])
        l_data = np.linspace(l_start, l_stop, Idata.shape[1])
        l_center = (l_start + l_stop)/2
        c = 299792458.0
        w0 = 2*np.pi*c/l_center

        w_data = 2 * np.pi * c / l_data[:].copy()
        Idata_i = Idata.copy()
        if filter_img is True:
            Idata_i = self.filter_frog_trace(Idata_i / Idata_i.max(), 3, thr)

        # Idata_i = np.flipud(Idata_i)
        # Fine center of gravity time delay for the frog trace time marginal
        Idata_t = Idata_i.sum(1)
        tau_center = np.trapz(tau_data * Idata_t) / np.trapz(Idata_t)
        tau_data -= tau_center      # Correct for off center time delay center of gravity
        root.debug("Found tau center at {0:.1f} fs".format(tau_center*1e15))

        # Find center wavelength
        Idata_l = Idata_i.sum(0)
        l_center = np.trapz(l_data * Idata_l) / np.trapz(Idata_l)

        # Find time resolution
        t_res = (tau_stop - tau_start) / n_frog
        # Generate a suitable time-frequency grid and start pulse
        # self.init_pulsefield_random(n_frog, t_res, l_center)

        root.info(''.join(('Interpolating frog trace to ', str(self.tau.shape[0]), 'x', str(self.w.shape[0]))))
        # Interpolate the values for the points in the reconstruction matrix
        # We shift the frequencies by the central frequency to make sure images are aligned.
        # Then fftshift is needed due to the way fft sorts its frequency vector (first positive frequencies
        # then the negative frequencies in the end)
        t0 = time.clock()
        Itmp_w = np.zeros((Idata_i.shape[0], n_frog))
        I_w_tau = np.zeros((n_frog, n_frog))
        # First do interpolation in w direction for each tau delay:
        for ind in range(Idata_i.shape[0]):
            Idata_interp = interp1d(w_data, Idata_i[ind, :], kind='linear', fill_value=0.0, bounds_error=False)
            Itmp_w[ind, :] = Idata_interp(self.w0 + self.w)
        # Then interpolate in tau direction using the newly constructed intensity matrix
        for ind in range(n_frog):
            Idata_interp = interp1d(tau_data, Itmp_w[:, ind], kind='linear', fill_value=0.0, bounds_error=False)
            I_w_tau[:, ind] = Idata_interp(self.tau)
        if filter_img is True:
            I_w_tau = self.filter_frog_trace(I_w_tau, 3, thr)
        I_w_tau = np.fft.fftshift(np.maximum(I_w_tau, 0.0), axes=1).astype(self.dtype_r)
        # self.I_w_tau = self.filter_frog_trace(I_w_tau, 3, thr)
        self.I_w_tau = I_w_tau
        self.I_w_tau /= self.I_w_tau.max()

        self.I_w_tau_cla = cla.to_device(self.q, self.I_w_tau)

        root.info(''.join(('Time spent: ', str(time.clock() - t0))))

        return Idata_i, w_data, tau_data

    def create_frog_trace_gaussian(self, n_t, t_res, l_center, tau_pulse, b=0, c=0, algo='SHG'):
        """ Initiate signal field with parameters:
        t_res: time resolution of the reconstruction
        n_t: number of points in time and wavelength axes
        l_center: center wavelength

        Creates the following variables:
        self.w
        self.dw
        self.t
        self.dt
        self.tau
        self.Et
        """
        t_span = n_t * t_res

        # Now we calculate the frequency resolution required by the
        # time span
        w_res = 2 * np.pi / t_span

        # Frequency span is given by the time resolution
        f_max = 1 / (2 * t_res)
        w_span = f_max * 2 * 2 * np.pi

        c_v = 299792458.0
        w0 = 2 * np.pi * c_v / l_center
        w_spectrum = np.linspace(-w_span / 2, -w_span / 2 + w_res * n_t, n_t)
        self.dw = w_res
        self.w = w_spectrum
        self.w0 = w0

        # Create time vector
        self.dt = t_res
        self.t = np.linspace(-t_span / 2, t_span / 2, n_t)

        # Et = np.exp(-(self.t/tau_pulse)**2).astype(self.dtype_c)
        Et = np.exp(-self.t**2 / tau_pulse**2 + 1j * (b * self.t**2 + c * self.t**3))
        Et_mat = np.tile(Et, (n_t, 1))  # Repeat Et into a matrix

        Et_mat_tau = np.zeros_like(Et_mat)
        shiftVec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shiftVec):
            if sh < 0:
                Et_mat_tau[ind, 0:n_t + sh] = Et[-sh:]
            else:
                Et_mat_tau[ind, sh:] = Et[0:n_t - sh]

        # Create signal trace in time:
        if algo == 'SHG':
            Esig_t_tau = Et_mat * Et_mat_tau
        elif algo == 'SD':
            Esig_t_tau = Et_mat**2 * np.conj(Et_mat_tau)
        elif algo == 'PG':
            Esig_t_tau = Et_mat * Et_mat_tau * np.conj(Et_mat_tau)

        # Convert to frequency - tau space
        Esig_w_tau = np.fft.fft(Esig_t_tau, axis=1).astype(self.dtype_c)
        # Store the intensity matrix as the frog trace
        self.I_w_tau = np.abs(Esig_w_tau)**2

    def create_frog_trace_gaussian_spectral(self, n_t, t_res, l_center, tau_pulse, b=0, c=0, algo='SHG'):
        """ Initiate signal field with parameters:
        t_res: time resolution of the reconstruction
        n_t: number of points in time and wavelength axes
        l_center: center wavelength

        Creates the following variables:
        self.w
        self.dw
        self.t
        self.dt
        self.tau
        self.Et
        """
        t_span = n_t * t_res

        # Now we calculate the frequency resolution required by the
        # time span
        w_res = 2 * np.pi / t_span

        # Frequency span is given by the time resolution
        f_max = 1 / (2 * t_res)
        w_span = f_max * 2 * 2 * np.pi

        c_v = 299792458.0
        w0 = 2 * np.pi * c_v / l_center
        #        w_spectrum = np.linspace(w0-w_span/2, w0+w_span/2, n_t)
        w_spectrum = np.linspace(-w_span / 2, -w_span / 2 + w_res * n_t, n_t)
        self.dw = w_res
        self.w = w_spectrum
        self.w0 = w0

        # Create time vector
        self.dt = t_res
        self.t = np.linspace(-t_span / 2, t_span / 2, n_t)

        # Et = np.exp(-(self.t/tau_pulse)**2).astype(self.dtype_c)
        dw = 2 * np.pi * 0.441 / tau_pulse
        ph = 0.0
        Eenv_w = np.exp(-self.w**2 / dw**2 + 1j * (b * self.w**2 + c * self.w**3) + ph)
        Et = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Eenv_w)))
        Et = Et / np.abs(Et).max()
        Et_mat = np.tile(Et, (n_t, 1))  # Repeat Et into a matrix

        Et_mat_tau = np.zeros_like(Et_mat)
        shiftVec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shiftVec):
            if sh < 0:
                Et_mat_tau[ind, 0:n_t + sh] = Et[-sh:]
            else:
                Et_mat_tau[ind, sh:] = Et[0:n_t - sh]

        # Create signal trace in time:
        if algo == 'SHG':
            Esig_t_tau = Et_mat * Et_mat_tau
        elif algo == 'SD':
            Esig_t_tau = Et_mat**2 * np.conj(Et_mat_tau)
        elif algo == 'PG':
            Esig_t_tau = Et_mat * Et_mat_tau * np.conj(Et_mat_tau)

        # Convert to frequency - tau space
        Esig_w_tau = np.fft.fft(Esig_t_tau, axis=1).astype(self.dtype_c)
        # Store the intensity matrix as the frog trace
        self.I_w_tau = np.abs(Esig_w_tau)**2

    @staticmethod
    def filter_frog_trace(Idata, kernel=5, thr=0.1):
        Idata_f = medfilt2d(Idata, kernel) - thr
        Idata_f[Idata_f < 0.0] = 0.0
        return Idata_f

    def generate_Esig_t_tau(self, algo='SHG'):
        """ Generate the time shifted E-field matrix for the SD process.

        Output:
        self.Esig_t_tau, a n_t x n_tau matrix where each row is Esig(t,tau)
        """
        root.debug('Generating new Esig_t_tau from ' + str(algo))
        t0 = time.clock()

        if algo == 'SHG':
            krn = self.progs.progs['generateEsig_t_tau_SHG'].generateEsig_t_tau_SHG
            krn.set_scalar_arg_dtypes((None, None, np.int32))
            krn.set_args(self.Et_cla.data, self.Esig_t_tau_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_t_tau.shape, None)
            ev.wait()
        elif algo == 'SD':
            krn = self.progs.progs['generateEsig_t_tau_SD'].generateEsig_t_tau_SD
            krn.set_scalar_arg_dtypes((None, None, np.int32))
            krn.set_args(self.Et_cla.data, self.Esig_t_tau_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_t_tau.shape, None)
            ev.wait()
        elif algo == 'PG':
            krn = self.progs.progs['generateEsig_t_tau_PG'].generateEsig_t_tau_PG
            krn.set_scalar_arg_dtypes((None, None, np.int32))
            krn.set_args(self.Et_cla.data, self.Esig_t_tau_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_t_tau.shape, None)
            ev.wait()
        else:
            raise ValueError("Unknown algorithm, should be SHG, SD, or PG")

        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def generate_Esig_w_tau(self):
        """
        Generate the electric field vs frequency and time shift from Esig_t_tau through FFT of each time row.

        :return:
        """
        root.debug('Generating Esig_w_tau')
        tic = time.clock()
        # self.Esig_t_tau_cla = cla.to_device(self.q, np.fft.fftshift(self.Esig_t_tau_cla.get(), axes=1))
        events = self.Esig_w_tau_fft.enqueue()
        for e in events:
            e.wait()
            #         if self.rollFFT == True:
            #             krn = self.progs.progs['rollEsigWTau'].rollEsigWTau
            #             krn.set_scalar_arg_dtypes((None, np.int32))
            #             krn.set_args(self.Esig_w_tau_cla.data, self.N)
            #             ev = cl.enqueue_nd_range_kernel(self.q, krn, [self.Esig_w_tau.shape[0]], None)
            #             ev.wait()

        toc = time.clock()
        root.debug(''.join(('Time spent: ', str(toc - tic))))

    def apply_intensity_data(self, I_w_tau=None):
        """
        Overwrite the magnitude of the generated electric field vs frequency and time shift
         with the measured intensity data.

        :param I_w_tau: Intensity data to overwrite magnitude of generated electric field. If None
         the intensity data stored in the class instance is used.
        :return:
        """
        root.debug('Applying intensity data from experiment')
        t0 = time.clock()
        if I_w_tau is not None:
            self.I_w_tau_cla = cla.to_device(self.q, I_w_tau)
        krn = self.progs.progs['applyIntensityData'].applyIntensityData
        krn.set_scalar_arg_dtypes((None, None, np.int32))
        krn.set_args(self.Esig_w_tau_cla.data, self.I_w_tau_cla.data, self.n_frog)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_w_tau.shape, None)
        ev.wait()

        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def update_Et_vanilla(self, algo='SHG'):
        """
        Calculate electric field vs time and time shift from Esig_w_tau through inverse FFT.
        Then generate electric field vs time vector by summing all time shift rows and normalizing.

        :return:
        """
        root.debug('Updating Et using vanilla algorithm')
        t0 = time.clock()
        self.Esig_t_tau_p = np.fft.ifft(self.Esig_w_tau_p, axis=1).astype(self.dtype_c)
        if algo == 'SHG' or algo == 'PG':
            self.Et = self.Esig_t_tau_p.sum(axis=0)
        elif algo == 'SD':
            # Change variables so that we have Esig(t+tau, tau) and integrate in tau direction
            Esig = self.Esig_t_tau_p
            self.Et = np.conj(Esig.sum(axis=0))/np.abs(self.Et.sum())
        else:
            raise ValueError("Unknown algorithm")
        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def update_Et_gp(self, algo='SHG'):
        root.debug('Updating Et using GP algorithm')
        t0 = time.clock()
        events = self.Esig_t_tau_p_fft.enqueue(forward=False)
        for e in events:
            e.wait()
        # self.Esig_t_tau_p_cla = cla.to_device(self.q, np.fft.fftshift(self.Esig_t_tau_p_cla.get(), axes=1))
        root.debug(''.join(('iFFT time spent: ', str(time.clock() - t0))))

        tic = time.clock()
        self.grad_z(algo)
        root.debug(''.join(('GradZ time spent: ', str(time.clock() - tic))))
        tic = time.clock()
        alpha = self.min_z(algo)
        root.debug(''.join(('MinZ time spent: ', str(time.clock() - tic))))

        root.debug(''.join(('Minimum found: ', str(alpha))))

        if abs(alpha) > 1e6:
            root.debug(''.join(('Alpha too large, using old value', str(self.alpha[-1]))))
            alpha = self.alpha[-1]
        self.alpha.append(alpha)

        tic = time.clock()
        krn = self.progs.progs['updateEtGP'].updateEtGP
        krn.set_scalar_arg_dtypes((None, None, self.dtype_r, np.int32))
        krn.set_args(self.Et_cla.data, self.dZ_cla.data, alpha, self.n_frog)
        ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
        ev.wait()
        root.debug(''.join(('Update Et time spent: ', str(time.clock() - tic))))
        root.debug(''.join(('Total update time spent: ', str(time.clock() - t0))))

    def grad_z(self, algo='SHG'):
        tic = time.clock()

        if algo == 'SHG':
            root.debug('Calculating dZ for SHG using gpu')
            krn = self.progs.progs['gradZSHG'].gradZSHG
            krn.set_scalar_arg_dtypes((None, None, None, np.int32))
            krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.dZ_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
            ev.wait()

        elif algo == 'SD':
            root.debug('Calculating dZ for SD using gpu')
            krn = self.progs.progs['gradZSD'].gradZSD
            krn.set_scalar_arg_dtypes((None, None, None, np.int32))
            krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.dZ_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
            ev.wait()

        elif algo == 'PG':
            root.debug('Calculating dZ for PG using gpu')
            krn = self.progs.progs['gradZPG'].gradZPG
            krn.set_scalar_arg_dtypes((None, None, None, np.int32))
            krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.dZ_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
            ev.wait()
        else:
            raise ValueError(''.join(('Algorithm ', str(algo), ' not valid.')))

        toc = time.clock()
        root.debug(''.join(('Grad_Z  time spent: ', str(toc - tic))))

    def min_z(self, algo='SHG'):
        if algo == 'SHG':
            krn = self.progs.progs['minZerrSHG'].minZerrSHG
            krn.set_scalar_arg_dtypes((None, None, None, None, None, None, None, None, np.int32))
            krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.dZ_cla.data,
                         self.X0_cla.data, self.X1_cla.data, self.X2_cla.data, self.X3_cla.data, self.X4_cla.data,
                         self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
            ev.wait()

            krn = self.progs.progs['normEsig'].normEsig
            krn.set_scalar_arg_dtypes((None, None, np.int32))
            krn.set_args(self.Esig_t_tau_p_cla.data, self.Esig_t_tau_norm_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_t_tau_p.shape, None)
            ev.wait()
            mx = cla.max(self.Esig_t_tau_norm_cla).get() * self.n_frog * self.n_frog

            X0 = cla.sum(self.X0_cla, queue=self.q).get() / mx
            X1 = cla.sum(self.X1_cla, queue=self.q).get() / mx
            X2 = cla.sum(self.X2_cla, queue=self.q).get() / mx
            X3 = cla.sum(self.X3_cla, queue=self.q).get() / mx
            X4 = cla.sum(self.X4_cla, queue=self.q).get() / mx

            root.debug(''.join(('Poly: ', str(X4), ' x^4 + ', str(X3), ' x^3 + ', str(X2), ' x^2 + ',
                                str(X1), ' x + ', str(X0))))
            # Polynomial in dZ (expansion of differential)
            X = np.array([X0, X1, X2, X3, X4]).astype(np.double)

        elif algo == 'SD':
            krn = self.progs.progs['minZerrSD'].minZerrSD
            krn.set_scalar_arg_dtypes((None, None, None, None, None, None, None, None, None, None, np.int32))
            krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.dZ_cla.data,
                         self.X0_cla.data, self.X1_cla.data, self.X2_cla.data, self.X3_cla.data,
                         self.X4_cla.data, self.X5_cla.data, self.X6_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
            ev.wait()

            krn = self.progs.progs['normEsig'].normEsig
            krn.set_scalar_arg_dtypes((None, None, np.int32))
            krn.set_args(self.Esig_t_tau_p_cla.data, self.Esig_t_tau_norm_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_t_tau_p.shape, None)
            ev.wait()
            mx = cla.max(self.Esig_t_tau_norm_cla).get() * self.n_frog * self.n_frog

            X0 = cla.sum(self.X0_cla, queue=self.q).get() / mx
            X1 = cla.sum(self.X1_cla, queue=self.q).get() / mx
            X2 = cla.sum(self.X2_cla, queue=self.q).get() / mx
            X3 = cla.sum(self.X3_cla, queue=self.q).get() / mx
            X4 = cla.sum(self.X4_cla, queue=self.q).get() / mx
            X5 = cla.sum(self.X5_cla, queue=self.q).get() / mx
            X6 = cla.sum(self.X6_cla, queue=self.q).get() / mx

            root.debug(
                ''.join(('Poly: ', str(X6), ' x^6 + ', str(X5), ' x^5 + ', str(X4), ' x^4 + ', str(X3), ' x^3 + ',
                         str(X2), ' x^2 + ', str(X1), ' x + ', str(X0))))
            # Polynomial in dZ (expansion of differential)
            X = np.array([X0, X1, X2, X3, X4, X5, X6]).astype(np.double)

        elif algo == 'PG':
            tic = time.clock()
            krn = self.progs.progs['minZerrPG'].minZerrPG
            krn.set_scalar_arg_dtypes((None, None, None, None, None, None, None, None, None, None, np.int32))
            krn.set_args(self.Esig_t_tau_p_cla.data, self.Et_cla.data, self.dZ_cla.data,
                         self.X0_cla.data, self.X1_cla.data, self.X2_cla.data, self.X3_cla.data,
                         self.X4_cla.data, self.X5_cla.data, self.X6_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Et.shape, None)
            ev.wait()
            root.debug(''.join(('minZerrPG time spent: ', str(time.clock() - tic))))

            tic = time.clock()
            krn = self.progs.progs['normEsig'].normEsig
            krn.set_scalar_arg_dtypes((None, None, np.int32))
            krn.set_args(self.Esig_t_tau_p_cla.data, self.Esig_t_tau_norm_cla.data, self.n_frog)
            ev = cl.enqueue_nd_range_kernel(self.q, krn, self.Esig_t_tau_p.shape, None)
            ev.wait()
            root.debug(''.join(('NormEsig kernel time spent: ', str(time.clock() - tic))))
            tic = time.clock()
            mx = cla.max(self.Esig_t_tau_norm_cla).get() / (self.n_frog * self.n_frog)
            # mx = (np.abs(self.Esig_t_tau_p_cla.get()) ** 2).max() / (self.n_frog * self.n_frog)
            root.debug(''.join(('NormEsig max time spent: ', str(time.clock() - tic))))

            tic = time.clock()
            X0 = cla.sum(self.X0_cla, queue=self.q).get() / mx
            X1 = cla.sum(self.X1_cla, queue=self.q).get() / mx
            X2 = cla.sum(self.X2_cla, queue=self.q).get() / mx
            X3 = cla.sum(self.X3_cla, queue=self.q).get() / mx
            X4 = cla.sum(self.X4_cla, queue=self.q).get() / mx
            X5 = cla.sum(self.X5_cla, queue=self.q).get() / mx
            X6 = cla.sum(self.X6_cla, queue=self.q).get() / mx
            root.debug(''.join(('X poly sum time spent: ', str(time.clock() - tic))))

            root.debug(
                ''.join(('Poly: ', str(X0), ' x^6 + ', str(X1), ' x^5 + ', str(X2), ' x^4 + ', str(X3), ' x^3 + ',
                         str(X4), ' x^2 + ', str(X5), ' x + ', str(X6))))
            # Polynomial in dZ (expansion of differential)
            X = np.array([X0, X1, X2, X3, X4, X5, X6]).astype(np.double)
        else:
            raise ValueError(''.join(('Algorithm ', str(algo), ' not valid.')))

        root.debug(''.join(('Esig_t_tau_p norm max: ', str(mx))))

        tic = time.clock()
        z_poly = np.polyder(X)
        r = np.roots(z_poly)
        # Only the real roots are considered (we want to go along the direction of dZ so alpha is a real number):
        r_real = r[np.abs(r.imag) < 1e-9].real
        root.debug(''.join(('Real roots: ', str(r_real))))

        # Evaluate the polynomial for these roots and find the minimum Z:
        Z1 = np.polyval(X, r_real)
        min_z_ind = Z1.argmin()

        toc = time.clock()
        root.debug(''.join(('Root finding  time spent: ', str(toc - tic))))

        alpha = r_real[min_z_ind].astype(self.dtype_r)

        return alpha

    def center_peaktime(self):
        """
        Center the reconstructed E-field E(t) maximum value in time.

        :return:
        """
        Et = self.Et_cla.get()
        ind = np.argmax(abs(Et))
        shift = (Et.shape[0] / 2 - ind).astype(np.int)
        Et = np.roll(Et, shift)
        self.Et_cla.set(Et)

    def calc_reconstruction_error(self):
        """
        Calculate the reconstruction error as the sum of the squared difference between experimental frog trace image
        I_w_tau and reconstructed frog image. The calculated error is appended to the class variable G_hist where
        the error history is stored as a list.

        :return: Reconstruction error G
        """
        root.debug('Calculating reconstruction error')
        tic = time.clock()
        Esig_w_tau = self.Esig_w_tau_cla.get()
        I_rec_w_tau = np.real(Esig_w_tau * np.conj(Esig_w_tau))
        I_w_tau = self.I_w_tau_cla.get()
        mx = I_w_tau.max()
        my = (I_w_tau * I_rec_w_tau).sum() / (I_rec_w_tau**2).sum()
        root.debug(''.join(('My=', str(my))))
        # my = 1.0
        G = np.sqrt(((I_w_tau - my * I_rec_w_tau) ** 2).sum() / (I_rec_w_tau.shape[0] * I_rec_w_tau.shape[1])) / mx
        toc = time.clock()
        root.debug(''.join(('Time spent: ', str(toc - tic))))
        self.G_hist.append(G)
        return G

    def get_trace_abs(self, norm=True):
        """
        Retrieve the magnitude of the reconstructed E-field (Et). Use get_t for the
        corresponding time vector.

        :param norm: If true, the return vector is normalized
        :return: Vector of the magnitude of the E-field abs(E(t))
        """
        if self.Et_cla is not None:
            # Center peak in time
            Et = self.Et_cla.get()
            ind = np.argmax(abs(Et))
            shift = (Et.shape[0] / 2 - ind).astype(np.int)
            Et = np.abs(np.roll(Et, shift))
            if norm is True:
                Et /= Et.max()
        else:
            Et = None
        return Et

    def get_trace_spectral_abs(self, norm=True):
        """
        Retrieve the spectral magnitude of the reconstructed E-field (fft of Et). Use get_w for the
        corresponding angular frequency vector.

        :param norm: If true, the return vector is normalized
        :return: Vector of the magnitude of the E-field abs(E(w))
        """
        if self.Et_cla is not None:
            # Center peak in time
            Et = self.Et_cla.get()
            ind = np.argmax(abs(Et))
            shift = (Et.shape[0] / 2 - ind).astype(np.int)
            Et = np.roll(Et, shift)

            N = Et.shape[0]
            wf = np.fft.fftfreq(N)
            w_ind = np.argsort(wf)

            Ew = np.abs(np.fft.fft(np.fft.fftshift(Et))[w_ind])

            if norm is True:
                Ew /= Ew.max()
        else:
            Ew = None
        return Ew

    def get_trace_phase(self, linear_comp=False):
        """
        Retrieve the temporal phase of the reconstructed E-field. The phase is zero at the peak field and NaN
        where the field magnitude is lower than the threshold phase_thr (class variable). Use get_t for the
        corresponding time vector.

        :param linear_comp: If true, the linear part of the phase (i.e. frequency shift) if removed
        :return: Temporal phase vector.
        """
        eps = self.phase_thr

        if self.Et_cla is not None:
            # Center peak in time
            Et = self.Et_cla.get()
            ind = np.argmax(abs(Et))
            shift = Et.shape[0] / 2 - ind
            Et = np.roll(Et, shift)

            # Unravelling 2*pi phase jumps
            ph0_ind = np.int(Et.shape[0] / 2)           # Center index
            ph = np.angle(Et)
            ph_diff = np.diff(ph)
            # We need to sample often enough that the difference in phase is less than 5 rad
            # A larger jump is taken as a 2*pi phase jump
            ph_ind = np.where(np.abs(ph_diff) > 5.0)
            # Loop through the 2*pi phase jumps
            for ind in ph_ind[0]:
                if ph_diff[ind] < 0:
                    ph[ind + 1:] += 2 * np.pi
                else:
                    ph[ind + 1:] -= 2 * np.pi

            # Find relevant portion of the pulse (intensity above a threshold value)
            ph0 = ph[ph0_ind]
            Et_mag = np.abs(Et)
            low_ind = np.where(Et_mag < eps)
            ph[low_ind] = np.nan

            # Here we could go through contiguous regions and make the phase connect at the edges...

            # Linear compensation is we have a frequency shift (remove 1st order phase)
            if linear_comp is True:
                idx = np.isfinite(ph)
                x = np.arange(Et.shape[0])
                ph_poly = np.polyfit(x[idx], ph[idx], 1)
                ph_out = ph - np.polyval(ph_poly, x)
            else:
                ph_out = ph - ph0
        else:
            ph_out = None
        return ph_out

    def get_trace_spectral_phase(self, linear_comp=True):
        """
        Retrieve the spectral phase of the reconstructed E-field. The phase is zero at the peak field and NaN
        where the field magnitude is lower than the threshold phase_thr (class variable). Use get_w for the
        corresponding angular frequency vector.

        :param linear_comp: If true, the linear part of the phase (i.e. time shift) if removed
        :return: Spectral phase vector.
        """
        eps = self.phase_thr    # Threshold for intensity where we have signal

        # Check if there is a reconstructed field:
        if self.Et_cla is not None:
            Et = self.Et_cla.get()
            N = Et.shape[0]
            w_ind = np.argsort(np.fft.fftfreq(N))   # Sorted index vector to unravel the fft:d E-field vector

            # Center peak in time
            ind = np.argmax(abs(Et))
            shift = (Et.shape[0] / 2 - ind).astype(np.int)
            Et = np.roll(Et, shift)

            Ew = np.fft.fft(np.fft.fftshift(Et))[w_ind]
            # Normalize
            Ew /= abs(Ew).max()

            # Unravelling 2*pi phase jumps
            ph0_ind = np.argmax(abs(Ew))
            ph = np.angle(Ew)
            ph_diff = np.diff(ph)
            # We need to sample often enough that the difference in phase is less than 5 rad
            # A larger jump is taken as a 2*pi phase jump
            ph_ind = np.where(np.abs(ph_diff) > 5.0)
            # Loop through the 2*pi phase jumps
            for ind in ph_ind[0]:
                if ph_diff[ind] < 0:
                    ph[ind + 1:] += 2 * np.pi
                else:
                    ph[ind + 1:] -= 2 * np.pi

            # Find relevant portion of the pulse (intensity above a threshold value)
            Ew_mag = np.abs(Ew)
            low_ind = np.where(Ew_mag < eps)
            ph[low_ind] = np.nan

            # Here we could go through contiguous regions and make the phase connect at the edges...

            # Linear compensation is we have a frequency shift (remove 1st order phase)
            if linear_comp is True:
                idx = np.isfinite(ph)
                x = np.arange(N)
                ph_poly = np.polyfit(x[idx], ph[idx], 1)
                ph_out = ph - np.polyval(ph_poly, x)
            else:
                ph_out = ph
            ph_out -= ph_out[ph0_ind]
        else:
            ph_out = None
        return ph_out

    def get_t(self):
        """
        Retrieve the time vector for the reconstructed E-field
        :return: time vector
        """
        return self.t

    def get_w(self):
        """
        Retrieve the angular frequency vector for the reconstructed E-field
        :return: angular frequency vector
        """
        return self.w

    def get_trace_summary(self, domain='temporal'):
        """
        Calculate trace parameters such as intensity FWHM and phase difference over the trace region.
        :param domain: 'temporal' for time domain parameters,
                     'spectral' for frequency domain parameters
        :return:
        trace_fwhm: full width at half maximum of the intensity trace (E-field squared)
        delta_ph: phase difference (max-min) of the phase trace
        """
        if domain == 'temporal':
            Eabs = self.get_trace_abs()
            ph = self.get_trace_phase()
            x = self.get_t()
        else:
            Eabs = self.get_trace_spectral_abs()
            ph = self.get_trace_spectral_phase()
            x = self.get_w()
        It = Eabs**2

        # Calculate FWHM
        t_ind = np.where(np.diff(np.sign(It - 0.5)))[0]
        if t_ind.shape[0] > 1:
            trace_fwhm = x[t_ind[-1]] - x[t_ind[0]]
            ph_fwhm = ph[t_ind[0]:t_ind[-1]]
            ph_good = ph_fwhm[np.isfinite(ph_fwhm)]
            delta_ph = ph_good.max() - ph_good.min()
        else:
            trace_fwhm = np.nan
            delta_ph = np.nan
        return trace_fwhm, delta_ph

    def get_temporal_phase_expansion(self, orders=4, prefix=1e-15):
        """
        Calculate a polynomial fit to the retrieved phase curve as function of time (temporal phase)
        :param orders: Number of orders to include in the fit
        :param prefix: Factor that the time is scaled with before the fit (1e-15 => fs)
        :return: Polynomial coefficients, highest order first
        """
        if self.Et_cla is not None:
            t_v = self.get_t()
            ph = self.get_trace_phase()
            ph_ind = np.isfinite(ph)
            ph_good = ph[ph_ind]
            t_good = t_v[ph_ind] / prefix
            ph_poly = np.polyfit(t_good, ph_good, orders)
        else:
            ph_poly = None
        return ph_poly

    def get_spectral_phase_expansion(self, orders=4, prefix=1e12):
        """
        Calculate a polynomial fit to the retrieved phase curve as function of angular frequency (spectral phase)
        :param orders: Number of orders to include in the fit
        :param prefix: Factor that the angular frequency is scaled with before the fit (1e12 => Trad)
        :return: Polynomial coefficients, highest order first
        """
        if self.Et_cla is not None:
            w = self.w
            ph = self.get_trace_spectral_phase()
            ph_ind = np.isfinite(ph)
            ph_good = ph[ph_ind]
            w_good = w[ph_ind] / prefix
            ph_poly = np.polyfit(w_good, ph_good, orders)
        else:
            ph_poly = None
        return ph_poly

    def get_reconstructed_intensity(self):
        return np.abs(self.Esig_w_tau_cla.get()) ** 2

    def setup_vanilla_algorithm(self):
        if self.Esig_t_tau_cla is None:
            self.initClBuffers()

    def run_cycle_vanilla(self, cycles=1, algo='SHG'):
        root.debug('Starting FROG reconstruction cycle using the vanilla algorithm')

        t0 = time.clock()
        error = []
        self.setup_vanilla_algorithm()
        for c in range(cycles):
            root.debug(''.join(('Cycle ', str(c + 1), '/', str(cycles))))
            self.generate_Esig_t_tau(algo)
            self.generate_Esig_w_tau()
            G = self.calc_reconstruction_error()
            self.apply_intensity_data()
            self.update_Et_vanilla(algo)
            self.center_peaktime()
            root.debug('-------------------------------------------')
            root.debug(''.join(('Error G = ', str(G))))
            root.debug('-------------------------------------------')
            error.append(G)
        deltaT = time.clock() - t0
        root.debug(''.join(('Total runtime ', str(deltaT))))
        root.debug(''.join((str(cycles / deltaT), ' iterations/s')))
        return np.array(error)

    def setup_gp_algorithm(self):
        if self.Esig_t_tau_cla is None:
            self.initClBuffers()

    def run_cycle_gp(self, cycles=1, algo='SHG', center_time=True):
        root.debug('Starting FROG reconstruction cycle using the GP algorithm')
        t0 = time.clock()
        error = []
        self.setup_gp_algorithm()
        for c in range(cycles):
            root.debug(''.join(('Cycle ', str(c + 1), '/', str(cycles))))
            self.generate_Esig_t_tau(algo)
            self.generate_Esig_w_tau()
            G = self.calc_reconstruction_error()
            self.apply_intensity_data()
            self.update_Et_gp(algo)
            if center_time is True:
                self.center_peaktime()
            root.debug('-------------------------------------------')
            root.debug(''.join(('Error G = ', str(G))))
            root.debug('-------------------------------------------')
            error.append(G)
        deltaT = time.clock() - t0
        root.debug(''.join(('Total runtime ', str(deltaT))))
        root.debug(''.join((str(cycles / deltaT), ' iterations/s')))
        return np.array(error)


if __name__ == '__main__':
    n_frog = 256
    dt_frog = 20e-15
    l0_frog = 263.5e-9
    tau_fwhm = 100e-15

    p_sim = sp.SimulatedPulse(n_frog, dt_frog, l0_frog, tau_fwhm)
    p_sim.generateGaussianCubicSpectralPhase(0.5e-27, 1.0e-40)
    # p.generateGaussianCubicPhase(5e24, 3e39)
    # p.generateGaussian(tau_pulse)
    # p.generateDoublePulse(tau_pulse, deltaT=0.5e-12)
    gt = sp.SimulatedFrogTrace(n_frog, dt_frog, l0_frog)
    gt.pulse = p_sim
    IfrogSHG = gt.generateSHGTraceDt(n_frog, dt_frog, l0_frog / 2)
    IfrogSD = gt.generateSDTraceDt(n_frog, dt_frog, l0_frog)
    IfrogPG = gt.generatePGTraceDt(n_frog, dt_frog, l0_frog)
    #     Ifrog = gt.addNoise(0.01, 0.1)
    l = gt.getWavelengths()
    t = gt.getTimedelays()

    frog_cl = FrogCalculation(use_gpu=True)

    frog_cl.init_pulsefield_random(n_frog, dt_frog, l0_frog+0e-9)

    # frog.condition_frog_trace2(IfrogSHG, l[0], l[-1], t[0], t[-1], n_frog=N, thr=0)
    # frog.condition_frog_trace2(IfrogSD, l[0], l[-1], t[0], t[-1], n_frog=N, thr=0)
    # frog.condition_frog_trace2(IfrogPG, l[0], l[-1], t[0], t[-1], n_frog=N, thr=0)
    # frog.create_frog_trace_gaussian(N, dt, l0, tau_pulse, algo='SD')
    # frog.create_frog_trace_gaussian_spectral(N, dt, l0, tau_pulse, b=1e-27, c=0.5e-40, algo='PG')
    # frog.init_pulsefield_perfect(N, dt, l0, tau_pulse)
    # frog.I_w_tau = np.abs(frog.Esig_w_tau)**2
    # N = 128
    # dt = 6e-15
    # l0 = 263.5e-9
    # frog.init_pulsefield_random(N, dt, l0)
    # frog_cl.load_frog_trace2('./data/frogtrace_2017-03-13_17h35_uv_pg_67mm_image',
    #                          thr=0.65, l_start_pixel=0, l_stop_pixel=-1,
    #                          t_start_pixel=0, t_stop_pixel=-1)

    frog_cl.load_frog_trace2('./data/frogtrace_2017-03-13_17h35_uv_pg_67mm_image',
                             thr=0.63, l_start_pixel=0, l_stop_pixel=-1,
                             t_start_pixel=0, t_stop_pixel=-1)
    er = np.array([])
    er = frog_cl.run_cycle_gp(61, 'PG')
    # er = frog.run_cycle_gp(20, 'SD', roll_fft=False)
    # er = np.hstack((er, frog.run_cycle_gp(1, 'SHG', roll_fft=False)))
    # er = np.hstack((er, frog.run_cycle_gp(50, 'PG', roll_fft=True)))
