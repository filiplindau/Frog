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
        self.roll_fft = False

        self.dtype_c = np.complex64
        self.dtype_r = np.float32

        self.Esignal_w = None
        self.Esignal_t = None
        self.dw = None      # Frequency resolution for frog trace
        self.w = None       # Frequency vector for frog trace
        self.w0 = None      # Central frequency for frog trace
        self.dt = None      # Time resolution for frog trace
        self.t = None       # Time vector for frog trace
        self.tau = None     # Time shift vector
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
        self.dZ = None
        self.alpha = []
        self.G_hist = []

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
        self.Et = np.random.rand(n_t) * np.exp(1j * 2 * np.pi * np.random.rand(n_t)).astype(self.dtype_c)

        self.init_shiftind(n_t)

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

        p = sp.SimulatedFrogTrace(n_t, dt, l0=l_center, tau=tau_pulse)
        p.pulse.generateGaussian(tau_pulse)
        Ifrog = p.generateSHGTraceDt(n_t, t_res, l_center)

        self.tau = self.t

        root.info(''.join(('t_span ', str(t_span))))
        root.info(''.join(('t_res ', str(t_res))))

        # Finally calculate a gaussian E-field from the
        self.Et = p.pulse.Et
        self.t = p.pulse.t

        self.init_shiftind(n_t)

        root.info('Finished')

    def init_shiftind(self, n_t):
        i = np.arange(n_t * n_t)
        i2 = np.arange(n_t).repeat(n_t)

        self.shiftinds = (i + i2 - n_t / 2) % n_t + i2 * n_t
        self.shiftinds_neg = (i + i2 - n_t / 2) % n_t + i2 * n_t
        self.shiftinds_pos = (-n_t / 2 + i - i2) % n_t + i2 * n_t

    def load_frog_trace(self, filename, thr=0.0, l_start_pixel=0, l_stop_pixel=-1, t_start_pixel=0, t_stop_pixel=-1):
        f_name_root = '_'.join((filename.split('_')[0:3]))
        t_data = np.loadtxt(''.join((f_name_root, '_timevector.txt')))
        t_data = t_data - t_data.mean()
        l_data = np.loadtxt(''.join((f_name_root, '_wavelengthvector.txt'))) * 1e-9
        pic = np.float32(imread(''.join((f_name_root, '_image.png'))))
        pic_n = pic / pic.max()

        if t_stop_pixel == -1:
            t_stop_pixel = pic_n.shape[0] - 1
        if l_stop_pixel == -1:
            l_stop_pixel = pic_n.shape[1] - 1

        picF = self.filter_frog_trace(pic_n, 3, thr)

        self.condition_frog_trace(picF[t_start_pixel:t_stop_pixel, l_start_pixel:l_stop_pixel],
                                  l_data[l_start_pixel], l_data[l_stop_pixel], t_data[t_start_pixel],
                                  t_data[t_stop_pixel], thr)

    def load_frog_trace2(self, filename, thr=0.0, l_start_pixel=0, l_stop_pixel=-1, t_start_pixel=0, t_stop_pixel=-1):
        f_name_root = '_'.join((filename.split('_')[0:3]))
        t_data = np.loadtxt(''.join((f_name_root, '_timevector.txt')))
        t_data = t_data - t_data.mean()
        l_data = np.loadtxt(''.join((f_name_root, '_wavelengthvector.txt'))) * 1e-9
        pic = np.float32(imread(''.join((f_name_root, '_image.png'))))
        pic_n = pic / pic.max()

        if t_stop_pixel == -1:
            t_stop_pixel = pic_n.shape[0] - 1
        if l_stop_pixel == -1:
            l_stop_pixel = pic_n.shape[1] - 1

        picF = self.filter_frog_trace(pic_n, 3, thr)

        self.condition_frog_trace2(picF[t_start_pixel:t_stop_pixel, l_start_pixel:l_stop_pixel],
                                  l_data[l_start_pixel], l_data[l_stop_pixel], t_data[t_start_pixel],
                                  t_data[t_stop_pixel], self.Et.shape[0], thr)

    def condition_frog_trace(self, Idata, l_start, l_stop, tau_start, tau_stop, n_frog=256, thr=0.15):
        """ Take the measured intensity data and interpolate it to the
        internal w, tau grid. The variables self.w and self.tau must be set up
        first (be e.g. calling one of the init_pulsefield functions).

        Idata.shape[0] = number of tau points
        Idata.shape[1] = number of spectrum points
        """
        # Setup trace frequency and time parameters
        tau_data = np.linspace(tau_start, tau_stop, Idata.shape[0])
        l_data = np.linspace(l_start, l_stop, Idata.shape[1])
        if l_start > l_stop:
            w_start = 2 * np.pi * 299792458.0 / l_start
            w_stop = 2 * np.pi * 299792458.0 / l_stop
            w_data = 2 * np.pi * 299792458.0 / l_data[:].copy()
            Idata_i = Idata.copy()

        else:
            w_start = 2 * np.pi * 299792458.0 / l_stop
            w_stop = 2 * np.pi * 299792458.0 / l_start
            w_data = 2 * np.pi * 299792458.0 / l_data[::-1].copy()
            Idata_i = np.fliplr(Idata).copy()

        # w0 = 2 * np.pi * 299792458.0 / ((l_stop + l_start) / 2)
        w0 = (w_start + w_stop)/2
        root.debug(''.join(('w_start: ', str(w_start))))
        root.debug(''.join(('w_stop: ', str(w_stop))))
        # w_data = np.linspace(w_start, w_stop, Idata.shape[1]) - w0
        # w_data = np.linspace(w_start, w_stop, Idata.shape[1])

        Idata_i = np.flipud(Idata_i).copy()
        Idata_i[0:2, :] = 0.0
        Idata_i[-2:, :] = 0.0
        Idata_i[:, 0:2] = 0.0
        Idata_i[:, -2:] = 0.0
        Idata_i = self.filter_frog_trace(Idata_i / Idata_i.max(), 3, thr)

        # Find center wavelength
        Idata_l = Idata_i.sum(0)
        l_center = np.trapz(l_data * Idata_l) / np.trapz(Idata_l)
        # Find time resolution
        t_res = (tau_stop - tau_start) / n_frog
        # Generate a suitable time-frequency grid and start pulse
        self.init_pulsefield_random(n_frog, t_res, l_center)

        root.info('Creating interpolator')
        t0 = time.clock()
        # Create a 2D bivariate spline interpolator for input image:
        Idata_interp = si.RectBivariateSpline(tau_data, w_data, Idata_i)
        root.info(''.join(('Time spent: ', str(time.clock() - t0))))

        root.info(''.join(('Interpolating frog trace to ', str(self.tau.shape[0]), 'x', str(self.w.shape[0]))))
        t0 = time.clock()
        # Interpolate the values for the points in the reconstruction matrix
        # We shift the frequencies by the central frequency to make sure images are aligned.
        # Then fftshift is needed due to the way fft sorts its frequency vector (first positive frequencies
        # then the negative frequencies in the end)
        I_w_tau_interp = np.fft.fftshift(np.maximum(Idata_interp(self.tau, self.w+self.w0), 0.0), axes=1).astype(self.dtype_r)
        # I_w_tau_interp = np.fft.fftshift(np.maximum(Idata_interp(self.tau, self.w), 0.0), axes=1).astype(self.dtype_r)
        self.I_w_tau = self.filter_frog_trace(I_w_tau_interp, 3, thr)

        if self.roll_fft is True:
            self.I_w_tau = np.roll(self.I_w_tau, 1, axis=0)

        root.info(''.join(('Time spent: ', str(time.clock() - t0))))

        return Idata_i, w_data, tau_data

    def condition_frog_trace2(self, Idata, l_start, l_stop, tau_start, tau_stop, n_frog=256, thr=0.15):
        """ Take the measured intensity data and interpolate it to the
        internal w, tau grid. The variables self.w and self.tau must be set up
        first (be e.g. calling one of the init_pulsefield functions).

        Idata.shape[0] = number of tau points
        Idata.shape[1] = number of spectrum points
        """
        # Setup trace frequency and time parameters
        tau_data = np.linspace(tau_start, tau_stop, Idata.shape[0])
        l_data = np.linspace(l_start, l_stop, Idata.shape[1])
        l_center = (l_start + l_stop)/2
        c = 299792458.0
        w0 = 2*np.pi*c/l_center

        w_data = 2 * np.pi * c / l_data[:].copy()
        Idata_i = Idata.copy()
        Idata_i = self.filter_frog_trace(Idata_i / Idata_i.max(), 3, thr)

        # Find center wavelength
        # Idata_l = Idata_i.sum(0)
        # l_center = np.trapz(l_data * Idata_l) / np.trapz(Idata_l)
        # Find time resolution
        t_res = (tau_stop - tau_start) / n_frog
        # Generate a suitable time-frequency grid and start pulse
        self.init_pulsefield_random(n_frog, t_res, l_center)

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
        I_w_tau = np.fft.fftshift(np.maximum(I_w_tau, 0.0), axes=1).astype(self.dtype_r)
        self.I_w_tau = self.filter_frog_trace(I_w_tau, 3, thr)
        self.I_w_tau = self.I_w_tau / self.I_w_tau.max()

        if self.roll_fft is True:
            self.I_w_tau = np.roll(self.I_w_tau, 1, axis=0)

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
        #        w_spectrum = np.linspace(w0-w_span/2, w0+w_span/2, n_t)
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

    def generate_Esig_t_tau_shg(self):
        """ Generate the time shifted E-field matrix for the SHG process.

        Output:
        self.Esig_t_tau, a n_t x n_tau matrix where each row is Esig(t,tau)
        """
        root.debug('Generating new Esig_t_tau from SHG')
        t0 = time.clock()

        Et_mat = np.tile(self.Et, (self.tau.shape[0], 1))  # Repeat Et into a matrix
        Et_mat_tau = np.zeros_like(Et_mat)
        n_t = self.Et.shape[0]

        shiftVec = (np.arange(n_t) - n_t / 2).astype(np.int)

        for ind, sh in enumerate(shiftVec):

            if sh < 0:
                Et_mat_tau[ind, 0:n_t + sh] = self.Et[-sh:]  # Here should be a factor exp(-1j*w0*tau)
            else:
                Et_mat_tau[ind, sh:] = self.Et[0:n_t - sh]  # Here should be a factor exp(-1j*w0*tau)

        self.Esig_t_tau = Et_mat * Et_mat_tau
        self.Et_mat = Et_mat
        self.Et_mat_tau = Et_mat_tau
        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def generate_Esig_t_tau_outer_shg(self):
        """ Generate the time shifted E-field matrix for the SHG process.

        Output:
        self.Esig_t_tau, a n_t x n_tau matrix where each row is Esig(t,tau)
        """
        root.debug('Generating new Esig_t_tau from SHG')
        t0 = time.clock()

        n_t = self.Et.shape[0]
        Et_mat = np.outer(self.Et, self.Et)  # Repeat Et into a matrix

        self.Esig_t_tau = np.rot90(Et_mat.flatten()[self.shiftinds].reshape(n_t, n_t))
        self.Et_mat = Et_mat
        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def generate_Esig_t_tau_outer_sd(self):
        """ Generate the time shifted E-field matrix for the SD process.

        Output:
        self.Esig_t_tau, a n_tau x n_t matrix where each row is Esig(t,tau)
        """
        root.debug('Generating new Esig_t_tau from SHG')
        t0 = time.clock()

        n_t = self.Et.shape[0]
        Et_mat = np.outer(self.Et, self.Et)  # Repeat Et into a matrix

        self.Esig_t_tau = np.rot90(Et_mat.flatten()[self.shiftInds].reshape(n_t, n_t)*Et_mat)
        self.Et_mat = Et_mat
        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def generate_Esig_t_tau_sd(self):
        """ Generate the time shifted E-field matrix for the SD process.

        Output:
        self.Esig_t_tau, a n_t x n_tau matrix where each row is Esig(t,tau)
        """
        root.debug('Generating new Esig_t_tau from SD')
        t0 = time.clock()

        Et_mat = np.tile(self.Et, (self.tau.shape[0], 1))  # Repeat Et into a matrix
        Et_mat_tau = np.zeros_like(Et_mat)
        n_t = self.Et.shape[0]

        shiftVec = (np.arange(n_t) - n_t / 2).astype(np.int)

        for ind, sh in enumerate(shiftVec):
            if sh < 0:
                Et_mat_tau[ind, 0:n_t + sh] = self.Et[-sh:]
            else:
                Et_mat_tau[ind, sh:] = self.Et[0:n_t - sh]

        self.Esig_t_tau = Et_mat**2 * np.conj(Et_mat_tau)
        self.Et_mat = Et_mat
        self.Et_mat_tau = Et_mat_tau
        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def generate_Esig_t_tau(self, algo='SHG'):
        """ Generate the time shifted E-field matrix for the SD process.

        Output:
        self.Esig_t_tau, a n_t x n_tau matrix where each row is Esig(t,tau)
        """
        root.debug('Generating new Esig_t_tau from SD')
        t0 = time.clock()

        Et_mat = np.tile(self.Et, (self.tau.shape[0], 1))  # Repeat Et into a matrix
        Et_mat_tau = np.zeros_like(Et_mat)
        n_t = self.Et.shape[0]

        shiftVec = (np.arange(n_t) - n_t / 2).astype(np.int)

        for ind, sh in enumerate(shiftVec):
            if sh < 0:
                Et_mat_tau[ind, 0:n_t + sh] = self.Et[-sh:]
            else:
                Et_mat_tau[ind, sh:] = self.Et[0:n_t - sh]

        if algo == 'SHG':
            self.Esig_t_tau = Et_mat * Et_mat_tau
        elif algo == 'SD':
            self.Esig_t_tau = Et_mat**2 * np.conj(Et_mat_tau)
        elif algo == 'PG':
            self.Esig_t_tau = Et_mat * np.abs(Et_mat_tau)**2
        else:
            raise ValueError("Unknown algorithm, should be SHG, SD, or PG")

        self.Et_mat = Et_mat
        self.Et_mat_tau = Et_mat_tau
        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def generate_Esig_w_tau(self):
        """
        Generate the electric field vs frequency and time shift from Esig_t_tau through FFT of each time row.

        :return:
        """
        root.debug('Generating new Esig_w_tau with fft')
        t0 = time.clock()
        n_t = self.Et.shape[0]
        if self.roll_fft is True:
            self.Esig_w_tau = np.roll(np.fft.fft(self.Esig_t_tau, axis=1).astype(self.dtype_c), 1, axis=0)
        else:
            self.Esig_w_tau = np.fft.fft(self.Esig_t_tau, axis=1).astype(self.dtype_c)
        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

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
        if I_w_tau is None:
            I_w_tau = self.I_w_tau
        eps = 0.000
        Esig_mag = np.abs(self.Esig_w_tau)
        self.Esig_w_tau_p = np.zeros_like(self.Esig_w_tau)
        # Ignore data below threshold eps:
        good_ind = np.where(Esig_mag > eps)
        self.Esig_w_tau_p[good_ind[0], good_ind[1]] = np.sqrt(I_w_tau[good_ind[0], good_ind[1]]) * self.Esig_w_tau[
            good_ind[0], good_ind[1]] / Esig_mag[good_ind[0], good_ind[1]]
        # self.Esig_w_tau_p = self.Esig_w_tau
        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def update_Et_vanilla(self, algo='SHG'):
        """
        Calculate electric field vs time and time shift from Esig_w_tau through inverse FFT.
        Then generate electric field vs time vector by summing all time shift rows and normalizing.

        :return:
        """
        root.debug('Updating Et using vanilla algorithm')
        t0 = time.clock()
        n_t = self.Et.shape[0]
        self.Esig_t_tau_p = np.fft.ifft(self.Esig_w_tau_p, axis=1).astype(self.dtype_c)
        if algo == 'SHG' or algo == 'PG':
            self.Et = self.Esig_t_tau_p.sum(axis=0)
        elif algo == 'SD':
            # Change variables so that we have Esig(t+tau, tau) and integrate in tau direction
            Esig = self.Esig_t_tau_p
            Esig_p = np.zeros_like(Esig)  # Esig(t+tau, tau)
            shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
            for ind, sh in enumerate(shift_vec):
                if sh < 0:
                    Esig_p[ind, -sh:] = Esig[ind, 0:n_t + sh]
                else:
                    Esig_p[ind, 0:n_t - sh] = Esig[ind, sh:]
            self.Et = np.conj(Esig_p.sum(axis=0))
        else:
            raise ValueError("Unknown algorithm")
        # self.Et = self.Et / np.abs(self.Et).max()
        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def update_Et_gp(self, algo='SHG'):
        root.debug('Updating Et using GP algorithm')
        t0 = time.clock()
        n_t = self.Et.shape[0]
        self.Esig_t_tau_p = np.fft.ifft(self.Esig_w_tau_p, axis=1)
        eps = 0.0
        bad_ind = np.abs(self.Esig_t_tau_p) < eps
        self.Esig_t_tau_p[bad_ind] = 0.0 + 0.0j

        if algo == 'SHG':
            self.grad_z_shg()
            alpha = self.min_z_shg()
            # alpha = fmin(self.functional_distance_proj, -1.0, args=(algo,))
        elif algo == 'SD':
            self.grad_z_sd()
            alpha = self.min_z_sd()
        elif algo == 'PG':
            self.grad_z_pg()
            alpha = self.min_z_pg()
            # alpha = fmin(self.functional_distance_proj, -1.0, args=(algo,))
        else:
            raise ValueError(''.join(('Algorithm ', str(algo), ' not valid.')))

        root.debug(''.join(('Minimum found: ', str(alpha))))
        self.alpha.append(alpha)
        self.Et += alpha * self.dZ
        root.debug(''.join(('Time spent: ', str(time.clock() - t0))))

    def functional_distance(self, Et, algo='SHG'):
        n_t = Et.shape[0]

        Et_mat = np.tile(Et, (n_t, 1))  # Repeat Et into a matrix
        Et_mat_tau = np.zeros_like(Et_mat)
        shiftVec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shiftVec):
            if sh < 0:
                Et_mat_tau[ind, 0:n_t + sh] = Et[-sh:]  # Here should be a factor exp(-1j*w0*tau)
            else:
                Et_mat_tau[ind, sh:] = Et[0:n_t - sh]  # Here should be a factor exp(-1j*w0*tau)

        if algo == 'SHG':
            Esig_t_tau_new = Et_mat * Et_mat_tau
        elif algo == 'SD':
            Esig_t_tau_new = Et_mat**2 * np.conj(Et_mat_tau)
        elif algo == 'PG':
            Esig_t_tau_new = Et_mat * Et_mat_tau * np.conj(Et_mat_tau)
        else:
            raise ValueError(''.join(('Algorithm ', str(algo), ' not valid.')))

        Z = np.sum(np.abs(self.Esig_t_tau_p - Esig_t_tau_new)**2)
        return Z

    def functional_distance_proj(self, alpha, algo='SHG'):
        """ Calculate the distance metric Z given current Esig, Et and
        gradient dZdE. Z(E+alpha*dZdE)
        """
        Ea = self.Et + alpha * self.dZ
        n_t = Ea.shape[0]

        # Ea_mat = np.tile(Ea, (n_t, 1))  # Repeat Ea into a matrix so it can be directly multiplied with the signal
        #                                 # matrix
        # z0 = self.Esig_t_tau_p - Ea_mat * Ea_mat.flatten()[self.shiftinds_neg].reshape(n_t, n_t)

        Ea_mat = np.tile(Ea, (n_t, 1))  # Repeat Et into a matrix
        Ea_mat_tau = np.zeros_like(Ea_mat)
        shiftVec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shiftVec):
            if sh < 0:
                Ea_mat_tau[ind, 0:n_t + sh] = Ea[-sh:]  # Here should be a factor exp(-1j*w0*tau)
            else:
                Ea_mat_tau[ind, sh:] = Ea[0:n_t - sh]  # Here should be a factor exp(-1j*w0*tau)

        if algo == 'SHG':
            Esig_t_tau_new = Ea_mat * Ea_mat_tau
        elif algo == 'SD':
            Esig_t_tau_new = Ea_mat**2 * np.conj(Ea_mat_tau)
        elif algo == 'PG':
            Esig_t_tau_new = Ea_mat * Ea_mat_tau * np.conj(Ea_mat_tau)
        else:
            raise ValueError(''.join(('Algorithm ', str(algo), ' not valid.')))
        z0 = self.Esig_t_tau_p - Esig_t_tau_new

        Z = np.real((z0 * np.conj(z0)).sum())
        return Z

    def grad_z_shg2(self):
        """ Calculate the gradient of the Z function (functional distance) for the SHG algorithm.
        """
        tic = time.clock()

        n_t = self.Et.shape[0]
        Et_mat = np.tile(self.Et, (self.tau.shape[0], 1))   # Copy Et for each tau value
        Esig = self.Esig_t_tau_p
        Et_tau_p = np.fliplr(
            np.triu(np.tril(np.fliplr(Et_mat.flatten()[self.shiftinds_neg].reshape(n_t, n_t)), n_t / 2 - 1),
                    -n_t / 2))
        Et_tau_n = np.triu(np.tril(Et_mat.flatten()[self.shiftinds_pos].reshape(n_t, n_t), n_t / 2 - 1),
                           -n_t / 2)
        Esig_p = np.fliplr(
            np.triu(np.tril(np.fliplr(Esig.flatten()[self.shiftinds_neg].reshape(n_t, n_t)), n_t / 2 - 1),
                    -n_t / 2))

        dZ_tmp = (Et_tau_n * (np.conj(Esig) - np.conj(Et_mat) * np.conj(Et_tau_n)) +
                  Et_tau_p * (np.conj(Esig_p) - np.conj(Et_mat) * np.conj(Et_tau_p)))

        dZ_r = -(dZ_tmp + np.conj(dZ_tmp)).sum(0)
        dZ_i = (dZ_tmp - np.conj(dZ_tmp)).sum(0)
        self.dZ = dZ_r + dZ_i
        self.dZ /= np.abs(self.dZ).max()

        toc = time.clock()
        root.debug(''.join(('Grad_Z_SHG time spent: ', str(toc - tic))))
        return dZ_r, dZ_i

    def grad_z_shg(self):
        """ Calculate the gradient of the Z function (functional distance) for the SHG algorithm.
        """
        tic = time.clock()

        n_t = self.Et.shape[0]
        Et = self.Et
        Esig = self.Esig_t_tau_p
        Et_mat = np.tile(Et, (n_t, 1))   # Copy Et for each tau value
        Et_mat_tau = np.zeros_like(Et_mat)
        Et_tau_p = np.zeros_like(Et_mat)    # E(t+tau)
        Et_tau_n = np.zeros_like(Et_mat)    # E(t-tau)
        Esig_p = np.zeros_like(Et_mat)      # Esig(t+tau, tau)

        shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shift_vec):
            if sh < 0:
                Et_tau_n[ind, 0:n_t + sh] = Et[-sh:]
            else:
                Et_tau_n[ind, sh:] = Et[0:n_t - sh]

        # shift_vec_p = n_t / 2 - np.arange(n_t)
        for ind, sh in enumerate(shift_vec):
            if sh > 0:
                Et_tau_p[ind, 0:n_t - sh] = Et[sh:]
                Esig_p[ind, 0:n_t - sh] = Esig[ind, sh:]
            else:
                Et_tau_p[ind, -sh:] = Et[0:n_t + sh]
                Esig_p[ind, -sh:] = Esig[ind, 0:n_t + sh]

        dZ_tmp = (Et_tau_n * (np.conj(Esig) - np.conj(Et_mat) * np.conj(Et_tau_n)) +
                  Et_tau_p * (np.conj(Esig_p) - np.conj(Et_mat) * np.conj(Et_tau_p)))

        dZ = (2*np.conj(dZ_tmp)).sum(0)
        self.dZ = dZ
        self.dZ /= n_t**2

        toc = time.clock()
        root.debug(''.join(('Grad_Z_SHGv2  time spent: ', str(toc - tic))))
        return dZ

    def grad_z_shg_naive(self):
        dZ_r = np.zeros(self.Et.shape[0])
        dZ_i = np.zeros(self.Et.shape[0])
        Z0 = self.functional_distance_shg(self.Et)
        eps_r = 1e-7
        eps_i = 1e-7j
        for t_ind in range(self.Et.shape[0]):
            Et_tmp = np.copy(self.Et)
            Et_tmp[t_ind] += eps_r
            Z = self.functional_distance_shg(Et_tmp)
            dZ_r[t_ind] = (Z-Z0)/eps_r
        for t_ind in range(self.Et.shape[0]):
            Et_tmp = np.copy(self.Et)
            Et_tmp[t_ind] += eps_i
            Z = self.functional_distance_shg(Et_tmp)
            dZ_i[t_ind] = (Z-Z0)/eps_r
        self.dZ = dZ_r + 1j*dZ_i
        return [dZ_r, dZ_i]

    def grad_z_sd(self):
        """ Calculate the gradient of the Z function (functional distance) for the SD algorithm.
        """
        n_t = self.Et.shape[0]
        Et = self.Et
        Esig = self.Esig_t_tau_p
        Ek = np.tile(Et, (n_t, 1))  # Copy Et for each tau value
        Ep = np.zeros_like(Ek)  # E(t+tau)
        En = np.zeros_like(Ek)  # E(t-tau)
        Esig_p = np.zeros_like(Ek)  # Esig(t+tau, tau)

        # shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        # for ind, sh in enumerate(shift_vec):
        #     if sh < 0:
        #         Et_tau_n[ind, 0:n_t + sh] = Et[-sh:]
        #         Et_tau_p[ind, -sh:] = Et[0:n_t + sh]
        #         Esig_p[ind, -sh:] = Esig[ind, 0:n_t + sh]
        #     else:
        #         Et_tau_n[ind, sh:] = Et[0:n_t - sh]
        #         Et_tau_p[ind, 0:n_t - sh] = Et[sh:]
        #         Esig_p[ind, 0:n_t - sh] = Esig[ind, sh:]
        shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shift_vec):
            if sh < 0:
                En[ind, 0:n_t + sh] = Et[-sh:]
                Ep[ind, -sh:] = Et[0:n_t + sh]
                Esig_p[ind, -sh:] = Esig[ind, 0:n_t + sh]
            else:
                En[ind, sh:] = Et[0:n_t - sh]
                Ep[ind, 0:n_t - sh] = Et[sh:]
                Esig_p[ind, 0:n_t - sh] = Esig[ind, sh:]

        # shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        # for ind, sh in enumerate(shift_vec):
        #     if sh < 0:
        #         Et_tau_n[ind, 0:n_t + sh] = Et[-sh:]
        #     else:
        #         Et_tau_n[ind, sh:] = Et[0:n_t - sh]
        #
        # for ind, sh in enumerate(shift_vec):
        #     if sh > 0:
        #         Et_tau_p[ind, 0:n_t - sh] = Et[sh:]
        #         Esig_p[ind, 0:n_t - sh] = Esig[ind, sh:]
        #     else:
        #         Et_tau_p[ind, -sh:] = Et[0:n_t + sh]
        #         Esig_p[ind, -sh:] = Esig[ind, 0:n_t + sh]

        dZ_tmp = (2 * np.conj(Ek) * En * (Esig - Ek**2 * np.conj(En)) +
                  Ep**2 * (np.conj(Esig_p) - np.conj(Ep)**2 * Ek))

        dZ = -(2*dZ_tmp).sum(0)
        self.dZ = dZ
        self.dZ /= n_t ** 2
        return dZ

    def grad_z_sd2(self):
        """ Calculate the gradient of the Z function (functional distance) for the SD algorithm.
        """
        n_t = self.Et.shape[0]
        Et = self.Et
        Esig = self.Esig_t_tau_p
        Et_mat = np.tile(Et, (n_t, 1))  # Copy Et for each tau value
        Et_tau_p = np.zeros_like(Et_mat)  # E(t+tau)
        Et_tau_n = np.zeros_like(Et_mat)  # E(t-tau)
        Esig_p = np.zeros_like(Et_mat)  # Esig(t+tau, tau)

        # shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        # for ind, sh in enumerate(shift_vec):
        #     if sh < 0:
        #         Et_tau_n[ind, 0:n_t + sh] = Et[-sh:]
        #         Et_tau_p[ind, -sh:] = Et[0:n_t + sh]
        #         Esig_p[ind, -sh:] = Esig[ind, 0:n_t + sh]
        #     else:
        #         Et_tau_n[ind, sh:] = Et[0:n_t - sh]
        #         Et_tau_p[ind, 0:n_t - sh] = Et[sh:]
        #         Esig_p[ind, 0:n_t - sh] = Esig[ind, sh:]

        shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shift_vec):
            if sh < 0:
                Et_tau_n[ind, 0:n_t + sh] = Et[-sh:]
            else:
                Et_tau_n[ind, sh:] = Et[0:n_t - sh]

        for ind, sh in enumerate(shift_vec):
            if sh > 0:
                Et_tau_p[ind, 0:n_t - sh] = Et[sh:]
                Esig_p[ind, 0:n_t - sh] = Esig[ind, sh:]
            else:
                Et_tau_p[ind, -sh:] = Et[0:n_t + sh]
                Esig_p[ind, -sh:] = Esig[ind, 0:n_t + sh]

        dZ_tmp_r = (2 * Et_mat * np.conj(Et_tau_n) * (np.conj(Esig) - np.conj(Et_mat)**2 * Et_tau_n) +
                    Et_tau_p**2 * (np.conj(Esig_p) - Et_mat * np.conj(Et_tau_p)**2))
        dZ_tmp_i = (-2 * Et_mat * np.conj(Et_tau_n) * (np.conj(Esig) - np.conj(Et_mat)**2 * Et_tau_n) +
                    Et_tau_p**2 * (np.conj(Esig_p) - Et_mat * np.conj(Et_tau_p)**2))

        dZ_r = -(dZ_tmp_r + np.conj(dZ_tmp_r)).sum(0)
        dZ_i = (dZ_tmp_i + np.conj(dZ_tmp_i)).sum(0)
        self.dZ = dZ_r + 1j * dZ_i
        # self.dZ /= np.abs(self.dZ).max()
        return dZ_r, dZ_i

    def grad_z_e(self, E, algo='SHG'):
        """ Calculate the gradient of the Z function (functional distance).
        """
        n_t = E.shape[0]
        Et = E
        Esig = self.Esig_t_tau_p
        Et_mat = np.tile(Et, (n_t, 1))  # Copy Et for each tau value
        Et_tau_p = np.zeros_like(Et_mat)  # E(t+tau)
        Et_tau_n = np.zeros_like(Et_mat)  # E(t-tau)
        Esig_p = np.zeros_like(Et_mat)  # Esig(t+tau, tau)

        # shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        # for ind, sh in enumerate(shift_vec):
        #     if sh < 0:
        #         Et_tau_n[ind, 0:n_t + sh] = Et[-sh:]
        #         Et_tau_p[ind, -sh:] = Et[0:n_t + sh]
        #         Esig_p[ind, -sh:] = Esig[ind, 0:n_t + sh]
        #     else:
        #         Et_tau_n[ind, sh:] = Et[0:n_t - sh]
        #         Et_tau_p[ind, 0:n_t - sh] = Et[sh:]
        #         Esig_p[ind, 0:n_t - sh] = Esig[ind, sh:]

        shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shift_vec):
            if sh < 0:
                Et_tau_n[ind, 0:n_t + sh] = Et[-sh:]
            else:
                Et_tau_n[ind, sh:] = Et[0:n_t - sh]

        # shift_vec_p = n_t / 2 - np.arange(n_t)
        for ind, sh in enumerate(shift_vec):
            if sh > 0:
                Et_tau_p[ind, 0:n_t - sh] = Et[sh:]
                Esig_p[ind, 0:n_t - sh] = Esig[ind, sh:]
            else:
                Et_tau_p[ind, -sh:] = Et[0:n_t + sh]
                Esig_p[ind, -sh:] = Esig[ind, 0:n_t + sh]

        if algo == 'SHG':
            dZ_tmp = (Et_tau_n * (np.conj(Esig) - np.conj(Et_mat) * np.conj(Et_tau_n)) +
                      Et_tau_p * (np.conj(Esig_p) - np.conj(Et_mat) * np.conj(Et_tau_p)))

            dZ = (2 * np.conj(dZ_tmp)).sum(0)
        elif algo == 'SD':
            dZ_tmp = (2 * Et_mat * np.conj(Et_tau_n) * (np.conj(Esig) - np.conj(Et_mat)**2 * Et_tau_n) +
                      np.conj(Et_tau_p)**2 * (Esig_p - np.conj(Et_mat) * Et_tau_p**2))

            dZ = -(2*dZ_tmp).sum(0)
        self.dZ = dZ
        self.dZ /= n_t ** 2
        return dZ

    def grad_z_pg(self):
        """ Calculate the gradient of the Z function (functional distance) for the PG algorithm.
        """
        n_t = self.Et.shape[0]
        Et = self.Et
        Esig = self.Esig_t_tau_p
        Et_mat = np.tile(Et, (n_t, 1))  # Copy Et for each tau value
        Et_tau_p = np.zeros_like(Et_mat)  # E(t+tau)
        Et_tau_n = np.zeros_like(Et_mat)  # E(t-tau)
        Esig_p = np.zeros_like(Et_mat)  # Esig(t+tau, tau)

        shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shift_vec):
            if sh < 0:
                Et_tau_n[ind, 0:n_t + sh] = Et[-sh:]
                Et_tau_p[ind, -sh:] = Et[0:n_t + sh]
                Esig_p[ind, -sh:] = Esig[ind, 0:n_t + sh]
            else:
                Et_tau_n[ind, sh:] = Et[0:n_t - sh]
                Et_tau_p[ind, 0:n_t - sh] = Et[sh:]
                Esig_p[ind, 0:n_t - sh] = Esig[ind, sh:]

        dZ_tmp = (Et_tau_n * np.conj(Et_tau_n) * (Esig - Et_mat * Et_tau_n * np.conj(Et_tau_n)) +
                  Et_mat * np.conj(Et_tau_p) * (Esig_p - Et_tau_p * Et_mat * np.conj(Et_mat)) +
                  Et_tau_p * Et_mat * (np.conj(Esig_p) - np.conj(Et_tau_p) * Et_mat * np.conj(Et_mat)))

        dZ = -(2*dZ_tmp).sum(0)
        self.dZ = dZ
        self.dZ /= n_t ** 2
        return dZ

    def grad_z_pg_naive(self):
        """ Calculate the gradient of the Z function (functional distance) for the PG algorithm.
        """
        n_t = self.Et.shape[0]
        Et = self.Et
        Esig = self.Esig_t_tau_p
        dZ = np.zeros_like(Et)

        for t0 in range(Esig.shape[1]):
            T = 0.0 + 0.0j
            for tau in range(Esig.shape[0]):
                tp = t0 - (tau - np.int(n_t/2))
                if 0 <= tp < Esig.shape[1]:
                    T += (Et[t0] * np.abs(Et[tp])**2 - Esig[t0, tau]) * np.abs(Et[tp])**2
                tp = t0 + (tau - np.int(n_t / 2))
                if 0 <= tp < Esig.shape[1]:
                    T += 2.0*(np.abs(Et[tp])**2 * np.abs(Et[t0])**2 - np.real(Esig[tp, tau] * np.conj(Et[tp]))) * Et[t0]
            dZ[t0] = T / n_t**2

        self.dZ = dZ
        return dZ

    def min_z_shg_naive(self):
        Et0 = self.Et
        Esig = self.Esig_t_tau_p
        dZ = self.dZ
        n_t = Esig.shape[0]

        mx = 0.0

        # Generate polynomial X that is an expansion of Z(alpha) = Z( E(t_k) + alpha*dZ/dE(t_k) ) in alpha
        X = np.zeros(5)
        for t_ind in range(n_t):
            for tau_ind in range(n_t):
                T = np.abs(Esig[tau_ind, t_ind]) ** 2
                if mx < T:
                    mx = T
                tp = t_ind - (tau_ind - n_t / 2)
                if 0 <= tp < n_t:
                    dZdZ = dZ[t_ind] * dZ[tp]
                    dZE = dZ[t_ind] * Et0[tp] + dZ[tp] * Et0[t_ind]
                    DEsig = Et0[t_ind] * Et0[tp] - Esig[tau_ind, t_ind]

                    X[0] += np.abs(dZdZ) ** 2
                    X[1] += 2.0 * np.real(dZE * np.conj(dZdZ))
                    X[2] += 2.0 * np.real(DEsig * np.conj(dZdZ)) + np.abs(dZE) ** 2
                    X[3] += 2.0 * np.real(DEsig * np.conj(dZE))
                    X[4] += np.abs(DEsig) ** 2
        T = n_t * n_t * mx
        X /= T
        root.debug(''.join(('Poly: ', str(X[4]), ' x^4 + ', str(X[3]), ' x^3 + ', str(X[2]), ' x^2 + ',
                            str(X[1]), ' x + ', str(X[0]))))

        # We take the derivative of the polynomial and find the zero roots
        # One of these roots is the minimum of Z
        z_poly = np.polyder(X)
        r = np.roots(z_poly)
        # Only the real roots are considered (we want to go along the direction of dZ so alpha is a real number):
        r_real = r[np.abs(r.imag) < 1e-9].real
        root.debug(''.join(('Real roots: ', str(r_real))))

        # Evaluate the polynomial for these roots and find the minimum Z:
        Z1 = np.polyval(X, r_real)
        min_z_ind = Z1.argmin()

        alpha = r_real[min_z_ind].astype(self.dtype_r)

        root.debug(''.join(('Moving distance alpha=', str(alpha))))
        root.debug(''.join(('Z_0=', str(np.polyval(X, 0.0)))))
        root.debug(''.join(('Z_a=', str(np.polyval(X, alpha)))))
        return alpha

    def min_z_shg(self):
        Et = self.Et
        Esig = self.Esig_t_tau_p
        dZ = self.dZ
        n_t = Esig.shape[0]

        Et_mat = np.tile(Et, (n_t, 1))
        dZ_mat = np.tile(dZ, (n_t, 1))
        Et_tau_n = np.zeros_like(Et_mat)
        dZ_tau_n = np.zeros_like(Et_mat)
        mx = (np.abs(Esig).max())**2

        shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shift_vec):
            if sh < 0:
                Et_tau_n[ind, 0:n_t + sh] = Et[-sh:]
                dZ_tau_n[ind, 0:n_t + sh] = dZ[-sh:]
            else:
                Et_tau_n[ind, sh:] = Et[0:n_t - sh]
                dZ_tau_n[ind, sh:] = dZ[0:n_t - sh]

        dZdZ = dZ_mat * dZ_tau_n
        dZE = dZ_mat * Et_tau_n + dZ_tau_n * Et_mat
        DEsig = Et_mat * Et_tau_n - Esig

        # Generate polynomial X that is an expansion of Z(alpha) = Z( E(t_k) + alpha*dZ/dE(t_k) ) in alpha
        X = np.zeros(5)
        X[0] = (np.abs(dZdZ)**2).sum()
        X[1] = (2.0 * np.real(dZE * np.conj(dZdZ))).sum()
        X[2] = (2.0 * np.real(DEsig * np.conj(dZdZ)) + np.abs(dZE)**2).sum()
        X[3] = (2.0 * np.real(DEsig * np.conj(dZE))).sum()
        X[4] = (np.abs(DEsig)**2).sum()
        X /= (mx * n_t * n_t)

        root.debug(''.join(('Poly: ', str(X[0]), ' x^4 + ', str(X[1]), ' x^3 + ', str(X[2]), ' x^2 + ',
                            str(X[3]), ' x + ', str(X[4]))))

        # We take the derivative of the polynomial and find the zero roots
        # One of these roots is the minimum of Z
        z_poly = np.polyder(X)
        r = np.roots(z_poly)
        # Only the real roots are considered (we want to go along the direction of dZ so alpha is a real number):
        r_real = r[np.abs(r.imag) < 1e-9].real
        root.debug(''.join(('Real roots: ', str(r_real))))

        # Evaluate the polynomial for these roots and find the minimum Z:
        Z1 = np.polyval(X, r_real)
        min_z_ind = Z1.argmin()

        alpha = r_real[min_z_ind].astype(self.dtype_r)

        root.debug(''.join(('Moving distance alpha=', str(alpha))))
        root.debug(''.join(('Z_0=', str(np.polyval(X, 0.0)))))
        root.debug(''.join(('Z_a=', str(np.polyval(X, alpha)))))
        return alpha

    def min_z_pg(self):
        Et = self.Et
        Esig = self.Esig_t_tau_p
        dZ = self.dZ
        n_t = Esig.shape[0]

        Ek = np.tile(Et, (n_t, 1))
        dZk = np.tile(dZ, (n_t, 1))
        En = np.zeros_like(Ek)
        dZn = np.zeros_like(Ek)
        mx = (np.abs(Esig).max())**2

        shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shift_vec):
            if sh < 0:
                En[ind, 0:n_t + sh] = Et[-sh:]
                dZn[ind, 0:n_t + sh] = dZ[-sh:]
            else:
                En[ind, sh:] = Et[0:n_t - sh]
                dZn[ind, sh:] = dZ[0:n_t - sh]

        a3 = dZk*dZn*np.conj(dZn)
        a2 = Ek*dZn*np.conj(dZn) + dZk*dZn*np.conj(En) + dZk*np.conj(dZn)*En
        a1 = dZk*En*np.conj(En) + Ek*dZn*np.conj(En) + Ek*En*np.conj(dZn)
        a0 = Ek*En*np.conj(En) - Esig

        # Generate polynomial X that is an expansion of Z(alpha) = Z( E(t_k) + alpha*dZ/dE(t_k) ) in alpha
        X = np.zeros(7)
        X[0] = (a3 * np.conj(a3)).sum()
        X[1] = (2.0*np.real(a3*np.conj(a2))).sum()
        X[2] = (2.0*np.real(a3*np.conj(a1)) + a2*np.conj(a2)).sum()
        X[3] = (2.0*np.real(a3*np.conj(a0) + a2*np.conj(a1))).sum()
        X[4] = (2.0*np.real(a2*np.conj(a0)) + a1*np.conj(a1)).sum()
        X[5] = (2.0*np.real(a1*np.conj(a0))).sum()
        X[6] = (a0*np.conj(a0)).sum()
        X /= (mx * n_t * n_t)

        root.debug(''.join(('Poly: ', str(X[0]), ' x^6 + ', str(X[1]), ' x^5 + ', str(X[2]), ' x^4 + ', str(X[3]),
                            ' x^3 + ', str(X[4]), ' x^2 + ', str(X[5]), ' x + ', str(X[6]))))

        # We take the derivative of the polynomial and find the zero roots
        # One of these roots is the minimum of Z
        z_poly = np.polyder(X)
        r = np.roots(z_poly)
        # Only the real roots are considered (we want to go along the direction of dZ so alpha is a real number):
        r_real = r[np.abs(r.imag) < 1e-9].real
        root.debug(''.join(('Real roots: ', str(r_real))))

        # Evaluate the polynomial for these roots and find the minimum Z:
        Z1 = np.polyval(X, r_real)
        min_z_ind = Z1.argmin()

        alpha = r_real[min_z_ind].astype(self.dtype_r)

        root.debug(''.join(('Moving distance alpha=', str(alpha))))
        root.debug(''.join(('Z_0=', str(np.polyval(X, 0.0)))))
        root.debug(''.join(('Z_a=', str(np.polyval(X, alpha)))))
        return alpha

    def min_z_sd(self):
        Et = self.Et
        Esig = self.Esig_t_tau_p
        dZ = self.dZ
        n_t = Esig.shape[0]

        Ek = np.tile(Et, (n_t, 1))
        dZk = np.tile(dZ, (n_t, 1))
        En = np.zeros_like(Ek)
        dZn = np.zeros_like(Ek)
        mx = (np.abs(Esig).max())**2

        shift_vec = (np.arange(n_t) - n_t / 2).astype(np.int)
        for ind, sh in enumerate(shift_vec):
            if sh < 0:
                En[ind, 0:n_t + sh] = Et[-sh:]
                dZn[ind, 0:n_t + sh] = dZ[-sh:]
            else:
                En[ind, sh:] = Et[0:n_t - sh]
                dZn[ind, sh:] = dZ[0:n_t - sh]

        a3 = dZk*dZk*np.conj(dZn)
        a2 = np.conj(En)*dZk*dZk + 2*Ek*dZk*np.conj(dZn)
        a1 = Ek*Ek*np.conj(dZn) + 2*Ek*np.conj(En)*dZk
        a0 = Ek*Ek*np.conj(En) - Esig

        # Generate polynomial X that is an expansion of Z(alpha) = Z( E(t_k) + alpha*dZ/dE(t_k) ) in alpha
        X = np.zeros(7)
        X[0] = (a3 * np.conj(a3)).sum()
        X[1] = (2.0*np.real(a3*np.conj(a2))).sum()
        X[2] = (2.0*np.real(a3*np.conj(a1)) + a2*np.conj(a2)).sum()
        X[3] = (2.0*np.real(a3*np.conj(a0) + a2*np.conj(a1))).sum()
        X[4] = (2.0*np.real(a2*np.conj(a0)) + a1*np.conj(a1)).sum()
        X[5] = (2.0*np.real(a1*np.conj(a0))).sum()
        X[6] = (a0*np.conj(a0)).sum()
        X /= (mx * n_t * n_t)

        root.debug(''.join(('Poly: ', str(X[0]), ' x^6 + ', str(X[1]), ' x^5 + ', str(X[2]), ' x^4 + ', str(X[3]),
                            ' x^3 + ', str(X[4]), ' x^2 + ', str(X[5]), ' x + ', str(X[6]))))

        # We take the derivative of the polynomial and find the zero roots
        # One of these roots is the minimum of Z
        z_poly = np.polyder(X)
        r = np.roots(z_poly)
        # Only the real roots are considered (we want to go along the direction of dZ so alpha is a real number):
        r_real = r[np.abs(r.imag) < 1e-9].real
        root.debug(''.join(('Real roots: ', str(r_real))))

        # Evaluate the polynomial for these roots and find the minimum Z:
        Z1 = np.polyval(X, r_real)
        min_z_ind = Z1.argmin()

        alpha = r_real[min_z_ind].astype(self.dtype_r)

        root.debug(''.join(('Moving distance alpha=', str(alpha))))
        root.debug(''.join(('Z_0=', str(np.polyval(X, 0.0)))))
        root.debug(''.join(('Z_a=', str(np.polyval(X, alpha)))))
        return alpha

    def center_peaktime(self):
        ind = np.argmax(abs(self.Et))
        shift = (self.Et.shape[0] / 2 - ind).astype(np.int)
        self.Et = np.roll(self.Et, shift)

    def calc_reconstruction_error(self):
        root.debug('Calculating reconstruction error')
        tic = time.clock()
        Esig_w_tau = self.Esig_w_tau
        I_rec_w_tau = np.real(Esig_w_tau * np.conj(Esig_w_tau))
        I_w_tau = self.I_w_tau
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
        # Center peak in time
        ind = np.argmax(abs(self.Et))
        shift = (self.Et.shape[0] / 2 - ind).astype(np.int)
        Et = np.abs(np.roll(self.Et, shift))
        if norm is True:
            Et /= Et.max()
        return Et

    def get_trace_phase(self):
        eps = 0.05

        # Center peak in time
        ind = np.argmax(abs(self.Et))
        shift = self.Et.shape[0] / 2 - ind
        Et = np.roll(self.Et, shift)

        ph0_ind = np.int(Et.shape[0] / 2)
        ph = np.angle(Et)
        ph_diff = np.diff(ph)
        ph_ind = np.where(np.abs(ph_diff) > 5.0)

        for ind in ph_ind[0]:
            if ph_diff[ind] < 0:
                ph[ind + 1:] += 2 * np.pi
            else:
                ph[ind + 1:] -= 2 * np.pi

        ph0 = ph[ph0_ind]
        Et_mag = np.abs(Et)
        low_ind = np.where(Et_mag < eps)
        ph[low_ind] = np.nan
        return ph - ph0

    def get_t(self):
        return self.t

    def setup_vanilla_algorithm(self):
        pass

    def run_cycle_vanilla(self, cycles=1, algo='SHG', roll_fft=False):
        root.debug('Starting FROG reconstruction cycle using the vanilla algorithm')

        self.roll_fft = roll_fft

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
        print(''.join((str(cycles / deltaT), ' iterations/s')))
        return np.array(error)

    def setup_gp_algorithm(self):
        pass

    def run_cycle_gp(self, cycles=1, algo='SHG', roll_fft=False):
        root.debug('Starting FROG reconstruction cycle using the GP algorithm')
        self.roll_fft = roll_fft
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
            self.center_peaktime()
            root.debug('-------------------------------------------')
            root.debug(''.join(('Error G = ', str(G))))
            root.debug('-------------------------------------------')
            error.append(G)
        deltaT = time.clock() - t0
        root.debug(''.join(('Total runtime ', str(deltaT))))
        root.debug(''.join((str(cycles / deltaT), ' iterations/s')))
        print(''.join((str(cycles / deltaT), ' iterations/s')))
        return np.array(error)


if __name__ == '__main__':
    N = 256
    dt = 8e-15
    l0 = 800e-9
    tau_pulse = 100e-15

    p = sp.SimulatedPulse(N, dt, l0, tau_pulse)
    p.generateGaussianCubicSpectralPhase(1e-27, 0.5e-40)
    # p.generateGaussianCubicPhase(5e24, 3e39)
    # p.generateGaussian(tau_pulse)
    # p.generateDoublePulse(tau_pulse, deltaT=0.5e-12)
    gt = sp.SimulatedFrogTrace(N, dt, l0)
    gt.pulse = p
    IfrogSHG = gt.generateSHGTraceDt(N, dt, l0 / 2)
    IfrogSD = gt.generateSDTraceDt(N, dt, l0)
    IfrogPG = gt.generatePGTraceDt(N, dt, l0)
    #     Ifrog = gt.addNoise(0.01, 0.1)
    l = gt.getWavelengths()
    t = gt.getTimedelays()

    frog = FrogCalculation()

    frog.init_pulsefield_random(N, dt, l0)

    # frog.condition_frog_trace2(IfrogSHG, l[0], l[-1], t[0], t[-1], thr=0)
    # frog.condition_frog_trace2(IfrogSD, l[0], l[-1], t[0], t[-1], thr=0)
    # frog.condition_frog_trace2(IfrogPG, l[0], l[-1], t[0], t[-1], thr=0)
    # frog.create_frog_trace_gaussian(N, dt, l0, tau_pulse, algo='SD')
    # frog.create_frog_trace_gaussian_spectral(N, dt, l0, tau_pulse, b=1e-27, c=0.5e-40, algo='PG')
    # frog.init_pulsefield_perfect(N, dt, l0, tau_pulse)
    # frog.I_w_tau = np.abs(frog.Esig_w_tau)**2
    N = 128
    dt = 6e-15
    l0 = 263.5e-9
    frog.init_pulsefield_random(N, dt, l0)
    frog.load_frog_trace2('./frogtrace_2017-02-16_14h44_image.png', thr=0.25, l_start_pixel=0, l_stop_pixel=-1,
                         t_start_pixel=0, t_stop_pixel=-1)
    er = np.array([])
    er = frog.run_cycle_vanilla(20, 'SD', roll_fft=True)
    # er = np.hstack((er, frog.run_cycle_gp(1, 'SHG', roll_fft=False)))
    # er = np.hstack((er, frog.run_cycle_gp(50, 'PG', roll_fft=True)))
