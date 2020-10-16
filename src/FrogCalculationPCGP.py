"""
Invert frog trace using principal component generalized projection according to
Kane IEEE 1998 Real time measurements of untrashort laser pulses using principal component
generalized projections

Created: 2020-09-08

@author: Filip Lindau
"""

import numpy as np
try:
    from scipy.misc import imread
except ImportError:
    from imageio import imread
from scipy.signal import medfilt2d
from scipy.interpolate import interp1d
from xml.etree import cElementTree as ElementTree
import os
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

logger = logging.getLogger()
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.INFO)


class FrogCalculation(object):
    def __init__(self):

        self.dtype_c = np.complex64     # datatype for complex values
        self.dtype_r = np.float32       # datatype for real values

        self.dw = None      # Frequency resolution for frog trace
        self.w = None       # Frequency vector for frog trace
        self.w0 = None      # Central frequency for frog trace
        self.dt = None      # Time resolution for frog trace
        self.t = None       # Time vector for frog trace
        self.tau = None     # Time shift vector
        self.Et = None          # Electric field vs time vector
        self.I_w_tau = None     # Measured FROG image, intensity for frequency and time shift
                                # Axis 0 is frequency, axis 1 is delay
        self.shiftinds = None       # Generated index vector for generating time shifted electric fields
        self.shiftinds_neg = None
        self.shiftinds_pos = None
        self.Esig_t_tau = None      # FROG signal electric field vs time and time shift tau
        self.Et_mat = None          # Matrix of repeated electric field
        self.Et_mat_tau = None      # Matrix of electric field vs time and time shift tau
        self.Esig_w_tau = None      # FROG signal electric field vs frequency and time shift tau
        self.Esig_w_tau_p = None    # FROG signal electric field vs frequency and time shift after intensity projection
        self.Esig_t_tau_p = None    # FROG signal electric field vs time and time shift after intensity projection
        self.G_hist = list()

        self.probe = None
        self.gate = None
        self.probe_hist = list()
        self.gate_hist = list()
        self.probe_mat_list = list()
        self.gate_mat_list = list()
        self.l_mat = np.linspace(200e-9, 2000e-9, 1000)
        self.materials = dict()
        self.materials_path = "./materials"
        self.generate_materials_dict()

        self.phase_thr = 0.1        # Threshold amplitude for phase information methods (get_phase etc.)
        self.filter_kernel = 1

    def init_pulsefield_random(self, n_t, t_res, l_center, seed=0, geometry="PG"):
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
        logger.info("Initializing random pulse field for {0} points, {1} ps res, {2} nm l_0".format(n_t, t_res * 1e12,
                                                                                                    l_center * 1e9))
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

        logger.info(''.join(('t_span ', str(t_span))))
        logger.info(''.join(('t_res ', str(t_res))))

        # self.Et = np.exp(-self.t ** 2 / tau_pulse ** 2 + 1j * 2 * np.pi * np.random.rand(n_t)).astype(self.dtype_c)
        self.Et = np.random.rand(n_t) * np.exp(-(np.arange(n_t) - n_t/2.0)**2 / (10 * n_t)) * \
                  np.exp(1j * 2 * np.pi * np.random.rand(n_t)).astype(self.dtype_c)

        self.probe = self.Et
        # self.probe_hist = [self.probe]
        # self.gate = (np.random.rand(n_t) * np.exp(-(np.arange(n_t) - n_t/2.0)**2 / (10 * n_t)))**2
        # self.gate_hist = [self.gate]

        self.init_probe_gate(geometry)

        self.init_shiftind(n_t)

        self.G_hist = list()

        logger.info('Finished')

    def init_pulsefield_perfect(self, n_t, t_res, l_center, tau_pulse=100e-15, geometry="PG"):
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
        logger.info("Initializing perfect pulse field for {0} points, {1} ps res, {2} nm l_0".format(n_t, t_res * 1e12,
                                                                                                    l_center * 1e9))
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

        logger.info(''.join(('t_span ', str(t_span))))
        logger.info(''.join(('t_res ', str(t_res))))

        # Finally calculate a gaussian E-field from the
        self.Et = p.pulse.Et
        self.t = p.pulse.t

        self.init_probe_gate(geometry)

        self.init_shiftind(n_t)

        logger.info('Finished')

    def init_probe_gate(self, geometry="PG"):
        if geometry == "PG":
            self.probe = self.Et
            self.probe_hist = [self.probe]
            self.gate = np.abs(self.Et)**2
            self.gate_hist = [self.gate]
        elif geometry == "SHG":
            self.probe = self.Et
            self.probe_hist = [self.probe]
            self.gate = self.Et
            self.gate_hist = [self.gate]

    def init_shiftind(self, n_t):
        """
        Generate shiftind matrixes for time shifting operations
        :param n_t:
        :return:
        """
        i = np.arange(n_t * n_t)
        i2 = np.arange(n_t).repeat(n_t)
        ik = np.arange(n_t).repeat(n_t)
        ii = np.arange(n_t)[np.newaxis].repeat(n_t, 0).flatten()

        si = ik * n_t + (ik + ii) % n_t
        self.shiftinds_fwd = np.roll(si.reshape((n_t, n_t)), int((n_t - 1) / 2), 1)[:, ::-1].flatten()

        si = ik * n_t + (ii - ik) % n_t
        self.shiftinds_back = np.roll(np.arange(n_t * n_t).reshape((n_t, n_t))[:, ::-1], -int((n_t - 1) / 2), 1).flatten()[si]

        self.shiftinds = ((i + i2 - n_t) % n_t + i2 * n_t).astype(int)
        self.shiftinds_neg = ((i + i2 - n_t) % n_t + i2 * n_t).astype(int)
        self.shiftinds_pos = ((-n_t + i - i2) % n_t + i2 * n_t).astype(int)
        # self.shiftinds = ((i + i2 - n_t) % n_t + i2 * n_t).astype(int).reshape((n_t, n_t)).transpose().flatten()
        # self.shiftinds_neg = ((i + i2 - n_t) % n_t + i2 * n_t).astype(int).reshape((n_t, n_t)).transpose().flatten()
        # self.shiftinds_pos = ((-n_t + i - i2) % n_t + i2 * n_t).astype(int).reshape((n_t, n_t)).transpose().flatten()


    def load_frog_trace(self, filename, thr=0.0,
                        l_start_pixel=0, l_stop_pixel=-1, t_start_pixel=0, t_stop_pixel=-1,
                        filter_img=True, transpose=False):
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
        :param filter_img:
        :param transpose:
        :return:
        """
        f_name_root = '_'.join((filename.split('_')[:-1]))
        logger.debug(f_name_root)
        t_data = np.loadtxt(''.join((f_name_root, '_timevector.txt')))
        t_data = t_data - t_data.mean()
        l_data = np.loadtxt(''.join((f_name_root, '_wavelengthvector.txt')))
        if l_data[0] > 1:
            l_data = l_data * 1e-9
        pic = np.float32(imread(''.join((f_name_root, '_image.png'))))
        if transpose is True:
            pic = pic.transpose()
        pic_n = pic / pic.max()

        if t_stop_pixel == -1:
            t_stop_pixel = pic_n.shape[0] - 1
        if l_stop_pixel == -1:
            l_stop_pixel = pic_n.shape[1] - 1

        if filter_img is True:
            picF = self.filter_frog_trace(pic_n, 3, thr)
        else:
            picF = pic_n.copy() - thr
            picF[picF < 0.0] = 0.0

        # self.condition_frog_trace(picF[t_start_pixel:t_stop_pixel, l_start_pixel:l_stop_pixel],
        #                           l_data[l_start_pixel], l_data[l_stop_pixel], t_data[t_start_pixel],
        #                           t_data[t_stop_pixel], self.Et.shape[0], thr, False)

        self.condition_frog_trace(picF[l_start_pixel:l_stop_pixel, t_start_pixel:t_stop_pixel],
                                  l_data[l_start_pixel], l_data[l_stop_pixel], t_data[t_start_pixel],
                                  t_data[t_stop_pixel], self.Et.shape[0], thr, False)

    def condition_frog_trace(self, Idata, l_start, l_stop, tau_start, tau_stop, n_frog=256, thr=0.15, filter_img=True):
        """ Take the measured intensity data and interpolate it to the
        internal w, tau grid. The variables self.w and self.tau must be set up
        first (be e.g. calling one of the init_pulsefield functions).

        Idata.shape[1] = number of tau points
        Idata.shape[0] = number of spectrum points
        """
        # Setup trace frequency and time parameters
        tau_data = np.linspace(tau_start, tau_stop, Idata.shape[1])
        l_data = np.linspace(l_start, l_stop, Idata.shape[0])
        l_center = (l_start + l_stop)/2
        c = 299792458.0
        w0 = 2*np.pi*c/l_center

        w_data = 2 * np.pi * c / l_data[:].copy()
        Idata_i = Idata.copy()
        if filter_img is True:
            Idata_i = self.filter_frog_trace(Idata_i / Idata_i.max(), self.filter_kernel, thr)

        # Idata_i = np.flipud(Idata_i)
        # Fine center of gravity time delay for the frog trace time marginal
        Idata_t = Idata_i.sum(0)
        tau_center = np.trapz(tau_data * Idata_t) / np.trapz(Idata_t)
        tau_data -= tau_center      # Correct for off center time delay center of gravity
        logger.debug("Found tau center at {0:.1f} fs".format(tau_center * 1e15))

        # Find center wavelength
        Idata_l = Idata_i.sum(1)
        l_center = np.trapz(l_data * Idata_l) / np.trapz(Idata_l)

        # Find time resolution
        t_res = (tau_stop - tau_start) / n_frog
        # Generate a suitable time-frequency grid and start pulse
        # self.init_pulsefield_random(n_frog, t_res, l_center)

        logger.info(''.join(('Interpolating frog trace to ', str(self.tau.shape[0]), 'x', str(self.w.shape[0]))))
        # Interpolate the values for the points in the reconstruction matrix
        # We shift the frequencies by the central frequency to make sure images are aligned.
        # Then fftshift is needed due to the way fft sorts its frequency vector (first positive frequencies
        # then the negative frequencies in the end)
        t0 = time.time()
        Itmp_w = np.zeros((n_frog, Idata_i.shape[1]))
        I_w_tau = np.zeros((n_frog, n_frog))
        # First do interpolation in w direction for each tau delay:
        for ind in range(Idata_i.shape[1]):
            Idata_interp = interp1d(w_data, Idata_i[:, ind], kind='linear', fill_value=0.0, bounds_error=False)
            Itmp_w[:, ind] = Idata_interp(self.w0 + self.w)
        # Then interpolate in tau direction using the newly constructed intensity matrix
        for ind in range(n_frog):
            Idata_interp = interp1d(tau_data, Itmp_w[ind, :], kind='linear', fill_value=0.0, bounds_error=False)
            I_w_tau[ind, :] = Idata_interp(self.tau)
        if filter_img is True:
            I_w_tau = self.filter_frog_trace(I_w_tau, 3, thr)
        self.Idata = Idata
        self.Iwt2 = Itmp_w
        I_w_tau = np.fft.fftshift(np.maximum(I_w_tau, 0.0), axes=0).astype(self.dtype_r)
        # self.I_w_tau = self.filter_frog_trace(I_w_tau, 3, thr)
        self.I_w_tau = I_w_tau
        self.I_w_tau /= self.I_w_tau.max()

        logger.info(''.join(('Time spent: ', str(time.time() - t0))))

        return Idata_i, w_data, tau_data

    def condition_frog_trace2(self, Idata, l_start, l_stop, tau_start, tau_stop, n_frog=256, thr=0.15, filter_img=True):
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
        if filter_img is True:
            Idata_i = self.filter_frog_trace(Idata_i / Idata_i.max(), 3, thr)

        # Idata_i = np.flipud(Idata_i)
        # Fine center of gravity time delay for the frog trace time marginal
        Idata_t = Idata_i.sum(1)
        tau_center = np.trapz(tau_data * Idata_t) / np.trapz(Idata_t)
        tau_data -= tau_center      # Correct for off center time delay center of gravity
        logger.debug("Found tau center at {0:.1f} fs".format(tau_center * 1e15))

        # Find center wavelength
        Idata_l = Idata_i.sum(0)
        l_center = np.trapz(l_data * Idata_l) / np.trapz(Idata_l)

        # Find time resolution
        t_res = (tau_stop - tau_start) / n_frog
        # Generate a suitable time-frequency grid and start pulse
        # self.init_pulsefield_random(n_frog, t_res, l_center)

        logger.info(''.join(('Interpolating frog trace to ', str(self.tau.shape[0]), 'x', str(self.w.shape[0]))))
        # Interpolate the values for the points in the reconstruction matrix
        # We shift the frequencies by the central frequency to make sure images are aligned.
        # Then fftshift is needed due to the way fft sorts its frequency vector (first positive frequencies
        # then the negative frequencies in the end)
        t0 = time.time()
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
        self.Idata = Idata
        self.Iwt2 = Itmp_w
        I_w_tau = np.fft.fftshift(np.maximum(I_w_tau, 0.0), axes=1).astype(self.dtype_r)
        # self.I_w_tau = self.filter_frog_trace(I_w_tau, 3, thr)
        self.I_w_tau = I_w_tau
        self.I_w_tau /= self.I_w_tau.max()

        logger.info(''.join(('Time spent: ', str(time.time() - t0))))

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
        Idata_f = Idata / np.max(Idata)
        Idata_f = medfilt2d(Idata_f, kernel) - thr
        Idata_f[Idata_f < 0.0] = 0.0
        Idata_f = Idata_f / np.max(Idata_f)
        return Idata_f

    def apply_intensity_data(self, I_w_tau=None):
        """
        Overwrite the magnitude of the generated electric field vs frequency and time shift
         with the measured intensity data.

        :param I_w_tau: Intensity data to overwrite magnitude of generated electric field. If None
         the intensity data stored in the class instance is used.
        :return:
        """
        logger.debug('Applying intensity data from experiment')
        t0 = time.time()
        if I_w_tau is None:
            I_w_tau = self.I_w_tau
        eps = 1e-10
        Esig_mag = np.abs(self.Esig_w_tau)
        self.Esig_w_tau_p = np.zeros_like(self.Esig_w_tau)
        # Ignore data below threshold eps:
        good_ind = np.where(Esig_mag > eps)
        self.Esig_w_tau_p[good_ind[0], good_ind[1]] = np.sqrt(I_w_tau[good_ind[0], good_ind[1]]) * self.Esig_w_tau[
            good_ind[0], good_ind[1]] / Esig_mag[good_ind[0], good_ind[1]]
        # self.Esig_w_tau_p = self.Esig_w_tau
        logger.debug(''.join(('Time spent: ', str(time.time() - t0))))

    def op_step(self, geometry="PG"):
        if geometry == "PG":
            self.op_step_pg()
        elif geometry == "SHG":
            self.op_step_shg()
        else:
            self.op_step_pg()

    def op_step_pg(self):
        # Get current probe and gate and generate outer product

        probe_m = self.propagate_material_list(self.probe, self.probe_mat_list, True)
        gate_m = self.propagate_material_list(self.gate, self.gate_mat_list, True)

        # Symmetrize gate and probe
        gp = np.abs(probe_m)**2
        pg_phase = np.exp(1j * np.angle(probe_m))
        pg_amp = np.sqrt(gate_m)
        eps = 1e-2
        good_pg = pg_amp > eps
        pg = probe_m
        pg.flatten()[good_pg] = (pg_amp.flatten()[good_pg] * pg_phase.flatten()[good_pg])
        pg.reshape(gate_m.shape)
        O = np.outer(probe_m, gate_m) + np.outer(pg, gp)
        # O = np.outer(self.probe, self.gate)
        # Row rotate to get time domain FROG trace
        self.Esig_t_tau = O.flatten()[self.shiftinds_fwd].reshape(O.shape)

        # FFT columns
        self.Esig_w_tau = np.fft.fft(self.Esig_t_tau, axis=0).astype(self.dtype_c)
        self.apply_intensity_data()
        self.Esig_t_tau_p = np.fft.ifft(self.Esig_w_tau_p, axis=0).astype(self.dtype_c)
        O_p = self.Esig_t_tau_p.flatten()[self.shiftinds_back].reshape(self.Esig_t_tau_p.shape)
        U, l, V = np.linalg.svd(O_p)
        self.probe = self.propagate_material_list(U[:, 0], self.probe_mat_list, False)
        self.gate = self.propagate_material_list(np.abs(V[0, :]), self.gate_mat_list, False)
        # self.probe = np.dot(np.matmul(O_p, O_p.transpose()), self.probe)
        self.probe /= np.max(self.probe)
        self.Et = self.probe
        # self.gate = np.abs(np.dot(np.matmul(O_p.transpose(), O_p), self.gate))
        self.gate /= np.max(self.gate)
        self.probe_hist.append(self.probe)
        self.gate_hist.append(self.gate)
        self.calc_reconstruction_error()

    def op_step_shg(self):
        # Get current probe and gate and generate outer product

        probe_m = self.propagate_material_list(self.probe, self.probe_mat_list, True)
        gate_m = self.propagate_material_list(self.gate, self.gate_mat_list, True)

        # Symmetrize gate and probe
        gp = probe_m
        pg = gate_m
        O = np.outer(probe_m, gate_m) + np.outer(pg, gp)
        # O = np.outer(self.probe, self.gate)
        # Row rotate to get time domain FROG trace
        self.Esig_t_tau = O.flatten()[self.shiftinds_fwd].reshape(O.shape)

        # FFT columns
        self.Esig_w_tau = np.fft.fft(self.Esig_t_tau, axis=0).astype(self.dtype_c)
        self.apply_intensity_data()
        self.Esig_t_tau_p = np.fft.ifft(self.Esig_w_tau_p, axis=0).astype(self.dtype_c)
        O_p = self.Esig_t_tau_p.flatten()[self.shiftinds_back].reshape(self.Esig_t_tau_p.shape)
        U, l, V = np.linalg.svd(O_p)
        self.probe = self.propagate_material_list(U[:, 0], self.probe_mat_list, False)
        self.gate = self.propagate_material_list(V[0, :], self.gate_mat_list, False)
        # self.probe = np.dot(np.matmul(O_p, O_p.transpose()), self.probe)
        self.probe /= np.max(self.probe)
        self.Et = self.probe
        # self.gate = np.abs(np.dot(np.matmul(O_p.transpose(), O_p), self.gate))
        self.gate /= np.max(self.gate)
        self.probe_hist.append(self.probe)
        self.gate_hist.append(self.gate)
        self.calc_reconstruction_error()

    def vanilla_step(self):
        # Get current probe and gate and generate outer product

        # Symmetrize gate and probe
        gp = np.abs(self.probe)**2
        pg_phase = np.exp(1j * np.angle(self.probe))
        pg_amp = np.sqrt(self.gate)
        eps = 1e-2
        good_pg = pg_amp > eps
        pg = self.probe
        pg.flatten()[good_pg] = (pg_amp.flatten()[good_pg] * pg_phase.flatten()[good_pg])
        pg.reshape(self.gate.shape)
        O = np.outer(self.probe, self.gate) + np.outer(pg, gp)
        # O = np.outer(self.probe, self.gate)
        # Row rotate to get time domain FROG trace
        self.Esig_t_tau = O.flatten()[self.shiftinds_pos].reshape(O.shape).transpose()

        # Et_mat = np.tile(self.probe, (self.tau.shape[0], 1))  # Repeat Et into a matrix
        # Et_mat_tau = np.zeros(Et_mat.shape)
        # n_t = self.probe.shape[0]
        #
        # shiftVec = (np.arange(n_t) - n_t / 2).astype(np.int)
        #
        # for ind, sh in enumerate(shiftVec):
        #     if sh < 0:
        #         Et_mat_tau[ind, 0:n_t + sh] = self.gate[-sh:]
        #     else:
        #         Et_mat_tau[ind, sh:] = self.gate[0:n_t - sh]
        #
        # self.Esig_t_tau = Et_mat * Et_mat_tau

        # FFT columns
        self.Esig_w_tau = np.fft.fft(self.Esig_t_tau, axis=1).astype(self.dtype_c)
        self.apply_intensity_data()
        self.Esig_t_tau_p = np.fft.ifft(self.Esig_w_tau_p, axis=1).astype(self.dtype_c)
        self.probe = self.Esig_t_tau_p.sum(axis=0)
        O_p = self.Esig_t_tau_p.flatten()[self.shiftinds_neg].reshape(self.Esig_t_tau_p.shape)
        U, l, V = np.linalg.svd(O_p)
        self.probe = U[:, 0]
        self.gate = np.abs(V[0, :])
        # self.probe = np.dot(np.matmul(O_p, O_p.transpose()), self.probe)
        self.probe /= np.max(self.probe)
        self.Et = self.probe
        # self.gate = np.abs(np.dot(np.matmul(O_p.transpose(), O_p), self.gate))
        self.gate /= np.max(self.gate)
        self.probe_hist.append(self.probe)
        self.gate_hist.append(self.gate)
        self.calc_reconstruction_error()

    def center_peaktime(self):
        """
        Center the reconstructed E-field E(t) maximum value in time.

        :return:
        """
        ind = np.argmax(abs(self.Et))
        shift = (self.Et.shape[0] / 2 - ind).astype(np.int)
        self.Et = np.roll(self.Et, shift)

    def calc_reconstruction_error(self):
        """
        Calculate the reconstruction error as the sum of the squared difference between experimental frog trace image
        I_w_tau and reconstructed frog image. The calculated error is appended to the class variable G_hist where
        the error history is stored as a list.

        :return: Reconstruction error G
        """
        logger.debug('Calculating reconstruction error')
        tic = time.time()
        Esig_w_tau = self.Esig_w_tau
        I_rec_w_tau = np.real(Esig_w_tau * np.conj(Esig_w_tau))
        I_w_tau = self.I_w_tau
        mx = I_w_tau.max()
        my = (I_w_tau * I_rec_w_tau).sum() / (I_rec_w_tau**2).sum()
        logger.debug(''.join(('My=', str(my))))
        # my = 1.0
        G = np.sqrt(((I_w_tau - my * I_rec_w_tau) ** 2).sum() / (I_rec_w_tau.shape[0] * I_rec_w_tau.shape[1])) / mx
        toc = time.time()
        logger.debug(''.join(('Time spent: ', str(toc - tic))))
        self.G_hist.append(G)
        return G

    def get_trace_abs(self, norm=True):
        """
        Retrieve the magnitude of the reconstructed E-field (Et). Use get_t for the
        corresponding time vector.

        :param norm: If true, the return vector is normalized
        :return: Vector of the magnitude of the E-field abs(E(t))
        """
        if self.Et is not None:
            # Center peak in time
            ind = np.argmax(abs(self.Et))
            shift = (self.Et.shape[0] / 2 - ind).astype(np.int)
            Et = np.abs(np.roll(self.Et, shift))
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
        if self.Et is not None:
            # Center peak in time
            ind = np.argmax(abs(self.Et))
            shift = (self.Et.shape[0] / 2 - ind).astype(np.int)
            Et = np.roll(self.Et, shift)

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

        if self.Et is not None:
            # Center peak in time
            ind = np.argmax(abs(self.Et))
            shift = (self.Et.shape[0] / 2 - ind).astype(int)
            Et = np.roll(self.Et, shift)

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
        if self.Et is not None:
            N = self.Et.shape[0]
            w_ind = np.argsort(np.fft.fftfreq(N))   # Sorted index vector to unravel the fft:d E-field vector

            # Center peak in time
            ind = np.argmax(abs(self.Et))
            shift = (self.Et.shape[0] / 2 - ind).astype(np.int)
            Et = np.roll(self.Et, shift)

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
        if self.Et is not None:
            t = self.get_t()
            ph = self.get_trace_phase()
            ph_ind = np.isfinite(ph)
            ph_good = ph[ph_ind]
            t_good = t[ph_ind] / prefix
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
        if self.Et is not None:
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
        return np.abs(self.Esig_w_tau) ** 2

    def generate_materials_dict(self):
        """
        Generates the internal materials dict from a set of non-sellmeier materials (air, sapphire, bbo..)
        and the files in the materials directory. The dict stores scipy interp1d interpolators that are
        used to find the refractive index at specific angular frequencies later.
        :return:
        """
        c = 299792458.0
        w_mat = 2 * np.pi * c / self.l_mat - self.w0
        l2_mat = (self.l_mat * 1e6) ** 2

        n_air = 1 + 0.05792105 * l2_mat / (238.0185 * l2_mat - 1) + 0.00167917 * l2_mat / (57.362 * l2_mat - 1)
        air_ip = interp1d(w_mat, n_air, bounds_error=False, fill_value=np.nan, kind="quadratic")
        self.materials['air'] = air_ip

        n_fs = np.sqrt(1 + 0.6961663 * l2_mat / (l2_mat - 0.0684043 ** 2) +
                       0.4079426 * l2_mat / (l2_mat - 0.1162414 ** 2) +
                       0.8974794 * l2_mat / (l2_mat - 9.896161 ** 2))
        fs_ip = interp1d(w_mat, n_fs, bounds_error=False, fill_value=np.nan, kind="quadratic")
        self.materials['fs'] = fs_ip

        n_mgf2 = np.sqrt(1 + 0.48755108 * l2_mat / (l2_mat - 0.04338408 ** 2) +
                         0.39875031 * l2_mat / (l2_mat - 0.09461442 ** 2) +
                         2.3120353 * l2_mat / (l2_mat - 23.793604 ** 2))
        mgf2_ip = interp1d(w_mat, n_mgf2, bounds_error=False, fill_value=np.nan, kind="quadratic")
        self.materials['mgf2'] = mgf2_ip

        n_sapphire_o = np.sqrt(1 + 1.4313493 * l2_mat / (l2_mat - 0.0726631 ** 2) +
                               0.65054713 * l2_mat / (l2_mat - 0.1193242 ** 2) +
                               5.3414021 * l2_mat / (l2_mat - 18.028251 ** 2))
        sapphire_o_ip = interp1d(w_mat, n_sapphire_o, bounds_error=False, fill_value=np.nan, kind="quadratic")
        self.materials['sapphire_o'] = sapphire_o_ip

        n_sapphire_e = np.sqrt(1 + 1.5039759 * l2_mat / (l2_mat - 0.0740288 ** 2) +
                               0.55069141 * l2_mat / (l2_mat - 0.1216529 ** 2) +
                               6.5927379 * l2_mat / (l2_mat - 20.072248 ** 2))
        sapphire_e_ip = interp1d(w_mat, n_sapphire_e, bounds_error=False, fill_value=np.nan, kind="quadratic")
        self.materials['sapphire_e'] = sapphire_e_ip

        n_bbo_o = np.sqrt(2.7405 + 0.0184 / (l2_mat - 0.0179) - 0.0155 * l2_mat)
        bbo_o_ip = interp1d(w_mat, n_bbo_o, bounds_error=False, fill_value=np.nan, kind="quadratic")
        self.materials['bbo_o'] = bbo_o_ip

        n_bbo_e = np.sqrt(2.3730 + 0.0128 / (l2_mat - 0.0156) - 0.0044 * l2_mat)
        bbo_e_ip = interp1d(w_mat, n_bbo_e, bounds_error=False, fill_value=np.nan, kind="quadratic")
        self.materials['bbo_e'] = bbo_e_ip

        materials_files = os.listdir(self.materials_path)
        logger.info("Found {0:d}".format(materials_files.__len__()))
        for mat_file in materials_files:
            logger.debug(mat_file)
            self.read_material(''.join((self.materials_path, '/', mat_file)))

    def add_material(self, name, b_coeff, c_coeff):
        """
        Adds a material to the internal materials dict. The material is specified with it's sellmeier
        coefficients: n = sqrt(1 + sum(B * l**2 / (l**2 - C))

        The wavelengths are in um as customary in Sellmeier equations.

        The dict stores scipy interp1d interpolators that are used to find the refractive index at
        specific angular frequencies later.

        :param name: String containing the name of the material (used as key in the dict)
        :param b_coeff: Vector of B-coefficients for the Sellmeier equation (for lambda in um)
        :param c_coeff: Vector of C-coefficients for the Sellmeier equation (for lambda in um)
        :return:
        """
        """

        :return:
        """
        l_mat = np.linspace(200e-9, 2000e-9, 5000)
        c = 299792458.0
        w_mat = 2 * np.pi * c / l_mat
        l2_mat = (l_mat * 1e6) ** 2
        n_tmp = 0.0
        for ind, b in enumerate(b_coeff):
            n_tmp += b*l2_mat / (l2_mat - c_coeff[ind])
        n = np.sqrt(1 + n_tmp)
        n_ip = interp1d(w_mat, n, bounds_error=False, fill_value=np.nan, kind="quadratic")
        self.materials[name] = n_ip

    def read_material(self, filename):
        """
        Read an xml file and extract the sellmeier coeffients from it. The file should have
        elements called sellmeier with tags called A, B, and C. The refractive index is then
        calculated as:
        n = sqrt(1 + sum(A + B * l**2 / (l**2 - C))

        The wavelengths are in um as customary in Sellmeier equations.

        The A coefficients were added to allow certain types of materials in the refractiveindex.info
        database.

        :param filename: String containing the filename
        :return:
        """
        l_mat = np.linspace(200e-9, 2000e-9, 5000)
        c = 299792458.0
        w_mat = 2 * np.pi * c / l_mat - self.w0
        l2_mat = (l_mat * 1e6) ** 2
        n_tmp = 0.0

        e = ElementTree.parse(filename)
        mat = e.getroot()
        name = mat.get('name')
        sm = mat.findall('sellmeier')
        for s in sm:
            at = s.find('A')
            if at is not None:
                a = np.double(at.text)
            else:
                a = 0.0
            bt = s.find('B')
            if bt is not None:
                b = np.double(bt.text)
            else:
                b = 0.0
            ct = s.find('C')
            if ct is not None:
                c = np.double(ct.text)
            else:
                c = 0.0
            n_tmp += a + b*l2_mat / (l2_mat - c)
        n = np.sqrt(1 + n_tmp)
        n_ip = interp1d(w_mat, n, bounds_error=False, fill_value=np.nan)
        self.materials[name] = n_ip

    def propagate_material(self, E_w_in, name, thickness):
        """
        Propagate the current pulse through a thickness of material. The propagation is performed
        in the fourier domain by spectral filtering. The pulse is then inverse transformed to
        the time domain.

        :param E_w_in: Electric field in angular frequency space (with frequencies as in self.w)
        :param name: String containing the name of the material (to match a key in the materials dict)
        :param thickness: Thickness of the material (SI units)
        :return:
        """
        logger.debug("Entering propagate_material {0:.1f} mm {1}".format(thickness*1e3, name))
        if len(self.materials) == 0:
            self.generate_materials_dict()

        c = 299792458.0
        try:
            k_w = self.w * self.materials[name](self.w) / c
        except KeyError:
            return
        # Remove 0th and 1st order phase
        x = np.arange(k_w.shape[0])
        pf = np.polyfit(x, k_w, 1)
        H_w = np.exp(-1j * (k_w - np.polyval(pf, x)) * thickness)
        H_w[np.isnan(H_w)] = 0
        E_w_out = H_w * E_w_in
        return E_w_out

    def propagate_material_list(self, Et, mat_list, fwd=True):
        N = Et.shape[0]
        Ew = np.fft.fftshift(np.fft.fft(np.fft.fftshift(Et), N))
        for mat in mat_list:
            if fwd:
                Ew = self.propagate_material(Ew, mat[0], mat[1])
            else:
                Ew = self.propagate_material(Ew, mat[0], -mat[1])
        Et = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ew), N))
        em = abs(Et).argmax()
        self.Et = np.roll(Et / np.abs(Et).max(), int(N / 2 - em))
        return self.Et

    def run_cycle_pc(self, cycles, geometry="PG"):
        for k in range(cycles):
            self.op_step(geometry)
        return self.G_hist


if __name__ == '__main__':
    N = 128
    dt = 7.7e-15
    l0 = 780e-9
    tau_pulse = 50e-15 / np.sqrt(4*np.log(2))

    p = sp.SimulatedPulse(N, dt, l0, tau_pulse)
    p.generateGaussianCubicSpectralPhase(0.0e-27, 0.0e-40)
    # p.generateGaussianCubicPhase(5e24, 3e39)
    # p.generateGaussian(tau_pulse)
    # p.generateDoublePulse(tau_pulse, deltaT=0.5e-12)
    gt = sp.SimulatedFrogTrace(N, dt, l0)
    gt.pulse = p
    IfrogSHG = gt.generateSHGTraceDt(N, dt, l0 / 2)
    IfrogSD = gt.generateSDTraceDt(N, dt, l0)
    IfrogPG = gt.generatePGTraceDt(N, dt, l0)
    IfrogPGw = gt.generatePGTraceDt_w(N, dt, l0)
    #     Ifrog = gt.addNoise(0.01, 0.1)
    l = gt.getWavelengths()
    t = gt.getTimedelays()

    frog = FrogCalculation()

    frog.init_pulsefield_random(N, dt, l0+0e-9)

    # frog.condition_frog_trace2(IfrogSHG, l[0], l[-1], t[0], t[-1], n_frog=N, thr=0)
    # frog.condition_frog_trace2(IfrogSD, l[0], l[-1], t[0], t[-1], n_frog=N, thr=0)
    frog.condition_frog_trace(IfrogPG, l[0], l[-1], t[0], t[-1], n_frog=N, thr=0)
    # frog.create_frog_trace_gaussian(N, dt, l0, tau_pulse, algo='SD')
    # frog.create_frog_trace_gaussian_spectral(N, dt, l0, tau_pulse, b=0.2e-27, c=0.2e-40, algo='PG')
    # frog.init_pulsefield_perfect(N, dt, l0, tau_pulse)
    # frog.I_w_tau = np.abs(frog.Esig_w_tau)**2
    # N = 128
    # dt = 6e-15
    # l0 = 263.5e-9
    # frog.init_pulsefield_random(N, dt, l0)
    # frog.load_frog_trace('./data/frogtrace_2017-03-13_17h35_uv_pg_67mm_image',
    #                      thr=0.65, l_start_pixel=0, l_stop_pixel=-1,
    #                      t_start_pixel=0, t_stop_pixel=-1, filter_img=False)
    # frog.I_w_tau = IfrogPGw

    # Data from IR PG FROG
    pic = imread("./data/pg_frog/frog_ir_pg_test_bs_1.png")
    if len(pic.shape) > 2:
        pic = pic[:, :, 0]
    picb = imread("./data/pg_frog/frog_ir_pg_test_bs_bkg.png")
    if len(picb.shape) > 2:
        picb = picb[:, :, 0]
    picr = np.maximum(0, (pic.astype(int) - picb))[100:400, 700:900]
    pic = imread("./data/pg_frog/frog_ir_pg_test_0.png")
    if len(pic.shape) > 2:
        pic = pic[:, :, 0]
    picl = imread("./data/pg_frog/frog_ir_pg_test_0_glanlaser.png")
    if len(picl.shape) > 2:
        picl = picl[:, :, 0]
    picb = imread("./data/pg_frog/frog_ir_pg_test_0_bkg.png")
    if len(picb.shape) > 2:
        picb = picb[:, :, 0]
    picr = np.maximum(0, (pic.astype(np.double) - picb))[100:400, 700:900]
    piclr = np.maximum(0, (picl.astype(np.double) - picb))[100:400, 700:900]

    dt_pg = 7.7e-15
    l0_pg = 782e-9
    dl_pg = 0.34e-9
    N_pg = 128
    t_pg = dt_pg * (np.arange(picr.shape[1]) - 90)
    l_pg = l0_pg + dl_pg * (np.arange(picr.shape[0]) - 156)
    frog.init_pulsefield_random(N_pg, dt_pg, l0_pg + 0e-9)
    # frog.condition_frog_trace(picr, l_pg[0], l_pg[-1], t_pg[0], t_pg[-1], n_frog=N_pg, thr=0.05, filter_img=True)
    frog.condition_frog_trace(piclr, l_pg[0], l_pg[-1], t_pg[0], t_pg[-1], n_frog=N_pg, thr=0.1, filter_img=True)
    frog.generate_materials_dict()
    frog.probe_mat_list.append(("bk7", 13e-3))
    frog.probe_mat_list.append(("caco3_e", 14e-3))
    frog.gate_mat_list.append(("bk7", 6e-3))
    frog.gate_mat_list.append(("fs", 2e-3))
    # frog.probe_mat_list.append(("bk7", 0e-3))
    # frog.probe_mat_list.append(("caco3_e", 0e-3))
    # frog.gate_mat_list.append(("bk7", 0e-3))

    er = np.array([])
    er = frog.run_cycle_pc(50, 'PG')
    # er = frog.run_cycle_gp(20, 'SD', roll_fft=False)
    # er = np.hstack((er, frog.run_cycle_gp(1, 'SHG', roll_fft=False)))
    # er = np.hstack((er, frog.run_cycle_gp(50, 'PG', roll_fft=True)))