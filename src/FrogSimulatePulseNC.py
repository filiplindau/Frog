'''
Created on 3 Feb 2016

@author: Filip Lindau

Calculate FROG with carrier frequency removed
'''

import numpy as np
import logging
from scipy.interpolate import interp1d
from xml.etree import cElementTree as ElementTree
import os


logger = logging.getLogger()
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class SimulatedPulse(object):
    def __init__(self, N=128, dt=10e-15, l0=800e-9, tau=50e-15):
        """

        :param N: number of points in the trace
        :param dt: time resolution
        :param l0: central wavelength of the peak
        :param tau: tentative pulse duration
        """
        self.Et = np.zeros(N, dtype=np.complex)
        self.N = N
        self.tau = tau
        self.l0 = l0
        self.tspan = N*dt
        self.dt = dt
        self.t = np.linspace(-self.tspan / 2, self.tspan / 2, self.N)

        # Now we calculate the frequency resolution required by the
        # time span
        self.w_res = 2 * np.pi / self.tspan

        # Frequency span is given by the time resolution
        f_max = 1 / (2 * self.dt)
        self.w_span = f_max * 2 * 2 * np.pi

        c = 299792458.0
        w0 = 2 * np.pi * c / self.l0
        # delta_l = -2*np.pi*c/self.w_span + np.sqrt(l0**2 + (2*np.pi*c/self.w_span)**2)
        # w0 = 2*np.pi*c*l0/(l0**2 - delta_l**2)
        self.w0 = w0
        self.w = np.fft.fftshift((self.w0 + 2 * np.pi * np.fft.fftfreq(self.N, d=self.dt)))

        self.l_mat = np.linspace(200e-9, 2000e-9, 1000)
        self.materials = dict()
        self.materials_path = "./materials"

        self.generateGaussian(tau)
        
    def generateGaussian(self, tau):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2/2 + ph) * np.exp(0j)
        self.setEt(Eenv, t)
        
    def generateGaussianQuadraticPhase(self, b=0):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2/2 + 1j*(b*t**2) + ph)
        self.setEt(Eenv, t)

    def generateGaussianCubicPhase(self, b=0, c=0):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2/2 + 1j*(b*t**2 + c*t**3) + ph)
        self.setEt(Eenv, t)

    def generateGaussianCubicSpectralPhase(self, b=0, c=0):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)        
        f_max = 1/(2*self.dt)
        w = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(self.N, d=self.dt))
        ph = 0.0
        Eenv_w = np.exp(-w ** 2 * self.tau**2 / 2 + 1j * (b * w ** 2 + c * w ** 3) + ph)
        Eenv = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Eenv_w)))
        self.setEt(Eenv, t)
        
    def generateDoublePulse(self, tau, deltaT):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        ph = 0.0
        Eenv = np.exp(-(t+deltaT/2)**2/self.tau**2/2 + ph) + np.exp(-(t-deltaT/2)**2/self.tau**2/2 + ph)
        self.setEt(Eenv, t)
        
    def addChirp(self, a):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)                
        Eenv = self.Et.copy()*np.exp(1j*a*t*t)
        self.setEt(Eenv, t)        
        
    def getFreqSpectrum(self, Nw=None):
        if Nw is None:
            Nw = self.t.shape[0]
        Ew = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.Et), Nw))
#        w0 = 0*2*np.pi*299792458.0/self.l0
        w = np.fft.fftshift((self.w0 + 2*np.pi*np.fft.fftfreq(Nw, d=self.dt)))
        return w, Ew

    def getSpectrogram(self, Nl=None):
        if Nl is None:
            Nl = self.t.shape[0]
        w, Ew = self.getFreqSpectrum(Nl)
        c = 299792458.0
        l = 2*np.pi*c/w

        # Resample over linear grid in wavelength
        E_interp = interp1d(l, Ew, kind='linear', fill_value=0.0, bounds_error=False)
        l_start = 2*np.pi*c / (self.w0 + self.w_span/2)
        l_stop = 2 * np.pi * c / (self.w0 - self.w_span/2)
        delta_l = np.min([np.abs(self.l0-l_start), np.abs(self.l0-l_stop)])
        l_sample = np.linspace(self.l0-delta_l, self.l0+delta_l, Nl)
        El = E_interp(l_sample)
        Il = El*El.conj()
#        Il = Il/max(Il)
        return l_sample, Il

    def setEt(self, Et, t=None):
        """
        Manually electric field vector. If no time vector is supplied, the internal is used. Then the
        new vector must have the same number of elements.
        :param Et: New electric field vector
        :param t: New time vector or None
        :return:
        """
        if t is not None:
            self.t = t
            self.tspan = np.abs(t.max()-t.min())
            self.N = t.shape[0]
            self.dt = self.tspan/self.N
            self.w = np.fft.fftshift((self.w0 + 2 * np.pi * np.fft.fftfreq(self.N, d=self.dt)))
        self.Et = Et
        
    def getShiftedEt(self, shift):
        """
        Return shifted electric field vector. Useful for generating time delays
        :param shift: Number of timesteps to shift the vector
        :return: Shifted electroc field. Exposed edges are filled with zeros.
        """
        sh = np.int(shift)
        Ets = np.zeros_like(self.Et)
        # logger.debug(''.join(('Shift: ', str(sh))))
        if sh < 0:
            Ets[0:self.N+sh] = self.Et[-sh:]
        else:
            Ets[sh:] = self.Et[0:self.N-sh]
        return Ets

    def generate_materials_dict(self):
        """
        Generates the internal materials dict from a set of non-sellmeier materials (air, sapphire, bbo..)
        and the files in the materials directory. The dict stores scipy interp1d interpolators that are
        used to find the refractive index at specific angular frequencies later.
        :return:
        """
        c = 299792458.0
        w_mat = 2 * np.pi * c / self.l_mat
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
        w_mat = 2 * np.pi * c / l_mat
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
        H_w = np.exp(-1j * k_w * thickness)
        H_w[np.isnan(H_w)] = 0
        E_w_out = H_w * E_w_in
        return E_w_out

    def propagate_material_list(self, mat_list):
        Ew = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.Et), self.N))
        for mat in mat_list:
            Ew = p.propagate_material(Ew, mat[0], mat[1])
        Et = np.fft.ifft(Ew)
        em = abs(Et).argmax()
        self.Et = np.roll(Et / np.abs(Et).max(), int(N / 2 - em))
        return self.Et


class SimulatedFrogTrace(object):
    def __init__(self, N=128, dt=10e-15, l0=800e-9, tau=50e-15):
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
        l_shift = signalPulse.l0 - l0
        nl_shift = np.int(l_shift/np.abs(l[1]-l[0]))
        self.l_vec = l + l_shift
        
        shiftVec = np.arange(N) - N/2
        
        logger.debug(''.join(('l_shift: ', str(l_shift))))
        logger.debug(''.join(('nl_shift: ', str(nl_shift))))
        
        for sh in shiftVec:
            Ils = np.zeros_like(Il)
            signalPulse.setEt(Et*self.pulse.getShiftedEt(sh), t)
            l, Iln = signalPulse.getSpectrogram(t.shape[0])
            Ils = np.roll(Iln, nl_shift)
            Ifrog.append(Ils)
                        
        Ifrog = np.array(Ifrog).real.transpose()
        self.Ifrog = Ifrog/Ifrog.max()
        
        return self.Ifrog

    def generateSDTraceDt(self, N, dt, l0):
        signalPulse = SimulatedPulse(self.pulse.N, self.pulse.dt, self.pulse.l0, self.pulse.tau)
        tspan = N*dt
        self.tau_vec = np.linspace(-tspan/2.0, tspan/2.0, N)
        t = self.pulse.t
        Et = self.pulse.Et
        Ifrog = []
        
        signalPulse.setEt(Et*Et*np.conj(Et), t)
        l, Il = signalPulse.getSpectrogram(t.shape[0])
        l_shift = signalPulse.l0 - l0
        nl_shift = np.int(l_shift/np.abs(l[1]-l[0]))
        self.l_vec = l + l_shift
        
        shiftVec = (np.arange(N) - N/2).astype(np.int)
        
        logger.debug(''.join(('l_shift: ', str(l_shift))))
        logger.debug(''.join(('nl_shift: ', str(nl_shift))))
        
        Esig_t_tau = []
        for sh in shiftVec:
            Ils = np.zeros_like(Il)
            signalPulse.setEt(Et*Et*np.conj(self.pulse.getShiftedEt(sh)), t)
            l, Iln = signalPulse.getSpectrogram(t.shape[0])
            Ils = np.roll(Iln, nl_shift)
            Ifrog.append(Ils)
            Esig_t_tau.append(signalPulse.Et)
            
        Ifrog = np.array(Ifrog).real.transpose()
        self.Ifrog = Ifrog/Ifrog.max()       
        self.Esig_t_tau = np.array(Esig_t_tau) 
        
        return self.Ifrog
    
    def generatePGTraceDt(self, N, dt, l0):
        signalPulse = SimulatedPulse(self.pulse.N, self.pulse.dt, self.pulse.l0, self.pulse.tau)
        tspan = N * dt
        self.tau_vec = np.linspace(-tspan / 2.0, tspan / 2.0, N)
        t = self.pulse.t
        Et = self.pulse.Et
        Ifrog = []

        signalPulse.setEt(Et * Et * np.conj(Et), t)
        l, Il = signalPulse.getSpectrogram(t.shape[0])
        l_shift = signalPulse.l0 - l0
        nl_shift = np.int(l_shift / np.abs(l[1] - l[0]))
        self.l_vec = l + l_shift

        shiftVec = (np.arange(N) - N / 2).astype(np.int)

        logger.debug(''.join(('l_shift: ', str(l_shift))))
        logger.debug(''.join(('nl_shift: ', str(nl_shift))))

        Esig_t_tau = []
        for sh in shiftVec:
            signalPulse.setEt(Et*np.abs(self.pulse.getShiftedEt(sh))**2, t)
            l, Iln = signalPulse.getSpectrogram(t.shape[0])
            Ils = np.roll(Iln, nl_shift)
            Ifrog.append(Ils)
            Esig_t_tau.append(signalPulse.Et)

        Ifrog = np.array(Ifrog).real.transpose()
        self.Ifrog = Ifrog / Ifrog.max()
        self.Esig_t_tau = np.array(Esig_t_tau)

        return self.Ifrog

    def generatePGTraceDt_w(self, N, dt, l0):
        signalPulse = SimulatedPulse(self.pulse.N, self.pulse.dt, self.pulse.l0, self.pulse.tau)
        tspan = N * dt

        # Now we calculate the frequency resolution required by the
        # time span
        w_res = 2 * np.pi / tspan

        # Frequency span is given by the time resolution
        f_max = 1 / (2 * dt)
        w_span = f_max * 2 * 2 * np.pi
        c = 299792458.0
        w0 = 2 * np.pi * c / l0   # Keep this so that the frog trace is centered in the field of view
        # w0 = w0 / 2 * (1 + np.sqrt(1 + w_span**2 / 4 / (w0/2)**2))
        w_spectrum = np.linspace(-w_span / 2, -w_span / 2 + w_res * N, N)

        self.tau_vec = np.linspace(-tspan / 2.0, tspan / 2.0, N)
        t = self.pulse.t
        Et = self.pulse.Et
        Ifrog = []

        signalPulse.setEt(Et * Et * np.conj(Et), t)
        l, Il = signalPulse.getSpectrogram(t.shape[0])
        l_shift = signalPulse.l0 - l0
        nl_shift = np.int(l_shift / np.abs(l[1] - l[0]))
        self.l_vec = l + l_shift

        shiftVec = (np.arange(N) - N / 2).astype(np.int)

        logger.debug(''.join(('l_shift: ', str(l_shift))))
        logger.debug(''.join(('nl_shift: ', str(nl_shift))))

        Esig_t_tau = []
        for sh in shiftVec:
            Iws = np.zeros(N)
            signalPulse.setEt(Et*np.abs(self.pulse.getShiftedEt(sh))**2, t)
            w, Ewn = signalPulse.getFreqSpectrum(t.shape[0])
            Iwn = abs(Ewn)**2
            Iws = np.roll(Iwn, nl_shift)
            Ifrog.append(Iws)
            Esig_t_tau.append(signalPulse.Et)

        Ifrog = np.array(Ifrog).real.transpose()
        self.Ifrog = Ifrog / Ifrog.max()
        self.Esig_t_tau = np.array(Esig_t_tau)

        return self.Ifrog

    def addNoise(self, shotAmp=0.1, readAmp=0.05):
        self.Ifrog = np.maximum(0, self.Ifrog + 
                                np.random.poisson(self.Ifrog/shotAmp)*shotAmp + 
                                np.random.standard_normal(self.Ifrog.shape)*readAmp)
        return self.Ifrog
    
    def getWavelengths(self):
        return self.l_vec
    
    def getTimedelays(self):
        return self.tau_vec


if __name__ == '__main__':
    Ns = 512
    dt = 5e-15
    tau = 100e-15 / (2*np.sqrt(np.log(2)))
    l0 = 263.5e-9
    p = SimulatedPulse(Ns, dt, l0, tau)
#     p.generateGaussianCubicPhase(5e24, 1e40)
#     p.generateGaussianCubicPhase(-5e26, 0)
    p.generateGaussianCubicSpectralPhase(0e-27, 0.3e-40)
#     p.generateDoublePulse(tau, 0.5e-12)
#     p.generateGaussian(tau)
#     p.addChirp(1e26)
    gt = SimulatedFrogTrace(Ns, dt, l0)
    gt.pulse = p
    IfrogSHG = gt.generateSHGTraceDt(Ns, dt, l0/2)
    IfrogSD = gt.generateSDTraceDt(Ns, dt, l0)
    IfrogPG = gt.generatePGTraceDt(Ns, dt, l0)
    l = gt.getWavelengths()
    t = gt.getTimedelays()

    N = 256
    dt = 4e-15
    l0 = 780e-9
    tau = 30e-15 / (2 * np.sqrt(np.log(2)))
    p = SimulatedPulse(N, dt, l0, tau)

    Et1 = p.Et.copy()
    Ew = np.fft.fftshift(np.fft.fft(np.fft.fftshift(p.Et), p.N))
    Ew2 = p.propagate_material(Ew, "bk7", 20e-3)
    Et2 = np.fft.ifft(Ew2*np.exp(1j*p.w0*p.t))
    em = abs(Et2).argmax()
    Et2 = np.roll(Et2 / np.abs(Et2).max(), int(N/2-em))
    Ew3 = p.propagate_material(Ew2, "bk7", -20e-3)
    Et3 = np.fft.ifft(Ew3*np.exp(1j*p.w0*p.t))
    em = abs(Et3).argmax()
    Et3 = np.roll(Et3 / np.abs(Et3).max(), int(N/2-em))

    # Et1 = p.Et.copy()
    # mat_list = [("bk7", 10e-3)]
    # Et2 = p.propagate_material_list(mat_list)
    # mat_list2 = [("bk7", -10e-3)]
    # Et3 = p.propagate_material_list(mat_list2)
