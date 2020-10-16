"""
Created on Jan 4, 2018

@author: Filip Lindau
"""
# -*- coding:utf-8 -*-

from PyQt5 import QtGui, QtCore, QtWidgets

from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter
try:
    from scipy.misc import imread
except ImportError:
    from imageio import imread
from PIL import Image

import time
import sys
import os

import PyTango as pt
import threading
import numpy as np
import pyqtgraph as pq

import FrogCalculationSimpleGP as FrogCalculation

# import FrogCalculationCLGP as FrogCalculation

sys.path.insert(0, '../../TangoWidgetsQt5')
from AttributeReadThreadClass import AttributeClass

import logging

root = logging.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
root.addHandler(fh)
root.setLevel(logging.DEBUG)

# camera_name = "b-v0-gunlaser-csdb-0:10000/gunlaser/cameras/jai_test"
camera_name = "b-v0-gunlaser-csdb-0:10000/gunlaser/cameras/spectrometer_camera"
# motor_name = "b-v0-gunlaser-csdb-0:10000/testgun/motors/zst25"
motor_name = "b-v0-gunlaser-csdb-0:10000/gunlaser/motors/zaber01"

pq.graphicsItems.GradientEditorItem.Gradients['greyclip2'] = {
    'ticks': [(0.0, (0, 0, 50, 255)), (0.0001, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}
pq.graphicsItems.GradientEditorItem.Gradients['thermalclip'] = {
    'ticks': [(0, (0, 0, 50, 255)), (0.0001, (0, 0, 0, 255)), (0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)),
              (1, (255, 255, 255, 255))], 'mode': 'rgb'}


class TangoDeviceClient(QtWidgets.QWidget):
    """
    Class for scanning a motor while grabbing images to produce a frog trace. It can also analyse the scanned trace
    or saved traces.
    """

    def __init__(self, camera_name, motor_name, parent=None):
        root.debug("Init")
        QtWidgets.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings('Maxlab', 'Frog')
        #        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.time_vector = None
        self.x_data = None
        self.x_data_temp = None
        self.camera_name = camera_name
        self.motor_name = motor_name

        self.gui_lock = threading.Lock()

        self.position_offset = 0.0

        self.devices = dict()
        self.attributes = dict()

        self.roi_data = np.zeros((2, 2))
        self.scan_wavelengths = None  # Wavelengths corresponding to pixels in the selected roi in the image
        self.spectrum_data_avg = None
        self.trend_data2 = None
        self.scan_data = np.array([])
        self.frog_raw_data = np.array([])
        self.frog_filtered_data = np.array([])
        self.frog_wavelengths = np.array([])
        self.frog_time_data = np.array([])
        self.frog_central_wavelength = 0.0
        self.scan_time_data = np.array([])
        self.time_marginal = np.array([])
        self.spectrum_marginal = np.array([])
        self.time_marginal_tmp = 0
        self.pos_data = np.array([])
        self.current_pos = 0.0
        self.current_sample = 0
        self.avg_data = 0
        self.avg_samples = None
        self.target_pos = 0.0
        self.measure_update_times = np.array([time.time()])
        self.lock = threading.Lock()

        self.spectrum_data = None  # Last recorded spectrum

        self.running = False
        self.scanning = False
        self.moving = False
        self.move_start = False
        self.scan_timer = QtCore.QTimer()
        self.scan_timer.timeout.connect(self.scan_update_action)

        self.frog_calc = FrogCalculation.FrogCalculation()  # Frog calculations object. Implements all the
        # frog inversion calculations.

        # Predefine all widgets
        self.frog_n_spinbox = None
        self.frog_method_combobox = None
        self.frog_algo_combobox = None
        self.frog_start_button = None
        self.frog_continue_button = None
        self.frog_iterations_spinbox = None
        self.frog_threshold_spinbox = None
        self.frog_fit_roi_button = None
        self.frog_dt_label = None
        self.frog_dt_spinbox = None
        self.frog_kernel_median_spinbox = None
        self.frog_kernel_gaussian_spinbox = None
        self.trace_source_label = None
        self.frog_load_button = None
        self.frog_use_scan_button = None
        self.frog_et_checkbox = None
        self.frog_phase_checkbox = None
        self.frog_et_fwhm_label = None
        self.frog_et_phase_label = None
        self.frog_temporal_spectral_combobox = None
        self.frog_expansion1 = None
        self.frog_expansion2 = None
        self.frog_expansion3 = None
        self.frog_expansion4 = None
        self.frog_grid_layout1 = None
        self.frog_grid_layout2 = None
        self.frog_grid_layout3 = None
        self.frog_grid_layout4 = None
        self.frog_grid_layout5 = None
        self.frog_error_widget = None
        self.frog_error_plot = None
        self.frog_result_widget = None
        self.frog_result_plotitem = None
        self.frog_result_viewbox = None
        self.frog_result_plot_phase = None
        self.frog_image_widget = None
        self.frog_roi = None
        self.frog_roi_image_widget = None
        self.frog_calc_image_widget = None
        self.frog_calc_result_image_widget = None
        self.frog_layout2 = None
        self.frog_result_plot_abs = None
        self.layout = None
        self.tab_widget = None
        self.grid_layout1 = None
        self.grid_layout2 = None
        self.grid_layout3 = None
        self.grid_layout4 = None
        self.grid_layout5 = None
        self.fps_label = None
        self.average_spinbox = None
        self.avg_samples = None
        self.start_pos_spinbox = None
        self.step_size_label = None
        self.step_size_spinbox = None
        self.set_pos_spinbox = None
        self.current_pos_label = None
        self.current_speed_label = None
        self.export_filename_edit = None
        self.export_file_location_edit = None
        self.center_wavelength_spinbox = None
        self.dispersion_spinbox = None
        self.shutter_label = None
        self.shutter_spinbox = None
        self.gain_label = None
        self.gain_spinbox = None
        self.normalize_pump_check = None
        self.time_units_radio = None
        self.pos_units_radio = None
        self.start_button = None
        self.stop_button = None
        self.export_button = None
        self.spectrum_image_widget = None
        self.roi = None
        self.spectrum_plot_widget = None
        self.plot1 = None
        self.plot2 = None
        self.scan_image_widget = None
        self.timemarginal_plot_widget = None
        self.timemarginal_plot = None
        self.time_marginal_fwhm_label = None
        self.spectrummarginal_plot_widget = None
        self.spectrummarginal_plot = None
        self.spectrum_marginal_fwhm_label = None
        self.bandwidth_limited_fwhm_label = None
        self.plot_layout = None
        self.spectrum_layout = None
        self.cam_grid_layout1 = None
        self.marginals_layout = None
        self.tab_camera_widget = None
        self.tab_scan_widget = None
        self.tab_frog_widget = None
        self.invisible_layout = None
        self.camera_start_button = None
        self.camera_stop_button = None
        self.camera_init_button = None
        self.camera_select_lineedit = None
        self.motor_select_lineedit = None
        self.camera_change_button = None
        self.motor_change_button = None
        self.device_select_layout = None

        self.setup_layout()
        self.select_camera()
        self.select_motor()

        root.debug("Init finished")

    def read_image(self, data):
        # root.debug("Read image")
        im_data = np.transpose(data.value)
        self.spectrum_image_widget.setImage(im_data, autoRange=False, autoLevels=True)
        self.roi_data = self.roi.getArrayRegion(im_data, self.spectrum_image_widget.getImageItem())
        self.measure_data()
        self.spectrum_image_widget.update()

    def read_shutter(self, data):
        self.shutter_label.setText(str(data.value))

    def write_shutter(self):
        w = self.shutter_spinbox.value()
        self.attributes['shutter'].attr_write(w)
        root.debug(''.join(('Write shutter ', str(w))))

    def read_gain(self, data):
        self.gain_label.setText(str(data.value))

    def write_gain(self):
        w = self.gain_spinbox.value()
        self.attributes['gain'].attr_write(w)
        root.debug(''.join(('Write gain ', str(w))))

    def read_position(self, data):
        if data is not None:
            if data.value is not None:
                self.current_pos_label.setText("{0:.3f} mm".format(data.value))
                self.current_pos = data.value
                if np.abs(self.target_pos - data.value) < 0.001:
                    self.moving = False

    def read_speed(self, data):
        if data is not None:
            if data.value is not None:
                self.current_speed_label.setText("{0:.3f}".format(data.value))
                if self.move_start is True:
                    if data.value > 0.01:
                        self.moving = True
                        self.move_start = False
                if self.moving is True:
                    if np.abs(data.value) < 0.001:
                        self.moving = False

    def write_position(self):
        w = self.set_pos_spinbox.value()
        self.attributes['position'].attr_write(w)

    def start_camera(self):
        root.debug("Sending start command to camera")
        self.devices["camera"].command_inout("start")

    def stop_camera(self):
        root.debug("Sending stop command to camera")
        self.devices["camera"].command_inout("stop")

    def init_camera(self):
        root.debug("Sending init command to camera")
        self.devices["camera"].command_inout("init")

    def select_camera(self):
        root.debug("Closing down camera attributes")
        if "image" in self.attributes:
            self.attributes["image"].stop_read()
            self.attributes["shutter"].stop_read()
            self.attributes["gain"].stop_read()
            self.attributes["image"].read_thread.join(0.5)
            self.attributes["shutter"].read_thread.join(0.5)
            self.attributes["gain"].read_thread.join(0.5)
        camera_tango_name = str(self.camera_select_lineedit.text())
        root.debug("Connecting to {0}".format(camera_tango_name))
        try:
            new_device = pt.DeviceProxy(camera_tango_name)
        except pt.DevFailed:
            root.debug("Could not connect")
            return
        self.devices["camera"] = new_device
        self.attributes['image'] = AttributeClass('image', self.devices['camera'], 0.1)
        self.attributes['shutter'] = AttributeClass('exposuretime', self.devices['camera'], 0.5)
        self.attributes['gain'] = AttributeClass('gain', self.devices['camera'], 0.5)

        self.attributes['image'].attrSignal.connect(self.read_image)
        self.attributes['shutter'].attrSignal.connect(self.read_shutter)
        self.attributes['gain'].attrSignal.connect(self.read_gain)

    def select_motor(self):
        root.debug("Closing down motor attributes")
        if "position" in self.attributes:
            self.attributes["position"].stop_read()
            self.attributes["speed"].stop_read()
            self.attributes["position"].read_thread.join(0.5)
            self.attributes["speed"].read_thread.join(0.5)
        motor_tango_name = str(self.motor_select_lineedit.text())
        root.debug("Connecting to {0}".format(motor_tango_name))
        try:
            new_device = pt.DeviceProxy(motor_tango_name)
        except pt.DevFailed:
            root.debug("Could not connect")
            return
        self.devices["motor"] = new_device
        self.attributes['position'] = AttributeClass('position', self.devices['motor'], 0.05)
        attr_list = [x.lower() for x in self.devices["motor"].get_attribute_list()]
        if "speed" in attr_list:
            speed_attr = AttributeClass('speed', self.devices['motor'], 0.05)
        elif "velocity" in attr_list:
            speed_attr = AttributeClass('velocity', self.devices['motor'], 0.05)
        else:
            root.debug("Could not find speed attribute name")
            return

        self.attributes['speed'] = speed_attr

        self.attributes['position'].attrSignal.connect(self.read_position)
        self.attributes['speed'].attrSignal.connect(self.read_speed)

    def set_average(self):
        self.avg_samples = self.average_spinbox.value()
        root.debug("Averging " + str(self.avg_samples) + " samples")

    def set_step_size(self):
        time_step = self.step_size_spinbox.value() * 1e-3 * 2 / 299792458.0 * 1e15
        root.debug("Time step " + str(time_step) + " fs")
        self.step_size_label.setText("Step size ({0:.1f} fs)".format(time_step))

    def generate_wavelengths(self):
        l0 = self.center_wavelength_spinbox.value() * 1e-9
        dl = self.dispersion_spinbox.value() * 1e-9
        im_size = self.spectrum_image_widget.getImageItem().image.shape[0]
        #         root.debug(''.join(('Image size: ', str(im_size))))
        l_range = dl * im_size
        global_wavelengths = np.linspace(l0 - l_range / 2, l0 + l_range / 2, im_size)
        start_ind = np.int(self.roi.pos()[0])
        stop_ind = np.int(self.roi.pos()[0] + np.ceil(self.roi.size()[0]))
        root.debug("start_ind: " + str(start_ind) + ", stop_ind: " + str(stop_ind))
        self.scan_wavelengths = global_wavelengths[start_ind:stop_ind]

        if self.scan_wavelengths.shape[0] == self.spectrum_data.shape[0]:
            self.plot1.setData(x=self.scan_wavelengths, y=self.spectrum_data)

    def start_scan(self):
        self.scan_data = None
        self.spectrum_data_avg = None
        self.trend_data2 = None
        self.time_marginal_tmp = 0
        self.scan_time_data = np.array([])
        self.time_marginal = np.array([])
        self.spectrum_marginal = np.array([])
        self.pos_data = np.array([])
        self.scanning = True
        self.move_start = True
        self.moving = True
        self.running = True
        self.target_pos = self.start_pos_spinbox.value()
        self.attributes['position'].attr_write(self.target_pos)
        self.trace_source_label.setText('Scan')

    #        self.scanTimer.start(100 * self.avgSamples)

    def stop_scan(self):
        root.debug("Stopping scan")
        self.running = False
        self.scanning = False
        self.scan_timer.stop()

    def export_scan(self):
        root.debug('Exporting scan data')
        if self.scan_data.max() > 256:
            data = np.uint8(np.double(self.scan_data) / self.scan_data.max() * 256)
        else:
            data = np.uint8(self.scan_data)
        path_name = str(self.export_file_location_edit.text())
        file_name_base = ''.join(('frogtrace_', time.strftime('%Y-%m-%d_%Hh%M_'),
                                  str(self.export_filename_edit.text())))
        filename = os.path.join(path_name, file_name_base + '_image.png')
        im = Image.fromarray(data)
        im.save(filename)
        data = self.scan_time_data
        filename = os.path.join(path_name, file_name_base + '_timevector.txt')
        np.savetxt(filename, data)
        data = self.scan_wavelengths
        filename = os.path.join(path_name, file_name_base + '_wavelengthvector.txt')
        np.savetxt(filename, data)

    def scan_update_action(self):
        self.scan_timer.stop()
        while self.running is True:
            time.sleep(0.02)
        new_pos = self.target_pos + self.step_size_spinbox.value()
        root.debug('New pos: ' + str(new_pos))
        self.attributes['position'].attr_write(new_pos)
        self.target_pos = new_pos
        self.running = True
        self.move_start = True

    def measure_scan_data(self):
        """
        Update frag raw data image and time marginal.

        Called by measure_data if a scan is running.
        :return:
        """
        self.avg_data = self.spectrum_data_avg / self.avg_samples
        if self.scan_data is None:
            self.scan_data = np.array([self.avg_data])
            self.time_marginal = np.hstack((self.time_marginal, self.time_marginal_tmp / self.avg_samples))
            self.time_marginal_tmp = 0.0
        else:
            self.scan_data = np.vstack((self.scan_data, self.avg_data))
            self.time_marginal = np.hstack((self.time_marginal, self.time_marginal_tmp / self.avg_samples))
            self.time_marginal_tmp = 0.0
        pos = self.current_pos
        new_time = (pos - self.start_pos_spinbox.value()) * 2 * 1e-3 / 299792458.0
        self.scan_time_data = np.hstack((self.scan_time_data, new_time))
        self.pos_data = np.hstack((self.pos_data, pos * 1e-3))
        root.debug(''.join(('Time vector: ', str(self.scan_time_data))))
        root.debug(''.join(('Time marginal: ', str(self.time_marginal))))
        x0, x1 = (self.scan_wavelengths[0], self.scan_wavelengths[-1])
        y0, y1 = (self.scan_time_data[0], self.scan_time_data[-1])
        xscale, yscale = (x1 - x0) / self.scan_wavelengths.shape[0], (y1 - y0) / self.scan_time_data.shape[0]
        self.scan_image_widget.setImage(np.transpose(self.scan_data), autoRange=False, autoLevels=True,
                                        scale=[xscale, yscale], pos=[x0, y0])
        self.calculate_marginal_parameters()

    def measure_data(self):
        """
        Function called whenever new data is available. Updates the spectrum plot and frog raw data
        and time marginal if a scan is running
        :return:
        """
        # root.debug("Measure data")
        self.spectrum_data = np.sum(self.roi_data, 1) / self.roi_data.shape[1]
        if self.scan_wavelengths is None:
            self.generate_wavelengths()
        if self.scan_wavelengths.shape[0] == self.spectrum_data.shape[0]:
            self.plot1.setData(x=self.scan_wavelengths, y=self.spectrum_data)
            self.plot1.update()

        # Evaluate the fps
        t = time.time()
        if self.measure_update_times.shape[0] > 10:
            self.measure_update_times = np.hstack((self.measure_update_times[1:], t))
        else:
            self.measure_update_times = np.hstack((self.measure_update_times, t))
        fps = 1 / np.diff(self.measure_update_times).mean()
        self.fps_label.setText("{0:.1f}".format(fps))

        # If we are running a scan, update the scan data
        if self.running is True:
            if self.moving is False and self.move_start is False:
                if self.spectrum_data.shape[0] > 100:
                    bkg_ind = 10
                else:
                    bkg_ind = np.int(self.spectrum_data.shape[0] * 0.1)
                spectrum_bkg = self.spectrum_data[0:bkg_ind].mean()*1.05
                if self.spectrum_marginal.shape[0] == 0:
                    self.spectrum_marginal = self.spectrum_data - spectrum_bkg
                else:
                    self.spectrum_marginal += self.spectrum_data - spectrum_bkg

                if self.spectrum_data_avg is None:
                    self.spectrum_data_avg = self.spectrum_data
                    self.time_marginal_tmp = (self.spectrum_data - spectrum_bkg).sum()
                else:
                    self.spectrum_data_avg += self.spectrum_data
                    self.time_marginal_tmp += (self.spectrum_data - spectrum_bkg).sum()
                self.current_sample += 1
                if self.current_sample >= self.avg_samples:
                    self.running = False
                    self.measure_scan_data()
                    self.spectrum_data_avg = None
                    self.current_sample = 0
                    self.scan_update_action()
            elif self.moving is False and self.move_start is True:
                # This condition discards the first image when starting a scan.
                # It allows a fresh image to be captured.
                self.move_start = False

    def x_axis_units_toggle(self):
        """
        Set the unit of the x-axis in the time marginal plot, ps or mm
        :return:
        """
        if self.time_units_radio.isChecked() is True:
            self.timemarginal_plot.setData(x=self.scan_time_data * 1, y=self.time_marginal)
            self.timemarginal_plot_widget.setLabel(axis="bottom", text="Time delay", units="s")
        else:
            self.timemarginal_plot.setData(x=self.pos_data * 1, y=self.time_marginal)
            self.timemarginal_plot_widget.setLabel(axis="bottom", text="Delay position", units="m")

    def update_frog_plot_view(self):
        """
        Update view in the frog data and result data images
        :return:
        """
        self.frog_result_viewbox.setGeometry(self.frog_result_plotitem.vb.sceneBoundingRect())
        self.frog_result_viewbox.linkedViewChanged(self.frog_result_plotitem.vb, self.frog_result_viewbox.XAxis)

    def update_frog_result_plot(self):
        """
        Update the calculated field intensity and phase plots
        :return:
        """
        if self.frog_temporal_spectral_combobox.currentText() == "Temporal":
            x = self.frog_calc.get_t()
            efield_abs = self.frog_calc.get_trace_abs() ** 2
            efield_phi = self.frog_calc.get_trace_phase(linear_comp=True)
        else:
            x = self.frog_calc.get_w()
            efield_abs = self.frog_calc.get_trace_spectral_abs() ** 2
            efield_phi = self.frog_calc.get_trace_spectral_phase(linear_comp=True)

        self.frog_result_plot_abs.setData(x=x, y=efield_abs)
        self.frog_result_plot_abs.update()
        self.frog_result_plot_phase.setData(x=x, y=efield_phi)
        self.frog_result_plot_phase.update()

        if self.frog_et_checkbox.isChecked() is True:
            self.frog_result_plot_abs.show()
        else:
            self.frog_result_plot_abs.hide()

        if self.frog_phase_checkbox.isChecked() is True:
            self.frog_result_plot_phase.show()
        else:
            self.frog_result_plot_phase.hide()

    def calculate_pulse_parameters(self):
        """
        Calculate fwhm duration and phase for the calculated electric field
        :return:
        """
        t_fwhm, delta_ph = self.frog_calc.get_trace_summary()
        s_t = '%.3f' % (t_fwhm * 1e12)
        self.frog_et_fwhm_label.setText(''.join((s_t, ' ps')))

        if self.frog_temporal_spectral_combobox.currentText() == 'Temporal':
            ph = self.frog_calc.get_trace_phase()
        else:
            ph = self.frog_calc.get_trace_spectral_phase()
        ph_good = ph[np.isfinite(ph)]
        ph_good
        self.frog_et_phase_label.setText('{0:.2f} rad'.format(delta_ph))

        self.update_expansion_coefficients()

    def find_fwhm(self, y, x=None, algo="intersect", bkg_subtract=False):
        """
        Find the FWHM of the supplied data vector.

        :param y: data vector
        :param x: optional x vector
        :param algo: algorithm used to find the fwhm.
                        "intersect":   data index intersecting the 0.5 height
                        "moment":      use second moment of the distribution to find width
                                       (assumes gaussian-like distribution)
                         # "gaussian": use gaussian fit # not implemented
        :param bkg_subtract: Indicates whether background subtraction should be done on the data before calculation
        :return: fwhm, central x
        """
        root.debug("Entering find_fwhm")
        if bkg_subtract is True:
            y_bkg_ind = np.minimum(10, np.int(y.shape[0] * 0.1))
            if y_bkg_ind > 0:
                y_bkg = y[0:y_bkg_ind].mean()
                y_m = y - y_bkg
            else:
                y_m = y
        else:
            y_m = y
        if x is None:
            x = np.arange(y_m.shape[0])
        if algo == "intersect":
            root.debug("Using intersect algo")
            y_max_ind = y_m.argmax()
            y_n = y_m / y_m[y_max_ind]
            x0 = x[y_max_ind]
            x_int = np.where(np.diff(np.signbit(y_n - 0.5)))[0]
            root.debug("Zero crossings: {0}".format(x_int))
            if len(x_int) > 1:
                fwhm = x[x_int[-1]] - x[x_int[0]]
            elif len(x_int) == 1:
                fwhm = x[len(y)-1] - x[x_int[0]]
            else:
                fwhm = 0.0
        else:
            root.debug("Using central moment algo")
            x0 = (x * y_m).sum() / y_m.sum()
            x_std = np.sqrt(y_m * (x - x0)**2 / y_m.sum())
            fwhm = 2 * np.sqrt(2*np.log(2)) * x_std
        root.debug("FWHM of data: {0}, central x: {1}".format(fwhm, x0))
        return fwhm, x0

    def calculate_marginal_parameters(self):
        """
        Calculate FWHM duration and FWHM spectral width of marginal traces. The first 10% or 10 samples (lowest)
        are used as the background. It is not useful to use the last samples as the background since we want to
        calculate during the scan that builds up the marginal traces.
        :return:
        """
        if self.time_units_radio.isChecked() is True:
            self.timemarginal_plot.setData(x=self.scan_time_data * 1, y=self.time_marginal)
            t_fwhm, t0 = self.find_fwhm(self.time_marginal, self.scan_time_data, bkg_subtract=True)
        else:
            self.timemarginal_plot.setData(x=self.pos_data * 1, y=self.time_marginal)
            t_fwhm, t0 = self.find_fwhm(self.time_marginal, self.pos_data, bkg_subtract=True)
        self.time_marginal_fwhm_label.setText("{0:.3f} ps".format(t_fwhm*1e12))

        self.spectrummarginal_plot.setData(x=self.scan_wavelengths, y=self.spectrum_marginal)
        # Use np.abs on the spectrum fwhm since the wavelength vector could be inverted.
        s_fwhm, l0 = np.abs(self.find_fwhm(self.spectrum_marginal, self.scan_wavelengths, bkg_subtract=True))
        self.spectrum_marginal_fwhm_label.setText("{0:.2f} nm".format(s_fwhm*1e9))
        t_limited = 0.44 / (299792458.0 / l0**2 * s_fwhm)
        self.bandwidth_limited_fwhm_label.setText("{0:.3f} ps".format(t_limited * 1e12))

    def update_expansion_coefficients(self):
        """
        Calculate field polynomial expansion coefficients
        :return:
        """
        if self.frog_temporal_spectral_combobox.currentText() == 'Temporal':
            p = self.frog_calc.get_temporal_phase_expansion(orders=4, prefix=1e-15)
            self.frog_expansion1.setText("{0:.2e} fs".format(p[3]))
            self.frog_expansion2.setText("{0:.2e} fs^2".format(p[2]))
            self.frog_expansion3.setText("{0:.2e} fs^3".format(p[1]))
            self.frog_expansion4.setText("{0:.2e} fs^4".format(p[0]))
        else:
            p = self.frog_calc.get_spectral_phase_expansion(orders=4, prefix=2 * np.pi * 1e15)
            self.frog_expansion1.setText("{0:.2e} fs^-1".format(p[3]))
            self.frog_expansion2.setText("{0:.2e} fs^-2".format(p[2]))
            self.frog_expansion3.setText("{0:.2e} fs^-3".format(p[1]))
            self.frog_expansion4.setText("{0:.2e} fs^-4".format(p[0]))

        self.update_frog_result_plot()

    def update_image_threshold(self):
        """
        Update the frog images due to new threshold value
        :return:
        """
        h = self.frog_image_widget.getHistogramWidget()
        levels = h.item.getLevels()
        root.debug('Levels: ' + str(levels))
        root.debug('frogRawData.max() ' + str(self.frog_raw_data.max()))
        self.frog_threshold_spinbox.setValue(levels[0] / self.frog_raw_data.max())
        self.update_frog_roi()

    def update_frog_roi(self):
        """
        Update frog images due to new roi set.
        :return:
        """
        root.debug(
            "Roi pos: {0:.2f} nm, {1:.2f} ps".format(self.frog_roi.pos()[0] * 1e9, self.frog_roi.pos()[1] * 1e12))
        root.debug(
            "Roi size: {0:.2f} nm, {1:.2f} ps".format(self.frog_roi.size()[0] * 1e9, self.frog_roi.size()[1] * 1e12))
        if self.frog_raw_data.size != 0:
            root.debug(''.join(('Raw data: ', str(self.frog_raw_data.shape))))

            bkg = self.frog_raw_data[0, :]
            bkg_img = self.frog_raw_data - np.tile(bkg, (self.frog_raw_data.shape[0], 1))
            x0, y0 = self.frog_roi.pos()
            x1, y1 = self.frog_roi.size()
            x1 += x0
            y1 += y0
            dx = self.frog_wavelengths[1] - self.frog_wavelengths[0]
            dy = (self.frog_time_data[-1] - self.frog_time_data[0]) / self.frog_time_data.shape[0]
            x0_ind = np.maximum(np.int((x0 - self.frog_wavelengths[0]) / dx), 0)
            x1_ind = np.minimum(np.int((x1 - self.frog_wavelengths[0]) / dx), self.frog_wavelengths.shape[0] - 1)
            y0_ind = np.maximum(np.int((y0 - self.frog_time_data[0]) / dy), 0)
            y1_ind = np.minimum(np.int((y1 - self.frog_time_data[0]) / dy), self.frog_time_data.shape[0] - 1)

            if x0_ind > x1_ind:
                x0_ind, x1_ind = x1_ind, x0_ind
            if y0_ind > y1_ind:
                y0_ind, y1_ind = y1_ind, y0_ind

            # x0_ind = np.int(np.maximum(x0, 0))
            # x1_ind = np.int(np.maximum(x1, self.frog_wavelengths.shape[0]-1))
            # y0_ind = np.int(np.maximum(y0, 0))
            # y1_ind = np.int(np.maximum(y1, self.frog_time_data.shape[0]-1))

            root.debug('x0_ind ' + str(x0_ind))
            root.debug('x1_ind ' + str(x1_ind))
            root.debug('y0_ind ' + str(y0_ind))
            root.debug('y1_ind ' + str(y1_ind))
            root.debug("bkg_img shape {0}".format(bkg_img.shape))
            roi_img = bkg_img[y0_ind:y1_ind, x0_ind:x1_ind]
            root.debug("roi_img bkg subtracted")
            root.debug("roi_img shape {0}".format(roi_img.shape))
            try:
                roi_img = roi_img / roi_img.max()
                root.debug(''.join(('Roi pos: ', str(x0), 'm x ', str(y0), 's')))
                root.debug(''.join(('Roi size: ', str(x1 - x0), 'm x ', str(y1 - y0), 's')))
                dt_est = self.estimate_frog_dt(self.frog_n_spinbox.value(), y1 - y0, x0, x1)
                self.frog_dt_label.setText('dt (fs) [{0:.1f} - {1:.1f}]'.format(dt_est[0] * 1e15, dt_est[1] * 1e15))
                root.debug('Slice complete')
                thr = np.maximum(self.frog_threshold_spinbox.value() - bkg.mean() / self.frog_raw_data.max(), 0)
                root.debug('Threshold: ' + str(thr))
                kernel = self.frog_kernel_median_spinbox.value()
                root.debug('Starting medfilt...')
                if kernel > 1:
                    filtered_img = medfilt2d(roi_img, kernel) - thr
                else:
                    filtered_img = roi_img - thr
                gauss_kernel = self.frog_kernel_gaussian_spinbox.value()
                if gauss_kernel > 1:
                    filtered_img = gaussian_filter(filtered_img, gauss_kernel)

                root.debug('Filtering complete')
                filtered_img[filtered_img < 0.0] = 0.0
                self.frog_filtered_data = filtered_img
                root.debug('Threshold complete')
                x0, x1 = (self.frog_wavelengths[0], self.frog_wavelengths[-1])
                y0, y1 = (self.frog_time_data[0], self.frog_time_data[-1])
                xscale, yscale = (x1 - x0) / self.frog_wavelengths.shape[0], (y1 - y0) / self.frog_time_data.shape[0]
                self.frog_roi_image_widget.setImage(filtered_img.transpose(), scale=[xscale, yscale], pos=[x0, y0])
                root.debug('Set image complete')
                self.frog_roi_image_widget.autoRange()
                root.debug('Autorange complete')
            except ValueError:
                return

    def fit_frog_roi(self):
        root.debug("Entering fit_frog_roi")
        try:
            x0 = np.min(self.frog_wavelengths)
            y0 = np.min(self.frog_time_data)
        except NameError:
            return
        except ValueError:
            return
        dx = np.abs(self.frog_wavelengths[-1] - self.frog_wavelengths[0])
        dy = np.abs(self.frog_time_data[-1] - self.frog_time_data[0])
        root.debug("x0={0} , dx={1}, y0={2}".format(x0, dx, y0))
        self.frog_roi.setPos([x0, y0], finish=False, update=False)
        self.frog_roi.setSize([dx, dy])
        root.debug(''.join(('Roi pos: ', str(x0), 'm x ', str(y0), 's')))
        root.debug(''.join(('Roi size: ', str(dx), 'm x ', str(dy), 's')))

    def estimate_frog_dt(self, n=None, t_span=None, l_min=None, l_max=None):
        update = False
        if n is None:
            n = self.frog_n_spinbox.value()
            update = True
        if t_span is None:
            t_span = self.frog_roi.size()[1]
        if l_min is None:
            l_min = self.frog_roi.pos()[0]
            l_max = l_min + self.frog_roi.size()[0]
        dt_t = t_span / n
        c = 299792458.0
        dt_l = np.abs(1 / (1 / l_min - 1 / l_max) / c)
        root.debug('tspan {0:.3g}'.format(t_span))
        root.debug('N {0:.3g}'.format(n))
        root.debug('l_min {0:.3g}'.format(l_min))
        root.debug('l_max {0:.3g}'.format(l_max))
        # If N is None we assume that the widget should be updated here.
        if update is True:
            self.frog_dt_label.setText('dt (fs) [{0:.1f} - {1:.1f}]'.format(dt_t * 1e15, dt_l * 1e15))
        # root.debug('dt_t {0:.3g}'.format(dt_t))
        # root.debug('dt_l {0:.3g}'.format(dt_l))
        return dt_t, dt_l

    def start_frog_inversion(self):
        """
        Start new frog inversion calculation using currently selected parameters and roi. Clears all previous data.
        :return:
        """
        n = self.frog_n_spinbox.value()
        frog_img = self.frog_filtered_data
        if frog_img.size > 0:
            x0, y0 = self.frog_roi.pos()
            x1, y1 = self.frog_roi.size()
            x1 += x0
            y1 += y0
            dx = self.frog_wavelengths[1] - self.frog_wavelengths[0]
            dy = (self.frog_time_data[-1] - self.frog_time_data[0]) / self.frog_time_data.shape[0]
            x0_ind = np.maximum(np.int((x0 - self.frog_wavelengths[0]) / dx), 0)
            x1_ind = np.minimum(np.int((x1 - self.frog_wavelengths[0]) / dx), self.frog_wavelengths.shape[0] - 1)
            y0_ind = np.maximum(np.int((y0 - self.frog_time_data[0]) / dy), 0)
            y1_ind = np.minimum(np.int((y1 - self.frog_time_data[0]) / dy), self.frog_time_data.shape[0] - 1)

            l_start_ind = x0_ind
            l_stop_ind = x1_ind
            root.debug(''.join(('Wavelength range: ', str(self.frog_wavelengths[l_start_ind]), ' - ',
                                str(self.frog_wavelengths[l_stop_ind]))))
            if self.frog_wavelengths[l_start_ind] > 1:
                l_start = self.frog_wavelengths[l_start_ind] * 1e-9
                l_stop = self.frog_wavelengths[l_stop_ind] * 1e-9
            else:
                l_start = self.frog_wavelengths[l_start_ind]
                l_stop = self.frog_wavelengths[l_stop_ind]
            l0 = (l_stop + l_start) / 2

            t_start_ind = y0_ind
            t_stop_ind = y1_ind

            root.debug(''.join(('Time range: ', str(self.frog_time_data[t_start_ind]), ' - ',
                                str(self.frog_time_data[t_stop_ind]))))
            tau_mean = (self.frog_time_data[t_start_ind] + self.frog_time_data[t_stop_ind]) / 2
            tau_start = self.frog_time_data[t_start_ind] - tau_mean
            tau_stop = self.frog_time_data[t_stop_ind] - tau_mean
            if self.frog_time_data.shape != 0:
                dt = self.frog_dt_spinbox.value() * 1e-15
            else:
                dt = 1e-15

            root.debug('Wavelength input data: l_start=' + str(l_start) + ', l_stop=' + str(l_stop) +
                       ', type: ' + str(type(l_start_ind)))
            root.debug('Time input data: tau_start=' + str(tau_start) + ', tau_stop=' + str(tau_stop) + ', dt=' +
                       str(dt))

            self.frog_calc.init_pulsefield_random(n, dt, l0)
            self.frog_calc.condition_frog_trace(frog_img, l_start, l_stop, tau_start, tau_stop, n,
                                                thr=0 * self.frog_threshold_spinbox.value())

            root.debug('frog_img shape: ' + str(frog_img.shape) + ', 10 values: ' + str(frog_img[0, 0:10]))
            root.debug('I_w_tau shape: ' + str(self.frog_calc.I_w_tau.shape) + ', 10 values: ' +
                       str(self.frog_calc.I_w_tau[0, 0:10]))
            self.frog_calc_image_widget.setImage(np.transpose(self.frog_calc.I_w_tau))
            self.frog_calc_image_widget.autoRange()
            self.frog_calc_image_widget.update()

            self.frog_calc.G_hist = []
            self.frog_error_plot.setData([])
            self.frog_error_plot.update()

            self.continue_frog_inversion()

    def continue_frog_inversion(self):
        """
        Continue previously started frog calculation, allowing additional iterations to reach convergence.
        New algorithm and number of interations can be selected.
        :return:
        """
        if self.frog_calc.I_w_tau is not None:
            algo = str(self.frog_algo_combobox.currentText())
            if str(self.frog_method_combobox.currentText()) == 'Vanilla':
                er = self.frog_calc.run_cycle_vanilla(self.frog_iterations_spinbox.value(), algo=algo)
            elif str(self.frog_method_combobox.currentText()) == 'GP':
                er = self.frog_calc.run_cycle_gp(self.frog_iterations_spinbox.value(), algo=algo)
            er
            self.frog_error_plot.setData(self.frog_calc.G_hist)
            self.frog_error_plot.update()

            self.frog_calc_result_image_widget.setImage(np.transpose(self.frog_calc.get_reconstructed_intensity()))
            self.frog_calc_result_image_widget.autoRange()
            self.frog_calc_result_image_widget.update()

            self.calculate_pulse_parameters()

    def load_frog_trace(self):
        """
        Load a previously measured frog trace image.

        The image file is selected, but two text files with the same name except ending in _timevector.txt
        and _wavelengthvector.txt are expected to exist.
        :return:
        """
        file_location = str(self.export_file_location_edit.text())
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Select frog trace', file_location, 'frogtrace_*.png')
        root.debug(''.join(('File selected: ', str(filename))))

        if filename[0] != "":
            root.debug("Loading auxiliary files")
            try:
                f_name_root = '_'.join((filename[0].split('_')[:-1]))
                t_name = ''.join((f_name_root, '_timevector.txt'))
                l_name = ''.join((f_name_root, '_wavelengthvector.txt'))
                root.debug("{0}".format(filename[0]))
                root.debug("f_name_root : {0}".format(f_name_root))
                root.debug("Timevector name: {0}".format(t_name))
                root.debug(" Wavelengthvector name: {0}".format(l_name))
                t_data = np.loadtxt(t_name)
                t_data = t_data - t_data.mean()
                l_data = np.loadtxt(l_name)
                if l_data[0] > 1:
                    l_data = l_data * 1e-9
                pic = np.float32(imread(''.join((f_name_root, '_image.png'))))

                root.debug(''.join(('Pic: ', str(pic.shape))))
                root.debug(''.join(('Time data: ', str(t_data.shape), ' ', str(t_data[0]))))
                root.debug(''.join(('l data: ', str(l_data.shape), ' ', str(l_data[0]))))

                self.frog_raw_data = pic
                self.frog_time_data = t_data
                self.frog_wavelengths = l_data
                self.frog_central_wavelength = 1e9 * l_data.mean()

                x0, x1 = (self.frog_wavelengths[0], self.frog_wavelengths[-1])
                y0, y1 = (self.frog_time_data[0], self.frog_time_data[-1])
                xscale, yscale = (x1 - x0) / self.frog_wavelengths.shape[0], (y1 - y0) / self.frog_time_data.shape[0]
                self.frog_image_widget.setImage(np.transpose(pic), scale=[xscale, yscale], pos=[x0, y0])
                self.frog_image_widget.autoLevels()
                self.frog_image_widget.autoRange()
                self.fit_frog_roi()
                # self.updateFrogRoi()
                self.trace_source_label.setText(f_name_root.split('/')[-1])
            except:
                pass

    def load_scandata_to_frog(self):
        """
        Load the last recorded frog trace to the frog calculation widgets.
        :return:
        """
        root.debug("Loading scan data to frog")
        self.frog_raw_data = self.scan_data
        self.frog_time_data = self.scan_time_data
        self.frog_wavelengths = self.scan_wavelengths
        self.frog_central_wavelength = self.center_wavelength_spinbox.value()

        root.debug("Data loaded")

        x0, x1 = (self.frog_wavelengths[0], self.frog_wavelengths[-1])
        y0, y1 = (self.frog_time_data[0], self.frog_time_data[-1])
        xscale, yscale = (x1 - x0) / self.frog_wavelengths.shape[0], (y1 - y0) / self.frog_time_data.shape[0]
        root.debug("l_0 = {0}, t_0 = {1}".format(x0, y0))
        self.frog_image_widget.setImage(np.transpose(self.frog_raw_data), scale=[xscale, yscale], pos=[x0, y0])
        root.debug("setImage")
        self.frog_image_widget.autoLevels()
        root.debug("autoLevels")
        self.frog_image_widget.autoRange()
        root.debug("Image set")
        self.fit_frog_roi()
        # self.updateFrogRoi()
        self.trace_source_label.setText("Scan")

    def tab_changed(self, i):
        """
        The tab was changed. Need to shuffle around the widgets to accomodate the new view.
        :param i:
        :return:
        """
        root.debug(''.join(('Tab changed: ', str(i))))
        root.debug(''.join(('Found ', str(self.plot_layout.count()), ' widgets')))
        # Remove all widgets:
        for widget_ind in range(self.plot_layout.count()):
            lay_item = self.plot_layout.itemAt(0)
            if type(lay_item) is QtWidgets.QVBoxLayout:
                root.debug(''.join(('Found ', str(lay_item.count()), ' inner widgets')))
                for w_ind2 in range(lay_item.count()):
                    lay_item.takeAt(0)

            self.plot_layout.takeAt(0)
        self.plot_layout.removeWidget(self.timemarginal_plot_widget)

        # Re-populate
        if i == 0 or i == 1:
            # self.plot_layout.addWidget(self.spectrum_image_widget)
            # self.plot_layout.addWidget(self.spectrum_plot_widget)
            self.spectrum_layout.addWidget(self.spectrum_image_widget)
            self.spectrum_layout.addWidget(self.spectrum_plot_widget)
            self.plot_layout.addLayout(self.spectrum_layout)

            self.plot_layout.addWidget(self.scan_image_widget)
            self.marginals_layout.addWidget(self.timemarginal_plot_widget)
            self.marginals_layout.addWidget(self.spectrummarginal_plot_widget)
            self.plot_layout.addLayout(self.marginals_layout)
            self.timemarginal_plot_widget.setVisible(True)
            self.spectrummarginal_plot_widget.setVisible(True)
            self.spectrum_plot_widget.setVisible(True)
            self.spectrum_image_widget.setVisible(True)
            self.scan_image_widget.setVisible(True)
            self.scan_image_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
            self.frog_error_widget.setVisible(False)
            self.frog_result_widget.setVisible(False)
            self.frog_image_widget.setVisible(False)
            self.frog_roi_image_widget.setVisible(False)
            self.frog_calc_image_widget.setVisible(False)
            self.frog_calc_result_image_widget.setVisible(False)
            self.frog_roi.setVisible(False)

            if "image" in self.attributes:
                self.attributes['image'].unpause_read()
                self.attributes['position'].unpause_read()
                self.attributes['speed'].unpause_read()

        elif i == 2:
            self.timemarginal_plot_widget.setVisible(False)
            self.spectrummarginal_plot_widget.setVisible(False)
            self.spectrum_plot_widget.setVisible(False)
            self.spectrum_image_widget.setVisible(False)
            self.scan_image_widget.setVisible(False)
            self.frog_error_widget.setVisible(True)
            self.frog_roi_image_widget.setVisible(True)
            self.frog_image_widget.setVisible(True)
            self.frog_roi.setVisible(True)
            self.frog_result_widget.setVisible(True)
            self.frog_calc_image_widget.setVisible(True)
            self.frog_calc_result_image_widget.setVisible(True)

            self.marginals_layout.addWidget(self.frog_image_widget)
            self.marginals_layout.addWidget(self.frog_roi_image_widget)
            self.frog_image_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
            self.plot_layout.addLayout(self.marginals_layout)
            self.plot_layout.setStretchFactor(self.marginals_layout, 3)
            self.frog_layout2.addWidget(self.frog_calc_image_widget)
            self.frog_layout2.addWidget(self.frog_calc_result_image_widget)
            self.plot_layout.addLayout(self.frog_layout2)
            self.plot_layout.setStretchFactor(self.frog_layout2, 2)
            self.plot_layout.addWidget(self.frog_result_widget)
            self.plot_layout.setStretchFactor(self.frog_result_widget, 3)

            if "image" in self.attributes:
                self.attributes['image'].pause_read()
                self.attributes['position'].pause_read()
                self.attributes['speed'].pause_read()

    def closeEvent(self, event):
        """
        Closing the applications. Stopping the tango measurement threads and saving the settings.
        :param event:
        :return:
        """
        for a in self.attributes.itervalues():
            root.debug('Stopping' + str(a.name))
            a.stop_read()
        for a in self.attributes.itervalues():
            a.read_thread.join()

        self.settings.setValue('start_pos', self.start_pos_spinbox.value())
        self.settings.setValue('set_pos', self.set_pos_spinbox.value())
        self.settings.setValue('averages', self.average_spinbox.value())
        self.settings.setValue('step', self.step_size_spinbox.value())
        self.settings.setValue('export_filename', str(self.export_filename_edit.text()))
        self.settings.setValue('export_file_location', str(self.export_file_location_edit.text()))
        self.settings.setValue('x_unit_time', self.time_units_radio.isChecked())
        self.settings.setValue('dispersion', self.dispersion_spinbox.value())
        self.settings.setValue('center_wavelength', self.center_wavelength_spinbox.value())
        self.settings.setValue('camera_name', str(self.camera_select_lineedit.text()))
        self.settings.setValue('motor_name', str(self.motor_select_lineedit.text()))
        root.debug(''.join(('roi_pos: ', str(self.roi.pos()))))
        root.debug(''.join(('roi_size: ', str(self.roi.size()))))
        self.settings.setValue('roi_pos_x', np.float(self.roi.pos()[0]))
        self.settings.setValue('roi_pos_y', np.float(self.roi.pos()[1]))
        self.settings.setValue('roi_size_w', np.float(self.roi.size()[0]))
        self.settings.setValue('roi_size_h', np.float(self.roi.size()[1]))
        root.debug(''.join(('Window size: ', str(self.size()))))
        root.debug(''.join(('Window pos: ', str(self.pos().y()))))
        self.settings.setValue('window_size_w', np.int(self.size().width()))
        self.settings.setValue('window_size_h', np.int(self.size().height()))
        self.settings.setValue('window_pos_x', np.int(self.pos().x()))
        self.settings.setValue('window_pos_y', np.int(self.pos().y()))

        self.settings.setValue('frog_iterations', np.int(self.frog_iterations_spinbox.value()))
        self.settings.setValue('frog_size', np.int(self.frog_n_spinbox.value()))
        self.settings.setValue('frog_method', np.int(self.frog_method_combobox.currentIndex()))
        self.settings.setValue('frog_algo', np.int(self.frog_algo_combobox.currentIndex()))
        self.settings.setValue('frog_dt', np.float(self.frog_dt_spinbox.value()))
        self.settings.setValue('frog_threshold', np.float(self.frog_threshold_spinbox.value()))
        self.settings.setValue('frog_kernel', np.int(self.frog_kernel_median_spinbox.value()))
        self.settings.setValue('frog_kernel_gaussian', np.int(self.frog_kernel_gaussian_spinbox.value()))
        self.settings.setValue('frog_roi_pos_x', np.float(self.frog_roi.pos()[0]))
        self.settings.setValue('frog_roi_pos_y', np.float(self.frog_roi.pos()[1]))
        self.settings.setValue('frog_roi_size_w', np.float(self.frog_roi.size()[0]))
        self.settings.setValue('frog_roi_size_h', np.float(self.frog_roi.size()[1]))
        self.settings.setValue('frog_temporal_spectral', np.int(self.frog_temporal_spectral_combobox.currentIndex()))

        self.settings.sync()
        event.accept()

    def setup_frog_layout(self):
        """
        Setup the frog calculation layout
        :return:
        """
        self.frog_n_spinbox = QtWidgets.QSpinBox()
        self.frog_n_spinbox.setMinimum(0)
        self.frog_n_spinbox.setMaximum(4096)
        #         self.frogNSpinbox.setValue(128)
        self.frog_n_spinbox.setValue(self.settings.value('frog_size', 128, type=int))
        self.frog_n_spinbox.editingFinished.connect(self.estimate_frog_dt)
        self.frog_method_combobox = QtWidgets.QComboBox()
        self.frog_method_combobox.addItem('Vanilla')
        self.frog_method_combobox.addItem('GP')
        self.frog_method_combobox.setCurrentIndex(self.settings.value('frog_method', 0, type=int))
        self.frog_algo_combobox = QtWidgets.QComboBox()
        self.frog_algo_combobox.addItem('PG')
        self.frog_algo_combobox.addItem('SHG')
        self.frog_algo_combobox.addItem('SD')
        self.frog_algo_combobox.setCurrentIndex(self.settings.value('frog_algo', 0, type=int))
        self.frog_start_button = QtWidgets.QPushButton('Start')
        self.frog_start_button.clicked.connect(self.start_frog_inversion)
        self.frog_continue_button = QtWidgets.QPushButton('Continue')
        self.frog_continue_button.clicked.connect(self.continue_frog_inversion)
        self.frog_iterations_spinbox = QtWidgets.QSpinBox()
        #         self.frogIterationsSpinbox.setValue(20)
        self.frog_iterations_spinbox.setMinimum(0)
        self.frog_iterations_spinbox.setMaximum(999)
        self.frog_iterations_spinbox.setValue(self.settings.value('frog_iterations', 20, type=int))
        self.frog_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.frog_threshold_spinbox.setMinimum(0.0)
        self.frog_threshold_spinbox.setMaximum(1.0)
        self.frog_threshold_spinbox.setSingleStep(0.01)
        self.frog_threshold_spinbox.setDecimals(3)
        self.frog_threshold_spinbox.setValue(self.settings.value('frog_threshold', 0.05, type=float))
        self.frog_threshold_spinbox.editingFinished.connect(self.update_frog_roi)
        self.frog_fit_roi_button = QtWidgets.QPushButton('Fit')
        self.frog_fit_roi_button.clicked.connect(self.fit_frog_roi)

        self.frog_dt_label = QtWidgets.QLabel('dt (fs) []')
        self.frog_dt_spinbox = QtWidgets.QDoubleSpinBox()
        self.frog_dt_spinbox.setMinimum(0.0)
        self.frog_dt_spinbox.setMaximum(10000.0)
        self.frog_dt_spinbox.setSingleStep(1.0)
        self.frog_dt_spinbox.setDecimals(1)
        self.frog_dt_spinbox.setValue(self.settings.value('frog_dt', 1.0, type=float))
        self.frog_kernel_median_spinbox = QtWidgets.QSpinBox()
        self.frog_kernel_median_spinbox.setMinimum(1)
        self.frog_kernel_median_spinbox.setMaximum(99)
        self.frog_kernel_median_spinbox.setSingleStep(2)
        self.frog_kernel_median_spinbox.setValue(self.settings.value('frog_kernel', 3, type=int))
        self.frog_kernel_median_spinbox.editingFinished.connect(self.update_frog_roi)
        self.frog_kernel_gaussian_spinbox = QtWidgets.QSpinBox()
        self.frog_kernel_gaussian_spinbox.setMinimum(1)
        self.frog_kernel_gaussian_spinbox.setMaximum(99)
        self.frog_kernel_gaussian_spinbox.setSingleStep(2)
        self.frog_kernel_gaussian_spinbox.setValue(self.settings.value('frog_kernel_gaussian', 3, type=int))
        self.frog_kernel_gaussian_spinbox.editingFinished.connect(self.update_frog_roi)
        self.trace_source_label = QtWidgets.QLabel('Scan')
        self.trace_source_label.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.trace_source_label.setMaximumWidth(250)
        self.trace_source_label.setMinimumWidth(250)
        self.frog_load_button = QtWidgets.QPushButton('Load')
        self.frog_load_button.clicked.connect(self.load_frog_trace)
        self.frog_use_scan_button = QtWidgets.QPushButton('Scan data')
        self.frog_use_scan_button.clicked.connect(self.load_scandata_to_frog)

        self.frog_et_checkbox = QtWidgets.QCheckBox()
        self.frog_et_checkbox.setChecked(True)
        self.frog_et_checkbox.stateChanged.connect(self.update_frog_result_plot)
        self.frog_phase_checkbox = QtWidgets.QCheckBox()
        self.frog_phase_checkbox.setChecked(True)
        self.frog_phase_checkbox.stateChanged.connect(self.update_frog_result_plot)

        self.frog_et_fwhm_label = QtWidgets.QLabel()
        self.frog_et_phase_label = QtWidgets.QLabel()

        self.frog_temporal_spectral_combobox = QtWidgets.QComboBox()
        self.frog_temporal_spectral_combobox.addItem('Temporal')
        self.frog_temporal_spectral_combobox.addItem('Spectral')
        self.frog_temporal_spectral_combobox.setCurrentIndex(self.settings.value('frog_temporal_spectral', 0, type=int))
        self.frog_temporal_spectral_combobox.currentIndexChanged.connect(self.update_expansion_coefficients)
        self.frog_expansion1 = QtWidgets.QLabel()
        self.frog_expansion2 = QtWidgets.QLabel()
        self.frog_expansion3 = QtWidgets.QLabel()
        self.frog_expansion4 = QtWidgets.QLabel()

        self.frog_grid_layout1 = QtWidgets.QGridLayout()
        self.frog_grid_layout1.addWidget(QtWidgets.QLabel("Image threshold"), 0, 0)
        self.frog_grid_layout1.addWidget(self.frog_threshold_spinbox, 0, 1)
        self.frog_grid_layout1.addWidget(QtWidgets.QLabel("Median kernel"), 1, 0)
        self.frog_grid_layout1.addWidget(self.frog_kernel_median_spinbox, 1, 1)
        self.frog_grid_layout1.addWidget(QtWidgets.QLabel("Gaussian kernel"), 2, 0)
        self.frog_grid_layout1.addWidget(self.frog_kernel_gaussian_spinbox, 2, 1)
        self.frog_grid_layout1.addItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                             QtWidgets.QSizePolicy.Expanding), 3, 0)
        self.frog_grid_layout1.addWidget(QtWidgets.QLabel("Fit ROI to data"), 4, 0)
        self.frog_grid_layout1.addWidget(self.frog_fit_roi_button, 4, 1)

        self.frog_grid_layout2 = QtWidgets.QGridLayout()
        self.frog_grid_layout2.addWidget(QtWidgets.QLabel("Frog algorithm"), 0, 0)
        self.frog_grid_layout2.addWidget(self.frog_algo_combobox, 0, 1)
        self.frog_grid_layout2.addWidget(QtWidgets.QLabel("Frog method"), 1, 0)
        self.frog_grid_layout2.addWidget(self.frog_method_combobox, 1, 1)
        self.frog_grid_layout2.addWidget(QtWidgets.QLabel("Frog size"), 2, 0)
        self.frog_grid_layout2.addWidget(self.frog_n_spinbox, 2, 1)
        self.frog_grid_layout2.addWidget(self.frog_dt_label, 3, 0)
        self.frog_grid_layout2.addWidget(self.frog_dt_spinbox, 3, 1)
        self.frog_grid_layout2.addItem(QtWidgets.QSpacerItem(30, 10, QtWidgets.QSizePolicy.Minimum,
                                                             QtWidgets.QSizePolicy.Expanding), 4, 0)

        self.frog_grid_layout3 = QtWidgets.QGridLayout()
        self.frog_grid_layout3.addWidget(QtWidgets.QLabel("Trace source:"), 0, 0)
        self.frog_grid_layout3.addWidget(self.trace_source_label, 0, 1, 1, 2)
        self.frog_grid_layout3.addWidget(QtWidgets.QLabel("Load trace"), 1, 0)
        self.frog_grid_layout3.addWidget(self.frog_load_button, 1, 1)
        self.frog_grid_layout3.addWidget(QtWidgets.QLabel("Load scan data"), 2, 0)
        self.frog_grid_layout3.addWidget(self.frog_use_scan_button, 2, 1)
        self.frog_grid_layout3.addItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                             QtWidgets.QSizePolicy.Expanding), 3, 0)
        self.frog_grid_layout3.addWidget(QtWidgets.QLabel("Iterations"), 4, 0)
        self.frog_grid_layout3.addWidget(self.frog_iterations_spinbox, 4, 1)
        self.frog_grid_layout3.addWidget(QtWidgets.QLabel("Invert trace"), 5, 0)
        self.frog_grid_layout3.addWidget(self.frog_start_button, 5, 1)
        self.frog_grid_layout3.addWidget(self.frog_continue_button, 6, 1)

        self.frog_grid_layout4 = QtWidgets.QGridLayout()
        self.frog_grid_layout4.addWidget(QtWidgets.QLabel("Plot type"), 0, 0)
        self.frog_grid_layout4.addWidget(self.frog_temporal_spectral_combobox, 0, 1)
        self.frog_grid_layout4.addWidget(QtWidgets.QLabel("Show intensity"), 1, 0)
        self.frog_grid_layout4.addWidget(self.frog_et_checkbox, 1, 1)
        self.frog_grid_layout4.addWidget(QtWidgets.QLabel("Show phase"), 2, 0)
        self.frog_grid_layout4.addWidget(self.frog_phase_checkbox, 2, 1)
        self.frog_grid_layout4.addItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                             QtWidgets.QSizePolicy.Minimum), 3, 0)
        self.frog_grid_layout4.addWidget(QtWidgets.QLabel("t_FWHM (intensity)"), 4, 0)
        self.frog_grid_layout4.addWidget(self.frog_et_fwhm_label, 4, 1)
        self.frog_grid_layout4.addWidget(QtWidgets.QLabel("Phase diff"), 5, 0)
        self.frog_grid_layout4.addWidget(self.frog_et_phase_label, 5, 1)
        self.frog_grid_layout4.addItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Expanding,
                                                             QtWidgets.QSizePolicy.Expanding), 6, 0, 1, 2)

        self.frog_grid_layout5 = QtWidgets.QGridLayout()
        self.frog_grid_layout5.addWidget(QtWidgets.QLabel("Phase expansion"), 0, 0)
        self.frog_grid_layout5.addItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                             QtWidgets.QSizePolicy.Minimum), 1, 0)
        self.frog_grid_layout5.addWidget(self.frog_expansion1, 2, 0, 1, 2)
        self.frog_grid_layout5.addWidget(self.frog_expansion2, 3, 0, 1, 2)
        self.frog_grid_layout5.addWidget(self.frog_expansion3, 4, 0, 1, 2)
        self.frog_grid_layout5.addWidget(self.frog_expansion4, 5, 0, 1, 2)
        self.frog_grid_layout5.addItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Expanding,
                                                             QtWidgets.QSizePolicy.Expanding), 6, 0, 1, 2)

        # Error plot widget
        self.frog_error_widget = pq.PlotWidget(useOpenGL=True)
        self.frog_error_plot = self.frog_error_widget.plot()
        self.frog_error_plot.setPen((10, 200, 70))
        self.frog_error_widget.setAntialiasing(True)
        self.frog_error_widget.showGrid(True, True)
        self.frog_error_widget.plotItem.setLabels(left='Reconstruction error')
        self.frog_error_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.frog_error_widget.setMinimumHeight(80)
        self.frog_error_widget.setMaximumHeight(120)

        # E field plot widget
        self.frog_result_widget = pq.PlotWidget(useOpenGL=True)
        self.frog_result_plot_abs = self.frog_result_widget.plot()
        self.frog_result_plot_abs.setPen((10, 200, 70))

        self.frog_result_plotitem = self.frog_result_widget.plotItem
        self.frog_result_plotitem.setLabels(left='abs(Et)')
        self.frog_result_viewbox = pq.ViewBox()
        self.frog_result_plotitem.showAxis('right')
        self.frog_result_plotitem.scene().addItem(self.frog_result_viewbox)
        self.frog_result_plotitem.getAxis('right').linkToView(self.frog_result_viewbox)
        self.frog_result_viewbox.setXLink(self.frog_result_plotitem)
        self.frog_result_plotitem.getAxis('right').setLabel('Phase / rad')
        self.frog_result_plot_phase = pq.PlotCurveItem()
        self.frog_result_plot_phase.setPen((200, 70, 10))
        self.frog_result_viewbox.addItem(self.frog_result_plot_phase)
        self.frog_result_plotitem.vb.sigResized.connect(self.update_frog_plot_view)
        self.frog_result_widget.setAntialiasing(True)
        self.frog_result_widget.showGrid(True, True)
        self.frog_result_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.update_frog_plot_view()

        # Raw frog trace with roi selector
        plt1 = pq.PlotItem(labels={'bottom': ('Spectrum', 'm'), 'left': ('Time delay', 's')})
        self.frog_image_widget = pq.ImageView(view=plt1)
        self.frog_image_widget.ui.histogram.gradient.loadPreset('thermalclip')
        self.frog_image_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.frog_image_widget.getView().setAspectLocked(False)
        h = self.frog_image_widget.getHistogramWidget()
        h.item.sigLevelChangeFinished.connect(self.update_image_threshold)
        self.frog_roi = pq.RectROI([0, 300], [600, 20], pen=(0, 9))
        self.frog_roi.sigRegionChangeFinished.connect(self.update_frog_roi)
        self.frog_roi.blockSignals(True)

        #         roi_pos_x = 10
        #         roi_pos_y = 10
        #         roi_size_h = 20
        #         roi_size_w = 500
        roi_pos_x = self.settings.value('frog_roi_pos_x', 10, type=float)
        roi_pos_y = self.settings.value('frog_roi_pos_y', 10, type=float)
        root.debug(''.join(('roiPos: ', str(roi_pos_x))))
        roi_size_w = self.settings.value('frog_roi_size_w', 100, type=float)
        roi_size_h = self.settings.value('frog_roi_size_h', 10, type=float)

        self.frog_roi.setPos([roi_pos_x, roi_pos_y], update=True)
        self.frog_roi.setSize([roi_size_w, roi_size_h], update=True)
        self.frog_image_widget.getView().addItem(self.frog_roi)

        self.frog_roi.blockSignals(False)

        # Filtered frog trace roi
        plt2 = pq.PlotItem(labels={'bottom': ('Spectrum', 'm'), 'left': ('Time delay', 's')})
        self.frog_roi_image_widget = pq.ImageView(view=plt2)
        self.frog_roi_image_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.frog_roi_image_widget.getView().setAspectLocked(False)
        # pos = np.array([0.0, 0.0001, 1.0])
        # colors = np.array([[0, 0, 50, 255], [0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte)
        # colormap = pq.ColorMap(pos, colors)
        # self.frogRoiImageWidget.ui.histogram.gradient.setColorMap(colormap)
        self.frog_roi_image_widget.ui.histogram.gradient.loadPreset('thermalclip')

        plt3 = pq.PlotItem(labels={'bottom': ('Frequency', 'px'), 'left': ('Time delay', 'px')})
        self.frog_calc_image_widget = pq.ImageView(view=plt3)
        self.frog_calc_image_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.frog_calc_image_widget.getView().setAspectLocked(False)
        self.frog_calc_image_widget.ui.histogram.gradient.loadPreset('thermalclip')
        plt4 = pq.PlotItem(labels={'bottom': ('Frequency', 'px'), 'left': ('Time delay', 'px')})
        self.frog_calc_result_image_widget = pq.ImageView(view=plt4)
        self.frog_calc_result_image_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                                         QtWidgets.QSizePolicy.Expanding)
        self.frog_calc_result_image_widget.getView().setAspectLocked(False)
        self.frog_calc_result_image_widget.view.setXLink(self.frog_calc_image_widget.view)
        self.frog_calc_result_image_widget.view.setYLink(self.frog_calc_image_widget.view)
        self.frog_calc_result_image_widget.ui.histogram.gradient.loadPreset('thermalclip')

        self.frog_layout2 = QtWidgets.QVBoxLayout()

    def setup_layout(self):
        """
        Setup the initial image capture and camera settings layout
        :return:
        """
        root.debug('Setting up layout')
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English))
        self.layout = QtWidgets.QVBoxLayout(self)
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.grid_layout1 = QtWidgets.QGridLayout()
        self.grid_layout2 = QtWidgets.QGridLayout()
        self.grid_layout3 = QtWidgets.QGridLayout()
        self.grid_layout4 = QtWidgets.QGridLayout()
        self.grid_layout5 = QtWidgets.QGridLayout()
        grid_layout6 = QtWidgets.QGridLayout()

        self.fps_label = QtWidgets.QLabel()
        self.average_spinbox = QtWidgets.QSpinBox()
        self.average_spinbox.setMaximum(100)
        self.average_spinbox.setValue(self.settings.value('averages', 5, type=int))
        self.avg_samples = self.settings.value('averages', 5, type=int)
        self.average_spinbox.editingFinished.connect(self.set_average)
        self.set_average()
        self.time_marginal_fwhm_label = QtWidgets.QLabel()
        self.spectrum_marginal_fwhm_label = QtWidgets.QLabel()
        self.bandwidth_limited_fwhm_label = QtWidgets.QLabel()

        self.start_pos_spinbox = QtWidgets.QDoubleSpinBox()
        self.start_pos_spinbox.setDecimals(3)
        self.start_pos_spinbox.setMaximum(2000000)
        self.start_pos_spinbox.setMinimum(-2000000)
        self.start_pos_spinbox.setSuffix(" mm")
        self.start_pos_spinbox.setValue(self.settings.value('start_pos', 0.0, type=float))
        self.step_size_label = QtWidgets.QLabel("Step size")
        self.step_size_spinbox = QtWidgets.QDoubleSpinBox()
        self.step_size_spinbox.setDecimals(4)
        self.step_size_spinbox.setMaximum(2000000)
        self.step_size_spinbox.setMinimum(-2000000)
        self.step_size_spinbox.setSuffix(" mm")
        self.step_size_spinbox.editingFinished.connect(self.set_step_size)
        self.step_size_spinbox.setValue(self.settings.value('step', 0.05, type=float))
        self.set_step_size()
        self.set_pos_spinbox = QtWidgets.QDoubleSpinBox()
        self.set_pos_spinbox.setDecimals(3)
        self.set_pos_spinbox.setMaximum(2000000)
        self.set_pos_spinbox.setMinimum(-2000000)
        self.set_pos_spinbox.setValue(63.7)
        self.set_pos_spinbox.setSuffix(" mm")
        self.set_pos_spinbox.setValue(self.settings.value('set_pos', 63, type=float))
        self.set_pos_spinbox.editingFinished.connect(self.write_position)
        self.current_pos_label = QtWidgets.QLabel()
        font_postext = self.current_pos_label.font()
        font_postext.setPointSize(28)
        current_pos_text_label = QtWidgets.QLabel('Current pos ')
        current_pos_text_label.setFont(font_postext)
        self.current_pos_label.setFont(font_postext)
        self.current_speed_label = QtWidgets.QLabel()
        #        self.currentSpeedLabel.setFont(f)
        self.export_filename_edit = QtWidgets.QLineEdit()
        self.export_filename_edit.setText(self.settings.value('export_filename', '', type="QString"))
        self.export_file_location_edit = QtWidgets.QLineEdit()
        self.export_file_location_edit.setText(self.settings.value('export_file_location', './', type="QString"))

        self.center_wavelength_spinbox = QtWidgets.QDoubleSpinBox()
        self.center_wavelength_spinbox.setMinimum(200)
        self.center_wavelength_spinbox.setMaximum(2000)
        self.center_wavelength_spinbox.setValue(self.settings.value('center_wavelength', 400, type=float))
        self.center_wavelength_spinbox.editingFinished.connect(self.generate_wavelengths)

        self.dispersion_spinbox = QtWidgets.QDoubleSpinBox()
        self.dispersion_spinbox.setMinimum(-10)
        self.dispersion_spinbox.setMaximum(10)
        self.dispersion_spinbox.setDecimals(4)
        self.dispersion_spinbox.setValue(self.settings.value('dispersion', 0.03, type=float))
        self.dispersion_spinbox.editingFinished.connect(self.generate_wavelengths)

        self.shutter_label = QtWidgets.QLabel()
        self.shutter_spinbox = QtWidgets.QDoubleSpinBox()
        self.shutter_spinbox.setMinimum(0)
        self.shutter_spinbox.setMaximum(1000000)
        self.shutter_spinbox.editingFinished.connect(self.write_shutter)

        self.gain_label = QtWidgets.QLabel()
        self.gain_spinbox = QtWidgets.QDoubleSpinBox()
        self.gain_spinbox.setMinimum(0)
        self.gain_spinbox.setMaximum(100000)
        self.gain_spinbox.editingFinished.connect(self.write_gain)

        self.camera_start_button = QtWidgets.QPushButton("Start")
        self.camera_stop_button = QtWidgets.QPushButton("Stop")
        self.camera_init_button = QtWidgets.QPushButton("Init")
        self.camera_start_button.clicked.connect(self.start_camera)
        self.camera_stop_button.clicked.connect(self.stop_camera)
        self.camera_init_button.clicked.connect(self.init_camera)

        self.camera_select_lineedit = QtWidgets.QLineEdit()
        self.camera_select_lineedit.setText(self.settings.value("camera_name",
                                                                "gunlaser/cameras/spectrometer_camera",
                                                                type="QString"))
        self.camera_change_button = QtWidgets.QPushButton("Select")
        self.camera_change_button.clicked.connect(self.select_camera)
        self.motor_select_lineedit = QtWidgets.QLineEdit()
        self.motor_select_lineedit.setText(self.settings.value("motor_name",
                                                                "gunlaser/motors/zaber01",
                                                                type="QString"))
        self.motor_change_button = QtWidgets.QPushButton("Select")
        self.motor_change_button.clicked.connect(self.select_motor)

        self.normalize_pump_check = QtWidgets.QCheckBox('Normalize')
        self.time_units_radio = QtWidgets.QRadioButton('ps')
        self.pos_units_radio = QtWidgets.QRadioButton('mm')
        if self.settings.value('x_unit_time', True, type=bool) is True:
            self.time_units_radio.setChecked(True)
        else:
            self.pos_units_radio.setChecked(True)
        self.time_units_radio.toggled.connect(self.x_axis_units_toggle)

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.clicked.connect(self.start_scan)
        self.stop_button = QtWidgets.QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_scan)
        self.export_button = QtWidgets.QPushButton('Export')
        self.export_button.clicked.connect(self.export_scan)

        self.grid_layout1.addWidget(QtWidgets.QLabel("Start position"), 0, 0)
        self.grid_layout1.addWidget(self.start_pos_spinbox, 0, 1)
        self.grid_layout1.addWidget(self.step_size_label, 1, 0)
        self.grid_layout1.addWidget(self.step_size_spinbox, 1, 1)
        self.grid_layout1.addWidget(QtWidgets.QLabel("Averages"), 2, 0)
        self.grid_layout1.addWidget(self.average_spinbox, 2, 1)
        self.grid_layout1.addWidget(QtWidgets.QLabel("Normalize"), 3, 0)
        self.grid_layout1.addWidget(self.normalize_pump_check, 3, 1)
        self.grid_layout1.addItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum,
                                                        QtWidgets.QSizePolicy.MinimumExpanding), 4, 0)
        self.grid_layout2.addWidget(QtWidgets.QLabel("Set position"), 0, 0)
        self.grid_layout2.addWidget(self.set_pos_spinbox, 0, 1)
        self.grid_layout2.addItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum,
                                                        QtWidgets.QSizePolicy.MinimumExpanding), 1, 0)
        self.grid_layout2.addWidget(QtWidgets.QLabel("Start scan"), 2, 0)
        self.grid_layout2.addWidget(self.start_button, 2, 1)
        self.grid_layout2.addWidget(QtWidgets.QLabel("Stop scan"), 3, 0)
        self.grid_layout2.addWidget(self.stop_button, 3, 1)
        self.grid_layout2.addWidget(QtWidgets.QLabel("Export scan"), 4, 0)
        self.grid_layout2.addWidget(self.export_button, 4, 1)
        self.grid_layout3.addWidget(current_pos_text_label, 0, 0)
        self.grid_layout3.addWidget(self.current_pos_label, 0, 1)
        self.grid_layout3.addWidget(QtWidgets.QLabel("Current speed"), 1, 0)
        self.grid_layout3.addWidget(self.current_speed_label, 1, 1)
        self.grid_layout3.addWidget(QtWidgets.QLabel("Filename: frogtrace_yyyy-mm-dd_hh-mm_"), 2, 0)
        self.grid_layout3.addWidget(self.export_filename_edit, 2, 1)
        self.grid_layout3.addWidget(QtWidgets.QLabel("File location: "), 3, 0)
        self.grid_layout3.addWidget(self.export_file_location_edit, 3, 1)

        self.grid_layout5.addWidget(QtWidgets.QLabel("X-axis units"), 0, 0)
        self.grid_layout5.addWidget(self.time_units_radio, 0, 1)
        self.grid_layout5.addWidget(self.pos_units_radio, 1, 1)
        self.grid_layout5.addWidget(QtWidgets.QLabel("FPS"), 2, 0)
        self.grid_layout5.addWidget(self.fps_label, 2, 1)

        grid_layout6.addItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum,
                                                   QtWidgets.QSizePolicy.MinimumExpanding), 0, 0)
        grid_layout6.addWidget(QtWidgets.QLabel("Time marginal pulse duration FWHM"), 1, 0)
        grid_layout6.addWidget(self.time_marginal_fwhm_label, 1, 1)
        grid_layout6.addWidget(QtWidgets.QLabel("Spectrum marginal FWHM"), 2, 0)
        grid_layout6.addWidget(self.spectrum_marginal_fwhm_label, 2, 1)
        grid_layout6.addWidget(QtWidgets.QLabel("Bandwidth limited pulse duration FWHM"), 3, 0)
        grid_layout6.addWidget(self.bandwidth_limited_fwhm_label, 3, 1)

        root.debug('Plot widgets')

        self.spectrum_image_widget = pq.ImageView()
        self.spectrum_image_widget.ui.histogram.gradient.loadPreset('thermal')
        self.roi = pq.RectROI([0, 300], [600, 20], pen=(0, 9))
        self.roi.sigRegionChanged.connect(self.generate_wavelengths)
        self.roi.blockSignals(True)

        roi_pos_x = self.settings.value('roi_pos_x', 0, type=float)
        roi_pos_y = self.settings.value('roi_pos_y', 0, type=float)
        root.debug(''.join(('roiPos: ', str(roi_pos_x))))
        roi_size_w = self.settings.value('roi_size_w', 0, type=float)
        roi_size_h = self.settings.value('roi_size_h', 0, type=float)
        root.debug(''.join(('roiSize: ', str(roi_size_w), 'x', str(roi_size_h))))
        self.roi.setPos([roi_pos_x, roi_pos_y], update=True)
        self.roi.setSize([roi_size_w, roi_size_h], update=True)
        self.spectrum_image_widget.getView().addItem(self.roi)
        self.spectrum_image_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.spectrum_image_widget.getView().setAspectLocked(False)
        self.roi.blockSignals(False)

        self.spectrum_plot_widget = pq.PlotWidget(useOpenGL=True, labels={'bottom': ('Spectrum', 'm')})
        self.plot1 = self.spectrum_plot_widget.plot()
        self.plot1.setPen((200, 25, 10))
        self.plot2 = self.spectrum_plot_widget.plot()
        self.plot2.setPen((10, 200, 25))
        self.plot1.antialiasing = True
        self.spectrum_plot_widget.setAntialiasing(True)
        self.spectrum_plot_widget.showGrid(True, True)
        self.spectrum_plot_widget.setMinimumWidth(200)
        self.spectrum_plot_widget.setMaximumWidth(500)
        self.spectrum_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.scan_image_widget = pq.ImageView()
        self.scan_image_widget.ui.histogram.gradient.loadPreset('thermal')
        self.scan_image_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.scan_image_widget.getView().setAspectLocked(False)

        self.timemarginal_plot_widget = pq.PlotWidget(useOpenGL=True, labels={'bottom': ('Time delay', 's')})
        if self.time_units_radio.isChecked() is True:
            self.timemarginal_plot_widget.setLabel("bottom", "Time delay", "s")
        else:
            self.timemarginal_plot_widget.setLabel("bottom", "Delay position", "m")
        self.timemarginal_plot = self.timemarginal_plot_widget.plot()
        self.timemarginal_plot.setPen((50, 99, 200))
        self.timemarginal_plot_widget.setAntialiasing(True)
        self.timemarginal_plot_widget.showGrid(True, True)
        # self.timemarginal_plot_widget.setMaximumHeight(200)
        self.timemarginal_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

        self.spectrummarginal_plot_widget = pq.PlotWidget(useOpenGL=True, labels={'bottom': ('Wavelength', 'm')})
        self.spectrummarginal_plot = self.spectrummarginal_plot_widget.plot()
        self.spectrummarginal_plot.setPen((50, 99, 200))
        self.spectrummarginal_plot_widget.setAntialiasing(True)
        self.spectrummarginal_plot_widget.showGrid(True, True)
        self.spectrummarginal_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

        self.plot_layout = QtWidgets.QHBoxLayout()
        self.plot_layout.addWidget(self.spectrum_image_widget)

        self.spectrum_layout = QtWidgets.QVBoxLayout()
        self.spectrum_layout.addWidget(self.spectrum_plot_widget)
        self.plot_layout.addLayout(self.spectrum_layout)

        scan_lay = QtWidgets.QHBoxLayout()
        scan_lay.addLayout(self.grid_layout1)
        scan_lay.addSpacerItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                     QtWidgets.QSizePolicy.Minimum))
        scan_lay.addLayout(self.grid_layout2)
        scan_lay.addSpacerItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                     QtWidgets.QSizePolicy.Minimum))
        scan_lay.addLayout(self.grid_layout4)
        scan_lay.addSpacerItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                     QtWidgets.QSizePolicy.Minimum))
        scan_lay.addLayout(self.grid_layout3)
        scan_lay.addSpacerItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                     QtWidgets.QSizePolicy.Minimum))
        scan_lay.addLayout(self.grid_layout5)
        scan_lay.addSpacerItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.MinimumExpanding,
                                                     QtWidgets.QSizePolicy.Minimum))
        scan_lay.addLayout(grid_layout6)

        camera_lay = QtWidgets.QHBoxLayout()
        self.cam_grid_layout1 = QtWidgets.QGridLayout()
        self.cam_grid_layout1.addWidget(QtWidgets.QLabel("Center wavelength (nm)"), 0, 0)
        self.cam_grid_layout1.addWidget(self.center_wavelength_spinbox, 0, 1)
        self.cam_grid_layout1.addWidget(QtWidgets.QLabel("Dispersion (nm/px)"), 1, 0)
        self.cam_grid_layout1.addWidget(self.dispersion_spinbox, 1, 1)
        self.cam_grid_layout1.addWidget(QtWidgets.QLabel("-0.0585 for IR FROG, -0.018 for UV FROG"), 1, 2)
        self.cam_grid_layout1.addWidget(QtWidgets.QLabel("Shutter time (us)"), 2, 0)
        self.cam_grid_layout1.addWidget(self.shutter_spinbox, 2, 1)
        self.cam_grid_layout1.addWidget(self.shutter_label, 2, 2)
        self.cam_grid_layout1.addWidget(QtWidgets.QLabel("Gain"), 3, 0)
        self.cam_grid_layout1.addWidget(self.gain_spinbox, 3, 1)
        self.cam_grid_layout1.addWidget(self.gain_label, 3, 2)
        self.cam_grid_layout1.addWidget(QtWidgets.QLabel("Camera control"), 4, 0)
        cam_ctl_lay = QtWidgets.QHBoxLayout()
        cam_ctl_lay.addWidget(self.camera_start_button)
        cam_ctl_lay.addWidget(self.camera_stop_button)
        cam_ctl_lay.addWidget(self.camera_init_button)
        self.cam_grid_layout1.addLayout(cam_ctl_lay, 4, 1, 1, 2)
        self.cam_grid_layout1.addItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum,
                                                            QtWidgets.QSizePolicy.MinimumExpanding), 5, 0)
        camera_lay.addLayout(self.cam_grid_layout1)

        camera_lay.addSpacerItem(
            QtWidgets.QSpacerItem(20, 20, 60, 20))

        self.device_select_layout = QtWidgets.QGridLayout()
        self.device_select_layout.addWidget(QtWidgets.QLabel("Camera name"), 0, 0)
        self.device_select_layout.addWidget(self.camera_select_lineedit, 0, 1)
        self.device_select_layout.addWidget(self.camera_change_button, 0, 2)
        self.device_select_layout.addWidget(QtWidgets.QLabel("Motor name"), 1, 0)
        self.device_select_layout.addWidget(self.motor_select_lineedit, 1, 1)
        self.device_select_layout.addWidget(self.motor_change_button, 1, 2)
        self.device_select_layout.addItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum,
                                                            QtWidgets.QSizePolicy.MinimumExpanding), 5, 0)
        camera_lay.addLayout(self.device_select_layout)

        camera_lay.addSpacerItem(
            QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum))

        self.setup_frog_layout()

        self.marginals_layout = QtWidgets.QVBoxLayout()
        self.marginals_layout.addWidget(self.scan_image_widget)
        self.marginals_layout.addWidget(self.timemarginal_plot_widget)
        self.plot_layout.addLayout(self.marginals_layout)

        frog_lay = QtWidgets.QHBoxLayout()
        frog_lay.addLayout(self.frog_grid_layout1)
        frog_lay.addSpacerItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                     QtWidgets.QSizePolicy.Minimum))
        frog_lay.addLayout(self.frog_grid_layout2)
        frog_lay.addSpacerItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                     QtWidgets.QSizePolicy.Minimum))
        frog_lay.addLayout(self.frog_grid_layout3)
        frog_lay.addSpacerItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                     QtWidgets.QSizePolicy.Minimum))
        frog_lay.addWidget(self.frog_error_widget)
        frog_lay.addSpacerItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                     QtWidgets.QSizePolicy.Minimum))
        frog_lay.addLayout(self.frog_grid_layout4)
        frog_lay.addSpacerItem(QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Minimum,
                                                     QtWidgets.QSizePolicy.Minimum))
        frog_lay.addLayout(self.frog_grid_layout5)
        frog_lay.addSpacerItem(QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.MinimumExpanding,
                                                     QtWidgets.QSizePolicy.Minimum))

        self.tab_camera_widget = QtWidgets.QWidget()
        self.tab_camera_widget.setLayout(camera_lay)
        self.tab_scan_widget = QtWidgets.QWidget()
        self.tab_scan_widget.setLayout(scan_lay)
        self.tab_frog_widget = QtWidgets.QWidget()
        self.tab_frog_widget.setLayout(frog_lay)
        self.tab_widget.addTab(self.tab_scan_widget, 'Scan')
        self.tab_widget.addTab(self.tab_camera_widget, 'Devices')
        self.tab_widget.addTab(self.tab_frog_widget, 'Frog')
        self.tab_widget.currentChanged.connect(self.tab_changed)
        self.layout.addWidget(self.tab_widget)
        #         self.layout.addLayout(scan_lay)
        self.layout.addLayout(self.plot_layout)

        self.invisible_layout = QtWidgets.QHBoxLayout()

        window_pos_x = self.settings.value('window_pos_x', 100, type=int)
        window_pos_y = self.settings.value('window_pos_y', 100, type=int)
        window_size_w = self.settings.value('window_size_w', 800, type=int)
        window_size_h = self.settings.value('window_size_h', 300, type=int)
        if window_pos_x < 50:
            window_pos_x = 200
        if window_pos_y < 50:
            window_pos_y = 200
        self.setGeometry(window_pos_x, window_pos_y, window_size_w, window_size_h)

        self.tab_changed(0)

        self.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myapp = TangoDeviceClient(camera_name, motor_name)
    myapp.show()
    sys.exit(app.exec_())
