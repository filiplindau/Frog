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
from PyQt5.QtCore import pyqtSignal
import multiprocessing as mp

import FrogCalculationPCGP as FrogCalculation

from frog_pcgp_gui import Ui_frog_dialog

sys.path.insert(0, '../../TangoWidgetsQt5')
from AttributeReadThreadClass import AttributeClass

import logging

logger = logging.getLogger()
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

pq.graphicsItems.GradientEditorItem.Gradients['greyclip2'] = {
    'ticks': [(0.0, (0, 0, 50, 255)), (0.0001, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}
pq.graphicsItems.GradientEditorItem.Gradients['thermalclip'] = {
    'ticks': [(0, (0, 0, 50, 255)), (0.0001, (0, 0, 0, 255)), (0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)),
              (1, (255, 255, 255, 255))], 'mode': 'rgb'}


class StatusMessager(QtCore.QObject):
    status_signal = pyqtSignal(str)

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.msg_list = list()
        self.timer = threading.Timer(0.2, self.tick)
        self.status_string = ""
        self.stop_timer = False
        self.lock = threading.Lock()
        self.timer.start()

    def update_list(self):
        with self.lock:
            try:
                new_list = list()
                new_string = ""
                for msg in self.msg_list:
                    dt = time.time() - msg[0]
                    if dt < msg[2]:
                        new_list.append(msg)
                        new_string += "{1}... {0:.1f} s\n\n".format(dt, msg[1])
                if len(new_string) != len(self.status_string):
                    signal = True
                else:
                    signal = False
                self.msg_list = new_list
                self.status_string = new_string
            except Exception as e:
                logger.error("Status update error: {0}".format(e))
        self.status_signal.emit(self.status_string)

    def add_message(self, msg_text, timestamp, lifetime):
        msg = (timestamp, msg_text, lifetime)
        with self.lock:
            self.msg_list.append(msg)
        self.update_list()

    def tick(self):
        self.update_list()
        if not self.stop_timer:
            self.timer = threading.Timer(0.2, self.tick)
            self.timer.start()

    def stop(self):
        with self.lock:
            self.stop_timer = True
            if self.timer.is_alive():
                self.timer.cancel()
                self.timer.join(0.5)


class FrogCalculationWorker(QtCore.QObject):
    ready_signal = pyqtSignal()

    def __init__(self, frog_calc, iterations=10, geometry="PG", parent=None):
        QtCore.QObject.__init__(self, parent)
        self.frog_calc = frog_calc
        self.iterations = iterations
        self.frog_geometry = geometry

    def run(self):
        logger.info("Starting frog calculation")
        self.frog_calc.run_cycle_pc(self.iterations, self.frog_geometry)
        logger.info("Emitting ready signal")
        self.ready_signal.emit()


class FrogReconstruction(QtWidgets.QWidget):
    """
    Class for scanning a motor while grabbing images to produce a frog trace. It can also analyse the scanned trace
    or saved traces.
    """
    new_image_signal = pyqtSignal(np.ndarray)
    new_roi_image_signal = pyqtSignal(np.ndarray)
    new_bkg_image_signal = pyqtSignal(np.ndarray)
    new_frog_image_signal = pyqtSignal(np.ndarray)
    new_wavelengths_signal = pyqtSignal()
    frog_calc_ready_signal = pyqtSignal()

    def __init__(self, parent=None):
        logger.debug("Init")
        QtWidgets.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings('Maxlab', 'FrogPCGP')
        #        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.time_vector = None
        self.x_data = None
        self.x_data_temp = None
        self.camera_name = None

        self.gui_lock = threading.Lock()

        self.position_offset = 0.0

        self.devices = dict()
        self.attributes = dict()

        self.roi_data = np.zeros((2, 2))
        self.roi = None
        self.loaded_image = np.array([])
        self.camera_image = np.array([])
        self.bkg_image = np.array([])
        self.bkg_image_filename = ""
        self.frog_raw_data = np.array([])
        self.frog_filtered_data = np.array([])
        self.frog_reconstructed_data = np.array([])
        self.frog_wavelengths = np.array([])
        self.frog_time_data = np.array([])
        self.frog_central_wavelength = 0.0
        self.lock = threading.Lock()

        self.frog_calc = FrogCalculation.FrogCalculation()  # Frog calculations object. Implements all the
        # frog inversion calculations.
        self.frog_calc_thread = None
        self.frog_calc_worker = None
        self.frog_calc_running_flag = False

        self.ui = Ui_frog_dialog()
        self.ui.setupUi(self)
        self.setup_layout()
        self.select_camera()

        self.status_messenger = StatusMessager()
        self.status_messenger.status_signal.connect(self.update_status)

        self.load_bkg_image(self.bkg_image_filename)

        logger.debug("Init finished")

    def setup_layout(self):
        """
        Setup the initial image capture and camera settings layout
        :return:
        """
        logger.debug('Setting up layout')
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English))

        #        self.currentSpeedLabel.setFont(f)
        # self.export_filename_edit = QtWidgets.QLineEdit()
        # self.export_filename_edit.setText(self.settings.value('export_filename', '', type="QString"))
        # self.export_file_location_edit = QtWidgets.QLineEdit()
        # self.export_file_location_edit.setText(self.settings.value('export_file_location', './', type="QString"))

        self.ui.dispersion_spinbox.setValue(self.settings.value('dispersion', 0.034, type=float))
        self.ui.central_wavelength_spinbox.setValue(self.settings.value('central_wavelength', 782.0, type=float))
        self.ui.threshold_spinbox.setValue(self.settings.value('frog_threshold', 0.05, type=float))
        self.ui.time_resolution_spinbox.setValue(self.settings.value('time_resolution', 7.5, type=float))
        self.ui.frog_dt_spinbox.setValue(self.settings.value("frog_dt", 10.0, type=float))
        self.ui.frog_size_spinbox.setValue(self.settings.value("frog_size", 128, type=int))
        self.ui.frog_iterations_spinbox.setValue(self.settings.value("frog_iterations", 10, type=int))

        self.ui.camera_name_lineedit.setText(self.settings.value("camera_name",
                                                                 "gunlaser/cameras/astrella_frog",
                                                                 type="QString"))
        self.ui.directory_lineedit.setText(self.settings.value("directory_name",
                                                               "./data",
                                                               type="QString"))

        self.ui.frog_geometry_combobox.addItem("PG")
        self.ui.frog_geometry_combobox.addItem("SHG")
        self.ui.frog_geometry_combobox.setCurrentText(self.settings.value("frog_geometry", "PG", type=str))
        self.ui.auto_run_radiobutton.setChecked(self.settings.value("auto_run", True, type=bool))

        self.bkg_image_filename = self.settings.value("bkg_image_filename", "", type=str)

        # self.ui.primary_image_widget = pq.ImageView()
        self.ui.primary_image_widget.ui.histogram.gradient.loadPreset('thermal')
        self.roi = pq.RectROI([0, 300], [600, 20], pen=(0, 9))
        self.roi.sigRegionChangeFinished.connect(self.update_image_roi)
        self.roi.blockSignals(True)

        roi_pos_x = self.settings.value('roi_pos_x', 0, type=float)
        roi_pos_y = self.settings.value('roi_pos_y', 0, type=float)
        logger.debug(''.join(('roiPos: ', str(roi_pos_x))))
        roi_size_w = self.settings.value('roi_size_w', 0, type=float)
        roi_size_h = self.settings.value('roi_size_h', 0, type=float)
        logger.debug(''.join(('roiSize: ', str(roi_size_w), 'x', str(roi_size_h))))
        self.roi.setPos([roi_pos_x, roi_pos_y], update=True)
        self.roi.setSize([roi_size_w, roi_size_h], update=True)
        self.ui.roi_top_spinbox.setValue(roi_pos_x)
        self.ui.roi_left_spinbox.setValue(roi_pos_y)
        self.ui.roi_width_spinbox.setValue(roi_size_w)
        self.ui.roi_height_spinbox.setValue(roi_size_h)
        self.ui.primary_image_widget.getView().addItem(self.roi)
        self.ui.primary_image_widget.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.ui.primary_image_widget.getView().setAspectLocked(False)
        self.roi.blockSignals(False)

        self.ui.secondary_image_widget.ui.histogram.gradient.loadPreset('thermalclip')

        self.temporal_result_plot_intensity = self.ui.line_plot_widget.plot()
        self.temporal_result_plot_intensity.setPen((200, 25, 10))
        self.temporal_result_plot_intensity.antialiasing = True
        self.temporal_result_plotitem = self.ui.line_plot_widget.plotItem
        self.temporal_result_plotitem.setLabels(left='Intensity')

        self.temporal_result_viewbox = pq.ViewBox()
        self.temporal_result_plotitem.showAxis('right')
        self.temporal_result_plotitem.scene().addItem(self.temporal_result_viewbox)
        self.temporal_result_plotitem.getAxis('right').linkToView(self.temporal_result_viewbox)
        self.temporal_result_viewbox.setXLink(self.temporal_result_plotitem)
        self.temporal_result_plotitem.getAxis('right').setLabel('Phase / rad')
        self.temporal_result_plot_phase = pq.PlotCurveItem()
        self.temporal_result_plot_phase.setPen((20, 200, 70))
        self.temporal_result_viewbox.addItem(self.temporal_result_plot_phase)
        self.temporal_result_plotitem.vb.sigResized.connect(self.update_frog_plot_view)
        self.ui.line_plot_widget.setAntialiasing(True)
        self.ui.line_plot_widget.showGrid(True, True)
        self.ui.line_plot_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.update_frog_plot_view()

        self.plot2 = self.ui.line_plot_widget2.plot()
        self.plot2.setPen((10, 200, 25))
        self.plot2.antialiasing = True
        self.ui.line_plot_widget2.setAntialiasing(True)
        self.ui.line_plot_widget2.showGrid(True, True)
        # self.ui.line_plot_widget2.setMinimumWidth(200)
        # self.ui.line_plot_widget2.setMaximumWidth(500)
        self.ui.line_plot_widget2.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.error_plot = self.ui.error_line_plot_widget.plot()
        self.error_plot.setPen((0, 0, 0))
        self.error_plot.antialiasing = True
        self.ui.error_line_plot_widget.setAntialiasing(True)
        self.ui.error_line_plot_widget.showGrid(True, True)
        self.ui.error_line_plot_widget.setBackground("w")

        #
        # Connect signals
        #
        self.new_image_signal.connect(self.update_primary_image)
        self.new_frog_image_signal.connect(self.update_primary_image)
        self.new_frog_image_signal.connect(self.update_primary_plot)
        self.new_bkg_image_signal.connect(self.update_primary_image)
        self.new_wavelengths_signal.connect(self.generate_frog_image)
        self.frog_calc_ready_signal.connect(self.frog_calc_ready)
        # self.new_roi_image_signal.connect(self.update_secondary_image)
        self.ui.camera_name_lineedit.editingFinished.connect(self.select_camera)
        self.new_roi_image_signal.connect(self.generate_frog_image)
        self.ui.threshold_spinbox.editingFinished.connect(self.update_image_roi)
        self.ui.dispersion_spinbox.editingFinished.connect(self.generate_wavelengths)
        self.ui.time_resolution_spinbox.editingFinished.connect(self.generate_wavelengths)
        self.ui.central_wavelength_spinbox.editingFinished.connect(self.generate_wavelengths)
        self.ui.frog_dt_spinbox.editingFinished.connect(self.generate_frog_image)
        self.ui.frog_size_spinbox.editingFinished.connect(self.generate_frog_image)
        self.ui.bkg_subtract_radiobutton.toggled.connect(self.update_image_roi)
        self.ui.bkg_image_radiobutton.toggled.connect(self.update_primary_image)
        self.ui.frog_image_radiobutton.toggled.connect(self.update_primary_image)
        self.ui.loaded_image_radiobutton.toggled.connect(self.update_primary_image)
        self.ui.camera_image_radiobutton.toggled.connect(self.update_primary_image)
        self.ui.frog_geometry_combobox.currentIndexChanged.connect(self.generate_frog_image)
        self.ui.frog_materials_enable_radiobutton.toggled.connect(self.generate_frog_image)

        self.ui.load_button.clicked.connect(self.load_image)
        self.ui.camera_start_button.clicked.connect(self.start_camera)
        self.ui.camera_stop_button.clicked.connect(self.stop_camera)
        self.ui.camera_bkg_button.clicked.connect(self.camera_bkg_image)
        self.ui.load_bkg_button.clicked.connect(self.load_bkg_image)
        self.ui.frog_restart_button.clicked.connect(self.restart_frog)
        # self.ui.frog_run_button.clicked.connect(self.run_frog_calc)
        self.ui.frog_run_button.clicked.connect(self.start_frog_calc_thread)

        self.ui.save_image_button.enabled = False

        window_pos_x = self.settings.value('window_pos_x', 100, type=int)
        window_pos_y = self.settings.value('window_pos_y', 100, type=int)
        window_size_w = self.settings.value('window_size_w', 800, type=int)
        window_size_h = self.settings.value('window_size_h', 300, type=int)
        if window_pos_x < 50:
            window_pos_x = 200
        if window_pos_y < 50:
            window_pos_y = 200
        self.setGeometry(window_pos_x, window_pos_y, window_size_w, window_size_h)
        self.ui.vertical_splitter.setSizes(self.settings.value("vertical_splitter_sizes", [100, 100], type=int))
        self.ui.horizontal_splitter.setSizes(self.settings.value("horizontal_splitter_sizes", [100, 100], type=int))

        self.show()

    def closeEvent(self, event):
        """
        Closing the applications. Stopping the tango measurement threads and saving the settings.
        :param event:
        :return:
        """
        for a in self.attributes.values():
            logger.debug('Stopping' + str(a.name))
            a.stop_read()
        for a in self.attributes.values():
            a.read_thread.join()

        self.status_messenger.stop()

        self.settings.setValue('dispersion', self.ui.dispersion_spinbox.value())
        self.settings.setValue('central_wavelength', self.ui.central_wavelength_spinbox.value())
        self.settings.setValue('time_resolution', self.ui.time_resolution_spinbox.value())
        self.settings.setValue('camera_name', str(self.ui.camera_name_lineedit.text()))
        self.settings.setValue('directory_name', str(self.ui.directory_lineedit.text()))
        self.settings.setValue('bkg_image_filename', self.bkg_image_filename)
        self.settings.setValue('roi_pos_x', np.float(self.ui.roi_left_spinbox.value()))
        self.settings.setValue('roi_pos_y', np.float(self.ui.roi_top_spinbox.value()))
        self.settings.setValue('roi_size_w', np.float(self.ui.roi_width_spinbox.value()))
        self.settings.setValue('roi_size_h', np.float(self.ui.roi_height_spinbox.value()))
        self.settings.setValue("frog_geometry", str(self.ui.frog_geometry_combobox.currentText()))
        logger.debug(''.join(('Window size: ', str(self.size()))))
        logger.debug(''.join(('Window pos: ', str(self.pos().y()))))
        self.settings.setValue('window_size_w', np.int(self.size().width()))
        self.settings.setValue('window_size_h', np.int(self.size().height()))
        self.settings.setValue('window_pos_x', np.int(self.pos().x()))
        self.settings.setValue('window_pos_y', np.int(self.pos().y()))
        self.settings.setValue("vertical_splitter_sizes", (self.ui.vertical_splitter.sizes()))
        self.settings.setValue("horizontal_splitter_sizes", (self.ui.horizontal_splitter.sizes()))

        self.settings.setValue('frog_iterations', np.int(self.ui.frog_iterations_spinbox.value()))
        self.settings.setValue('frog_size', np.int(self.ui.frog_size_spinbox.value()))
        self.settings.setValue('frog_dt', np.float(self.ui.frog_dt_spinbox.value()))
        self.settings.setValue('frog_threshold', np.float(self.ui.threshold_spinbox.value()))
        self.settings.setValue("auto_run", bool(self.ui.auto_run_radiobutton.isChecked()))

        self.settings.sync()
        event.accept()

    def start_camera(self):
        logger.debug("Sending start command to camera")
        self.devices["camera"].command_inout("start")
        self.attributes["image"].unpause_read()

    def stop_camera(self):
        logger.debug("Sending stop command to camera")
        self.devices["camera"].command_inout("stop")
        self.attributes["image"].pause_read()

    def init_camera(self):
        logger.debug("Sending init command to camera")
        self.devices["camera"].command_inout("init")

    def select_camera(self):
        logger.debug("Closing down camera attributes")
        if "image" in self.attributes:
            try:
                self.attributes["image"].stop_read()
                self.attributes["gain"].stop_read()
                self.attributes["state"].stop_read()
                self.attributes["framecounter"].stop_read()
                self.attributes["image"].read_thread.join(0.5)
                self.attributes["gain"].read_thread.join(0.5)
            except KeyError as e:
                logger.info("Attribute close error: {0}".format(e))
        camera_tango_name = str(self.ui.camera_name_lineedit.text())
        logger.info("Connecting to {0}".format(camera_tango_name))
        try:
            new_device = pt.DeviceProxy(camera_tango_name)
        except pt.DevFailed:
            logger.error("Could not connect")
            return
        logger.info("Connected to camera {0}".format(camera_tango_name))
        self.devices["camera"] = new_device
        self.attributes['image'] = AttributeClass('image', self.devices['camera'], 0.5)
        # self.attributes['shutter'] = AttributeClass('exposuretime', self.devices['camera'], 0.5)
        self.attributes['gain'] = AttributeClass('gain', self.devices['camera'], 0.5)
        self.attributes['state'] = AttributeClass('state', self.devices['camera'], 0.5)
        self.attributes['framecounter'] = AttributeClass('framecounter', self.devices['camera'], 0.5)
        logger.info("Attributes created")

        self.attributes['image'].attrSignal.connect(self.read_camera_image)
        # self.attributes['shutter'].attrSignal.connect(self.read_camera_shutter)
        self.attributes['gain'].attrSignal.connect(self.read_camera_gain)
        self.attributes['state'].attrSignal.connect(self.read_camera_state)
        self.attributes['framecounter'].attrSignal.connect(self.read_camera_framecounter)
        logger.info("Attribute signals connected")

    def read_camera_image(self, data):
        logger.debug("New image")
        with self.lock:
            self.camera_image = data.value
        self.update_image_roi()
        self.new_image_signal.emit(data.value)

    def read_camera_shutter(self, data):
        logger.debug("Read shutter: {0}".format(data.value))

    def read_camera_gain(self, data):
        logger.debug("Read gain: {0}".format(data.value))
        self.ui.camera_gain_label.setText("{0}".format(data.value))

    def read_camera_state(self, data):
        self.ui.camera_state_label.setText("{0}".format(data.value))

    def read_camera_framecounter(self, data):
        self.ui.camera_framecounter_label.setText("{0}".format(data.value))

    def estimate_frog_dt(self, n=None, t_span=None, l_min=None, l_max=None):
        update = False
        if n is None:
            n = self.ui.frog_size_spinbox.value()
            update = True
        if t_span is None:
            t_span = self.ui.roi_width_spinbox.value()
        if l_min is None:
            l_min = self.ui.roi_top_spinbox.value()
            l_max = l_min + self.ui.roi_height_spinbox.value()
        dt_t = t_span / n
        c = 299792458.0
        dt_l = np.abs(1 / (1 / l_min - 1 / l_max) / c)
        logger.debug('tspan {0:.3g}'.format(t_span))
        logger.debug('N {0:.3g}'.format(n))
        logger.debug('l_min {0:.3g}'.format(l_min))
        logger.debug('l_max {0:.3g}'.format(l_max))
        # If N is None we assume that the widget should be updated here.
        if update is True:
            self.frog_dt_label.setText('dt (fs) [{0:.1f} - {1:.1f}]'.format(dt_t * 1e15, dt_l * 1e15))
        # logger.debug('dt_t {0:.3g}'.format(dt_t))
        # logger.debug('dt_l {0:.3g}'.format(dt_l))
        return dt_t, dt_l

    def generate_wavelengths(self):
        l0 = self.ui.central_wavelength_spinbox.value() * 1e-9
        dl = self.ui.dispersion_spinbox.value() * 1e-9
        im_size = self.roi_data.shape
        l_range = dl * im_size[0]
        global_wavelengths = np.linspace(l0 - l_range / 2, l0 + l_range / 2, im_size[0])
        self.frog_wavelengths = global_wavelengths
        self.frog_time_data = self.ui.time_resolution_spinbox.value() * np.arange(im_size[1]) * 1e-15
        logger.info("ROI size: {0} x {1}".format(im_size[0], im_size[1]))
        logger.info("New wavelengths: l0={0:.1e}, l_range={1:.1e}".format(l0, l_range))
        logger.info("New tau: frog_time_data[0]={0:.1e}, frog_time_data[-1]={1:.1e}".format(self.frog_time_data[0], self.frog_time_data[-1]))
        self.new_wavelengths_signal.emit()

    def update_status(self, new_status):
        self.ui.status_label.setText(new_status)

    def load_image(self):
        pathname = self.ui.directory_lineedit.text()
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Select frog trace', pathname, '*.png')[0]
        logger.info("File {0} selected".format(filename))
        if filename == "":
            return
        pic = imread(filename).astype(np.double)
        if len(pic.shape) > 2:
            # The image is RGB, convert by taking one channel
            pic = pic[:, :, 0]
        with self.lock:
            self.loaded_image = pic
        self.ui.load_filename_label.setText(filename.split("/")[-1])
        self.new_image_signal.emit(pic)
        self.update_image_roi()
        self.status_messenger.add_message("Image loaded", time.time(), 5.0)

    def load_bkg_image(self, filename=None):
        logger.info("Load background image {0}".format(filename))
        if not isinstance(filename, str):
            pathname = self.ui.directory_lineedit.text()
            filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Select background image', pathname, '*.png')[0]
            self.bkg_image_filename = filename
        self.ui.bkg_image_filename_label.setText(self.bkg_image_filename.split("/")[-1])
        logger.info("File {0} selected".format(filename))
        if filename == "":
            return
        pic = imread(filename).astype(np.double)
        if len(pic.shape) > 2:
            # The image is RGB, convert by taking one channel
            pic = pic[:, :, 0]
        with self.lock:
            self.bkg_image = np.double(pic)
        self.new_bkg_image_signal.emit(pic)
        self.update_image_roi()
        self.status_messenger.add_message("Background image loaded", time.time(), 5.0)

    def camera_bkg_image(self):
        logger.info("Camera background image")
        with self.lock:
            pic = self.camera_image.copy()
        if len(pic.shape) > 2:
            # The image is RGB, convert by taking one channel
            pic = pic[:, :, 0]
        with self.lock:
            self.bkg_image = np.double(pic)
        self.new_bkg_image_signal.emit(pic)
        self.update_image_roi()
        self.status_messenger.add_message("Background image loaded", time.time(), 5.0)

    def update_primary_image(self, pic):
        v = self.ui.primary_image_widget.getView()
        if self.roi in v.addedItems:
            self.ui.primary_image_widget.getView().removeItem(self.roi)
        if self.ui.loaded_image_radiobutton.isChecked():
            pic = self.loaded_image
            pic_sec = self.roi_data
            self.ui.primary_image_widget.getView().addItem(self.roi)
            self.ui.primary_image_label.setText("Loaded image")
            self.ui.secondary_image_label.setText("ROI image")
        elif self.ui.bkg_image_radiobutton.isChecked():
            pic = self.bkg_image
            pic_sec = self.roi_data
            self.ui.primary_image_label.setText("Background image")
            self.ui.secondary_image_label.setText("ROI image")
        elif self.ui.frog_image_radiobutton.isChecked():
            pic = self.frog_filtered_data
            pic_sec = self.frog_reconstructed_data
            self.ui.primary_image_label.setText("FROG input image")
            self.ui.secondary_image_label.setText("Reconstructed FROG image")
        elif self.ui.camera_image_radiobutton.isChecked():
            with self.lock:
                pic = self.camera_image.copy()
                pic_sec = self.roi_data.copy()
            self.ui.primary_image_widget.getView().addItem(self.roi)
            self.ui.primary_image_label.setText("Camera image")
            self.ui.secondary_image_label.setText("ROI image")
        else:
            pic = pic
            pic_sec = self.roi_data
        try:
            logger.debug("pic size {0}, max {1}".format(pic.shape, pic.max()))
        except ValueError:
            return
        xscale = 1
        yscale = 1
        x0 = 0
        y0 = 0
        self.ui.primary_image_widget.setImage(np.transpose(pic), scale=[xscale, yscale], pos=[x0, y0], autoRange=False)
        self.ui.primary_image_widget.autoLevels()
        # self.ui.primary_image_widget.autoRange()
        self.ui.primary_image_widget.update()
        self.update_secondary_image(pic_sec)
        # self.status_messenger.add_message("Primary image updated", time.time(), 0.5)

    def update_secondary_image(self, pic):
        try:
            logger.debug("pic size {0}, max {1}".format(pic.shape, pic.max()))
        except ValueError:
            pic = np.zeros((10, 10))
        xscale = 1
        yscale = 1
        x0 = 0
        y0 = 0
        self.ui.secondary_image_widget.setImage(np.transpose(pic), scale=[xscale, yscale], pos=[x0, y0], autoRange=False)
        # self.ui.secondary_image_widget.autoLevels()
        # self.ui.secondary_image_widget.autoRange()
        self.ui.secondary_image_widget.update()
        # self.status_messenger.add_message("Secondary image updated", time.time(), 0.5)

    def update_primary_plot(self, data):
        logger.debug("Updating plot widget")
        y_data = self.frog_calc.get_trace_abs() ** 2
        x_data = self.frog_calc.get_t()
        # y_data = np.random.random(x_data.shape)
        self.temporal_result_plot_intensity.setData(x=x_data, y=y_data)
        self.temporal_result_plot_intensity.update()
        y_data = self.frog_calc.get_trace_phase()
        good_ind = ~np.isnan(y_data)
        # y_data = np.random.random(x_data.shape)
        self.temporal_result_plot_phase.setData(x=x_data[good_ind], y=y_data[good_ind])
        self.temporal_result_plot_phase.update()
        y_data = self.frog_calc.get_trace_spectral_abs() ** 2
        x_data = self.frog_calc.get_w()
        good_ind = ~np.isnan(y_data)
        self.plot2.setData(x=x_data[good_ind], y=y_data[good_ind])
        self.plot2.update()

    def update_image_roi(self):
        """
        Update frog images due to new roi set.
        :return:
        """
        logger.debug(
            "Roi pos: {0:.2f} nm, {1:.2f} ps".format(self.roi.pos()[0], self.roi.pos()[1]))
        logger.debug(
            "Roi size: {0:.2f} nm, {1:.2f} ps".format(self.roi.size()[0], self.roi.size()[1]))
        self.ui.roi_top_spinbox.blockSignals(True)
        self.ui.roi_top_spinbox.setValue(self.roi.pos()[0])
        self.ui.roi_top_spinbox.blockSignals(False)
        self.ui.roi_left_spinbox.blockSignals(True)
        self.ui.roi_left_spinbox.setValue(self.roi.pos()[1])
        self.ui.roi_left_spinbox.blockSignals(False)
        self.ui.roi_width_spinbox.blockSignals(True)
        self.ui.roi_width_spinbox.setValue(self.roi.size()[0])
        self.ui.roi_width_spinbox.blockSignals(False)
        self.ui.roi_height_spinbox.blockSignals(True)
        self.ui.roi_height_spinbox.setValue(self.roi.size()[1])
        self.ui.roi_height_spinbox.blockSignals(False)

        x0, y0 = self.roi.pos()
        x1, y1 = self.roi.size()
        x1 += x0
        y1 += y0
        x0_ind = np.int(np.maximum(np.int(x0), 0))
        x1_ind = np.int(np.minimum(np.int(x1), x1 - 1))
        y0_ind = np.int(np.maximum(np.int(y0), 0))
        y1_ind = np.int(np.minimum(np.int(y1), y1 - 1))

        if x0_ind > x1_ind:
            x0_ind, x1_ind = x1_ind, x0_ind
        if y0_ind > y1_ind:
            y0_ind, y1_ind = y1_ind, y0_ind

        logger.debug('x0_ind ' + str(x0_ind))
        logger.debug('x1_ind ' + str(x1_ind))
        logger.debug('y0_ind ' + str(y0_ind))
        logger.debug('y1_ind ' + str(y1_ind))
        with self.lock:
            try:
                if self.ui.camera_image_radiobutton.isChecked():
                    roi_img = self.camera_image[y0_ind:y1_ind, x0_ind:x1_ind]
                else:
                    roi_img = self.loaded_image[y0_ind:y1_ind, x0_ind:x1_ind]
            except IndexError:
                return
        try:
            bkg_img = self.bkg_image[y0_ind:y1_ind, x0_ind:x1_ind]
        except IndexError:
            logger.debug("Background image size not compatible")
            bkg_img = np.zeros_like(roi_img)
            self.status_messenger.add_message("Background image size not compatible", time.time(), 5)
        logger.debug("bkg_img shape {0}".format(bkg_img.shape))
        logger.debug("roi_img bkg subtracted")
        logger.debug("roi_img shape {0}".format(roi_img.shape))
        try:
            if self.ui.bkg_subtract_radiobutton.isChecked():
                roi_img = (roi_img - bkg_img) / roi_img.max()
            else:
                roi_img = roi_img / roi_img.max()
            logger.debug(''.join(('Roi pos: ', str(x0), 'm x ', str(y0), 's')))
            logger.debug(''.join(('Roi size: ', str(x1 - x0), 'm x ', str(y1 - y0), 's')))
            # dt_est = self.estimate_frog_dt(self.ui.frog_size_spinbox.value(), y1 - y0, x0, x1)
            # self.frog_dt_label.setText('dt (fs) [{0:.1f} - {1:.1f}]'.format(dt_est[0] * 1e15, dt_est[1] * 1e15))
            logger.debug('Slice complete')
            thr = self.ui.threshold_spinbox.value()
            logger.debug('Threshold: ' + str(thr))
            filtered_img = roi_img - thr

            logger.debug('Filtering complete')
            filtered_img[filtered_img < 0.0] = 0.0
            self.roi_data = filtered_img
            self.generate_wavelengths()
            self.new_roi_image_signal.emit(filtered_img)
            logger.debug('Threshold complete')

        except ValueError:
            return
            # self.status_messenger.add_message("ROI updated", time.time(), 1.0)
        # if self.ui.auto_run_radiobutton.isChecked():
        #     self.generate_frog_image()

    def update_frog_plot_view(self):
        """
        Update view in the frog data and result data images
        :return:
        """
        self.temporal_result_viewbox.setGeometry(self.temporal_result_plotitem.vb.sceneBoundingRect())
        self.temporal_result_viewbox.linkedViewChanged(self.temporal_result_plotitem.vb, self.temporal_result_viewbox.XAxis)

    def restart_frog(self):
        self.frog_calc_running_flag = False
        self.generate_frog_image()

    def generate_frog_image(self):
        row_count = self.ui.probe_material_table.rowCount()
        probe_mat_list = list()
        gate_mat_list = list()
        if self.ui.frog_materials_enable_radiobutton.isChecked():
            for r in range(row_count):
                try:
                    mat = self.ui.probe_material_table.item(r, 0).text()
                    thickness = float(self.ui.probe_material_table.item(r, 1).text()) * 1e-3
                except AttributeError:
                    break
                probe_mat_list.append((mat, thickness))
            row_count = self.ui.gate_material_table.rowCount()
            for r in range(row_count):
                try:
                    mat = self.ui.gate_material_table.item(r, 0).text()
                    thickness = float(self.ui.gate_material_table.item(r, 1).text()) * 1e-3
                except AttributeError:
                    break
                gate_mat_list.append((mat, thickness))
        self.frog_calc.probe_mat_list = probe_mat_list
        self.frog_calc.gate_mat_list = gate_mat_list
        pic = self.roi_data
        l_start = self.frog_wavelengths[0]
        l_stop = self.frog_wavelengths[-1]
        tau_start = self.frog_time_data[0]
        tau_stop = self.frog_time_data[-1]
        dt = self.ui.frog_dt_spinbox.value() * 1e-15
        l0 = self.ui.central_wavelength_spinbox.value() * 1e-9
        N = self.ui.frog_size_spinbox.value()
        geometry = self.ui.frog_geometry_combobox.currentText()
        self.frog_calc.init_pulsefield_random(N, dt, l0, geometry=geometry)
        logger.debug("Condition {6} frog trace: \n"
                    "pic: {4}\n"
                    "l_start: {0:.1f} nm, l_stop: {1:.1f} nm\n"
                    "tau_start: {2:.1f} fs, tau_stop: {3:.1f} fs\n"
                    "dt: {5}\n"
                    "".format(l_start*1e9, l_stop*1e9, tau_start*1e15,
                              tau_stop*1e15, type(pic), dt, geometry))
        self.frog_calc.condition_frog_trace(pic, l_start, l_stop, tau_start, tau_stop, N,
                                            thr=0.1, filter_img=False)
        frog_pic = self.frog_calc.I_w_tau
        self.frog_filtered_data = np.fft.fftshift(frog_pic, 0)
        # self.new_frog_image_signal.emit(frog_pic)
        self.ui.frog_error_label.setText("--")
        self.error_plot.setData(y=[])
        if self.ui.auto_run_radiobutton.isChecked():
            # self.run_frog_calc()
            self.start_frog_calc_thread()

    def start_frog_calc_thread(self):
        if not self.frog_calc_running_flag:
            with self.lock:
                self.frog_calc_running_flag = True
            self.frog_calc_thread = QtCore.QThread()
            geometry = self.ui.frog_geometry_combobox.currentText()
            iterations = self.ui.frog_iterations_spinbox.value()
            self.frog_calc_worker = FrogCalculationWorker(self.frog_calc, iterations, geometry)
            self.frog_calc_worker.moveToThread(self.frog_calc_thread)
            self.frog_calc_worker.ready_signal.connect(self.frog_calc_thread.quit)
            self.frog_calc_worker.ready_signal.connect(self.frog_calc_ready)
            self.frog_calc_worker.ready_signal.connect(self.frog_calc_worker.deleteLater)
            self.frog_calc_worker.ready_signal.connect(self.frog_calc_thread.deleteLater)
            self.frog_calc_thread.started.connect(self.frog_calc_worker.run)
            self.frog_calc_thread.start()

    def run_frog_calc(self):
        logger.info("frog_calc_running_flag: {0}".format(self.frog_calc_running_flag))
        if not self.frog_calc_running_flag:
            with self.lock:
                self.frog_calc_running_flag = True
            geometry = self.ui.frog_geometry_combobox.currentText()
            iterations = self.ui.frog_iterations_spinbox.value()
            self.frog_calc.run_cycle_pc(iterations, geometry=geometry)
            self.frog_calc_ready_signal.emit()

    def frog_calc_ready(self):
        logger.info("Frog calc ready.")
        self.frog_reconstructed_data = np.fft.fftshift(np.abs(self.frog_calc.Esig_w_tau) ** 2, 0)
        self.new_frog_image_signal.emit(self.frog_filtered_data)
        error = self.frog_calc.G_hist
        with self.lock:
            try:
                self.ui.frog_error_label.setText("{0:.4f}".format(error[-1]))
            except IndexError:
                pass
            self.error_plot.setData(y=error)
            self.error_plot.update()
            self.ui.frog_pulse_duration_label.setText("{0:.1f} fs".format(1e15 * self.frog_calc.get_trace_summary()[0]))
            self.frog_calc_running_flag = False
            logger.info("frog_calc_running_flag: {0}".format(self.frog_calc_running_flag))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myapp = FrogReconstruction()
    myapp.show()
    sys.exit(app.exec_())

