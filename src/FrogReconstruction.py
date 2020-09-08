'''
Created on 3 May 2016

@author: Filip Lindau
'''
# -*- coding:utf-8 -*-


from PyQt4 import QtGui, QtCore

from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter
from matplotlib.pyplot import imsave
from scipy.misc import imread
from PIL import Image

import time
import sys
import os

sys.path.insert(0, '../../guitests/src/QTangoWidgets')

import logging

root = logging.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
root.addHandler(fh)
root.setLevel(logging.CRITICAL)

import pyqtgraph as pq

pq.graphicsItems.GradientEditorItem.Gradients['greyclip2'] = {
    'ticks': [(0.0, (0, 0, 50, 255)), (0.0001, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))], 'mode': 'rgb'}
pq.graphicsItems.GradientEditorItem.Gradients['thermalclip'] = {
    'ticks': [(0, (0, 0, 50, 255)), (0.0001, (0, 0, 0, 255)), (0.3333, (185, 0, 0, 255)), (0.6666, (255, 220, 0, 255)),
              (1, (255, 255, 255, 255))], 'mode': 'rgb'}

import PyTango as pt
import threading
import numpy as np

import FrogCalculationSimpleGP as FrogCalculation
# import FrogCalculationCLGP as FrogCalculation


class FrogReconstructionClient(QtGui.QWidget):
    """
    Class for scanning a motor while grabbing images to produce a frog trace. It can also analyse the scanned trace
    or saved traces.
    """
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings('Maxlab', 'FrogReconstruction')
        #        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.timeVector = None
        self.xData = None
        self.xDataTemp = None

        t0 = time.clock()

        self.positionOffset = 0.0

        self.roiData = np.zeros((2, 2))
        self.scanWavelengths = None
        self.trendData1 = None
        self.trendData2 = None
        self.scanData = np.array([])
        self.frogRawData = np.array([])
        self.frogFilteredData = np.array([])
        self.frogWavelengths = np.array([])
        self.frogTimeData = np.array([])
        self.frogCentralWavelength = 0.0
        self.scanTimeData = np.array([])
        self.timeMarginal = np.array([])
        self.timeMarginalTmp = 0
        self.posData = np.array([])
        self.currentPos = 0.0
        self.currentSample = 0
        self.avgData = 0
        self.avgSamples = None
        self.targetPos = 0.0
        self.measureUpdateTimes = np.array([time.time()])
        self.lock = threading.Lock()

        self.frogCalc = FrogCalculation.FrogCalculation()
        self.exportFileLocation = "."

        self.setupLayout()

    def generateWavelengths(self):
        l0 = self.centerWavelengthSpinbox.value() * 1e-9
        dl = self.dispersionSpinbox.value() * 1e-9
        imSize = self.spectrumImageWidget.getImageItem().image.shape[0]
        #         root.debug(''.join(('Image size: ', str(imSize))))
        lrange = dl * imSize
        globalWavelengths = np.linspace(l0 - lrange / 2, l0 + lrange / 2, imSize)
        startInd = np.int(self.ROI.pos()[0])
        stopInd = np.int(self.ROI.pos()[0] + np.ceil(self.ROI.size()[0]))
        root.debug("startInd: " + str(startInd) + ", stopInd: " + str(stopInd))
        self.scanWavelengths = globalWavelengths[startInd:stopInd]

        if self.scanWavelengths.shape[0] == self.spectrumData.shape[0]:
            self.plot1.setData(x=self.scanWavelengths, y=self.spectrumData)

    def updateFrogPlotView(self):
        self.frogResultViewbox.setGeometry(self.frogResultPlotitem.vb.sceneBoundingRect())
        self.frogResultViewbox.linkedViewChanged(self.frogResultPlotitem.vb, self.frogResultViewbox.XAxis)

    def updateFrogResultPlot(self):
        if self.frogTemporalSpectralCombobox.currentText() == "Temporal":
            x = self.frogCalc.get_t()
            Eabs = self.frogCalc.get_trace_abs()**2
            Ephi = self.frogCalc.get_trace_phase(linear_comp=True)
        else:
            x = self.frogCalc.get_w()
            Eabs = self.frogCalc.get_trace_spectral_abs()**2
            Ephi = self.frogCalc.get_trace_spectral_phase(linear_comp=True)

        self.frogResultPlotAbs.setData(x=x, y=Eabs)
        self.frogResultPlotAbs.update()
        self.frogResultPlotPhase.setData(x=x, y=Ephi)
        self.frogResultPlotPhase.update()

        if self.frogEtCheckbox.isChecked() is True:
            self.frogResultPlotAbs.show()
        else:
            self.frogResultPlotAbs.hide()

        if self.frogPhaseCheckbox.isChecked() is True:
            self.frogResultPlotPhase.show()
        else:
            self.frogResultPlotPhase.hide()

    def calculatePulseParameters(self):
        tFWHM, delta_ph = self.frogCalc.get_trace_summary()
        s_t = '%.3f' % (tFWHM * 1e12)
        self.frogEtFWHMLabel.setText(''.join((s_t, ' ps')))

        if self.frogTemporalSpectralCombobox.currentText() == 'Temporal':
            ph = self.frogCalc.get_trace_phase()
        else:
            ph = self.frogCalc.get_trace_spectral_phase()
        ph_good = ph[np.isfinite(ph)]
        self.frogEtPhaseLabel.setText('{0:.2f} rad'.format(delta_ph))

        self.updateExpansionCoefficients()

    def updateExpansionCoefficients(self):
        if self.frogTemporalSpectralCombobox.currentText() == 'Temporal':
            p = self.frogCalc.get_temporal_phase_expansion(orders=4, prefix=1e-15)
            self.frogExpansion1.setText("{0:.2e} fs".format(p[3]))
            self.frogExpansion2.setText("{0:.2e} fs^2".format(p[2]))
            self.frogExpansion3.setText("{0:.2e} fs^3".format(p[1]))
            self.frogExpansion4.setText("{0:.2e} fs^4".format(p[0]))
        else:
            p = self.frogCalc.get_spectral_phase_expansion(orders=4, prefix=2*np.pi*1e15)
            self.frogExpansion1.setText("{0:.2e} fs^-1".format(p[3]))
            self.frogExpansion2.setText("{0:.2e} fs^-2".format(p[2]))
            self.frogExpansion3.setText("{0:.2e} fs^-3".format(p[1]))
            self.frogExpansion4.setText("{0:.2e} fs^-4".format(p[0]))

        self.updateFrogResultPlot()

    def updateImageThreshold(self):
        h = self.frogImageWidget.getHistogramWidget()
        levels = h.item.getLevels()
        root.debug('Levels: ' + str(levels))
        root.debug('frogRawData.max() ' + str(self.frogRawData.max()))
        self.frogThresholdSpinbox.setValue(levels[0] / self.frogRawData.max())
        self.updateFrogRoi()

    def updateFrogRoi(self):
        root.debug(''.join(('Roi pos: ', str(self.frogRoi.pos()))))
        root.debug(''.join(('Roi size: ', str(self.frogRoi.size()))))
        if self.frogRawData.size != 0:
            root.debug(''.join(('Raw data: ', str(self.frogRawData.shape))))

            bkg = self.frogRawData[0, :]
            bkgImg = self.frogRawData - np.tile(bkg, (self.frogRawData.shape[0], 1))
            x0, y0 = self.frogRoi.pos()
            x1, y1 = self.frogRoi.size()
            x1 += x0
            y1 += y0
            dx = self.frogWavelengths[1] - self.frogWavelengths[0]
            dy = (self.frogTimeData[-1] - self.frogTimeData[0]) / self.frogTimeData.shape[0]
            x0_ind = np.maximum(np.int((x0 - self.frogWavelengths[0]) / dx), 0)
            x1_ind = np.minimum(np.int((x1 - self.frogWavelengths[0]) / dx), self.frogWavelengths.shape[0] - 1)
            y0_ind = np.maximum(np.int((y0 - self.frogTimeData[0]) / dy), 0)
            y1_ind = np.minimum(np.int((y1 - self.frogTimeData[0]) / dy), self.frogTimeData.shape[0] - 1)
            root.debug('x0_ind ' + str(x0_ind))
            root.debug('x1_ind ' + str(x1_ind))
            root.debug('y0_ind ' + str(y0_ind))
            root.debug('y1_ind ' + str(y1_ind))
            roiImg = bkgImg[y0_ind:y1_ind, x0_ind:x1_ind]
            roiImg = roiImg / roiImg.max()
            root.debug(''.join(('Roi pos: ', str(x0), 'm x ', str(y0), 's')))
            root.debug(''.join(('Roi size: ', str(x1-x0), 'm x ', str(y1-y0), 's')))
            dt_est = self.estimateFrogDt(self.frogNSpinbox.value(), y1 - y0, x0, x1)
            self.frogDtLabel.setText('dt (fs) [{0:.1f} - {1:.1f}]'.format(dt_est[0]*1e15, dt_est[1]*1e15))
            root.debug('Slice complete')
            thr = np.maximum(self.frogThresholdSpinbox.value() - bkg.mean() / self.frogRawData.max(), 0)
            root.debug('Threshold: ' + str(thr))
            kernel = self.frogKernelMedianSpinbox.value()
            root.debug('Starting medfilt...')
            if kernel > 1:
                filteredImg = medfilt2d(roiImg, kernel) - thr
            else:
                filteredImg = roiImg - thr
            gauss_kernel = self.frogKernelGaussianSpinbox.value()
            if gauss_kernel > 1:
                filteredImg = gaussian_filter(filteredImg, gauss_kernel)

            #             filteredImg = roiImg - thr
            root.debug('Filtering complete')
            filteredImg[filteredImg < 0.0] = 0.0
            self.frogFilteredData = filteredImg
            root.debug('Threshold complete')
            x0, x1 = (self.frogWavelengths[0], self.frogWavelengths[-1])
            y0, y1 = (self.frogTimeData[0], self.frogTimeData[-1])
            xscale, yscale = (x1 - x0) / self.frogWavelengths.shape[0], (y1 - y0) / self.frogTimeData.shape[0]
            self.frogRoiImageWidget.setImage(filteredImg.transpose(), scale=[xscale, yscale], pos=[x0, y0])
            root.debug('Set image complete')
            self.frogRoiImageWidget.autoRange()
            root.debug('Autorange complete')

    def fitFrogRoi(self):
        x0 = self.frogWavelengths[0]
        y0 = self.frogTimeData[0]
        dx = self.frogWavelengths[-1] - self.frogWavelengths[0]
        dy = (self.frogTimeData[-1] - self.frogTimeData[0])
        self.frogRoi.setPos([x0, y0], finish=False, update=False)
        self.frogRoi.setSize([dx, dy])
        root.debug(''.join(('Roi pos: ', str(x0), 'm x ', str(y0), 's')))
        root.debug(''.join(('Roi size: ', str(dx), 'm x ', str(dy), 's')))

    def estimateFrogDt(self, N=None, t_span=None, l_min=None, l_max=None):
        update = False
        if N is None:
            N = self.frogNSpinbox.value()
            update = True
        if t_span is None:
            t_span = self.frogRoi.size()[1]
        if l_min is None:
            l_min = self.frogRoi.pos()[0]
            l_max = l_min + self.frogRoi.size()[0]
        dt_t = t_span / N
        c = 299792458.0
        dt_l = np.abs(1 / (1/l_min - 1/l_max) / c)
        root.debug('tspan {0:.3g}'.format(t_span))
        root.debug('N {0:.3g}'.format(N))
        root.debug('l_min {0:.3g}'.format(l_min))
        root.debug('l_max {0:.3g}'.format(l_max))
        # If N is None we assume that the widget should be updated here.
        if update is True:
            self.frogDtLabel.setText('dt (fs) [{0:.1f} - {1:.1f}]'.format(dt_t * 1e15, dt_l * 1e15))
        # root.debug('dt_t {0:.3g}'.format(dt_t))
        # root.debug('dt_l {0:.3g}'.format(dt_l))
        return dt_t, dt_l

    def startFrogInversion(self):
        N = self.frogNSpinbox.value()
        # frogImg = self.frogRoiImageWidget.getImageItem().image.transpose()
        frogImg = self.frogFilteredData
        if frogImg.size > 0:
            x0, y0 = self.frogRoi.pos()
            x1, y1 = self.frogRoi.size()
            x1 += x0
            y1 += y0
            dx = self.frogWavelengths[1] - self.frogWavelengths[0]
            dy = (self.frogTimeData[-1] - self.frogTimeData[0]) / self.frogTimeData.shape[0]
            x0_ind = np.maximum(np.int((x0 - self.frogWavelengths[0]) / dx), 0)
            x1_ind = np.minimum(np.int((x1 - self.frogWavelengths[0]) / dx), self.frogWavelengths.shape[0] - 1)
            y0_ind = np.maximum(np.int((y0 - self.frogTimeData[0]) / dy), 0)
            y1_ind = np.minimum(np.int((y1 - self.frogTimeData[0]) / dy), self.frogTimeData.shape[0] - 1)

            # lStartInd = np.int(self.frogRoi.pos()[0])
            # lStopInd = np.int(self.frogRoi.pos()[0] + np.ceil(self.frogRoi.size()[0]))
            lStartInd = x0_ind
            lStopInd = x1_ind
            root.debug(''.join(('Wavelength range: ', str(self.frogWavelengths[lStartInd]), ' - ',
                                str(self.frogWavelengths[lStopInd]))))
            if self.frogWavelengths[lStartInd] > 1:
                l_start = self.frogWavelengths[lStartInd] * 1e-9
                l_stop = self.frogWavelengths[lStopInd] * 1e-9
            else:
                l_start = self.frogWavelengths[lStartInd]
                l_stop = self.frogWavelengths[lStopInd]
            l0 = (l_stop + l_start) / 2

            # tStartInd = np.int(self.frogRoi.pos()[1])
            # tStopInd = np.int(self.frogRoi.pos()[1] + np.ceil(self.frogRoi.size()[1]))
            tStartInd = y0_ind
            tStopInd = y1_ind

            root.debug(''.join(('Time range: ', str(self.frogTimeData[tStartInd]), ' - ', str(self.frogTimeData[tStopInd]))))
            tau_mean = (self.frogTimeData[tStartInd] + self.frogTimeData[tStopInd]) / 2
            tau_start = self.frogTimeData[tStartInd] - tau_mean
            tau_stop = self.frogTimeData[tStopInd] - tau_mean
            if self.frogTimeData.shape != 0:
                dt = self.frogDtSpinbox.value() * 1e-15
            else:
                dt = 1e-15

            root.debug('Wavelength input data: l_start=' + str(l_start) + ', l_stop=' + str(l_stop) +
                       ', type: ' + str(type(lStartInd)))
            root.debug('Time input data: tau_start=' + str(tau_start) + ', tau_stop=' + str(tau_stop) + ', dt=' +
                       str(dt))

            self.frogCalc.init_pulsefield_random(N, dt, l0)
            self.frogCalc.condition_frog_trace(frogImg, l_start, l_stop, tau_start, tau_stop, N,
                                               thr=0*self.frogThresholdSpinbox.value())

            root.debug('frogImg shape: ' + str(frogImg.shape) + ', 10 values: ' + str(frogImg[0, 0:10]))
            root.debug('I_w_tau shape: ' + str(self.frogCalc.I_w_tau.shape) + ', 10 values: ' +
                       str(self.frogCalc.I_w_tau[0, 0:10]))
            self.frogCalcImageWidget.setImage(np.transpose(self.frogCalc.I_w_tau))
            self.frogCalcImageWidget.autoRange()
            self.frogCalcImageWidget.update()

            self.frogCalc.G_hist = []
            self.frogErrorPlot.setData([])
            self.frogErrorPlot.update()

            self.continueFrogInversion()

    def continueFrogInversion(self):
        if self.frogCalc.I_w_tau is not None:
            algo = str(self.frogAlgoCombobox.currentText())
            if str(self.frogMethodCombobox.currentText()) == 'Vanilla':
                er = self.frogCalc.run_cycle_vanilla(self.frogIterationsSpinbox.value(), algo=algo)
            elif str(self.frogMethodCombobox.currentText()) == 'GP':
                er = self.frogCalc.run_cycle_gp(self.frogIterationsSpinbox.value(), algo=algo)

            self.frogErrorPlot.setData(self.frogCalc.G_hist)
            self.frogErrorPlot.update()

            self.frogCalcResultImageWidget.setImage(np.transpose(self.frogCalc.get_reconstructed_intensity()))
            self.frogCalcResultImageWidget.autoRange()
            self.frogCalcResultImageWidget.update()

            self.calculatePulseParameters()

    def loadFrogTrace(self):
        fileLocation = self.exportFileLocation
        filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Select frog trace', fileLocation, 'frogtrace_*.png'))
        root.debug(''.join(('File selected: ', str(filename))))

        fNameRoot = '_'.join((filename.split('_')[:-1]))
        tData = np.loadtxt(''.join((fNameRoot, '_timevector.txt')))
        tData = tData - tData.mean()
        lData = np.loadtxt(''.join((fNameRoot, '_wavelengthvector.txt')))
        if lData[0] > 1:
            lData = lData * 1e-9
        pic = np.float32(imread(''.join((fNameRoot, '_image.png'))))

        root.debug(''.join(('Pic: ', str(pic.shape))))
        root.debug(''.join(('Time data: ', str(tData.shape), ' ', str(tData[0]))))
        root.debug(''.join(('l data: ', str(lData.shape), ' ', str(lData[0]))))

        self.frogRawData = pic
        self.frogTimeData = tData
        self.frogWavelengths = lData
        self.frogCentralWavelength = 1e9*lData.mean()

        x0, x1 = (self.frogWavelengths[0], self.frogWavelengths[-1])
        y0, y1 = (self.frogTimeData[0], self.frogTimeData[-1])
        xscale, yscale = (x1 - x0) / self.frogWavelengths.shape[0], (y1 - y0) / self.frogTimeData.shape[0]
        self.frogImageWidget.setImage(np.transpose(pic), scale=[xscale, yscale], pos=[x0, y0])
        self.frogImageWidget.autoLevels()
        self.frogImageWidget.autoRange()
        self.fitFrogRoi()
        # self.updateFrogRoi()
        self.traceSourceLabel.setText(fNameRoot.split('/')[-1])

    def loadScandataToFrog(self):
        self.frogRawData = self.scanData
        self.frogTimeData = self.scanTimeData
        self.frogWavelengths = self.scanWavelengths
        self.frogCentralWavelength = self.centerWavelengthSpinbox.value()

        x0, x1 = (self.frogWavelengths[0], self.frogWavelengths[-1])
        y0, y1 = (self.frogTimeData[0], self.frogTimeData[-1])
        xscale, yscale = (x1 - x0) / self.frogWavelengths.shape[0], (y1 - y0) / self.frogTimeData.shape[0]
        self.frogImageWidget.setImage(np.transpose(self.frogRawData), scale=[xscale, yscale], pos=[x0, y0])
        self.frogImageWidget.autoLevels()
        self.frogImageWidget.autoRange()
        self.fitFrogRoi()
        # self.updateFrogRoi()
        self.traceSourceLabel.setText("Scan")

    def closeEvent(self, event):

        root.debug(''.join(('Window size: ', str(self.size()))))
        root.debug(''.join(('Window pos: ', str(self.pos().y()))))
        self.settings.setValue('windowSizeW', np.int(self.size().width()))
        self.settings.setValue('windowSizeH', np.int(self.size().height()))
        self.settings.setValue('windowPosX', np.int(self.pos().x()))
        self.settings.setValue('windowPosY', np.int(self.pos().y()))

        self.settings.setValue('frogIterations', np.int(self.frogIterationsSpinbox.value()))
        self.settings.setValue('frogSize', np.int(self.frogNSpinbox.value()))
        self.settings.setValue('frogMethod', np.int(self.frogMethodCombobox.currentIndex()))
        self.settings.setValue('frogDt', np.float(self.frogDtSpinbox.value()))
        self.settings.setValue('frogThreshold', np.float(self.frogThresholdSpinbox.value()))
        self.settings.setValue('frogKernel', np.int(self.frogKernelMedianSpinbox.value()))
        self.settings.setValue('frogKernelGaussian', np.int(self.frogKernelGaussianSpinbox.value()))
        self.settings.setValue('frogRoiPosX', np.float(self.frogRoi.pos()[0]))
        self.settings.setValue('frogRoiPosY', np.float(self.frogRoi.pos()[1]))
        self.settings.setValue('frogRoiSizeW', np.float(self.frogRoi.size()[0]))
        self.settings.setValue('frogRoiSizeH', np.float(self.frogRoi.size()[1]))
        self.settings.setValue('frogTemporalSpectral', np.int(self.frogTemporalSpectralCombobox.currentIndex()))

        self.settings.sync()
        event.accept()

    def setupFrogLayout(self):
        self.frogNSpinbox = QtGui.QSpinBox()
        self.frogNSpinbox.setMinimum(0)
        self.frogNSpinbox.setMaximum(4096)
        #         self.frogNSpinbox.setValue(128)
        self.frogNSpinbox.setValue(self.settings.value('frogSize', 128).toInt()[0])
        self.frogNSpinbox.editingFinished.connect(self.estimateFrogDt)
        self.frogMethodCombobox = QtGui.QComboBox()
        self.frogMethodCombobox.addItem('Vanilla')
        self.frogMethodCombobox.addItem('GP')
        self.frogMethodCombobox.setCurrentIndex(self.settings.value('frogMethod', 0).toInt()[0])
        self.frogAlgoCombobox = QtGui.QComboBox()
        self.frogAlgoCombobox.addItem('PG')
        self.frogAlgoCombobox.addItem('SHG')
        self.frogAlgoCombobox.addItem('SD')
        self.frogAlgoCombobox.setCurrentIndex(self.settings.value('frogAlgo', 0).toInt()[0])
        self.frogStartButton = QtGui.QPushButton('Start')
        self.frogStartButton.clicked.connect(self.startFrogInversion)
        self.frogContinueButton = QtGui.QPushButton('Continue')
        self.frogContinueButton.clicked.connect(self.continueFrogInversion)
        self.frogIterationsSpinbox = QtGui.QSpinBox()
        #         self.frogIterationsSpinbox.setValue(20)
        self.frogIterationsSpinbox.setMinimum(0)
        self.frogIterationsSpinbox.setMaximum(999)
        self.frogIterationsSpinbox.setValue(self.settings.value('frogIterations', 20).toInt()[0])
        self.frogThresholdSpinbox = QtGui.QDoubleSpinBox()
        self.frogThresholdSpinbox.setMinimum(0.0)
        self.frogThresholdSpinbox.setMaximum(1.0)
        self.frogThresholdSpinbox.setSingleStep(0.01)
        self.frogThresholdSpinbox.setDecimals(3)
        self.frogThresholdSpinbox.setValue(self.settings.value('frogThreshold', 0.05).toDouble()[0])
        self.frogThresholdSpinbox.editingFinished.connect(self.updateFrogRoi)
        self.frogFitRoiButton = QtGui.QPushButton('Fit')
        self.frogFitRoiButton.clicked.connect(self.fitFrogRoi)

        self.frogDtLabel = QtGui.QLabel('dt (fs) []')
        self.frogDtSpinbox = QtGui.QDoubleSpinBox()
        self.frogDtSpinbox.setMinimum(0.0)
        self.frogDtSpinbox.setMaximum(10000.0)
        self.frogDtSpinbox.setSingleStep(1.0)
        self.frogDtSpinbox.setDecimals(1)
        self.frogDtSpinbox.setValue(self.settings.value('frogDt', 1.0).toDouble()[0])
        self.frogKernelMedianSpinbox = QtGui.QSpinBox()
        self.frogKernelMedianSpinbox.setMinimum(1)
        self.frogKernelMedianSpinbox.setMaximum(99)
        self.frogKernelMedianSpinbox.setSingleStep(2)
        self.frogKernelMedianSpinbox.setValue(self.settings.value('frogKernel', 3).toInt()[0])
        self.frogKernelMedianSpinbox.editingFinished.connect(self.updateFrogRoi)
        self.frogKernelGaussianSpinbox = QtGui.QSpinBox()
        self.frogKernelGaussianSpinbox.setMinimum(1)
        self.frogKernelGaussianSpinbox.setMaximum(99)
        self.frogKernelGaussianSpinbox.setSingleStep(2)
        self.frogKernelGaussianSpinbox.setValue(self.settings.value('frogKernelGaussian', 3).toInt()[0])
        self.frogKernelGaussianSpinbox.editingFinished.connect(self.updateFrogRoi)
        self.traceSourceLabel = QtGui.QLabel('Scan')
        self.traceSourceLabel.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.traceSourceLabel.setMaximumWidth(250)
        self.traceSourceLabel.setMinimumWidth(250)
        self.frogLoadButton = QtGui.QPushButton('Load')
        self.frogLoadButton.clicked.connect(self.loadFrogTrace)
        self.frogUseScanButton = QtGui.QPushButton('Scan data')
        self.frogUseScanButton.clicked.connect(self.loadScandataToFrog)

        self.frogEtCheckbox = QtGui.QCheckBox()
        self.frogEtCheckbox.setChecked(True)
        self.frogEtCheckbox.stateChanged.connect(self.updateFrogResultPlot)
        self.frogPhaseCheckbox = QtGui.QCheckBox()
        self.frogPhaseCheckbox.setChecked(True)
        self.frogPhaseCheckbox.stateChanged.connect(self.updateFrogResultPlot)

        self.frogEtFWHMLabel = QtGui.QLabel()
        self.frogEtPhaseLabel = QtGui.QLabel()

        self.frogTemporalSpectralCombobox = QtGui.QComboBox()
        self.frogTemporalSpectralCombobox.addItem('Temporal')
        self.frogTemporalSpectralCombobox.addItem('Spectral')
        self.frogTemporalSpectralCombobox.setCurrentIndex(self.settings.value('frogTemporalSpectral', 0).toInt()[0])
        self.frogTemporalSpectralCombobox.currentIndexChanged.connect(self.updateExpansionCoefficients)
        self.frogExpansion1 = QtGui.QLabel()
        self.frogExpansion2 = QtGui.QLabel()
        self.frogExpansion3 = QtGui.QLabel()
        self.frogExpansion4 = QtGui.QLabel()

        self.frogGridLayout1 = QtGui.QGridLayout()
        self.frogGridLayout1.addWidget(QtGui.QLabel("Image threshold"), 0, 0)
        self.frogGridLayout1.addWidget(self.frogThresholdSpinbox, 0, 1)
        self.frogGridLayout1.addWidget(QtGui.QLabel("Median kernel"), 1, 0)
        self.frogGridLayout1.addWidget(self.frogKernelMedianSpinbox, 1, 1)
        self.frogGridLayout1.addWidget(QtGui.QLabel("Gaussian kernel"), 2, 0)
        self.frogGridLayout1.addWidget(self.frogKernelGaussianSpinbox, 2, 1)
        self.frogGridLayout1.addItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding),
                                     3, 0)
        self.frogGridLayout1.addWidget(QtGui.QLabel("Fit ROI to data"), 4, 0)
        self.frogGridLayout1.addWidget(self.frogFitRoiButton, 4, 1)

        self.frogGridLayout2 = QtGui.QGridLayout()
        self.frogGridLayout2.addWidget(QtGui.QLabel("Frog algorithm"), 0, 0)
        self.frogGridLayout2.addWidget(self.frogAlgoCombobox, 0, 1)
        self.frogGridLayout2.addWidget(QtGui.QLabel("Frog method"), 1, 0)
        self.frogGridLayout2.addWidget(self.frogMethodCombobox, 1, 1)
        self.frogGridLayout2.addWidget(QtGui.QLabel("Frog size"), 2, 0)
        self.frogGridLayout2.addWidget(self.frogNSpinbox, 2, 1)
        self.frogGridLayout2.addWidget(self.frogDtLabel, 3, 0)
        self.frogGridLayout2.addWidget(self.frogDtSpinbox, 3, 1)
        self.frogGridLayout2.addItem(QtGui.QSpacerItem(30, 10, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding),
                                     4, 0)

        self.frogGridLayout3 = QtGui.QGridLayout()
        self.frogGridLayout3.addWidget(QtGui.QLabel("Trace source:"), 0, 0)
        self.frogGridLayout3.addWidget(self.traceSourceLabel, 0, 1, 1, 2)
        self.frogGridLayout3.addWidget(QtGui.QLabel("Load trace"), 1, 0)
        self.frogGridLayout3.addWidget(self.frogLoadButton, 1, 1)
        self.frogGridLayout3.addWidget(QtGui.QLabel("Load scan data"), 2, 0)
        self.frogGridLayout3.addWidget(self.frogUseScanButton, 2, 1)
        self.frogGridLayout3.addItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding),
                                     3, 0)
        self.frogGridLayout3.addWidget(QtGui.QLabel("Iterations"), 4, 0)
        self.frogGridLayout3.addWidget(self.frogIterationsSpinbox, 4, 1)
        self.frogGridLayout3.addWidget(QtGui.QLabel("Invert trace"), 5, 0)
        self.frogGridLayout3.addWidget(self.frogStartButton, 5, 1)
        self.frogGridLayout3.addWidget(self.frogContinueButton, 6, 1)

        self.frogGridLayout4 = QtGui.QGridLayout()
        self.frogGridLayout4.addWidget(QtGui.QLabel("Plot type"), 0, 0)
        self.frogGridLayout4.addWidget(self.frogTemporalSpectralCombobox, 0, 1)
        self.frogGridLayout4.addWidget(QtGui.QLabel("Show intensity"), 1, 0)
        self.frogGridLayout4.addWidget(self.frogEtCheckbox, 1, 1)
        self.frogGridLayout4.addWidget(QtGui.QLabel("Show phase"), 2, 0)
        self.frogGridLayout4.addWidget(self.frogPhaseCheckbox, 2, 1)
        self.frogGridLayout4.addItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum), 3,
                                     0)
        self.frogGridLayout4.addWidget(QtGui.QLabel("t_FWHM (intensity)"), 4, 0)
        self.frogGridLayout4.addWidget(self.frogEtFWHMLabel, 4, 1)
        self.frogGridLayout4.addWidget(QtGui.QLabel("Phase diff"), 5, 0)
        self.frogGridLayout4.addWidget(self.frogEtPhaseLabel, 5, 1)
        self.frogGridLayout4.addItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding),
                                     6, 0, 1, 2)

        self.frogGridLayout5 = QtGui.QGridLayout()
        self.frogGridLayout5.addWidget(QtGui.QLabel("Phase expansion"), 0, 0)
        self.frogGridLayout5.addItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum), 1,
                                     0)
        self.frogGridLayout5.addWidget(self.frogExpansion1, 2, 0, 1, 2)
        self.frogGridLayout5.addWidget(self.frogExpansion2, 3, 0, 1, 2)
        self.frogGridLayout5.addWidget(self.frogExpansion3, 4, 0, 1, 2)
        self.frogGridLayout5.addWidget(self.frogExpansion4, 5, 0, 1, 2)
        self.frogGridLayout5.addItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding),
                                     6, 0, 1, 2)

        # Error plot widget
        self.frogErrorWidget = pq.PlotWidget(useOpenGL=True)
        self.frogErrorPlot = self.frogErrorWidget.plot()
        self.frogErrorPlot.setPen((10, 200, 70))
        self.frogErrorWidget.setAntialiasing(True)
        self.frogErrorWidget.showGrid(True, True)
        self.frogErrorWidget.plotItem.setLabels(left='Reconstruction error')
        self.frogErrorWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.frogErrorWidget.setMinimumHeight(80)
        self.frogErrorWidget.setMaximumHeight(120)

        # E field plot widget
        self.frogResultWidget = pq.PlotWidget(useOpenGL=True)
        self.frogResultPlotAbs = self.frogResultWidget.plot()
        self.frogResultPlotAbs.setPen((10, 200, 70))

        self.frogResultPlotitem = self.frogResultWidget.plotItem
        self.frogResultPlotitem.setLabels(left='abs(Et)')
        self.frogResultViewbox = pq.ViewBox()
        self.frogResultPlotitem.showAxis('right')
        self.frogResultPlotitem.scene().addItem(self.frogResultViewbox)
        self.frogResultPlotitem.getAxis('right').linkToView(self.frogResultViewbox)
        self.frogResultViewbox.setXLink(self.frogResultPlotitem)
        self.frogResultPlotitem.getAxis('right').setLabel('Phase / rad')
        self.frogResultPlotPhase = pq.PlotCurveItem()
        self.frogResultPlotPhase.setPen((200, 70, 10))
        self.frogResultViewbox.addItem(self.frogResultPlotPhase)
        self.frogResultPlotitem.vb.sigResized.connect(self.updateFrogPlotView)
        self.frogResultWidget.setAntialiasing(True)
        self.frogResultWidget.showGrid(True, True)
        self.frogResultWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.updateFrogPlotView()

        # Raw frog trace with roi selector
        plt1 = pq.PlotItem(labels={'bottom': ('Spectrum', 'm'), 'left': ('Time delay', 's')})
        self.frogImageWidget = pq.ImageView(view=plt1)
        self.frogImageWidget.ui.histogram.gradient.loadPreset('thermalclip')
        self.frogImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.frogImageWidget.getView().setAspectLocked(False)
        h = self.frogImageWidget.getHistogramWidget()
        h.item.sigLevelChangeFinished.connect(self.updateImageThreshold)
        self.frogRoi = pq.RectROI([0, 300], [600, 20], pen=(0, 9))
        self.frogRoi.sigRegionChangeFinished.connect(self.updateFrogRoi)
        self.frogRoi.blockSignals(True)

        #         roiPosX = 10
        #         roiPosY = 10
        #         roiSizeH = 20
        #         roiSizeW = 500
        roiPosX = self.settings.value('frogRoiPosX', 10).toDouble()[0]
        roiPosY = self.settings.value('frogRoiPosY', 10).toDouble()[0]
        root.debug(''.join(('roiPos: ', str(roiPosX))))
        roiSizeW = self.settings.value('frogRoiSizeW', 100).toDouble()[0]
        roiSizeH = self.settings.value('frogRoiSizeH', 10).toDouble()[0]

        self.frogRoi.setPos([roiPosX, roiPosY], update=True)
        self.frogRoi.setSize([roiSizeW, roiSizeH], update=True)
        self.frogImageWidget.getView().addItem(self.frogRoi)

        self.frogRoi.blockSignals(False)

        # Filtered frog trace roi
        plt2 = pq.PlotItem(labels={'bottom': ('Spectrum', 'm'), 'left': ('Time delay', 's')})
        self.frogRoiImageWidget = pq.ImageView(view=plt2)
        self.frogRoiImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.frogRoiImageWidget.getView().setAspectLocked(False)
        pos = np.array([0.0, 0.0001, 1.0])
        colors = np.array([[0, 0, 50, 255], [0, 0, 0, 255], [255, 255, 255, 255]], dtype=np.ubyte)
        colormap = pq.ColorMap(pos, colors)
        # self.frogRoiImageWidget.ui.histogram.gradient.setColorMap(colormap)
        self.frogRoiImageWidget.ui.histogram.gradient.loadPreset('thermalclip')

        plt3 = pq.PlotItem(labels={'bottom': ('Frequency', 'px'), 'left': ('Time delay', 'px')})
        self.frogCalcImageWidget = pq.ImageView(view=plt3)
        self.frogCalcImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.frogCalcImageWidget.getView().setAspectLocked(False)
        self.frogCalcImageWidget.ui.histogram.gradient.loadPreset('thermalclip')
        plt4 = pq.PlotItem(labels={'bottom': ('Frequency', 'px'), 'left': ('Time delay', 'px')})
        self.frogCalcResultImageWidget = pq.ImageView(view=plt4)
        self.frogCalcResultImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.frogCalcResultImageWidget.getView().setAspectLocked(False)
        self.frogCalcResultImageWidget.view.setXLink(self.frogCalcImageWidget.view)
        self.frogCalcResultImageWidget.view.setYLink(self.frogCalcImageWidget.view)
        self.frogCalcResultImageWidget.ui.histogram.gradient.loadPreset('thermalclip')

        self.frogLayout2 = QtGui.QVBoxLayout()

    def setupLayout(self):
        root.debug('Setting up layout')
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English))
        self.layout = QtGui.QVBoxLayout(self)

        self.setupFrogLayout()

        frogLay = QtGui.QHBoxLayout()
        frogLay.addLayout(self.frogGridLayout1)
        frogLay.addSpacerItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum))
        frogLay.addLayout(self.frogGridLayout2)
        frogLay.addSpacerItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum))
        frogLay.addLayout(self.frogGridLayout3)
        frogLay.addSpacerItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum))
        frogLay.addWidget(self.frogErrorWidget)
        frogLay.addSpacerItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum))
        frogLay.addLayout(self.frogGridLayout4)
        frogLay.addSpacerItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum))
        frogLay.addLayout(self.frogGridLayout5)
        frogLay.addSpacerItem(QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum))

        self.plotLayout = QtGui.QHBoxLayout()

        self.frogLayout = QtGui.QVBoxLayout()

        self.frogLayout.addWidget(self.frogImageWidget)
        self.frogLayout.addWidget(self.frogRoiImageWidget)
        self.frogImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.plotLayout.addLayout(self.frogLayout)
        self.plotLayout.setStretchFactor(self.frogLayout, 3)
        self.frogLayout2.addWidget(self.frogCalcImageWidget)
        self.frogLayout2.addWidget(self.frogCalcResultImageWidget)
        self.plotLayout.addLayout(self.frogLayout2)
        self.plotLayout.setStretchFactor(self.frogLayout2, 2)
        self.plotLayout.addWidget(self.frogResultWidget)
        self.plotLayout.setStretchFactor(self.frogResultWidget, 3)

        self.layout.addLayout(frogLay)
        self.layout.addLayout(self.plotLayout)

        self.invisibleLayout = QtGui.QHBoxLayout()

        windowPosX = self.settings.value('windowPosX', 100).toInt()[0]
        windowPosY = self.settings.value('windowPosY', 100).toInt()[0]
        windowSizeW = self.settings.value('windowSizeW', 800).toInt()[0]
        windowSizeH = self.settings.value('windowSizeH', 300).toInt()[0]
        if windowPosX < 50:
            windowPosX = 200
        if windowPosY < 50:
            windowPosY = 200
        self.setGeometry(windowPosX, windowPosY, windowSizeW, windowSizeH)

        self.show()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    myapp = FrogReconstructionClient()
    myapp.show()
    sys.exit(app.exec_())
