'''
Created on 7 Mar 2016

@author: Filip Lindau
'''
# -*- coding:utf-8 -*-

"""
Created on Oct 1, 2015

@author: Filip Lindau
"""
cameraName = 'b-v0-gunlaser-csdb-0:10000/gunlaser/cameras/jai_test'
motorName = 'b-v0-gunlaser-csdb-0:10000/testgun/motors/zst25'

from PyQt4 import QtGui, QtCore

from scipy.signal import medfilt2d
from matplotlib.pyplot import imsave
from scipy.misc import imread
from PIL import Image

import time
import sys

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

from AttributeReadThreadClass import AttributeClass
import pyqtgraph as pq
import PyTango as pt
import threading
import numpy as np

import FrogCalculationSimpleGP as FrogCalculation


class MyQSplitter(QtGui.QSplitter):
    def __init__(self, parent=None):
        QtGui.QSplitter.__init__(self, parent)

    def dragEnterEvent(self, e):
        root.debug(''.join(('Drag enter event: ', str(e))))

    def resizeEvent(self, e):
        root.debug(''.join(('Resize event: ', str(e))))

    def changeEvent(self, e):
        root.debug(''.join(('Change event: ', str(e))))


class TangoDeviceClient(QtGui.QWidget):
    def __init__(self, cameraName, motorName, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings('Maxlab', 'Frog')
        #        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.timeVector = None
        self.xData = None
        self.xDataTemp = None
        self.cameraName = cameraName
        self.motorName = motorName

        self.setupLayout()

        t0 = time.clock()
        print time.clock() - t0, ' s'

        self.guiLock = threading.Lock()

        self.positionOffset = 0.0

        self.devices = {}
        self.devices['camera'] = pt.DeviceProxy(self.cameraName)
        self.devices['motor'] = pt.DeviceProxy(self.motorName)

        self.attributes = {}
        self.attributes['image'] = AttributeClass('image', self.devices['camera'], 0.1)
        self.attributes['shutter'] = AttributeClass('exposuretime', self.devices['camera'], 0.5)
        self.attributes['gain'] = AttributeClass('gain', self.devices['camera'], 0.5)
        self.attributes['position'] = AttributeClass('position', self.devices['motor'], 0.05)
        self.attributes['speed'] = AttributeClass('velocity', self.devices['motor'], 0.05)
        # self.attributes['triggerdelay'] = AttributeClass('triggerdelay', self.devices['redpitaya'], 0.3)
        # self.attributes['triggermode'] = AttributeClass('triggermode', self.devices['redpitaya'], 1.0)
        # self.attributes['triggersource'] = AttributeClass('triggersource', self.devices['redpitaya'], 1.0)
        # self.attributes['recordlength'] = AttributeClass('recordlength', self.devices['redpitaya'], 0.3)
        # self.attributes['samplerate'] = AttributeClass('samplerate', self.devices['redpitaya'], 0.3)

        self.attributes['image'].attrSignal.connect(self.readImage)
        self.attributes['shutter'].attrSignal.connect(self.readShutter)
        self.attributes['gain'].attrSignal.connect(self.readGain)
        self.attributes['position'].attrSignal.connect(self.readPosition)
        self.attributes['speed'].attrSignal.connect(self.readSpeed)
        # self.attributes['triggerdelay'].attrSignal.connect(self.readTrigDelay)
        # self.attributes['triggermode'].attrSignal.connect(self.readTrigMode)
        # self.attributes['triggersource'].attrSignal.connect(self.readTrigSource)
        # self.attributes['recordlength'].attrSignal.connect(self.readRecordLength)
        # self.attributes['samplerate'].attrSignal.connect(self.readSampleRate)

        self.roiData = np.zeros((2, 2))
        self.wavelengths = None
        self.trendData1 = None
        self.trendData2 = None
        self.scanData = np.array([])
        self.scanDataRoi = np.array([])
        self.timeData = np.array([])
        self.timeMarginal = np.array([])
        self.timeMarginalTmp = 0
        self.posData = np.array([])
        self.currentSample = 0
        self.avgData = 0
        self.targetPos = 0.0
        self.measureUpdateTimes = np.array([time.time()])
        self.lock = threading.Lock()

        self.running = False
        self.scanning = False
        self.moving = False
        self.moveStart = False
        self.scanTimer = QtCore.QTimer()
        self.scanTimer.timeout.connect(self.scanUpdateAction)

        self.frogCalc = FrogCalculation.FrogCalculation()

    #         splitterSizes = [self.settings.value('splitterSize0',200).toInt()[0], self.settings.value('splitterSize1',200).toInt()[0], self.settings.value('splitterSize2',200).toInt()[0]]
    #         root.debug(''.join(('New splitter sizes: ', str(splitterSizes))))
    #         self.plotSplitter.setSizes(splitterSizes)
    #         self.plotSplitter.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Expanding)
    #         self.plotSplitter.update()
    #         root.debug(''.join(('Splitter sizes: ', str(self.plotSplitter.sizes()))))

    def readImage(self, data):
        #         root.debug(''.join(('Image data type: ', str(type(data.value)))))
        #         root.debug(''.join(('Image size: ', str(type(data.value.shape)))))

        imData = np.transpose(data.value)
        self.spectrumImageWidget.setImage(imData, autoRange=False, autoLevels=True)
        self.roiData = self.ROI.getArrayRegion(imData, self.spectrumImageWidget.getImageItem())
        self.measureData()
        self.spectrumImageWidget.update()

    def readShutter(self, data):
        self.shutterLabel.setText(str(data.value))

    def writeShutter(self):
        w = self.shutterSpinbox.value()
        self.attributes['shutter'].attr_write(w)
        root.debug(''.join(('Write shutter ', str(w))))

    def readGain(self, data):
        self.gainLabel.setText(str(data.value))

    def writeGain(self):
        w = self.gainSpinbox.value()
        self.attributes['gain'].attr_write(w)
        root.debug(''.join(('Write gain ', str(w))))

    def readPosition(self, data):
        self.currentPosLabel.setText(QtCore.QString.number(data.value, 'f', 3))
        if np.abs(self.targetPos - data.value) < 0.001:
            self.moveStart = False
            self.moving = False

    def readSpeed(self, data):
        self.currentSpeedLabel.setText(QtCore.QString.number(data.value, 'f', 3))
        if self.moveStart == True:
            if data.value > 0.01:
                self.moving = True
                self.moveStart = False
        if self.moving == True:
            if np.abs(data.value) < 0.001:
                self.moving = False

    def writePosition(self):
        w = self.setPosSpinbox.value()
        self.attributes['position'].attr_write(w)

    def setAverage(self):
        self.avgSamples = self.averageSpinbox.value()

    def generateWavelengths(self):
        l0 = self.centerWavelengthSpinbox.value()
        dl = self.dispersionSpinbox.value()
        imSize = self.spectrumImageWidget.getImageItem().image.shape[0]
        #         root.debug(''.join(('Image size: ', str(imSize))))
        lrange = dl * imSize
        globalWavelengths = np.linspace(l0 - lrange / 2, l0 + lrange / 2, imSize)
        startInd = np.int(self.ROI.pos()[0])
        stopInd = np.int(self.ROI.pos()[0]) + np.ceil(self.ROI.size()[0])
        #         root.debug(''.join(('Wavelength range: ', str(globalWavelengths[startInd]),  '-', str(globalWavelengths[stopInd]))))
        self.wavelengths = globalWavelengths[startInd:stopInd]

        self.plot1.setData(x=self.wavelengths, y=self.spectrumData)

    def startScan(self):
        self.scanData = None
        self.trendData1 = None
        self.trendData2 = None
        self.timeMarginalTmp = 0
        self.timeData = np.array([])
        self.timeMarginal = np.array([])
        self.posData = np.array([])
        self.scanning = True
        self.moveStart = True
        self.running = True
        self.targetPos = self.startPosSpinbox.value()
        self.attributes['position'].attr_write(self.targetPos)

    #        self.scanTimer.start(100 * self.avgSamples)

    def stopScan(self):
        print 'Stopping scan'
        self.running = False
        self.scanning = False
        self.scanTimer.stop()

    def exportScan(self):
        print 'Exporting scan data'
        if self.scanData.max() > 256:
            data = np.uint8(np.double(self.scanData) / self.scanData.max() * 256)
        else:
            data = np.uint8(self.scanData)
        filename = ''.join(('frogtrace_', time.strftime('%Y-%m-%d_%Hh%M'), '_image.png'))
        im = Image.fromarray(data)
        im.save(filename)
        #         imsave(filename, data)
        data = self.timeData
        filename = ''.join(('frogtrace_', time.strftime('%Y-%m-%d_%Hh%M'), '_timevector.txt'))
        np.savetxt(filename, data)
        data = self.wavelengths
        filename = ''.join(('frogtrace_', time.strftime('%Y-%m-%d_%Hh%M'), '_wavelengthvector.txt'))
        np.savetxt(filename, data)

    def scanUpdateAction(self):
        self.scanTimer.stop()
        while self.running == True:
            time.sleep(0.1)
        newPos = self.targetPos + self.stepSizeSpinbox.value()
        print 'New pos: ', newPos
        self.attributes['position'].attr_write(newPos)
        self.targetPos = newPos
        self.running = True
        self.moveStart = True

    def measureScanData(self):
        self.avgData = self.trendData1 / self.avgSamples
        if self.scanData is None:
            self.scanData = np.array([self.avgData])
            self.timeMarginal = np.hstack((self.timeMarginal, self.timeMarginalTmp / self.avgSamples))
            self.timeMarginalTmp = 0.0
        else:
            self.scanData = np.vstack((self.scanData, self.avgData))
            self.timeMarginal = np.hstack((self.timeMarginal, self.timeMarginalTmp / self.avgSamples))
            self.timeMarginalTmp = 0.0
        pos = np.double(str(self.currentPosLabel.text()))
        newTime = (pos - self.startPosSpinbox.value()) * 2 * 1e-3 / 299792458.0
        self.timeData = np.hstack((self.timeData, newTime))
        self.posData = np.hstack((self.posData, pos * 1e-3))
        root.debug(''.join(('Time vector: ', str(self.timeData))))
        root.debug(''.join(('Time marginal: ', str(self.timeMarginal))))
        if self.timeUnitsRadio.isChecked() is True:
            self.frogImageWidget.setImage(np.transpose(self.scanData), autoRange=False, autoLevels=True)
            self.plot3.setData(x=self.timeData * 1e12, y=self.timeMarginal)
        else:
            self.frogImageWidget.setImage(np.transpose(self.scanData), autoRange=False, autoLevels=True)
            self.plot3.setData(x=self.posData * 1e3, y=self.timeMarginal)

    def measureData(self):
        roiDataFilt = medfilt2d(np.double(self.roiData), 5)
        self.spectrumData = np.sum(self.roiData, 1) / self.roiData.shape[1]
        if self.wavelengths is None:
            self.generateWavelengths()
        if self.wavelengths.shape[0] == self.spectrumData.shape[0]:
            self.plot1.setData(x=self.wavelengths, y=self.spectrumData)
            self.plot1.update()

        #         goodInd = np.arange(self.signalStartIndex.value(), self.signalEndIndex.value() + 1, 1)
        #         bkgInd = np.arange(self.backgroundStartIndex.value(), self.backgroundEndIndex.value() + 1, 1)
        #         bkg = self.waveform1[bkgInd].mean()
        #         bkgPump = self.waveform2[bkgInd].mean()
        #         autoCorr = (self.waveform1[goodInd] - bkg).sum()
        #         pump = (self.waveform2[goodInd] - bkgPump).sum()
        # #        pump = 1.0
        #         if self.normalizePumpCheck.isChecked() == True:
        #             try:
        #                 self.trendData1 = np.hstack((self.trendData1[1:], autoCorr / pump))
        #             except:
        #                 pass
        #         else:
        #             self.trendData1 = np.hstack((self.trendData1[1:], autoCorr))
        #         self.plot3.setData(y=self.trendData1)

        # Evaluate the fps
        t = time.time()
        if self.measureUpdateTimes.shape[0] > 10:
            self.measureUpdateTimes = np.hstack((self.measureUpdateTimes[1:], t))
        else:
            self.measureUpdateTimes = np.hstack((self.measureUpdateTimes, t))
        fps = 1 / np.diff(self.measureUpdateTimes).mean()
        self.fpsLabel.setText(QtCore.QString.number(fps, 'f', 1))

        # If we are running a scan, update the scan data
        if self.running == True:

            if self.moving == False and self.moveStart == False:
                if self.trendData1 is None:
                    self.trendData1 = self.spectrumData
                    self.timeMarginalTmp = self.spectrumData.sum()
                else:
                    self.trendData1 += self.spectrumData
                    self.timeMarginalTmp += self.spectrumData.sum()
                self.currentSample += 1
                if self.currentSample >= self.avgSamples:
                    self.running = False
                    self.measureScanData()
                    self.trendData1 = None
                    self.currentSample = 0
                    self.scanUpdateAction()

    def xAxisUnitsToggle(self):
        if self.timeUnitsRadio.isChecked() == True:
            self.plot3.setData(x=self.timeData * 1e12, y=self.timeMarginal)
        else:
            self.plot3.setData(x=self.posData * 1e3, y=self.timeMarginal)

    def updateFrogPlotView(self):
        self.frogResultViewbox.setGeometry(self.frogResultPlotitem.vb.sceneBoundingRect())
        self.frogResultViewbox.linkedViewChanged(self.frogResultPlotitem.vb, self.frogResultViewbox.XAxis)

    def updateFrogResultPlot(self):
        if self.frogEtCheckbox.isChecked() == True:
            self.frogResultPlotAbs.show()
        else:
            self.frogResultPlotAbs.hide()

        if self.frogPhaseCheckbox.isChecked() == True:
            self.frogResultPlotPhase.show()
        else:
            self.frogResultPlotPhase.hide()

    def calculatePulseParameters(self):
        t = self.frogCalc.get_t()
        Et = self.frogCalc.get_trace_abs()

        # Use first 10% of trace as background level
        bkg = Et[0:np.maximum(2, np.int(Et.shape[0] * 0.1))].mean() * 1.1
        EtN = np.maximum(0, Et - bkg) / (Et.max() - bkg)
        aboveInd = np.argwhere(EtN > 0.5)
        tFWHM = np.abs(t[aboveInd[-1, 0]] - t[aboveInd[0, 0]])
        s_t = '%.3f' % (tFWHM * 1e12)
        self.frogEtFWHMLabel.setText(''.join((s_t, ' ps')))

        t0 = np.trapz(EtN * t, t) / np.trapz(EtN, t)
        tRMS = np.sqrt(np.trapz(EtN * (t - t0) ** 2, t) / np.trapz(EtN, t))
        s_t = '%.3f' % (tRMS * 1e12)
        self.frogEtRMSLabel.setText(''.join((s_t, ' ps')))

    def updateFrogRoi(self):
        root.debug(''.join(('Roi pos: ', str(self.frogRoi.pos()))))
        root.debug(''.join(('Roi size: ', str(self.frogRoi.size()))))
        if self.scanData.size != 0:
            root.debug(''.join(('Scan data: ', str(self.scanData.shape))))

            bkg = self.scanData[0, :]
            bkgImg = self.scanData - np.tile(bkg, (self.scanData.shape[0], 1))
            roiImg = self.frogRoi.getArrayRegion(bkgImg, self.frogImageWidget.getImageItem(), axes=(1, 0))
            roiImg = roiImg / roiImg.max()

            root.debug('Slice complete')
            thr = self.frogThresholdSpinbox.value()
            kernel = self.frogKernelSpinbox.value()
            root.debug('Starting medfilt...')
            filteredImg = medfilt2d(roiImg, kernel) - thr
            #             filteredImg = roiImg - thr
            root.debug('Filtering complete')
            filteredImg[filteredImg < 0.0] = 0.0
            root.debug('Threshold complete')
            self.frogRoiImageWidget.setImage(filteredImg)
            root.debug('Set image complete')
            self.frogRoiImageWidget.autoRange()
            root.debug('Autorange complete')

    def startFrogInversion(self):
        data = np.uint8(self.scanDataRoi)
        N = self.frogNSpinbox.value()
        frogImg = self.frogRoiImageWidget.getImageItem().image
        if frogImg.size > 0:
            lStartInd = np.int(self.frogRoi.pos()[0])
            lStopInd = np.int(self.frogRoi.pos()[0] + np.ceil(self.frogRoi.size()[0]))
            root.debug(''.join(('Wavelength range: ', str(self.wavelengths[lStartInd]), ' - ',
                       str(self.wavelengths[lStopInd]))))
            l_start = self.wavelengths[lStartInd] * 1e-9
            l_stop = self.wavelengths[lStopInd] * 1e-9
            l0 = (l_stop + l_start) / 2

            tStartInd = np.int(self.frogRoi.pos()[1])
            tStopInd = np.int(self.frogRoi.pos()[1] + np.ceil(self.frogRoi.size()[1]))
            root.debug(''.join(('Time range: ', str(self.timeData[tStartInd]), ' - ', str(self.timeData[tStopInd]))))
            tau_start = self.timeData[tStartInd]
            tau_stop = self.timeData[tStopInd]
            if self.timeData.shape != 0:
                dt = self.frogDtSpinbox.value() * 1e-15
            else:
                dt = 1e-15

            root.debug('Wavelength input data: l_start=' + str(l_start) + ', l_stop=' + str(l_stop) +
                       ', type: ' + str(type(lStartInd)))
            root.debug('Time input data: tau_start=' + str(tau_start) + ', tau_stop=' + str(tau_stop) + ', dt=' +
                       str(dt))

            self.frogCalc.init_pulsefield_random(N, dt, l0)
            self.frogCalc.condition_frog_trace2(frogImg, l_start, l_stop, tau_start, tau_stop, N,
                                                thr=self.frogThresholdSpinbox.value())

            root.debug('frogImg shape: ' + str(frogImg.shape) + ', 10 values: ',
                       str(frogImg[0, 0:10]))
            root.debug('I_w_tau shape: ' + str(self.frogCalc.I_w_tau.shape) + ', 10 values: ' +
                       str(self.frogCalc.I_w_tau[0, 0:10]))
            self.frogCalcImageWidget.setImage(np.transpose(self.frogCalc.I_w_tau))
            self.frogCalcImageWidget.autoRange()
            self.frogCalcImageWidget.update()

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

            t = self.frogCalc.get_t()
            Et = self.frogCalc.get_trace_abs()
            Ephi = self.frogCalc.get_trace_phase(linear_comp=True)

            self.frogResultPlotAbs.setData(x=t, y=Et)
            self.frogResultPlotAbs.update()
            self.frogResultPlotPhase.setData(x=t, y=Ephi)
            self.frogResultPlotPhase.update()

            self.frogCalcResultImageWidget.setImage(np.transpose(np.abs(self.frogCalc.Esig_w_tau) ** 2))
            self.frogCalcResultImageWidget.autoRange()
            self.frogCalcResultImageWidget.update()

            self.calculatePulseParameters()

    def loadFrogTrace(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Select frog trace', '.', 'frogtrace_*.png'))
        root.debug(''.join(('File selected: ', str(filename))))

        fNameRoot = '_'.join((filename.split('_')[0:3]))
        tData = np.loadtxt(''.join((fNameRoot, '_timevector.txt')))
        tData = tData - tData.mean()
        lData = np.loadtxt(''.join((fNameRoot, '_wavelengthvector.txt')))
        pic = np.float32(imread(''.join((fNameRoot, '_image.png'))))

        root.debug(''.join(('Pic: ', str(pic.shape))))
        root.debug(''.join(('Time data: ', str(tData.shape), ' ', str(tData[0]))))
        root.debug(''.join(('l data: ', str(lData.shape), ' ', str(lData[0]))))

        self.scanData = pic
        self.timeData = tData
        self.wavelengths = lData
        self.centerWavelengthSpinbox.setValue(lData.mean())

        self.frogImageWidget.setImage(np.transpose(pic))
        self.frogImageWidget.autoLevels()
        self.frogImageWidget.autoRange()
        self.updateFrogRoi()

    def tabChanged(self, i):
        root.debug(''.join(('Tab changed: ', str(i))))
        root.debug(''.join(('Found ', str(self.plotLayout.count()), ' widgets')))
        # Remove all widgets:
        for widgetInd in range(self.plotLayout.count()):
            layItem = self.plotLayout.itemAt(0)
            if type(layItem) is QtGui.QVBoxLayout:
                root.debug(''.join(('Found ', str(layItem.count()), ' inner widgets')))
                #                 self.frogLayout.removeWidget(self.plotWidget2)
                #                 self.frogLayout.removeWidget(self.frogImageWidget)
                for wInd2 in range(layItem.count()):
                    layItem.takeAt(0)

            self.plotLayout.takeAt(0)
        self.plotLayout.removeWidget(self.plotWidget2)

        # Re-populate
        if i == 0 or i == 1:
            self.plotLayout.addWidget(self.spectrumImageWidget)
            self.plotLayout.addWidget(self.plotWidget)
            self.frogLayout.addWidget(self.frogImageWidget)
            #             self.invisibleLayout.addWidget(self.plotWidget2)
            #             self.invisibleLayout.addWidget(self.plotWidget)
            self.frogLayout.addWidget(self.plotWidget2)
            self.plotLayout.addLayout(self.frogLayout)
            self.plotWidget2.setVisible(True)
            self.plotWidget.setVisible(True)
            self.spectrumImageWidget.setVisible(True)
            self.frogErrorWidget.setVisible(False)
            self.frogResultWidget.setVisible(False)
            self.frogRoiImageWidget.setVisible(False)
            self.frogCalcImageWidget.setVisible(False)
            self.frogCalcResultImageWidget.setVisible(False)
            self.frogRoi.setVisible(False)
        elif i == 2:
            self.plotWidget2.setVisible(False)
            self.plotWidget.setVisible(False)
            self.spectrumImageWidget.setVisible(False)
            self.frogErrorWidget.setVisible(True)
            self.frogRoiImageWidget.setVisible(True)
            self.frogRoi.setVisible(True)
            self.frogResultWidget.setVisible(True)
            self.frogCalcImageWidget.setVisible(True)
            self.frogCalcResultImageWidget.setVisible(True)

            self.frogLayout.addWidget(self.frogImageWidget)
            self.frogLayout.addWidget(self.frogRoiImageWidget)
            self.frogImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
            self.plotLayout.addLayout(self.frogLayout)
            self.plotLayout.setStretchFactor(self.frogLayout, 3)
            #             self.plotLayout.addWidget(self.frogImageWidget)
            #             self.plotLayout.setStretchFactor(self.frogImageWidget, 2)
            #             self.plotLayout.addWidget(self.frogRoiImageWidget)
            #             self.plotLayout.setStretchFactor(self.frogRoiImageWidget, 2)
            #             self.plotLayout.addWidget(self.frogErrorWidget)
            #             self.plotLayout.setStretchFactor(self.frogErrorWidget, 1)
            self.frogLayout2.addWidget(self.frogCalcImageWidget)
            self.frogLayout2.addWidget(self.frogCalcResultImageWidget)
            self.plotLayout.addLayout(self.frogLayout2)
            self.plotLayout.setStretchFactor(self.frogLayout2, 2)
            self.plotLayout.addWidget(self.frogResultWidget)
            self.plotLayout.setStretchFactor(self.frogResultWidget, 3)

    def closeEvent(self, event):
        for a in self.attributes.itervalues():
            print 'Stopping', a.name
            a.stopRead()
        for a in self.attributes.itervalues():
            a.readThread.join()

        self.settings.setValue('startPos', self.startPosSpinbox.value())
        self.settings.setValue('setPos', self.setPosSpinbox.value())
        self.settings.setValue('averages', self.averageSpinbox.value())
        self.settings.setValue('step', self.stepSizeSpinbox.value())
        self.settings.setValue('startInd', self.signalStartIndex.value())
        self.settings.setValue('endInd', self.signalEndIndex.value())
        self.settings.setValue('bkgStartInd', self.backgroundStartIndex.value())
        self.settings.setValue('bkgEndInd', self.backgroundEndIndex.value())
        self.settings.setValue('xUnitTime', self.timeUnitsRadio.isChecked())
        self.settings.setValue('dispersion', self.dispersionSpinbox.value())
        self.settings.setValue('centerWavelength', self.centerWavelengthSpinbox.value())
        root.debug(''.join(('roiPos: ', str(self.ROI.pos()))))
        root.debug(''.join(('roiSize: ', str(self.ROI.size()))))
        self.settings.setValue('roiPosX', np.int(self.ROI.pos()[0]))
        self.settings.setValue('roiPosY', np.int(self.ROI.pos()[1]))
        self.settings.setValue('roiSizeW', np.int(self.ROI.size()[0]))
        self.settings.setValue('roiSizeH', np.int(self.ROI.size()[1]))
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
        self.settings.setValue('frogKernel', np.int(self.frogKernelSpinbox.value()))
        self.settings.setValue('frogRoiPosX', np.int(self.frogRoi.pos()[0]))
        self.settings.setValue('frogRoiPosY', np.int(self.frogRoi.pos()[1]))
        self.settings.setValue('frogRoiSizeW', np.int(self.frogRoi.size()[0]))
        self.settings.setValue('frogRoiSizeH', np.int(self.frogRoi.size()[1]))

        self.settings.sync()
        event.accept()

    def setupFrogLayout(self):
        self.frogNSpinbox = QtGui.QSpinBox()
        self.frogNSpinbox.setMinimum(0)
        self.frogNSpinbox.setMaximum(4096)
        #         self.frogNSpinbox.setValue(128)
        self.frogNSpinbox.setValue(self.settings.value('frogSize', 128).toInt()[0])
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
        self.frogDtSpinbox = QtGui.QDoubleSpinBox()
        self.frogDtSpinbox.setMinimum(0.0)
        self.frogDtSpinbox.setMaximum(10000.0)
        self.frogDtSpinbox.setSingleStep(1.0)
        self.frogDtSpinbox.setDecimals(1)
        self.frogDtSpinbox.setValue(self.settings.value('frogDt', 1.0).toDouble()[0])
        self.frogKernelSpinbox = QtGui.QSpinBox()
        self.frogKernelSpinbox.setMinimum(1)
        self.frogKernelSpinbox.setMaximum(99)
        self.frogKernelSpinbox.setSingleStep(2)
        self.frogKernelSpinbox.setValue(self.settings.value('frogKernel', 3).toInt()[0])
        self.frogKernelSpinbox.editingFinished.connect(self.updateFrogRoi)
        self.frogLoadButton = QtGui.QPushButton('Load')
        self.frogLoadButton.clicked.connect(self.loadFrogTrace)

        self.frogEtCheckbox = QtGui.QCheckBox()
        self.frogEtCheckbox.setChecked(True)
        self.frogEtCheckbox.stateChanged.connect(self.updateFrogResultPlot)
        self.frogPhaseCheckbox = QtGui.QCheckBox()
        self.frogPhaseCheckbox.setChecked(True)
        self.frogPhaseCheckbox.stateChanged.connect(self.updateFrogResultPlot)

        self.frogEtFWHMLabel = QtGui.QLabel()
        self.frogEtRMSLabel = QtGui.QLabel()

        self.frogGridLayout1 = QtGui.QGridLayout()
        self.frogGridLayout1.addWidget(QtGui.QLabel("Image threshold"), 0, 0)
        self.frogGridLayout1.addWidget(self.frogThresholdSpinbox, 0, 1)
        self.frogGridLayout1.addWidget(QtGui.QLabel("Median kernel"), 1, 0)
        self.frogGridLayout1.addWidget(self.frogKernelSpinbox, 1, 1)
        self.frogGridLayout1.addItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding),
                                     2, 0)

        self.frogGridLayout2 = QtGui.QGridLayout()
        self.frogGridLayout2.addWidget(QtGui.QLabel("Frog algorithm"), 0, 0)
        self.frogGridLayout2.addWidget(self.frogAlgoCombobox, 0, 1)
        self.frogGridLayout2.addWidget(QtGui.QLabel("Frog method"), 1, 0)
        self.frogGridLayout2.addWidget(self.frogMethodCombobox, 1, 1)
        self.frogGridLayout2.addWidget(QtGui.QLabel("Frog size (pow 2)"), 2, 0)
        self.frogGridLayout2.addWidget(self.frogNSpinbox, 2, 1)
        self.frogGridLayout2.addWidget(QtGui.QLabel("dt (fs)"), 3, 0)
        self.frogGridLayout2.addWidget(self.frogDtSpinbox, 3, 1)

        self.frogGridLayout3 = QtGui.QGridLayout()
        self.frogGridLayout3.addWidget(QtGui.QLabel("Load trace"), 0, 0)
        self.frogGridLayout3.addWidget(self.frogLoadButton, 0, 1)
        self.frogGridLayout3.addItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding),
                                     2, 0)
        self.frogGridLayout3.addWidget(QtGui.QLabel("Iterations"), 3, 0)
        self.frogGridLayout3.addWidget(self.frogIterationsSpinbox, 3, 1)
        self.frogGridLayout3.addWidget(QtGui.QLabel("Invert trace"), 4, 0)
        self.frogGridLayout3.addWidget(self.frogStartButton, 4, 1)
        self.frogGridLayout3.addWidget(self.frogContinueButton, 5, 1)

        self.frogGridLayout4 = QtGui.QGridLayout()
        self.frogGridLayout4.addWidget(QtGui.QLabel("Show Et"), 0, 0)
        self.frogGridLayout4.addWidget(self.frogEtCheckbox, 0, 1)
        self.frogGridLayout4.addWidget(QtGui.QLabel("Show phase"), 1, 0)
        self.frogGridLayout4.addWidget(self.frogPhaseCheckbox, 1, 1)
        self.frogGridLayout4.addItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum), 2,
                                     0)
        self.frogGridLayout4.addWidget(QtGui.QLabel("t_FWHM"), 3, 0)
        self.frogGridLayout4.addWidget(self.frogEtFWHMLabel, 3, 1)
        self.frogGridLayout4.addWidget(QtGui.QLabel("t_RMS"), 4, 0)
        self.frogGridLayout4.addWidget(self.frogEtRMSLabel, 4, 1)
        self.frogGridLayout4.addItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding),
                                     5, 0)

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
        self.frogImageWidget = pq.ImageView()
        self.frogImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.frogImageWidget.getView().setAspectLocked(False)
        self.frogRoi = pq.RectROI([0, 300], [600, 20], pen=(0, 9))
        self.frogRoi.sigRegionChangeFinished.connect(self.updateFrogRoi)
        self.frogRoi.blockSignals(True)

        #         roiPosX = 10
        #         roiPosY = 10
        #         roiSizeH = 20
        #         roiSizeW = 500
        roiPosX = self.settings.value('frogRoiPosX', 10).toInt()[0]
        roiPosY = self.settings.value('frogRoiPosY', 10).toInt()[0]
        root.debug(''.join(('roiPos: ', str(roiPosX))))
        roiSizeW = self.settings.value('frogRoiSizeW', 100).toInt()[0]
        roiSizeH = self.settings.value('frogRoiSizeH', 10).toInt()[0]

        self.frogRoi.setPos([roiPosX, roiPosY], update=True)
        self.frogRoi.setSize([roiSizeW, roiSizeH], update=True)
        self.frogImageWidget.getView().addItem(self.frogRoi)

        self.frogRoi.blockSignals(False)

        # Filtered frog trace roi 
        self.frogRoiImageWidget = pq.ImageView()
        self.frogRoiImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.frogRoiImageWidget.getView().setAspectLocked(False)

        self.frogCalcImageWidget = pq.ImageView()
        self.frogCalcImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.frogCalcImageWidget.getView().setAspectLocked(False)
        self.frogCalcResultImageWidget = pq.ImageView()
        self.frogCalcResultImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.frogCalcResultImageWidget.getView().setAspectLocked(False)
        self.frogCalcResultImageWidget.view.setXLink(self.frogCalcImageWidget.view)
        self.frogCalcResultImageWidget.view.setYLink(self.frogCalcImageWidget.view)

        self.frogLayout2 = QtGui.QVBoxLayout()

    def setupLayout(self):
        root.debug('Setting up layout')
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English))
        self.layout = QtGui.QVBoxLayout(self)
        self.tabWidget = QtGui.QTabWidget()
        self.tabWidget.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.gridLayout1 = QtGui.QGridLayout()
        self.gridLayout2 = QtGui.QGridLayout()
        self.gridLayout3 = QtGui.QGridLayout()
        self.gridLayout4 = QtGui.QGridLayout()
        self.gridLayout5 = QtGui.QGridLayout()

        self.fpsLabel = QtGui.QLabel()
        self.averageSpinbox = QtGui.QSpinBox()
        self.averageSpinbox.setMaximum(100)
        self.averageSpinbox.setValue(self.settings.value('averages', 5).toInt()[0])
        self.avgSamples = self.settings.value('averages', 5).toInt()[0]
        self.averageSpinbox.editingFinished.connect(self.setAverage)

        self.startPosSpinbox = QtGui.QDoubleSpinBox()
        self.startPosSpinbox.setDecimals(3)
        self.startPosSpinbox.setMaximum(2000000)
        self.startPosSpinbox.setMinimum(-2000000)
        self.startPosSpinbox.setValue(self.settings.value('startPos', 0.0).toDouble()[0])
        self.stepSizeSpinbox = QtGui.QDoubleSpinBox()
        self.stepSizeSpinbox.setDecimals(4)
        self.stepSizeSpinbox.setMaximum(2000000)
        self.stepSizeSpinbox.setMinimum(-2000000)
        self.stepSizeSpinbox.setValue(self.settings.value('step', 0.05).toDouble()[0])
        self.setPosSpinbox = QtGui.QDoubleSpinBox()
        self.setPosSpinbox.setDecimals(3)
        self.setPosSpinbox.setMaximum(2000000)
        self.setPosSpinbox.setMinimum(-2000000)
        self.setPosSpinbox.setValue(63.7)
        self.setPosSpinbox.setValue(self.settings.value('setPos', 63).toDouble()[0])
        self.setPosSpinbox.editingFinished.connect(self.writePosition)
        self.currentPosLabel = QtGui.QLabel()
        f = self.currentPosLabel.font()
        f.setPointSize(28)
        currentPosTextLabel = QtGui.QLabel('Current pos ')
        currentPosTextLabel.setFont(f)
        self.currentPosLabel.setFont(f)
        self.currentSpeedLabel = QtGui.QLabel()
        #        self.currentSpeedLabel.setFont(f)

        self.signalStartIndex = QtGui.QSpinBox()
        self.signalStartIndex.setMinimum(0)
        self.signalStartIndex.setMaximum(16384)
        self.signalStartIndex.setValue(self.settings.value('startInd', 1050).toInt()[0])
        self.signalEndIndex = QtGui.QSpinBox()
        self.signalEndIndex.setMinimum(0)
        self.signalEndIndex.setMaximum(16384)
        self.signalEndIndex.setValue(self.settings.value('endInd', 1150).toInt()[0])
        self.backgroundStartIndex = QtGui.QSpinBox()
        self.backgroundStartIndex.setMinimum(0)
        self.backgroundStartIndex.setMaximum(16384)
        self.backgroundStartIndex.setValue(self.settings.value('bkgStartInd', 900).toInt()[0])
        self.backgroundEndIndex = QtGui.QSpinBox()
        self.backgroundEndIndex.setMinimum(0)
        self.backgroundEndIndex.setMaximum(16384)
        self.backgroundEndIndex.setValue(self.settings.value('bkgEndInd', 1000).toInt()[0])

        self.centerWavelengthSpinbox = QtGui.QDoubleSpinBox()
        self.centerWavelengthSpinbox.setMinimum(200)
        self.centerWavelengthSpinbox.setMaximum(2000)
        self.centerWavelengthSpinbox.setValue(self.settings.value('centerWavelength', 400).toDouble()[0])
        self.centerWavelengthSpinbox.editingFinished.connect(self.generateWavelengths)

        self.dispersionSpinbox = QtGui.QDoubleSpinBox()
        self.dispersionSpinbox.setMinimum(0)
        self.dispersionSpinbox.setMaximum(10)
        self.dispersionSpinbox.setDecimals(4)
        self.dispersionSpinbox.setValue(self.settings.value('dispersion', 0.03).toDouble()[0])
        self.dispersionSpinbox.editingFinished.connect(self.generateWavelengths)

        self.shutterLabel = QtGui.QLabel()
        self.shutterSpinbox = QtGui.QDoubleSpinBox()
        self.shutterSpinbox.setMinimum(0)
        self.shutterSpinbox.setMaximum(1000000)
        self.shutterSpinbox.editingFinished.connect(self.writeShutter)

        self.gainLabel = QtGui.QLabel()
        self.gainSpinbox = QtGui.QDoubleSpinBox()
        self.gainSpinbox.setMinimum(0)
        self.gainSpinbox.setMaximum(100000)
        self.gainSpinbox.editingFinished.connect(self.writeGain)

        self.normalizePumpCheck = QtGui.QCheckBox('Normalize')
        self.timeUnitsRadio = QtGui.QRadioButton('ps')
        self.posUnitsRadio = QtGui.QRadioButton('mm')
        if self.settings.value('xUnitTime', True).toBool() == True:
            self.timeUnitsRadio.setChecked(True)
        else:
            self.posUnitsRadio.setChecked(True)
        self.timeUnitsRadio.toggled.connect(self.xAxisUnitsToggle)

        self.startButton = QtGui.QPushButton('Start')
        self.startButton.clicked.connect(self.startScan)
        self.stopButton = QtGui.QPushButton('Stop')
        self.stopButton.clicked.connect(self.stopScan)
        self.exportButton = QtGui.QPushButton('Export')
        self.exportButton.clicked.connect(self.exportScan)

        self.gridLayout1.addWidget(QtGui.QLabel("Start position"), 0, 0)
        self.gridLayout1.addWidget(self.startPosSpinbox, 0, 1)
        self.gridLayout1.addWidget(QtGui.QLabel("Step size"), 1, 0)
        self.gridLayout1.addWidget(self.stepSizeSpinbox, 1, 1)
        self.gridLayout1.addWidget(QtGui.QLabel("Averages"), 2, 0)
        self.gridLayout1.addWidget(self.averageSpinbox, 2, 1)
        self.gridLayout1.addWidget(QtGui.QLabel("Normalize"), 3, 0)
        self.gridLayout1.addWidget(self.normalizePumpCheck, 3, 1)
        self.gridLayout2.addWidget(QtGui.QLabel("Set position"), 0, 0)
        self.gridLayout2.addWidget(self.setPosSpinbox, 0, 1)
        self.gridLayout2.addWidget(QtGui.QLabel("Start scan"), 1, 0)
        self.gridLayout2.addWidget(self.startButton, 1, 1)
        self.gridLayout2.addWidget(QtGui.QLabel("Stop scan"), 2, 0)
        self.gridLayout2.addWidget(self.stopButton, 2, 1)
        self.gridLayout2.addWidget(QtGui.QLabel("Export scan"), 3, 0)
        self.gridLayout2.addWidget(self.exportButton, 3, 1)
        self.gridLayout3.addWidget(currentPosTextLabel, 0, 0)
        self.gridLayout3.addWidget(self.currentPosLabel, 0, 1)
        self.gridLayout3.addWidget(QtGui.QLabel("Current speed"), 1, 0)
        self.gridLayout3.addWidget(self.currentSpeedLabel, 1, 1)

        self.gridLayout4.addWidget(QtGui.QLabel("Signal start index"), 0, 0)
        self.gridLayout4.addWidget(self.signalStartIndex, 0, 1)
        self.gridLayout4.addWidget(QtGui.QLabel("Signal end index"), 1, 0)
        self.gridLayout4.addWidget(self.signalEndIndex, 1, 1)
        self.gridLayout4.addWidget(QtGui.QLabel("Background start index"), 2, 0)
        self.gridLayout4.addWidget(self.backgroundStartIndex, 2, 1)
        self.gridLayout4.addWidget(QtGui.QLabel("Background end index"), 3, 0)
        self.gridLayout4.addWidget(self.backgroundEndIndex, 3, 1)

        self.gridLayout5.addWidget(QtGui.QLabel("X-axis units"), 0, 0)
        self.gridLayout5.addWidget(self.timeUnitsRadio, 0, 1)
        self.gridLayout5.addWidget(self.posUnitsRadio, 1, 1)
        self.gridLayout5.addWidget(QtGui.QLabel("FPS"), 2, 0)
        self.gridLayout5.addWidget(self.fpsLabel, 2, 1)

        root.debug('Plot widgets')

        self.spectrumImageWidget = pq.ImageView()
        gradEditor = pq.GradientEditorItem()
        gradEditor.loadPreset('flame')
        self.spectrumImageWidget.ui.histogram.setColorMap = gradEditor.colorMap()
        self.ROI = pq.RectROI([0, 300], [600, 20], pen=(0, 9))
        self.ROI.sigRegionChanged.connect(self.generateWavelengths)
        self.ROI.blockSignals(True)

        roiPosX = self.settings.value('roiPosX', 0).toInt()[0]
        roiPosY = self.settings.value('roiPosY', 0).toInt()[0]
        root.debug(''.join(('roiPos: ', str(roiPosX))))
        roiSizeW = self.settings.value('roiSizeW', 0).toInt()[0]
        roiSizeH = self.settings.value('roiSizeH', 0).toInt()[0]
        root.debug(''.join(('roiSize: ', str(roiSizeW), 'x', str(roiSizeH))))
        self.ROI.setPos([roiPosX, roiPosY], update=True)
        self.ROI.setSize([roiSizeW, roiSizeH], update=True)
        self.spectrumImageWidget.getView().addItem(self.ROI)
        self.spectrumImageWidget.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.spectrumImageWidget.getView().setAspectLocked(False)
        self.ROI.blockSignals(False)

        self.plotWidget = pq.PlotWidget(useOpenGL=True)
        self.plot1 = self.plotWidget.plot()
        self.plot1.setPen((200, 25, 10))
        self.plot2 = self.plotWidget.plot()
        self.plot2.setPen((10, 200, 25))
        self.plot1.antialiasing = True
        self.plotWidget.setAntialiasing(True)
        self.plotWidget.showGrid(True, True)
        self.plotWidget.setMinimumWidth(200)
        self.plotWidget.setMaximumWidth(500)
        self.plotWidget.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)

        self.plotWidget2 = pq.PlotWidget(useOpenGL=True)
        self.plot3 = self.plotWidget2.plot()
        self.plot3.setPen((50, 99, 200))
        self.plotWidget2.setAntialiasing(True)
        self.plotWidget2.showGrid(True, True)
        self.plotWidget2.setMaximumHeight(200)
        self.plotWidget2.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Maximum)

        self.plotLayout = QtGui.QHBoxLayout()
        self.plotLayout.addWidget(self.spectrumImageWidget)
        self.plotLayout.addWidget(self.plotWidget)

        scanLay = QtGui.QHBoxLayout()
        scanLay.addLayout(self.gridLayout1)
        scanLay.addSpacerItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum))
        scanLay.addLayout(self.gridLayout2)
        scanLay.addSpacerItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum))
        scanLay.addLayout(self.gridLayout4)
        scanLay.addSpacerItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum))
        scanLay.addLayout(self.gridLayout3)
        scanLay.addSpacerItem(QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum))
        scanLay.addLayout(self.gridLayout5)
        scanLay.addSpacerItem(QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum))

        cameraLay = QtGui.QHBoxLayout()
        self.camGridLayout1 = QtGui.QGridLayout()
        self.camGridLayout1.addWidget(QtGui.QLabel("Center wavelength"), 0, 0)
        self.camGridLayout1.addWidget(self.centerWavelengthSpinbox, 0, 1)
        self.camGridLayout1.addWidget(QtGui.QLabel("Dispersion"), 1, 0)
        self.camGridLayout1.addWidget(self.dispersionSpinbox, 1, 1)
        self.camGridLayout1.addWidget(QtGui.QLabel("Shutter time"), 2, 0)
        self.camGridLayout1.addWidget(self.shutterSpinbox, 2, 1)
        self.camGridLayout1.addWidget(self.shutterLabel, 2, 2)
        self.camGridLayout1.addWidget(QtGui.QLabel("Gain"), 3, 0)
        self.camGridLayout1.addWidget(self.gainSpinbox, 3, 1)
        self.camGridLayout1.addWidget(self.gainLabel, 3, 2)
        cameraLay.addLayout(self.camGridLayout1)
        cameraLay.addSpacerItem(
            QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum))

        self.setupFrogLayout()

        self.frogLayout = QtGui.QVBoxLayout()
        self.frogLayout.addWidget(self.frogImageWidget)
        self.frogLayout.addWidget(self.plotWidget2)
        self.plotLayout.addLayout(self.frogLayout)

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
        frogLay.addSpacerItem(QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum))

        self.tabCameraWidget = QtGui.QWidget()
        self.tabCameraWidget.setLayout(cameraLay)
        self.tabScanWidget = QtGui.QWidget()
        self.tabScanWidget.setLayout(scanLay)
        self.tabFrogWidget = QtGui.QWidget()
        self.tabFrogWidget.setLayout(frogLay)
        self.tabWidget.addTab(self.tabScanWidget, 'Scan')
        self.tabWidget.addTab(self.tabCameraWidget, 'Camera')
        self.tabWidget.addTab(self.tabFrogWidget, 'Frog')
        self.tabWidget.currentChanged.connect(self.tabChanged)
        self.layout.addWidget(self.tabWidget)
        #         self.layout.addLayout(scanLay)
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

        self.tabChanged(0)

        self.show()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    myapp = TangoDeviceClient(cameraName, motorName)
    myapp.show()
    sys.exit(app.exec_())
