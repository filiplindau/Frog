'''
Created on 7 Mar 2016

@author: Filip Lindau
'''
# -*- coding:utf-8 -*-
"""
Created on Oct 1, 2015

@author: Filip Lindau
"""
cameraName = 'b-v0-gunlaser-csdb-0:10000/gunlaser/cameras/cam0'
motorName = 'b-v0-gunlaser-csdb-0:10000/gunlaser/motors/zaber01'

from PyQt4 import QtGui, QtCore

from scipy.signal import medfilt2d
from matplotlib.pyplot import imsave

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
root.setLevel(logging.DEBUG)

from AttributeReadThreadClass import AttributeClass
import pyqtgraph as pq
import PyTango as pt
import threading
import numpy as np

class TangoDeviceClient(QtGui.QWidget):
    def __init__(self, redpitayaName, motorName, parent=None):
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
        self.attributes['image'] = AttributeClass('image', self.devices['camera'], 0.2)
        self.attributes['shutter'] = AttributeClass('shutter', self.devices['camera'], 0.5)
        self.attributes['gain'] = AttributeClass('gain', self.devices['camera'], 0.5)
        self.attributes['position'] = AttributeClass('position', self.devices['motor'], 0.05)
        self.attributes['speed'] = AttributeClass('speed', self.devices['motor'], 0.05)
        #self.attributes['triggerdelay'] = AttributeClass('triggerdelay', self.devices['redpitaya'], 0.3)
        #self.attributes['triggermode'] = AttributeClass('triggermode', self.devices['redpitaya'], 1.0)
        #self.attributes['triggersource'] = AttributeClass('triggersource', self.devices['redpitaya'], 1.0)
        #self.attributes['recordlength'] = AttributeClass('recordlength', self.devices['redpitaya'], 0.3)
        #self.attributes['samplerate'] = AttributeClass('samplerate', self.devices['redpitaya'], 0.3)


        self.attributes['image'].attrSignal.connect(self.readImage)
        self.attributes['shutter'].attrSignal.connect(self.readShutter)
        self.attributes['gain'].attrSignal.connect(self.readGain)
        self.attributes['position'].attrSignal.connect(self.readPosition)
        self.attributes['speed'].attrSignal.connect(self.readSpeed)
        #self.attributes['triggerdelay'].attrSignal.connect(self.readTrigDelay)
        #self.attributes['triggermode'].attrSignal.connect(self.readTrigMode)
        #self.attributes['triggersource'].attrSignal.connect(self.readTrigSource)
        #self.attributes['recordlength'].attrSignal.connect(self.readRecordLength)
        #self.attributes['samplerate'].attrSignal.connect(self.readSampleRate)


        self.roiData = np.zeros((2,2))
        self.wavelengths = None
        self.trendData1 = None
        self.trendData2 = np.zeros(600)
        self.scanData = np.array([])
        self.timeData = np.array([])
        self.posData = np.array([])
        self.avgSamples = 5
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

        self.settings = QtCore.QSettings('Maxlab', 'Frog')
        



    def readImage(self, data):
#         root.debug(''.join(('Image data type: ', str(type(data.value)))))
#         root.debug(''.join(('Image size: ', str(type(data.value.shape)))))
        
        imData = np.transpose(data.value)
        self.spectrumImageWidget.setImage(imData, autoRange=False, autoLevels=False)
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
        if np.abs(self.targetPos - data.value) < 0.01:
            self.moveStart = False
            self.moving = False

    def readSpeed(self, data):
        self.currentSpeedLabel.setText(QtCore.QString.number(data.value, 'f', 3))
        if self.moveStart == True:
            if data.value > 0.01:
                self.moving = True
                self.moveStart = False
        if self.moving == True:
            if np.abs(data.value) < 0.01:
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
        lrange = dl*imSize
        globalWavelengths = np.linspace(l0-lrange/2, l0+lrange/2, imSize)
        startInd = np.int(self.ROI.pos()[0])
        stopInd = np.int(self.ROI.pos()[0])+np.ceil(self.ROI.size()[0])
#         root.debug(''.join(('Wavelength range: ', str(globalWavelengths[startInd]),  '-', str(globalWavelengths[stopInd])))) 
        self.wavelengths = globalWavelengths[startInd:stopInd]
        
        self.plot1.setData(x=self.wavelengths, y=self.spectrumData)

    def startScan(self):
        self.scanData = None
        self.trendData1 = None
        self.timeData = np.array([])
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
        data = self.scanData
        filename = ''.join(('frogimage_', time.strftime('%Y-%m-%d_%Hh%M'), '.png'))
        imsave(filename, data)
        data = self.timeData
        filename = ''.join(('frogtime_', time.strftime('%Y-%m-%d_%Hh%M'), '.txt'))
        np.savetxt(filename, data)
        data = self.wavelengths
        filename = ''.join(('frogwavlengths_', time.strftime('%Y-%m-%d_%Hh%M'), '.txt'))
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
        self.avgData = self.trendData1/self.avgSamples
        if self.scanData is None:
            self.scanData = np.array([self.avgData])
        else:
            self.scanData = np.vstack((self.scanData, self.avgData))
        pos = np.double(str(self.currentPosLabel.text()))
        newTime = (pos - self.startPosSpinbox.value()) * 2 * 1e-3 / 299792458.0
        self.timeData = np.hstack((self.timeData, newTime))
        self.posData = np.hstack((self.posData, pos * 1e-3))
        if self.timeUnitsRadio.isChecked() == True:
            self.frogImageWidget.setImage(self.scanData, autoRange=False, autoLevels=False)
#             self.plot5.setData(x=self.timeData * 1e12, y=self.scanData)
        else:
            self.frogImageWidget.setImage(self.scanData, autoRange=False, autoLevels=False)
#             self.plot5.setData(x=self.posData * 1e3, y=self.scanData)

    def measureData(self):
        roiDataFilt = medfilt2d(self.roiData, 5)
        self.spectrumData = np.sum(self.roiData, 1) / self.roiData.shape[1]
        if self.wavelengths is None:
            self.generateWavelengths()
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
            if self.trendData1 is None:
                self.trendData1 = self.spectrumData
            else:
                self.trendData1 += self.spectrumData
            if self.moving == False and self.moveStart == False:
                self.currentSample += 1
                if self.currentSample >= self.avgSamples:
                    self.running = False
                    self.measureScanData()
                    self.trendData1 = None
                    self.currentSample = 0
                    self.scanUpdateAction()

    def xAxisUnitsToggle(self):
        if self.timeUnitsRadio.isChecked() == True:
            self.plot5.setData(x=self.timeData * 1e12, y=self.scanData)
        else:
            self.plot5.setData(x=self.posData * 1e3, y=self.scanData)

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
        self.settings.sync()
        event.accept()

    def setupLayout(self):
        root.debug('Setting up layout')
        self.setLocale(QtCore.QLocale(QtCore.QLocale.English))
        self.layout = QtGui.QVBoxLayout(self)
        self.tabWidget = QtGui.QTabWidget()
        self.gridLayout1 = QtGui.QGridLayout()
        self.gridLayout2 = QtGui.QGridLayout()
        self.gridLayout3 = QtGui.QGridLayout()
        self.gridLayout4 = QtGui.QGridLayout()
        self.gridLayout5 = QtGui.QGridLayout()

        self.fpsLabel = QtGui.QLabel()
        self.averageSpinbox = QtGui.QSpinBox()
        self.averageSpinbox.setMaximum(100)
        self.averageSpinbox.setValue(self.settings.value('averages', 5).toInt()[0])
        self.averageSpinbox.editingFinished.connect(self.setAverage)

        self.startPosSpinbox = QtGui.QDoubleSpinBox()
        self.startPosSpinbox.setDecimals(3)
        self.startPosSpinbox.setMaximum(2000000)
        self.startPosSpinbox.setMinimum(-2000000)
        self.startPosSpinbox.setValue(self.settings.value('startPos', 0.0).toDouble()[0])
        self.stepSizeSpinbox = QtGui.QDoubleSpinBox()
        self.stepSizeSpinbox.setDecimals(3)
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
        self.centerWavelengthSpinbox.setMaximum(1000)
        self.centerWavelengthSpinbox.setValue(self.settings.value('centerWavelength', 400).toDouble()[0])
        self.centerWavelengthSpinbox.editingFinished.connect(self.generateWavelengths)

        self.dispersionSpinbox = QtGui.QDoubleSpinBox()
        self.dispersionSpinbox.setMinimum(0)
        self.dispersionSpinbox.setMaximum(10)
        self.dispersionSpinbox.setValue(self.settings.value('dispersion', 0.03).toDouble()[0])
        self.dispersionSpinbox.editingFinished.connect(self.generateWavelengths)
        
        self.shutterLabel = QtGui.QLabel()
        self.shutterSpinbox = QtGui.QDoubleSpinBox()
        self.shutterSpinbox.setMinimum(0)
        self.shutterSpinbox.setMaximum(1000)
        self.shutterSpinbox.editingFinished.connect(self.writeShutter)

        self.gainLabel= QtGui.QLabel()
        self.gainSpinbox = QtGui.QDoubleSpinBox()
        self.gainSpinbox.setMinimum(0)
        self.gainSpinbox.setMaximum(100)
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
        self.ROI = pq.RectROI([0, 300], [600, 20], pen=(0,9))
        self.ROI.sigRegionChanged.connect(self.generateWavelengths)
        self.ROI.blockSignals(True)
        
        roiPosX = self.settings.value('roiPosX', 0).toInt()[0]
        roiPosY = self.settings.value('roiPosY', 0).toInt()[0]
        root.debug(''.join(('roiPos: ', str(roiPosX))))
        roiSizeW = self.settings.value('roiSizeW', 0).toInt()[0]
        roiSizeH = self.settings.value('roiSizeH', 0).toInt()[0]
        root.debug(''.join(('roiSize: ', str(roiSizeW), 'x', str(roiSizeH))))
        self.ROI.setPos([roiPosX, roiPosY], update = True)
        self.ROI.setSize([roiSizeW, roiSizeH], update = True)
        self.spectrumImageWidget.getView().addItem(self.ROI)
        self.spectrumImageWidget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.ROI.blockSignals(False)

        self.frogImageWidget = pq.ImageView()
        self.frogImageWidget.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.frogImageWidget.getView().setAspectLocked(False)

        self.plotWidget = pq.PlotWidget(useOpenGL=True)
        self.plot1 = self.plotWidget.plot()
        self.plot1.setPen((200, 25, 10))
        self.plot2 = self.plotWidget.plot()
        self.plot2.setPen((10, 200, 25))
        self.plot1.antialiasing = True
        self.plotWidget.setAntialiasing(True)
        self.plotWidget.showGrid(True, True)

#         self.plotWidget2 = pq.PlotWidget(useOpenGL=True)
#         self.plot3 = self.plotWidget2.plot()
#         self.plot3.setPen((50, 99, 200))
#         self.plot4 = self.plotWidget2.plot()
#         self.plot4.setPen((10, 200, 25))
#         self.plot3.antialiasing = True
#         self.plotWidget2.setAntialiasing(True)
#         self.plotWidget2.showGrid(True, True)
# 
#         self.plotWidget3 = pq.PlotWidget(useOpenGL=True)
#         self.plot5 = self.plotWidget3.plot()
#         self.plot5.setPen((10, 200, 70))
#         self.plotWidget3.setAntialiasing(True)
#         self.plotWidget3.showGrid(True, True)

        plotLayout = QtGui.QHBoxLayout()
        plotLayout.addWidget(self.spectrumImageWidget)
        plotLayout.addWidget(self.plotWidget)
        plotLayout.addWidget(self.frogImageWidget)

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
        cameraLay.addSpacerItem(QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Minimum))
        
        frogLay = QtGui.QHBoxLayout()
        
        self.tabCameraWidget = QtGui.QWidget()
        self.tabCameraWidget.setLayout(cameraLay)
        self.tabScanWidget = QtGui.QWidget()
        self.tabScanWidget.setLayout(scanLay)
        self.tabFrogWidget = QtGui.QWidget()
        self.tabFrogWidget.setLayout(frogLay)
        self.tabWidget.addTab(self.tabScanWidget, 'Scan')
        self.tabWidget.addTab(self.tabCameraWidget, 'Camera')        
        self.tabWidget.addTab(self.tabFrogWidget, 'Frog')
        self.layout.addWidget(self.tabWidget)
#         self.layout.addLayout(scanLay)
        self.layout.addLayout(plotLayout)
        
        windowPosX = self.settings.value('windowPosX', 100).toInt()[0]
        windowPosY = self.settings.value('windowPosY', 100).toInt()[0]
        windowSizeW = self.settings.value('windowSizeW', 800).toInt()[0]
        windowSizeH = self.settings.value('windowSizeH', 300).toInt()[0]
        self.setGeometry(windowPosX, windowPosY, windowSizeW, windowSizeH)

        self.update()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    myapp = TangoDeviceClient(cameraName, motorName)
    myapp.show()
    sys.exit(app.exec_())


