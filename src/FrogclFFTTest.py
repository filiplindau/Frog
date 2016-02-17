'''
Created on 15 Feb 2016

@author: Filip Lindau
Testing the gpyfft library, a wrapper for clFFT
'''
import time
import numpy as np
from numpy.fft import fftn as npfftn
from numpy.testing import assert_array_almost_equal
import pyopencl as cl
import pyopencl.array as cla
from gpyfft.fft import FFT
from gpyfft.gpyfftlib import GpyFFT_Error

pl = cl.get_platforms()
d = pl[1].get_devices()
context = cl.Context(devices = d)
#context = cl.create_some_context()
queue = cl.CommandQueue(context)


dataRe = np.random.rand(512,512)
dataIm = np.random.rand(512,512)
nd_dataC = (dataRe + 1j*dataIm).astype(np.complex64)
#nd_dataC = np.random.rand((1024, 1024), dtype = np.complex64)
dataC = cla.to_device(queue, nd_dataC)
nd_result = np.zeros_like(nd_dataC, dtype = np.complex64)
resultC = cla.to_device(queue, nd_result)
transform = FFT(context, queue, (dataC,), (resultC,), axes = [1])
tic = time.clock()
events = transform.enqueue()
for e in events:
    e.wait()
toc = time.clock()
clTime = toc-tic
print 'clTime: ', clTime
tic = time.clock()
resultCl = resultC.get()
toc = time.clock()
print "transfer time: ", toc-tic

ticNp = time.clock()
resultNp = np.fft.fft(nd_dataC, axis=1).astype(np.complex64)
tocNp = time.clock()
npTime = tocNp - ticNp
print 'npTime: ', npTime