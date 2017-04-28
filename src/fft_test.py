"""
Created on 24 Apr 2017

To test the open_cl fft routine clFFT

@author: Filip Lindau
"""

import numpy as np
import time
import sys
if sys.version_info > (3, 0):
    from importlib import reload

import pyopencl as cl
import pyopencl.array as cla
from gpyfft.fft import FFT

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

root.debug('Initializing opencl')
pl = cl.get_platforms()
d = None
v = None
root.debug(''.join(('Found ', str(pl.__len__()), ' platforms')))
vendorDict = {'amd': 3, 'nvidia': 2, 'intel': 1}
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
root.debug(''.join(('Using device ', str(d), ' from ', v)))
ctx = cl.Context(devices=d)
q = cl.CommandQueue(ctx)

dtype_c = np.complex64
N = 256
t = np.arange(N)-N/2
tau = 10.0
t_shift = 0
Et = (np.exp(-(t-t_shift)**2/tau**2)).astype(dtype_c)
Efft = np.zeros(N, dtype=dtype_c)
Et_cla = cla.to_device(q, np.fft.fftshift(Et))
Efft_cla = cla.to_device(q, Efft)

E_fft_transform = FFT(ctx, q, (Et_cla,), (Efft_cla,), axes=[0])
events = E_fft_transform.enqueue(forward=True)
for ev in events:
    ev.wait()

Efft = np.fft.fft(np.fft.fftshift(Et))
Efft = np.fft.fft(Et)

Et2 = np.fft.ifft(np.roll(Efft, 50))

