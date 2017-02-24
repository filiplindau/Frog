"""
Created on 19 Apr 2016

@author: Filip
"""

import numpy as np

def femtosoft_generator(image, timestep, wavelengthstep, centerwavelength, filename):
    n_t = image.shape[0]
    n_l = image.shape[1]
    header = ''.join((str(n_t), ' ', str(n_l), ' ', str(timestep*1e15), ' ', str(wavelengthstep*1e9), ' ', str(centerwavelength*1e9)))
    np.savetxt(filename, image, fmt='%i', header=header, comments='')
