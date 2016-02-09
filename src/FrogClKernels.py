'''
Created on 9 Feb 2016

@author: Filip Lindau
'''
import numpy as np
import pyopencl as cl

class FrogClKernels(object):
    def __init__(self, ctx):
        self.progs = {}
        self.ctx = ctx
        
        self.initEsigTau()
        
    def initEsigTau(self):        
        Source = """
        #include <pyopencl-complex.h>
        __kernel void generateEsig_t_tau_SHG(__global cfloat_t *Et, __global cfloat_t *Esig, __global int N){
        const int tau_i = get_global_id(0);
        const int t_i = get_global_id(1);
        cfloat_t Etmp = (cfloat_t)(0.0f, 0.0f);
        int ind = (t_i-(tau_i-N/2));
        
        if (ind >= 0 && ind < N)
        {
            Esig[tau_i, t_i] = cfloat_mul(Et[t_i], Et[ind]) ;
        }
        else
        {
            Esig[tau_i, t_i] = 0.0;
        }
        
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['generateEsig_t_tau'] = prg