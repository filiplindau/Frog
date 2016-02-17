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
        self.initApplyIntensityData()
        self.updateEtVanilla()
        
    def initEsigTau(self):        
        Source = """
        #include <pyopencl-complex.h>
        __kernel void generateEsig_t_tau_SHG(__global cfloat_t* Et, __global cfloat_t* Esig, int N){
            const int tau_i = get_global_id(0);
            const int t_i = get_global_id(1);
            cfloat_t Etmp = (cfloat_t)(0.0f, 0.0f);
//            int ind = (t_i+(tau_i-N/2));
            int ind = (t_i-(tau_i-N/2));
            
            if (ind >= 0 && ind < N)
            {
                Esig[t_i + N*tau_i] = cfloat_mul(Et[t_i], Et[ind]) ;
            }
            else
            {
                Esig[t_i + N*tau_i] = (cfloat_t)(0.0f, 0.0f);
            }
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['generateEsig_t_tau_SHG'] = prg
        
    def initApplyIntensityData(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void applyIntensityData(__global cfloat_t* Esig_w_tau, __global float* I_w_tau, int N){
            const int tau_i = get_global_id(0);
            const int t_i = get_global_id(1);
            
            float Esig_mag = cfloat_abs(Esig_w_tau[t_i + N*tau_i]);
            Esig_w_tau[t_i + N*tau_i] = cfloat_rmul(sqrt(I_w_tau[t_i + N*tau_i])/Esig_mag, Esig_w_tau[t_i + N*tau_i]);
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['applyIntensityData'] = prg
        
    def updateEtVanilla(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void updateEtVanillaSum(__global cfloat_t* Esig_t_tau_p, __global cfloat_t* Et, int N){
            const int t_i = get_global_id(0);
            
            cfloat_t sum = (cfloat_t)(0.0f, 0.0f);
            int i;
            for(i=0; i<N; i++) {
                sum = cfloat_add(sum, Esig_t_tau_p[t_i + N*i]);
            }
            Et[t_i] = sum;
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['updateEtVanillaSum'] = prg
        
        Source = """
        #include <pyopencl-complex.h>
        __kernel void updateEtVanillaNorm(__global cfloat_t* Et, int N){
            float m = 0.0f;
            float a = 0.0f;
            int i;
            for(i=0; i<N; i++) 
            {
                a = cfloat_abs(Et[i]);
                if (a > m) 
                {
                    m = a;
                }                
            }
            for(i=0; i<N; i++) 
            {
                Et[i] = cfloat_divider(Et[i], m);
            }
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['updateEtVanillaNorm'] = prg
