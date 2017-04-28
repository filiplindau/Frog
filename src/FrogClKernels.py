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
        
        self.initEsigTau()      # SHG, PG, and SD generation included
        self.initEsigWTauRoll()
        self.initApplyIntensityData()
        self.initUpdateEtVanilla()
        
        self.initGradZSHG()
        self.initGradZSD()
        self.initGradZPG()
        self.initMinZerrSHG()
        self.initMinZerrSD()
        self.initMinZerrPG()
        self.initUpdateEtGP()
        self.initNormEsig()
        
    def initEsigTau(self):        
        Source = """
        #include <pyopencl-complex.h>
        __kernel void generateEsig_t_tau_SHG(__global cfloat_t* Et, __global cfloat_t* Esig, int N){
            const int tau_i = get_global_id(0);
            const int t_i = get_global_id(1);
            cfloat_t Etmp = (cfloat_t)(0.0f, 0.0f);
            int ind = (t_i-tau_i+N/2+N)%N;
            Esig[t_i + N*tau_i] = cfloat_mul(Et[t_i], Et[ind]);
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['generateEsig_t_tau_SHG'] = prg

        Source = """
        #include <pyopencl-complex.h>
        __kernel void generateEsig_t_tau_SD(__global cfloat_t* Et, __global cfloat_t* Esig, int N){
            const int tau_i = get_global_id(0);
            const int t_i = get_global_id(1);
            cfloat_t Etmp = (cfloat_t)(0.0f, 0.0f);
            int ind = (t_i-tau_i+N/2+N)%N;
            Esig[t_i + N*tau_i] = cfloat_mul(Et[t_i], cfloat_mul(Et[t_i], cfloat_conj(Et[ind])));
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['generateEsig_t_tau_SD'] = prg

        Source = """
        #include <pyopencl-complex.h>
        __kernel void generateEsig_t_tau_PG(__global cfloat_t* Et, __global cfloat_t* Esig, int N){
            const int tau_i = get_global_id(0);
            const int t_i = get_global_id(1);
            cfloat_t Etmp = (cfloat_t)(1.0f, 1.0f);
            int ind = (t_i-tau_i+N/2+N)%N;

            if (t_i > (tau_i-N/2) && (t_i - tau_i) < N/2)
            {
                Esig[t_i + N*tau_i] = cfloat_mul(Et[t_i], cfloat_mul(Et[ind], cfloat_conj(Et[ind])));
            }
            else
            {
                Esig[t_i + N*tau_i] = cfloat_new(0.0f, 0.0f);
            };

//            Esig[t_i + N*tau_i] = cfloat_mul(Et[t_i], cfloat_mul(Et[ind], cfloat_conj(Et[ind])));
//            Esig[t_i + N*tau_i] = cfloat_new(cfloat_real(Et[t_i]), 0.0f);
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['generateEsig_t_tau_PG'] = prg

    def initEsigWTauRoll(self):        
        Source = """
        #include <pyopencl-complex.h>
        __kernel void rollEsigWTau(__global cfloat_t* Esig, int N){
            const int col = get_global_id(0);
            cfloat_t Etmp = Esig[col];
            for (int i=0; i<N-1; i++)
            {
                Esig[col+i*N]=Esig[col+(i+1)*N];
            }
            Esig[col+N*N]=Etmp;
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['rollEsigWTau'] = prg
                
    def initApplyIntensityData(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void applyIntensityData(__global cfloat_t* Esig_w_tau, __global float* I_w_tau, int N){
            const int tau_i = get_global_id(0);
            const int t_i = get_global_id(1);
            
            float Esig_mag = cfloat_abs(Esig_w_tau[t_i + N*tau_i]);
            if (Esig_mag > 1e-10)
            { 
                Esig_w_tau[t_i + N*tau_i] = cfloat_rmul(sqrt(I_w_tau[t_i + N*tau_i])/Esig_mag, Esig_w_tau[t_i + N*tau_i]);
            }
            else
            {
                Esig_w_tau[t_i + N*tau_i] = (cfloat_t)(0.0f, 0.0f);
            }
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['applyIntensityData'] = prg
        
    def initUpdateEtVanilla(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void updateEtVanillaSumSHG(__global cfloat_t* Esig_t_tau_p, __global cfloat_t* Et, int N){
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
        self.progs['updateEtVanillaSumSHG'] = prg

        Source = """
        #include <pyopencl-complex.h>
        __kernel void updateEtVanillaSumSD(__global cfloat_t* Esig_t_tau_p, __global cfloat_t* Et, int N){
            const int t_i = get_global_id(0);
            
            cfloat_t sum = (cfloat_t)(0.0f, 0.0f);
            int i;
            for(i=0; i<N; i++) {
                sum = cfloat_add(sum, Esig_t_tau_p[t_i + N*i]);
            }
//            Et[t_i] = cfloat_conj(sum);
            Et[t_i] = sum;
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['updateEtVanillaSumSD'] = prg
        
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
        
    def initGradZSHG(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void gradZSHG(__global cfloat_t* Esig_t_tau, __global cfloat_t* Et, __global cfloat_t* dZ, int N){
            const int t_i = get_global_id(0);
            
            cfloat_t T = (cfloat_t)(0.0f, 0.0f);
            cfloat_t tmp0, tmp1;
            int tp;
            for (int tau_i=0; tau_i<N; tau_i++)
            {
                tp = t_i-(tau_i-N/2);
                if (tp >= 0 && tp < N)
                {
                    tmp0 = cfloat_mul(Et[t_i], Et[tp]);
                    // Complex number subtraction routine doesn't work so I have to do like this:
                    tmp1 = cfloat_new(cfloat_real(tmp0) - cfloat_real(Esig_t_tau[t_i+N*tau_i]), cfloat_imag(tmp0) - cfloat_imag(Esig_t_tau[t_i+N*tau_i])); 
                    T = cfloat_add(T, cfloat_mul(tmp1, cfloat_conj(Et[tp])));
                }
                tp = t_i+(tau_i-N/2);
                if (tp >= 0 && tp < N)
                {
                    tmp0 = cfloat_mul(Et[t_i], Et[tp]);
                    // Complex number subtraction routine doesn't work so I have to do like this:
                    tmp1 = cfloat_new(cfloat_real(tmp0) - cfloat_real(Esig_t_tau[tp+N*tau_i]), cfloat_imag(tmp0) - cfloat_imag(Esig_t_tau[tp+N*tau_i]));
                    T = cfloat_add(T, cfloat_mul(tmp1, cfloat_conj(Et[tp])));
                }
                
            }
            dZ[t_i] = cfloat_divider(T, N*N);
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['gradZSHG'] = prg

    def initGradZSD(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void gradZSD(__global cfloat_t* Esig_t_tau, __global cfloat_t* Et, __global cfloat_t* dZ, int N){
            const int t_i = get_global_id(0);
            
            cfloat_t T = (cfloat_t)(0.0f, 0.0f);
            cfloat_t tmp0, tmp1, tmp2;
            int tp;
            for (int tau_i=0; tau_i<N; tau_i++)
            {
                tp = t_i-(tau_i-N/2);
                if (tp >= 0 && tp < N)
                {
                    tmp0 = cfloat_mul(cfloat_conj(Et[t_i]), Et[tp]);
                    // Complex number subtraction routine doesn't work so I have to do like this:
                    tmp2 = cfloat_mul(Et[t_i], cfloat_conj(tmp0));
                    tmp1 = cfloat_new(cfloat_real(tmp2) - cfloat_real(Esig_t_tau[t_i+N*tau_i]), cfloat_imag(tmp2) - cfloat_imag(Esig_t_tau[t_i+N*tau_i]));
                    tmp2 = cfloat_mulr(cfloat_mul(tmp1, tmp0), 4.0f);
                    T = cfloat_add(T, tmp2);
                }
                tp = t_i+(tau_i-N/2);
                if (tp >= 0 && tp < N)
                {
                    tmp0 = cfloat_mul(Et[tp], Et[tp]);
                    tmp2 = cfloat_mul(Et[t_i], cfloat_conj(tmp0));
                    // Complex number subtraction routine doesn't work so I have to do like this:
                    tmp1 = cfloat_new(cfloat_real(tmp2) - cfloat_real(Esig_t_tau[tp+N*tau_i]), cfloat_imag(tmp2) + cfloat_imag(Esig_t_tau[tp+N*tau_i]));
                    tmp2 = cfloat_mulr(cfloat_mul(tmp1, tmp0), 2.0f);
                    T = cfloat_add(T, tmp2);
                }
                
            }
            dZ[t_i] = cfloat_divider(T, N*N);
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['gradZSD'] = prg

    def initGradZPG(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void gradZPG(__global cfloat_t* Esig_t_tau, __global cfloat_t* Et, __global cfloat_t* dZ, int N){
            const int t_i = get_global_id(0);

            cfloat_t T = (cfloat_t)(0.0f, 0.0f);
            cfloat_t tmp0, tmp1, tmp2;
            int tp;

            for (int tau_i=0; tau_i<N; tau_i++)
            {
                tp = t_i-(tau_i-N/2);
                if (tp >= 0 && tp < N)
                {
                    tmp0 = cfloat_mul(cfloat_conj(Et[tp]), Et[tp]);
                    // Complex number subtraction routine doesn't work so I have to do like this:
                    tmp2 = cfloat_mul(Et[t_i], tmp0);
                    tmp1 = cfloat_new(cfloat_real(tmp2) - cfloat_real(Esig_t_tau[t_i+N*tau_i]), cfloat_imag(tmp2) - cfloat_imag(Esig_t_tau[t_i+N*tau_i]));
                    tmp2 = cfloat_mul(tmp1, tmp0);
                    T = cfloat_add(T, tmp2);
                }
                tp = t_i+(tau_i-N/2);
                if (tp >= 0 && tp < N)
                {
                    tmp0 = cfloat_mul(Et[tp], cfloat_conj(Et[tp]));
                    tmp1 = cfloat_mul(Et[t_i], cfloat_conj(Et[t_i]));
                    tmp2 = cfloat_mul(tmp0, tmp1);
                    // Complex number subtraction routine doesn't work so I have to do like this:
                    tmp0 = cfloat_mul(Esig_t_tau[tp+N*tau_i], cfloat_conj(Et[tp]));
                    tmp1 = cfloat_new(cfloat_real(tmp2) - cfloat_real(tmp0), cfloat_imag(tmp2));
                    tmp2 = cfloat_mulr(cfloat_mul(tmp1, Et[t_i]), 2.0f);
                    T = cfloat_add(T, tmp2);
                }
            }
            dZ[t_i] = cfloat_mulr(cfloat_divider(T, N*N), 2.0f);
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['gradZPG'] = prg
        
    def initMinZerrSHG(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void minZerrSHG(__global cfloat_t* Esig_t_tau, __global cfloat_t* Et, __global cfloat_t* dZ, __global float* X0, __global float* X1, __global float* X2, __global float* X3, __global float* X4, int N){
            const int t_i = get_global_id(0);

            int tp;
            cfloat_t dZdZ;
            cfloat_t dZE;
            cfloat_t DEsig;
            cfloat_t tmp0, tmp1;
            X0[t_i] = 0.0f;
            X1[t_i] = 0.0f;
            X2[t_i] = 0.0f;
            X3[t_i] = 0.0f;
            X4[t_i] = 0.0f;
            for (int tau_i=0; tau_i<N; tau_i++)
            {
                tp = t_i-(tau_i-N/2);
                if (tp >= 0 && tp < N)
                {
                    dZdZ = cfloat_mul(dZ[t_i], dZ[tp]);
                    dZE = cfloat_add(cfloat_mul(dZ[t_i], Et[tp]), cfloat_mul(dZ[tp], Et[t_i]));
                    tmp0 = cfloat_mul(Et[t_i], Et[tp]);
                    tmp1 = cfloat_new(cfloat_real(tmp0) - cfloat_real(Esig_t_tau[t_i+N*tau_i]), cfloat_imag(tmp0) - cfloat_imag(Esig_t_tau[t_i+N*tau_i]));
                    DEsig = tmp1;
                    X0[t_i] += cfloat_real(dZdZ)*cfloat_real(dZdZ)+cfloat_imag(dZdZ)*cfloat_imag(dZdZ);
                    X1[t_i] += 2*cfloat_real(cfloat_mul(dZE, cfloat_conj(dZdZ)));
                    X2[t_i] += 2*cfloat_real(cfloat_mul(DEsig, cfloat_conj(dZdZ))) + cfloat_real(dZE)*cfloat_real(dZE)+cfloat_imag(dZE)*cfloat_imag(dZE);
                    X3[t_i] += 2*cfloat_real(cfloat_mul(DEsig, cfloat_conj(dZE)));
                    X4[t_i] += cfloat_real(DEsig)*cfloat_real(DEsig)+cfloat_imag(DEsig)*cfloat_imag(DEsig);
                }
            }
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['minZerrSHG'] = prg
        
    def initMinZerrSD(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void minZerrSD(__global cfloat_t* Esig_t_tau, __global cfloat_t* Et, __global cfloat_t* dZ, __global float* X0, __global float* X1, __global float* X2, __global float* X3, __global float* X4, __global float* X5, __global float* X6, int N){
            const int t_i = get_global_id(0);

            int tp;
            cfloat_t a0, a1, a2, a3;
            cfloat_t tmp0, tmp1;
            X0[t_i] = 0.0f;
            X1[t_i] = 0.0f;
            X2[t_i] = 0.0f;
            X3[t_i] = 0.0f;
            X4[t_i] = 0.0f;
            X5[t_i] = 0.0f;
            X6[t_i] = 0.0f;
            for (int tau_i=0; tau_i<N; tau_i++)
            {
                tp = t_i-(tau_i-N/2);
                if (tp >= 0 && tp < N)
                {                    
                    tmp0 = cfloat_mul(Et[t_i], cfloat_mul(Et[t_i], cfloat_conj(Et[tp])));
                    a0 = cfloat_new(cfloat_real(tmp0) - cfloat_real(Esig_t_tau[t_i+N*tau_i]), cfloat_imag(tmp0) - cfloat_imag(Esig_t_tau[t_i+N*tau_i]));
                    
                    tmp0 = cfloat_rmul(2.0f, cfloat_mul(dZ[t_i], cfloat_conj(Et[tp])));
//                    tmp0 = cfloat_new(1.0f, 1.0f);
                    tmp1 = cfloat_mul(Et[t_i], cfloat_conj(dZ[tp]));
//                    tmp1 = cfloat_mul(Et[t_i], cfloat_new(1.0f, 1.0f));
                    a1 = cfloat_mul(Et[t_i], cfloat_add(tmp0, tmp1));
                    
                    tmp0 = cfloat_rmul(2.0f, cfloat_mul(Et[t_i], cfloat_conj(dZ[tp])));
                    tmp1 = cfloat_mul(dZ[t_i], cfloat_conj(Et[tp]));
                    a2 = cfloat_mul(dZ[t_i], cfloat_add(tmp0, tmp1));
                    
                    a3 = cfloat_mul(dZ[t_i], cfloat_mul(dZ[t_i], cfloat_conj(dZ[tp])));
                    
                    X6[t_i] += cfloat_real(cfloat_mul(a0, cfloat_conj(a0)));
                    X5[t_i] += cfloat_real(cfloat_add(cfloat_mul(a0, cfloat_conj(a1)), cfloat_mul(a1, cfloat_conj(a0))));
//                    X5[t_i] += cfloat_real(cfloat_mul(a1, cfloat_conj(a1)));
                    tmp0 = cfloat_add(cfloat_mul(a0, cfloat_conj(a2)), cfloat_mul(a2, cfloat_conj(a0)));
                    tmp1 = cfloat_add(tmp0, cfloat_mul(a1, cfloat_conj(a1)));
                    X4[t_i] += cfloat_real(tmp1);
                    tmp0 = cfloat_add(cfloat_mul(a0, cfloat_conj(a3)), cfloat_mul(a3, cfloat_conj(a0)));
                    tmp1 = cfloat_add(cfloat_mul(a1, cfloat_conj(a2)), cfloat_mul(a2, cfloat_conj(a1)));
                    X3[t_i] += cfloat_real(cfloat_add(tmp1, tmp0));
                    tmp0 = cfloat_add(cfloat_mul(a1, cfloat_conj(a3)), cfloat_mul(a3, cfloat_conj(a1)));
                    tmp1 = cfloat_add(tmp0, cfloat_mul(a2, cfloat_conj(a2)));
                    X2[t_i] += cfloat_real(tmp1);
                    X1[t_i] += cfloat_real(cfloat_add(cfloat_mul(a2, cfloat_conj(a3)), cfloat_mul(a3, cfloat_conj(a2))));
                    X0[t_i] += cfloat_real(cfloat_mul(a3, cfloat_conj(a3)));
                }
            }
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['minZerrSD'] = prg

    def initMinZerrPG(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void minZerrPG(__global cfloat_t* Esig_t_tau, __global cfloat_t* Et, __global cfloat_t* dZ, __global float* X0, __global float* X1, __global float* X2, __global float* X3, __global float* X4, __global float* X5, __global float* X6, int N){
            const int t_i = get_global_id(0);

            int tp;
            cfloat_t a0, a1, a2, a3;
            cfloat_t tmp0, tmp1, tmp2;
            X0[t_i] = 0.0f;
            X1[t_i] = 0.0f;
            X2[t_i] = 0.0f;
            X3[t_i] = 0.0f;
            X4[t_i] = 0.0f;
            X5[t_i] = 0.0f;
            X6[t_i] = 0.0f;

            for (int tau_i=0; tau_i<N; tau_i++)
            {
                tp = t_i-(tau_i-N/2);
                if (tp >= 0 && tp < N)
                {
                    tmp0 = cfloat_mul(Et[t_i], cfloat_mul(Et[tp], cfloat_conj(Et[tp])));
                    a0 = cfloat_new(cfloat_real(tmp0) - cfloat_real(Esig_t_tau[t_i+N*tau_i]), cfloat_imag(tmp0) - cfloat_imag(Esig_t_tau[t_i+N*tau_i]));

                    tmp0 = cfloat_mul(dZ[tp], cfloat_conj(Et[tp]));
                    tmp1 = cfloat_mul(Et[t_i], cfloat_add(tmp0, cfloat_conj(tmp0)));
                    tmp2 = cfloat_mul(dZ[t_i], cfloat_mul(Et[tp], cfloat_conj(Et[tp])));
                    a1 = cfloat_add(tmp1, tmp2);

                    tmp0 = cfloat_mul(dZ[tp], cfloat_conj(Et[tp]));
                    tmp1 = cfloat_mul(dZ[t_i], cfloat_add(tmp0, cfloat_conj(tmp0)));
                    tmp2 = cfloat_mul(Et[t_i], cfloat_mul(dZ[tp], cfloat_conj(dZ[tp])));
                    a2 = cfloat_add(tmp1, tmp2);

                    tmp0 = cfloat_mul(dZ[t_i], cfloat_mul(dZ[tp], cfloat_conj(dZ[tp])));
                    a3 = tmp0;

                    X0[t_i] += cfloat_real(cfloat_mul(a3, cfloat_conj(a3)));
                    X1[t_i] += cfloat_real(cfloat_mul(a3, cfloat_conj(a2)));
                    tmp0 = cfloat_mul(a3, cfloat_conj(a1));
                    tmp1 = cfloat_mul(a2, cfloat_conj(a2));
                    X2[t_i] += cfloat_real(cfloat_add(tmp0, tmp1));
                    tmp0 = cfloat_mul(a3, cfloat_conj(a0));
                    tmp1 = cfloat_mul(a2, cfloat_conj(a1));
                    X3[t_i] += cfloat_real(cfloat_add(tmp0, tmp1));
                    tmp0 = cfloat_mul(a2, cfloat_conj(a0));
                    tmp1 = cfloat_mul(a1, cfloat_conj(a1));
                    X4[t_i] += cfloat_real(cfloat_add(tmp0, tmp1));
                    X5[t_i] += cfloat_real(cfloat_mul(a1, cfloat_conj(a0)));
                    X6[t_i] += cfloat_real(cfloat_mul(a0, cfloat_conj(a0)));
                }
            }
            X1[t_i] = 2 * X1[t_i];
            X2[t_i] = 2 * X2[t_i];
            X3[t_i] = 2 * X3[t_i];
            X4[t_i] = 2 * X4[t_i];
            X5[t_i] = 2 * X5[t_i];
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['minZerrPG'] = prg

    def initUpdateEtGP(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void updateEtGP(__global cfloat_t* Et, __global cfloat_t* dZ, float X, int N){
            const int t_i = get_global_id(0);
            
            Et[t_i] = cfloat_add(Et[t_i], cfloat_rmul(X, dZ[t_i]));
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['updateEtGP'] = prg
        
    def initNormEsig(self):
        Source = """
        #include <pyopencl-complex.h>
        __kernel void normEsig(__global cfloat_t* Esig_t_tau_p, __global float* Esig_t_tau_norm, int N){
            const int tau_i = get_global_id(0);
            const int t_i = get_global_id(1);
            
            cfloat_t Esig = Esig_t_tau_p[t_i+N*tau_i];
            Esig_t_tau_norm[t_i+N*tau_i] = cfloat_real(Esig)*cfloat_real(Esig)+cfloat_imag(Esig)*cfloat_imag(Esig);
        }
        """
        prg = cl.Program(self.ctx, Source).build()
        self.progs['normEsig'] = prg
