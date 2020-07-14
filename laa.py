# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 18:34:17 2018

@author: jcranney
"""

import numpy as np
from scipy.linalg import block_diag as blkdiag
import pycuda.autoinit
import pycuda.driver as drv
import skcuda.linalg as linalg
from pycuda import gpuarray
from astropy.io.fits import open as fitsopen
from astropy.io.fits import writeto as fitswrite
from scipy.special import gamma, kv
import scipy.stats as stats
import matplotlib.pyplot as pp
from numpy.linalg import solve,inv
import sys
import time

print("Using GPU: %s" % (pycuda.autoinit.device.name(),))

typeSelect = 2 #0 = floast16 1 = float32 2 =  float64 

if(typeSelect == 0):
    numpyType =np.float16
    Ctype = "float"
elif(typeSelect == 1):
    numpyType = np.float32
    Ctype = "float"
elif(typeSelect == 2):
    numpyType = np.float64
    Ctype = "double"
from pycuda.compiler import SourceModule
linalg.init()


mod = SourceModule("""
    //#include "cublas_v2.h"
    //#DEFINE BESSEL_ORDER 7
    __device__ void gpuParSolveF("""+Ctype+""" *G, """+Ctype+""" *b ,int m,int n);
    __device__ void gpuParSolveB("""+Ctype+""" *G, """+Ctype+""" *b ,int m,int n);
    
    /*============================================================================
    Function to compute Gauss Map with GPU. 
    Inputs:
        y_out,x_out,y_in,x_in,dmpitchSQ,infl
    Output
        gauss_map
    Function
    
    gauss_map=infl^(((x_out-x_in)^2+(y_out-y_in)^2)/dmpitchSQ)
    
    =============================================================================*/
    __global__ void gpuGaussMap("""+Ctype+""" *gauss_map,"""+Ctype+""" *y_out, """+Ctype+""" *x_out,"""+Ctype+""" *y_in, """+Ctype+""" *x_in,const """+Ctype+""" *dmpitch,const """+Ctype+""" *infl, int *dims)
    {
    
        const int m = blockIdx.x*blockDim.x+threadIdx.x; // Dimension of the input
        const int n = blockIdx.y*blockDim.y+threadIdx.y; // Dimension of the output        
        if((m < (dims[0]))&&(n < (dims[1]))){                                                               
        """+Ctype+""" x_dist,y_dist,tmp;
     
            // Calculate exponent
            x_dist = x_out[m]-x_in[n];
            y_dist = y_out[m]-y_in[n];
            tmp = ( x_dist*x_dist + y_dist*y_dist ) / dmpitch[0];
            tmp = pow(infl[0],tmp);
            gauss_map[m*dims[1]+n] = tmp;
        }
    }

    /*============================================================================
    Function to compute Gauss Map with GPU. 
    Inputs:
        y_out,x_out,y_in,x_in,rS,L0
    Output
        cov_map
    Function
        
        r=sqrt(((x_out-x_in)^2+(y_out-y_in)^2))+1e-10/L0
        
        rS*
    
    =============================================================================*/

    __global__ void gpuCovMap("""+Ctype+""" *cov_map,"""+Ctype+""" *y_out, """+Ctype+""" *x_out,"""+Ctype+""" *y_in, """+Ctype+""" *x_in,"""+Ctype+""" *rS,"""+Ctype+""" *L0, int *dims)
    {
    
        const int m = blockIdx.x*blockDim.x+threadIdx.x; // Dimension of the input
        const int n = blockIdx.y*blockDim.y+threadIdx.y; // Dimension of the output  
        if((m < (dims[0]))&&(n < (dims[1]))){                                           
            """+Ctype+""" x_dist,y_dist,r_sqrt,tmp,f;
            // Bessel function coefficients
            const """+Ctype+""" besselCoeff[15] = {1.726297320500,-0.003237998143,0.273140140037,-23.463550287780,-38.834479060473,425.845739476153,-1344.011011285270,3302.687697164970,-6891.916880474140,10878.648247778900,-12060.778082656600,9056.166061506630,-4400.578764200840,1253.933383863860,-159.686900036592};
            // Calculate exponent
            x_dist = x_out[m]-x_in[n];
            y_dist = y_out[m]-y_in[n];
            r_sqrt = sqrt((sqrt( x_dist*x_dist + y_dist*y_dist ))/L0[0]);
            tmp = r_sqrt;
            f = besselCoeff[0];
            #pragma unroll
            for (int i=1;i<15;i++){
                    f += tmp*besselCoeff[i];
                    tmp *= r_sqrt;
            }
            cov_map[m*dims[1]+n] = rS[0]*f;
        }
    }
""")
gpuGaussMap = mod.get_function("gpuGaussMap")
gpuCovMap = mod.get_function("gpuCovMap")

"""
if len(sys.argv)==1:
    ao = __import__("unif_mcao")
    filename = "unif_mcao.py"
else:
    ao = __import__(sys.argv[1].replace('.py',''))
    filename = sys.argv[1]
"""
import argparsing as ao

verbose = 1

def log(message):
    if verbose:
        print(message)
        sys.stdout.flush()

def fitsread(filename):
    return (fitsopen(filename)[0].data)

def getAp(r,k=1.05): #1.05 for small system, 1.02 for big
    [xx,yy] = np.meshgrid(np.linspace(-1,1,r),np.linspace(-1,1,r))
    A = np.sqrt(xx**2+yy**2)<=k
    return A

def createMask(v):
    v = v.T.flatten()
    A = np.eye(v.shape[0],dtype=bool)
    v = [x for x in range(len(v)) if v[x]==1]
    A = np.delete(A,v,axis=0)
    return A

def threadMap(Dimension,count):
    if Dimension < count:
        threadCount = Dimension;
        gCount = 1;
    else:
        threadCount = count
        if(Dimension%threadCount):
                gCount = int((Dimension+(threadCount-Dimension%threadCount))/threadCount)
        else:
                gCount = int(Dimension/threadCount)
    return [gCount,threadCount]

def gaussMap(y_out,x_out,y_in,x_in,dmpitch,infl):
    count = 32
    m = x_out.shape[0]
    n = x_in.shape[0]    
    dims = gpuarray.to_gpu(np.asarray([m,n]).astype(np.int32))
    gauss_map = gpuarray.zeros((m,n),dtype=numpyType)
    [gCountm,threadCountm] = threadMap(x_out.shape[0],count)
    [gCountn,threadCountn] = threadMap(x_in.shape[0],count)
    gpuGaussMap(
        gauss_map, drv.In(y_out.astype(numpyType)), drv.In(x_out.astype(numpyType)),drv.In(y_in.astype(numpyType)),drv.In(x_in.astype(numpyType)),drv.In(numpyType(dmpitch**2)),drv.In(numpyType(infl)),dims,
        block=(threadCountm,threadCountn,1), grid=(gCountm,gCountn))
    return gauss_map

def covMap(y_out,x_out,y_in,x_in,r0,L0):
    count = 32
    m = x_out.shape[0]
    n = x_in.shape[0]
    dims = gpuarray.to_gpu(np.asarray([m,n]).astype(np.int32))
    cov_map = gpuarray.zeros((m,n),dtype=numpyType)
    [gCountm,threadCountm] = threadMap(x_out.shape[0],count)
    [gCountn,threadCountn] = threadMap(x_in.shape[0],count)
    rS = (L0/r0)**(5/3) # pre calculate this value so we only have to do it once
    gpuCovMap(
        cov_map, drv.In(y_out.astype(numpyType)), drv.In(x_out.astype(numpyType)),drv.In(y_in.astype(numpyType)),drv.In(x_in.astype(numpyType)),drv.In(numpyType(rS)),drv.In(numpyType(L0)),dims,
        block=(threadCountm,threadCountn,1), grid=(gCountm,gCountn))
    return cov_map

def kMCAO(dirs,proj,meta,dm,h_meta,h_dm,r0,L0,dm_pitch,infl,proj_reg):
    QTQ = [[gpuarray.zeros([dm[mi][0].shape[0],dm[ni][0].shape[0]],dtype=numpyType) for ni in range(len(dm))] for mi in range(len(dm))]
    QTP = [[gpuarray.zeros([dm[mi][0].shape[0],meta[ni][0].shape[0]],dtype=numpyType) for ni in range(len(meta))] for mi in range(len(dm))]
    log("Doing directions...")
    CovInv = [covInInv(meta[ni][0],meta[ni][1],r0[ni],L0[ni]) for ni in range(len(meta))]
    for mi in range(len(dirs)):  #loop over directions
        Q = [numpyType(dirs[mi][2])*gaussMap(proj[0][0] + h_dm[ni]*np.tan(dirs[mi][0]),proj[0][1] + h_dm[ni]*np.tan(dirs[mi][1]),dm[ni][0],dm[ni][1],dm_pitch[ni],infl[ni]) for ni in range(len(dm))]
        P = [numpyType(dirs[mi][2])*covMap(proj[0][0] + h_meta[ni]*np.tan(dirs[mi][0]),proj[0][1] + h_meta[ni]*np.tan(dirs[mi][1]),meta[ni][0],meta[ni][1],r0[ni],L0[ni])  for ni in range(len(meta))]
        gpuAddQTQ(QTQ,Q)
        gpuAddQTP(QTP,Q,P,CovInv)
        log("Done direction %d" % (mi+1,))
    for mi in range(len(dm)):
        QTQ[mi][mi]=QTQ[mi][mi]+proj_reg*linalg.eye(QTQ[mi][mi].shape[0],dtype=numpyType)
    QTQ = cell2mat([[QTQ[mi][ni].get() for ni in range(len(dm))]  for mi in range(len(dm))])
    QTP = cell2mat([[QTP[mi][ni].get() for ni in range(len(meta))]  for mi in range(len(dm))])
    print(np.linalg.cond(QTQ))
    return solve(QTQ,QTP)

def covInInv(y_in,x_in,r0,L0):
    cov_map = covMap(y_in,x_in,y_in,x_in,r0,L0)
    return linalg.inv(cov_map)

def gpuMul(gpu_A,gpu_B):
    return linalg.dot(gpu_A, gpu_B, transa='T')

def gpuSolve(gpu_A,gpu_B,gpu_C):
    gpu_D = linalg.dot(gpu_A, gpu_B.astype(numpyType), transa='T')
    gpu_D = linalg.dot(gpu_D, gpu_C.astype(numpyType), transa='N')
    return gpu_D

def gpuAddQTQ(gpu_QTQ,gpu_Q):
    for mi in range(len(gpu_Q)):
        for ni in range(len(gpu_Q)):
            gpu_QTQ[mi][ni] = gpu_QTQ[mi][ni] + gpuMul(gpu_Q[mi],gpu_Q[ni])

def tmp_covMap(y_out,x_out,y_in,x_in,r0,L0):
    cov_in = covMap(y_in,x_in,y_in,x_in,r0,L0).get()
    cov_out = covMap(y_out,x_out,y_in,x_in,r0,L0).get()
    return solve(cov_in,cov_out.T).T

def gpuAddQTP(gpu_QTP,gpu_Q,gpu_P,gpu_CovInv):
    for ni in range(len(gpu_P)):
        for mi in range(len(gpu_Q)):
            gpu_QTP[mi][ni] = gpu_QTP[mi][ni] + gpuSolve(gpu_Q[mi],gpu_P[ni],gpu_CovInv[ni])

def cell2mat(X):
    n = np.sum([x.shape[1] for x in X[0]])
    A = np.zeros([0,n],dtype=numpyType)
    for mi in range(len(X)):
        tmp = np.zeros([X[mi][0].shape[0],0],dtype=numpyType)
        for ni in range(len(X[0])):
            tmp = np.concatenate([tmp,X[mi][ni]],axis=1)
        A = np.concatenate([A,tmp],axis=0)
    return A

def getMetaCoords(D,h,rad,res,mask=True):
    f = 1+2*h*np.tan(rad)/D
    xx = np.linspace(-f*D/2,f*D/2,res,dtype=numpyType)
    [X,Y] = np.meshgrid(xx,xx)
    X = X.T.flatten()
    Y = Y.T.flatten()
    if mask:
        mask_mat = createMask((getAp(xx.shape[0],1.0)==0).T.flatten())
        X = mask_mat @ X
        Y = mask_mat @ Y
    return [Y,X]

def LGS_filter(wfs_coords):
    Y_meta = wfs_coords[0]
    X_meta = wfs_coords[1]
    X = np.array([np.ones(X_meta.shape),
                  X_meta,
                  Y_meta
                 ]).T
    return  (np.eye(X_meta.shape[0])-X@inv(X.T@X)@X.T)

def P_filter(wfs_coords):
    Y_meta = wfs_coords[0]
    X_meta = wfs_coords[1]
    X = np.array([np.ones(X_meta.shape)
                 ]).T
    return  (np.eye(X_meta.shape[0])-X@inv(X.T@X)@X.T)

def covLMMSE(y_out,x_out,y_in,x_in,r0,L0):
    cov_in = covMap(y_in,x_in,y_in,x_in,r0,L0).get()
    cov_out = covMap(y_out,x_out,y_in,x_in,r0,L0).get()
    return solve(cov_in,cov_out.T).T

log("Loaded " + ao.simname)

if __name__ == '__main__':
    wfs_coords = [[(ao.K_y[0]).copy(),(ao.K_x[0]).copy()]]
    dm_coords = [[ao.K_y[ni],ao.K_x[ni]] for ni in range(ao.N)]
    proj_coords = [getMetaCoords(ao.D,0.0,0.0,ao.proj_res,mask=True)]
    ao.dm_pitch = [stats.mode([x for x in abs(dm_coords[ni][1]-np.roll(dm_coords[ni][1],1)) if x > 0])[0][0] for ni in range(ao.N)]
    N_sml = gaussMap(ao.K_y[0],ao.K_x[0],ao.K_y[0],ao.K_x[0],ao.dm_pitch[0],ao.infl[0]).get()
    ao.C_sml = {}
    if ao.C_sml_N is not None:
        ao.C_sml_N = (solve(N_sml.T,ao.C_sml_N.T).T).astype(numpyType)
        ao.C_sml["N"] = ao.C_sml_N @ P_filter(wfs_coords[0])
    if ao.C_sml_L is not None:
        ao.C_sml_L = (solve(N_sml.T,ao.C_sml_L.T).T).astype(numpyType)
        if ao.filter_LGS_TT:
            ao.C_sml["L"] = ao.C_sml_L @ LGS_filter(wfs_coords[0])
        else:
            ao.C_sml["L"] = ao.C_sml_L @ P_filter(wfs_coords[0])

    log("Doing CMM")
    CMM = cell2mat([[ ao.C_sml[ao.whichCsml[mi]] @ np.sum([ covMap(
           wfs_coords[0][0]*(1-ao.h[li]/ao.h_gs[mi])+ao.h[li]*ao.yc[mi],
           wfs_coords[0][1]*(1-ao.h[li]/ao.h_gs[mi])+ao.h[li]*ao.xc[mi],
           wfs_coords[0][0]*(1-ao.h[li]/ao.h_gs[ni])+ao.h[li]*ao.yc[ni],
           wfs_coords[0][1]*(1-ao.h[li]/ao.h_gs[ni])+ao.h[li]*ao.xc[ni],
           ao.r0[li], ao.L0[li]).get() for li in range(ao.N_meta)],axis=0) @ ao.C_sml[ao.whichCsml[ni]].T
           for ni in range(ao.M)] for mi in range(ao.M)])
    log("Prepping CMM")
    CMM_diag = [(ao.sigwlgs**2)*np.eye(ao.C_sml_L.shape[0],dtype=numpyType) for x in range(ao.M_lgs)] + \
               [(ao.sigwngs**2)*np.eye(ao.C_sml_N.shape[0],dtype=numpyType) for x in range(ao.M_ngs)]  
    CMM = CMM + blkdiag(*CMM_diag)


    log("Doing TS iMat")
    # Interaction Matrix from commands to TS
    iMatTS = cell2mat([[ gaussMap(
           proj_coords[0][0]*(1-ao.h_dm[ni]/ao.h_ts[mi])+ao.h_dm[ni]*ao.dirs[mi][0],
           proj_coords[0][1]*(1-ao.h_dm[ni]/ao.h_ts[mi])+ao.h_dm[ni]*ao.dirs[mi][1],
           dm_coords[ni][0],dm_coords[ni][1],
           ao.dm_pitch[ni],ao.infl[ni]
    ).get()  for ni in range(ao.N)] for mi in range(len(ao.dirs))])

    log("Doing CTM")
    CTM = cell2mat([[ np.sum([covMap(
           proj_coords[0][0]*(1-ao.h[li]/ao.h_ts[mi])+ao.h[li]*ao.dirs[mi][0]+ao.framedelay*ao.vy[li]*ao.ittime,
           proj_coords[0][1]*(1-ao.h[li]/ao.h_ts[mi])+ao.h[li]*ao.dirs[mi][1]+ao.framedelay*ao.vx[li]*ao.ittime,
           wfs_coords[0][0]*(1-ao.h[li]/ao.h_gs[ni])+ao.h[li]*ao.yc[ni],
           wfs_coords[0][1]*(1-ao.h[li]/ao.h_gs[ni])+ao.h[li]*ao.xc[ni],
           ao.r0[li], ao.L0[li]).get() for li in range(ao.N_meta)],axis=0) @ ao.C_sml[ao.whichCsml[ni]].T 
           for ni in range(ao.M)] for mi in range(len(ao.dirs))]) 

    log("Doing GS iMat")
    iMat = cell2mat([[ ao.C_sml[ao.whichCsml[mi]] @ gaussMap(
           wfs_coords[0][0]*(1-ao.h_dm[ni]/ao.h_gs[mi])+ao.h_dm[ni]*ao.yc[mi],
           wfs_coords[0][1]*(1-ao.h_dm[ni]/ao.h_gs[mi])+ao.h_dm[ni]*ao.xc[mi],
           dm_coords[ni][0],dm_coords[ni][1],
           ao.dm_pitch[ni],ao.infl[ni]
    ).get()  for ni in range(ao.N)] for mi in range(ao.M)])

    log("Solving cMat")
    im1 = iMatTS.T @ iMatTS + ao.proj_reg*np.eye(iMatTS.shape[1])
    im2 = iMatTS.T @ CTM
    print(im2.shape)
    log("im p1")
    im2[:,:10000] = solve(im1,im2[:,:10000])
    log("im p2")
    im2[:,10000:] = solve(im1,im2[:,10000:])
    im1 = []
    iMatTS = []
    CTM = []

    log("Solving Reconstructor")
    R = solve( CMM, im2.T).T

    log("Computing command filter")
    E = R @ iMat
    log("saving E")
    fitswrite(ao.simname+"_E.fits",E,overwrite=True)
    log("saving R")
    fitswrite(ao.simname+"_R.fits",R,overwrite=True)
