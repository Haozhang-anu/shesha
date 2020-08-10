import sys
import time
import math
from astropy.io import fits
#import numpy as np
import os
import pycuda.autoinit
import skcuda.linalg as linalg
from pycuda import gpuarray
import numpy as np
linalg.init()
from matplotlib import pyplot as plt


##### This file is to compute LAA reconstructor from scratch
##### First get covariance matrices

prefix="mavis_TLR"
cmmfile = "cmm_"+prefix+".fits"
ctmfile = "ctm_"+prefix+".fits"

if not os.path.exists(cmmfile):
    os.system('yorick -i gen_covariance.i')
nwfs = 8  ##### This shall be modified if par file changed
cmm = fits.getdata(cmmfile) # direct invrersion
ctm = fits.getdata(ctmfile) # shall be transposed
#ctm = ctm.T
nmes = int(cmm.shape[0]/nwfs)
nmesxy = int(nmes/2)

##### define const block and const diag
##### No cons block now 
consblock = np.ones([int(nmes/2),int(nmes/2)])
consmat = np.zeros(cmm.shape)
consdiag = np.eye(cmm.shape[0])
#TTFblock = np.eye(nmesxy)
#TTFblock = TTFblock - 1/TTFblock.shape[0]
#TTFmat = np.kron(np.eye(nwfs*2),TTFblock)
for ii in range(2*nwfs):
    consmat[ii*nmesxy:(ii+1)*nmesxy,ii*nmesxy:(ii+1)*nmesxy] = consblock
##### Filter TT in Ctm
##### ctm = ctm @ TTFmat.T
#for ii in range(2*nwfs):
#    consmat[ii*nmesxy:(ii+1)*nmesxy,ii*nmesxy:(ii+1)*nmesxy] = consblock

##### read default Truth file iMat
#if os.path.exists("iMat_"+prefix+"_truth.fits"):
#    iMat = fits.getdata("iMat_"+prefix+"_truth.fits") #saved with Transpose, high order only
#print(iMat.shape)

##### generate and load iMat for Truth file
if not os.path.exists("iMat_py_"+prefix+"_truth.fits"):
    os.system('python3 gen_iMat_truth.py')
iMatfile = "iMat_py_"+prefix+"_truth.fits"
iMat = fits.getdata(iMatfile)

##### do direct inversion for cmm mat
def gen_cmmm1(cdiag=0.0005,cblock = 0.001):
    print("begin calculating cmmm1 for cdiag = %.0E, TT filtered!"%(cdiag))
    #cmm_reg = cmm + consdiag*cdiag
    cmm_reg = cmm + consdiag*cdiag+ consmat*cblock 
    #cmm_reg = TTFmat @ cmm_reg @ TTFmat.T
    cmmm1 = np.linalg.solve(cmm_reg,np.eye(cmm.shape[0]))
    print("max value of cmmm1 = %.2E"%(np.amax(cmmm1)))
    #hdu = fits.PrimaryHDU(cmmm1)
    #cmmm1file = "cmmm1_"+prefix+"_%.0E.fits"%(cdiag)
    #cmmm1file = "cmmm1_num_"+prefix+"_%.0E.fits"%(cdiag)
    #hdu.writeto(cmmm1file,overwrite=True)
    #print("cmmm1 calculated and saved!")
    return cmmm1

##### tSVD inversion for cmm mat

#def gen_cmmm1_SVD(cdiag = 0.0005,cond=1e+6):
def gen_cmmm1_SVD(cond=1e+6):
    #print("begin calculating cmmm1 for cdiag = %.0E with tSVD,cond = %.0E TTFmat applied!"%(cdiag,cond))
    print("begin calculating cmmm1 with tSVD,cond = %.0E TTFmat applied!"%(cond))
    bb = time.time()
    cmm_reg = cmm
    #cmm_reg = TTFmat @ cmm_reg @ TTFmat.T
    U,S,Vt =  np.linalg.svd(cmm_reg)
    ee = time.time()
    print("SVD calculation takes %.1f seconds, details about Evalues:"%(ee-bb))
    print("S[0]=%.2E,S[-17]=%.2E,S[-1]=%.2E"%(S[0],S[-17],S[-1]))
    Sm1 = np.array(S**-1);
    Sm1n = Sm1/min(Sm1);
    inde = Sm1n[Sm1n>cond]
    print("Here I get rid of ",inde.shape[0]," modes!")
    Sm1[Sm1n>cond] = 0.;
    cmmm1 = np.dot (Vt.transpose(), np.dot(np.diag(Sm1),U.transpose()))
    print("max value of cmmm1 = %.2E"%(np.amax(cmmm1)))
    hdu = fits.PrimaryHDU(cmmm1)
    #cmmm1file = "cmmm1_tSVD_"+prefix+"_%.0E_cond_%.0E.fits"%(cdiag,cond)
    cmmm1file = "cmmm1_tSVD_"+prefix+"cond_%.0E.fits"%(cond)
    hdu.writeto(cmmm1file,overwrite=True)
    print("cmmm1 calculated and saved,")
    return cmmm1

##### generate E mat for POLC
def gen_E (mRec):
    bb = time.time()
    ##### generate and load iMat for apply file, low and high order combined
    iMat_apply = fits.getdata("iMat_"+prefix+"_full.fits")
    cMat_apply = fits.getdata("cMat_"+prefix+"_full.fits")
    ##### generate high order iMat matrix for apply file
    if not os.path.exists("iMat_py_"+prefix+".fits"):
        os.system('python3 gen_iMat.py')
    iMatpyfile = "iMat_py_"+prefix+".fits"
    iMat_py = fits.getdata(iMatpyfile)
    mRec = np.array(mRec)
    print("begin calculating E mat!")
    ##### try seperate multiplication
    #cMat_apply[0:mRec.shape[0],0:mRec.shape[1]] = mRec
    #iMat_apply[0:iMat_py.shape[0],0:iMat_py.shape[1]] = iMat_py
    gpu_mRec = gpuarray.to_gpu(mRec.astype(np.float32))
    gpu_iPy = gpuarray.to_gpu(iMat_py.astype(np.float32))
    E1=linalg.dot(gpu_mRec,gpu_iPy)    
    gpu_cMat = gpuarray.to_gpu(cMat_apply.astype(np.float32))
    gpu_iMat = gpuarray.to_gpu(iMat_apply.astype(np.float32))
    E=linalg.dot(gpu_cMat,gpu_iMat)
    np_E=E.get()
    np_E1 = E1.get()
    np_E[0:np_E1.shape[0],0:np_E1.shape[1]] = np_E1
    ee = time.time()
    print("E mat calculation takes %.1f seconds"%(ee-bb))
    print("E mat got for POLC, dims of E:")
    print(np_E.shape)
    hdu = fits.PrimaryHDU(np_E.transpose())
    hdu.writeto("E_"+prefix+"_%.0E_%.0E_Jul.fits"%(iRegu,cdiag),overwrite=True)
    #hdu.writeto("E_num_"+prefix+"_%.0E_%.0E.fits"%(cdiag,iRegu),overwrite=True)

##### function for generating quadratic modes projection matrix
def gen_quad_mat (wfs_type="NGS"):
    ### this function should take the altitude of highest DM, coords for all wfs
    ### and return a 6-column mat, rotation included.
    as2r = 4.84814e-6
    gsposfile = "gspos_"+prefix+".fits"
    dmaltfile = "dmalt_"+prefix+".fits"
    gspos = fits.getdata(gsposfile) ## pos for LGS+NGS, last 3 WFSs are NGS
    dmalt = fits.getdata(dmaltfile)
    alt = dmalt[-1] # consider quad modes on highest DM altitude
    ### NGS coord [0,0]
    if (wfs_type == "NGS"):
        nwfs = 3
        ngs_pos_array = np.reshape(gspos[-nwfs:,],nwfs*2) # in arcsec
        ngs_pos_array = ngs_pos_array*as2r
        ngs_yx_array  = np.reshape(np.flip(gspos[-nwfs:,],1),nwfs*2)
        ngs_yx_array = ngs_yx_array*as2r
        tip_term = np.array([1,0]*nwfs)
        tilt_term = np.ones(tip_term.shape)-tip_term
        '''
        foc_term = 4*math.sqrt(3)*alt*ngs_pos_array
        astig_term1 = 2*math.sqrt(6)*alt*ngs_yx_array
        astig_term2 = -2*math.sqrt(6)*alt*np.multiply(ngs_pos_array,tip_term)\
                      +2*math.sqrt(6)*alt*np.multiply(ngs_pos_array,tilt_term)
        rotate_term = -alt*np.multiply(ngs_yx_array,tip_term)\
                      +alt*np.multiply(ngs_yx_array,tilt_term)
        '''
        ##### For NGS, x and y coords are 0, only shifting is considered
        foc_term = +4*math.sqrt(3)*alt*ngs_pos_array
        astig_term1 = +2*math.sqrt(6)*alt*ngs_yx_array
        astig_term2 = +2*math.sqrt(6)*alt*np.multiply(ngs_pos_array,tip_term)\
                      -2*math.sqrt(6)*alt*np.multiply(ngs_pos_array,tilt_term)
        rotate_term = +alt*np.multiply(ngs_yx_array,tip_term)\
                      -alt*np.multiply(ngs_yx_array,tilt_term)
        
        X = np.array([tip_term*2,tilt_term*2,foc_term,astig_term1,astig_term2])
        return np.linalg.inv(X@X.T) @ X
        #return X
    if (wfs_type == "LGS"):
        ### LGS coords, here we start from a 0 altitude 
        wfsxfile = "wfsx_"+prefix+".fits"
        wfsyfile = "wfsy_"+prefix+".fits"
        wfs_x = fits.getdata(wfsxfile)
        wfs_y = fits.getdata(wfsyfile)
        # number of LGS?
        nwfs = gspos.shape[0]-3
        gsaltfile = "gsalt_"+prefix+".fits"
        gsalts = fits.getdata(gsaltfile) # alt for all LGSs
        gspos = gspos * as2r
        for ii in range(nwfs):
            # for LGSs, get shifted and scaled x/y coord
            gsalt = gsalts[ii]
            if (gsalt == 0):
                #x_coord = wfs_x  + alt * gspos[ii,0]
                #y_coord = wfs_y  + alt * gspos[ii,1]
                cone_eff = 1
            else :
                #x_coord = wfs_x *(gsalt-alt)/gsalt + alt * gspos[ii,0]
                #y_coord = wfs_y *(gsalt-alt)/gsalt + alt * gspos[ii,1]
                cone_eff = (gsalt-alt)/gsalt
            #print(cone_eff) 
            cone_var1 = 1 - cone_eff**2
            cone_var0 = cone_eff
            tt_term1 = np.ones(wfs_x.shape)
            tt_term0 = np.zeros(wfs_x.shape)
            # get proj arrays, for LGS, cone effect shall be considered
            # now we are just dealing with 0 altitude situation
            '''
            proj_x = np.array([tt_term1*alt/gsalt,tt_term0,\
                              4*math.sqrt(3)*(cone_var1*wfs_x-cone_var0*alt*gspos[ii,0]*tt_term1),\
                              2*math.sqrt(6)*(cone_var1*wfs_y-cone_var0*alt*gspos[ii,1]*tt_term1),\
                              2*math.sqrt(6)*(cone_var1*wfs_x-cone_var0*alt*gspos[ii,0]*tt_term1)])
            proj_y = np.array([tt_term0,tt_term1*alt/gsalt,\
                              4*math.sqrt(3)*(cone_var1*wfs_y-cone_var0*alt*gspos[ii,1]*tt_term1),\
                              2*math.sqrt(6)*(cone_var1*wfs_x-cone_var0*alt*gspos[ii,0]*tt_term1),\
                              -2*math.sqrt(6)*(cone_var1*wfs_y-cone_var0*alt*gspos[ii,1]*tt_term1)])
            
            ###### Direct projection
            cone_var1 = cone_eff
            proj_x = np.array([tt_term1*2,tt_term0,\
                              4*math.sqrt(3)*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii,0]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii,1]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii,0])])
            proj_y = np.array([tt_term0,tt_term1*2,\
                              4*math.sqrt(3)*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii,1]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii,0]),\
                              -2*math.sqrt(6)*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii,1])])
            '''
            ###### Plate Scale projection
            cone_var1 = cone_eff
            proj_x = np.array([tt_term1*2,tt_term0,\
                              4*math.sqrt(3)*(cone_var1**2 *wfs_y*0+cone_var1*alt*gspos[ii,0]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_y*0+cone_var1*alt*gspos[ii,1]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_x*0+cone_var1*alt*gspos[ii,0])])
            proj_y = np.array([tt_term0,tt_term1*2,\
                              4*math.sqrt(3)*(cone_var1**2 *wfs_y*0+cone_var1*alt*gspos[ii,1]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_x*0+cone_var1*alt*gspos[ii,0]),\
                              -2*math.sqrt(6)*(cone_var1**2 *wfs_y*0+cone_var1*alt*gspos[ii,1])])

            if (ii == 0):
                #initialise proj_mat
                proj_mat = np.concatenate((proj_x,proj_y),axis=1)
                #print(proj_mat.shape)
            else :
                #proj_mat = np.array([proj_mat,proj_x,proj_y])
                proj_mat = np.concatenate((proj_mat,proj_x),axis=1)
                proj_mat = np.concatenate((proj_mat,proj_y),axis=1)
                #print(proj_mat.shape)
        print(proj_mat.shape)
        return proj_mat
    if (wfs_type == "MNGS"):
        ### MNGS coords, here we save NGS XY  
        wfsxfile = "wfsx_"+prefix+"_MNGS.fits"
        wfsyfile = "wfsy_"+prefix+"_MNGS.fits"
        wfs_x = fits.getdata(wfsxfile)
        wfs_y = fits.getdata(wfsyfile)
        # number of MNGS?
        nwfs = 3
        #gsaltfile = "gsalt_"+prefix+".fits"
        gsalts = 0 # NGS
        gspos = gspos * as2r
        for ii in range(nwfs):
            # for LGSs, get shifted and scaled x/y coord
            gsalt = gsalts
            if (gsalt == 0):
                #x_coord = wfs_x  + alt * gspos[ii,0]
                #y_coord = wfs_y  + alt * gspos[ii,1]
                cone_eff = 1
            else :
                #x_coord = wfs_x *(gsalt-alt)/gsalt + alt * gspos[ii,0]
                #y_coord = wfs_y *(gsalt-alt)/gsalt + alt * gspos[ii,1]
                cone_eff = (gsalt-alt)/gsalt
            #print(cone_eff) 
            cone_var1 = 1 - cone_eff**2
            cone_var0 = cone_eff
            tt_term1 = np.ones(wfs_x.shape)
            tt_term0 = np.zeros(wfs_x.shape)
            # get proj arrays, for MNGS
            ###### Direct projection
            cone_var1 = cone_eff
            proj_x = np.array([tt_term1*2,tt_term0,\
                              4*math.sqrt(3)*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii+8,0]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii+8,1]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii+8,0]),\
                              -1*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii+8,1])])
            proj_y = np.array([tt_term0,tt_term1*2,\
                              4*math.sqrt(3)*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii+8,1]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii+8,0]),\
                              -2*math.sqrt(6)*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii+8,1]),\
                              1*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii+8,0])])

            if (ii == 0):
                #initialise proj_mat
                proj_mat = np.concatenate((proj_x,proj_y),axis=1)
                #print(proj_mat.shape)
            else :
                #proj_mat = np.array([proj_mat,proj_x,proj_y])
                proj_mat = np.concatenate((proj_mat,proj_x),axis=1)
                proj_mat = np.concatenate((proj_mat,proj_y),axis=1)
                #print(proj_mat.shape)
                
        print(proj_mat.shape)
        return proj_mat

##### calculate LAA mRec
def gen_mRec(iReg = 0.1,cd = 0.005,cb = 0.001):
#def gen_mRec(iReg = 0.1,con=1000):
    bb = time.time()
    cmmm1 = gen_cmmm1(cdiag=cd,cblock = cb)
    #cmmm1 = gen_cmmm1_SVD(cond=con)
    print(iMat.shape)
    #iMatt    = np.ascontiguousarray(iMat)
    iSquare = iMat.T @ iMat
    iRegmat = np.eye(iSquare.shape[0])
    iAvg = np.average(np.diag(iSquare))
    iRegmat = iRegmat*iReg*iAvg
    iSquare = iSquare + iRegmat
    print("iSquare regularised with factor = %.0E"%iReg)
    gpu_iSq = gpuarray.to_gpu(iSquare.astype(np.float32))
    gpu_iMat = gpuarray.to_gpu(iMat.astype(np.float32))
    gpu_ctm = gpuarray.to_gpu(ctm.astype(np.float32))
    #gpu_TTF = gpuarray.to_gpu(TTFmat.astype(np.float32))
    gpu_cmmm1 = gpuarray.to_gpu(cmmm1.astype(np.float32))
    temp1 = linalg.inv(gpu_iSq)
    temp2 = linalg.dot(gpu_iMat,gpu_ctm,transa='T',transb='T')
    #temp2 = linalg.dot(gpu_iMat,linalg.dot(gpu_TTF,gpu_ctm),transa='T',transb='T')
    temp3 = linalg.dot(temp2,gpu_cmmm1)
    #temp2 = linalg.dot(gpu_ctm,gpu_TTF)
    #temp3 = linalg.dot(temp2,gpu_cmmm1,transa="T")
    #Tfile = "TT_filter_covariance.fits"
    #np_T  = temp3.get()
    #hdu = fits.PrimaryHDU(np_T.transpose())
    #hdu.writeto(Tfile,overwrite=True)
    M     = linalg.dot(temp1,temp3)
    np_M = M.get()
    print("mRec got! dims of mRec is:")
    print(np_M.shape)
    print("max value of M = %.2E,min value of M = %.2E"%(np.amax(np_M),np.amin(np_M)))
    Mfile = "mRec_"+prefix+"_%.0E_%.0E_Jul.fits"%(iReg,cd)
    #Mfile = "mRec_tSVD_"+prefix+"_%.0E_%.0E_cond_%.0E_nottf.fits"%(cd,iReg,con)
    hdu = fits.PrimaryHDU(np_M.transpose())
    hdu.writeto(Mfile,overwrite=True)
    ee = time.time()
    print("LAA reconstructor calculation takes %.1f seconds"%(ee-bb))
    gen_E(np_M)

    ##### split tomography part, get projection matrices
    Xfilename = "X_proj_"+prefix+".fits"
    if not os.path.exists(Xfilename):
        X_proj = gen_quad_mat (wfs_type="MNGS") # (X.T@X)**-1X.T 
        Y_proj = gen_quad_mat (wfs_type="LGS") # Y
        hdu = fits.PrimaryHDU(X_proj)
        hdu.writeto("X_proj_"+prefix+".fits",overwrite=True)
        hdu = fits.PrimaryHDU(Y_proj)
        hdu.writeto("Y_proj_"+prefix+".fits",overwrite=True)
    


#cblock = float(sys.argv[1])
#cdiag  = float(sys.argv[1])
#iRegu  = float(sys.argv[2])
#co     = float(sys.argv[3])
#cblocks  = [0.000001,0.0000001]
cdiags = [0.00005]
iRegus = [0.0005]
cblock = 0.01
#coS    = [100,1000,10000]
for ii in [1]:
    for iRegu in iRegus:
        for cdiag in cdiags:
            print("calculating matrices for iRegu = %.0E,cdiag=%.0E"%(iRegu,cdiag))
            #print("calculating matrices for cdiag = %.0E, iRegu = %.0E,con=%.0E"%(iRegu,co))
            gen_mRec(iReg = iRegu,cd=cdiag,cb = cblock)


#w_type = str(sys.argv[1])
#Y = gen_quad_mat (wfs_type="LGS")
#X = gen_quad_mat (wfs_type="NGS")
#U,S,Vt = np.linalg.svd(Y)
#print(S)
