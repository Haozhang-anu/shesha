import sys
import os
import time
#from astropy.io import fits
import numpy as np
import pycuda.autoinit
import skcuda.linalg as linalg
from pycuda import gpuarray
linalg.init()
import math
#from matplotlib import pyplot as plt

# This file generates only the iMat for high order subsystem
# iMat for truth file: number of WFS mes should fit truth file while 
# number of DM act should fit apply file

#if iMat_type == "lgs":
prefix = "mavis_TLR_truth"
#else:
#    prefix = "mavis_TLR_truth"
##### save coords from yorick
os.system("yorick -i save_coords_truth.i")

##### load files
#wfsx = fits.getdata("wfsx_"+prefix+".fits")
#wfsy = fits.getdata("wfsy_"+prefix+".fits")
nact = fits.getdata("nact_"+prefix+".fits")
actidx = [sum(nact[:ii]) for ii in range(nact.shape[0]+1)]
gspos = fits.getdata("gspos_"+prefix+".fits")
nwfs = gspos.shape[0]
gsalt = fits.getdata("gsalt_"+prefix+".fits")
dmalt = fits.getdata("dmalt_"+prefix+".fits")
iBlock = fits.getdata("iBlock_"+prefix+".fits")
iBlock = iBlock.transpose()
dmpitch = fits.getdata("dmpitch_"+prefix+".fits")
diam = fits.getdata("diam_"+prefix+".fits")
pixel_size = float(diam[0]/diam[1])
##### load dm x and y
dmxs  = fits.getdata("dmx_"+prefix+".fits")
dmys  = fits.getdata("dmy_"+prefix+".fits")

##### function: get distance mat from xy vector
def gen_dist(x1,y1,x2,y2):
    if (x1.shape[0] != y1.shape[0]) or (x2.shape[0] != y2.shape[0]):
        return
    x1mat = np.tile(x1,(x2.shape[0],1)) # dim #act2*#act1, every column is the same
    y1mat = np.tile(y1,(y2.shape[0],1))
    x2mat = np.tile(x2,(x1.shape[0],1))
    y2mat = np.tile(y2,(y1.shape[0],1))
    dtx  = x1mat-x2mat.transpose()    #diagonal entries = 0, x distance mat
    dty  = y1mat-y2mat.transpose()
    #dtx2  = x2mat-x2mat.transpose()
    #dty2  = y2mat-y2mat.transpose()
    r = np.sqrt(dtx**2+dty**2)
    return r

##### gen_iMat main function
def gen_iMat ():
    # first generate ground layer M_phi_u, dimension nact*nact
    # generate dist mat from x,y
    r = gen_dist(dmxs[actidx[0]:actidx[1]]/dmpitch[0],dmys[actidx[0]:actidx[1]]/dmpitch[0],\
                 dmxs[actidx[0]:actidx[1]]/dmpitch[0],dmys[actidx[0]:actidx[1]]/dmpitch[0])
    M_pugl = 0.3**(r**2)
    M_puglm1 = np.linalg.solve(M_pugl,np.eye(M_pugl.shape[0]))
    M_sp   = np.dot(iBlock,M_puglm1)


    # filter Tip-Tilt in Msp, TTF_mat = x@(xT@x)^(-1)@xT
    # first define x, which is the T_T_P terms
    #dm1x = dmxs[actidx[0]:actidx[1]]
    #dm1y = dmys[actidx[0]:actidx[1]]
    #X    = np.array([np.ones(dm1x.shape),dm1x,dm1y]).T
    #TTF_mat  = (np.eye(dm1x.shape[0])-X@np.linalg.inv(X.T@X)@X.T)
    #M_sp     = M_sp @ TTF_mat
    # no need to filter TT:
    iBlock_f = M_sp @ M_pugl 

    # now calculate blocks for altitude DM layer by M_sp*M_pushift*F_glal
    for ndm in range(3):
        if ndm == 0:
            iMat = np.tile(iBlock_f,(nwfs,1))
        elif ndm > 0:
            for nw in range(nwfs):
                # dm_xy in pixels
                dm1x = dmxs[actidx[0]:actidx[1]]
                dm1y = dmys[actidx[0]:actidx[1]]
                dmndmx = dmxs[actidx[ndm]:actidx[ndm+1]]/dmpitch[ndm]
                dmndmy = dmys[actidx[ndm]:actidx[ndm+1]]/dmpitch[ndm]
                # shift dm1x/y first, x and y not scaled
                #dm1x = dm1x/dmpitch[0]
                #dm1y = dm1y/dmpitch[0]
                if gsalt[nw]>0 :
                    dm1x_shifted = dm1x*(gsalt[nw]-dmalt[ndm])/gsalt[nw]+gspos[nw,0]*4.848*1e-6*dmalt[ndm]/pixel_size
                    dm1y_shifted = dm1y*(gsalt[nw]-dmalt[ndm])/gsalt[nw]+gspos[nw,1]*4.848*1e-6*dmalt[ndm]/pixel_size
                    # shifted coords in pixels
                else:
                    dm1x_shifted = dm1x+gspos[nw,0]*4.848*1e-6*dmalt[ndm]/pixel_size;
                    dm1y_shifted = dm1y+gspos[nw,1]*4.848*1e-6*dmalt[ndm]/pixel_size;
                    # for NGS, just do shifting
                # scale the shifted coords to dmpitch at altitude layer
                dm1x_shifted = dm1x_shifted/dmpitch[ndm];
                dm1y_shifted = dm1y_shifted/dmpitch[ndm];
                # get shifted M_phi_u
                r_sft = gen_dist(dm1x_shifted,dm1y_shifted,dm1x_shifted,dm1y_shifted)
                M_pusft = 0.3**(r_sft**2)
                # get F_ual_ugl mat
                r_trans = gen_dist(dm1x_shifted,dm1y_shifted,dmndmx,dmndmy)
                F_trans = 0.3**(r_trans.transpose()**2)
                # calculate iMat block by M_sp*M_pu_shift*F_trans
                iMat_block = M_sp @ F_trans
                #concatenate for all WFSs
                if nw == 0 :
                    Mati = iMat_block
                else :
                    Mati = np.concatenate((Mati,iMat_block))
            #concatenate iMat for DMs
            iMat = np.concatenate((iMat,Mati),axis=1)
    #return iMat

    #now save iMat
    hdu = fits.PrimaryHDU(iMat)
    iMatfile = "iMat_py_"+prefix+".fits"
    print("Writing output file to %s" % iMatfile)
    print(iMat.shape)
    hdu.writeto(iMatfile,overwrite=True)

gen_iMat()
quit
