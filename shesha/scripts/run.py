#!/usr/bin/env python

## @package   shesha.script.closed_loop
## @brief     script test to simulate a closed loop
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.3.2
## @date      2011/01/28
## @copyright GNU Lesser General Public License
#
#  This file is part of COMPASS <https://anr-compass.github.io/compass/>
#
#  Copyright (C) 2011-2019 COMPASS Team <https://github.com/ANR-COMPASS>
#  All rights reserved.
#  Distributed under GNU - LGPL
#
#  COMPASS is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser 
#  General Public License as published by the Free Software Foundation, either version 3 of the License, 
#  or any later version.
#
#  COMPASS: End-to-end AO simulation tool using GPU acceleration 
#  The COMPASS platform was designed to meet the need of high-performance for the simulation of AO systems. 
#  
#  The final product includes a software package for simulating all the critical subcomponents of AO, 
#  particularly in the context of the ELT and a real-time core based on several control approaches, 
#  with performances consistent with its integration into an instrument. Taking advantage of the specific 
#  hardware architecture of the GPU, the COMPASS tool allows to achieve adequate execution speeds to
#  conduct large simulation campaigns called to the ELT. 
#  
#  The COMPASS platform can be used to carry a wide variety of simulations to both testspecific components 
#  of AO of the E-ELT (such as wavefront analysis device with a pyramid or elongated Laser star), and 
#  various systems configurations such as multi-conjugate AO.
#
#  COMPASS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the 
#  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
#  See the GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License along with COMPASS. 
#  If not, see <https://www.gnu.org/licenses/lgpl-3.0.txt>.

"""
script test to simulate a closed loop

Usage:
  closed_loop.py <parameters_filename> [options]

with 'parameters_filename' the path to the parameters file

Options:
  -h --help          Show this help message and exit
  --brahma           Distribute data with BRAHMA
  --bench            For a timed call
  -i, --interactive  keep the script interactive
  -d, --devices devices      Specify the devices
  -n, --niter niter  Number of iterations
  --DB               Use database to skip init phase
  -g, --generic      Use generic controller
  -f, --fast         Compute PSF only during monitoring
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
from docopt import docopt
from matplotlib import pyplot as plt
#plt.ion()
from matplotlib.colors import Normalize as Norm
from shesha.util.CovMap_from_Mat import Map_and_Mat
from scipy import stats as st

if __name__ == "__main__":
    arguments = docopt(__doc__)
    param_file = arguments["<parameters_filename>"]
    use_DB = False
    compute_tar_psf = not arguments["--fast"]
    
    # Get parameters from file
    if arguments["--bench"]:
        from shesha.supervisor.benchSupervisor import BenchSupervisor as Supervisor
    elif arguments["--brahma"]:
        from shesha.supervisor.canapassSupervisor import CanapassSupervisor as Supervisor
    else:
        from shesha.supervisor.compassSupervisor import CompassSupervisor as Supervisor

    if arguments["--DB"]:
        use_DB = True

    supervisor = Supervisor(param_file, use_DB=use_DB)

    if arguments["--devices"]:
        supervisor.config.p_loop.set_devices([
                int(device) for device in arguments["--devices"].split(",")
        ])
    if arguments["--generic"]:
        supervisor.config.p_controllers[0].set_type("generic")
        print("Using GENERIC controller...")

    supervisor.initConfig()
    if arguments["--niter"]:
        supervisor.loop(int(arguments["--niter"]), compute_tar_psf=compute_tar_psf)
    else:
        supervisor.loop(supervisor.config.p_loop.niter, compute_tar_psf=compute_tar_psf)

    if arguments["--interactive"]:
        from shesha.util.ipython_embed import embed
        from os.path import basename
        embed(basename(__file__), locals())

    sup=supervisor  
    sim=sup._sim  
    conf=sup.config  
    sup.reset  
    from shesha.util.writers import yao  
    from shesha.util import fits_io  
    from importlib import reload  
    from shesha.util import tao  
    #from shesha.util import psfMap  
    #tao.VARS["GPUIDS"]="1"   
    #tao.VARS["INPUTPATH"]="./"   
    #tao.VARS["TAOPATH"]="/home/hzhang/moao_dev/chameleon/build_ndoucet/testHalf/install/bin"  
    #tao.VARS["STARPU_FLAGS"]="STARPU_SCHED=dmdas STARPU_SILENT=1"  
    #tao.check()
    if sim.config.p_atmos.get_windspeed()[0]>100:
        nniter = 10000
    else:
        nniter = 50000   
    #dd,ss = tao.run(sup,tao.mcao,nIter=nniter,initialisation=1,nfilt=150,WFS="lgs",DM_TT=False,lgstt=0.0) 
    M = fits_io.fitsread("M_mcao.fits")
    sup.setCommandMatrix(M)
    sim.rtc.d_control[0].set_polc(True)
    dd,ss = sup._sim.loopPOLC(nniter)

    #### have to use

    ''' 
    rand_id = str(int(np.random.random(1)*100000))
    imat = np.array(sup._sim.rtc.d_control[0].d_imat)
    lgs0_y = sim.config.p_wfs_lgs[0].get_validpuppixy()
    lgs0_x = sim.config.p_wfs_lgs[0].get_validpuppixx()
    np.save("buffer/imat_"+rand_id+".npy",imat)
    #print("imat saved!")
    np.save("buffer/dd_"+rand_id+".npy",dd)
    np.save("buffer/ss_"+rand_id+".npy",ss)
    np.save("buffer/lgs0_y"+rand_id+".npy",lgs0_y)
    np.save("buffer/lgs0_x"+rand_id+".npy",lgs0_x)

    print("imat, dd, ss, lgs_xy saved with id = "+rand_id)
    
    dd_cov = dd@dd.T / nniter
    ss_cov = ss@ss.T / nniter
    pols   = (-imat@dd +ss).astype("float32")
    pols   = pols.T
    pols = pols - pols.mean(axis=0)
    cmm = pols.T@pols / nniter
    cmm_diag = cmm.diagonal()
    dd_diag = dd_cov.diagonal()
    ss_diag = ss_cov.diagonal()
    fig=plt.figure();plt.imshow(dd_cov);plt.colorbar();plt.savefig("fig/dd_cov_"+rand_id+".png");plt.close(fig)
    fig=plt.figure();plt.imshow(ss_cov);plt.colorbar();plt.savefig("fig/ss_cov_"+rand_id+".png");plt.close(fig)
    fig=plt.figure();
    for ii in range(9):
        plt.subplot(3,3,ii+1)
        if ii == 0 :
            plt.plot(cmm_diag);plt.title("All WFS cmm diag");
        else:
            npix = sim.config.p_wfs_lgs[ii-1].get_npix()
            plt.plot(cmm_diag[1148*2*(ii-1):1148*2*ii]);plt.title("LGS"+str(ii)+": "+str(npix)+" pix");plt.ylim((0.04,0.1));
    plt.subplots_adjust(wspace = 0.5,hspace = 0.4)
    plt.savefig("fig/all_wfs_cmm_diag_"+rand_id+".png");plt.close(fig)
    
    nn  = Norm(0.04,0.08)
    pos_sub = np.array([15,9,3,7,11,17,23,19]).astype(int)
    fig=plt.figure()
    for ii in range(8):
        plt.subplot(5,5,int(pos_sub[ii]))
        plt.scatter(lgs0_x,lgs0_y,c=cmm_diag[1148*(2*ii):1148*(2*ii+1)],marker='x',s=0.7,norm=nn)
        plt.colorbar();plt.title("LGS # "+str(ii+1))
    plt.subplots_adjust(wspace = 0.6,hspace = 0.4)
    plt.savefig("fig/cmm_xx_diag_colored_aligned"+rand_id+".png");plt.close(fig)

    fig=plt.figure()
    for ii in range(8):
        plt.subplot(5,5,int(pos_sub[ii]))
        plt.scatter(lgs0_x,lgs0_y,c=cmm_diag[1148*(2*ii)+1148:1148*(2*ii+1)+1148],marker='x',s=0.7,norm=nn)
        plt.colorbar();plt.title("LGS # "+str(ii+1))
    plt.subplots_adjust(wspace = 0.6,hspace = 0.4)
    plt.savefig("fig/cmm_yy_diag_colored_aligned"+rand_id+".png");plt.close(fig)
    
    print("all figs saved with id = "+rand_id+"!")


    ######## below is the comparison between analytical values
    Cmap,Cmat = Map_and_Mat(sup)
    Cmat_full = np.tril(Cmat)+np.tril(Cmat).T-np.diag(Cmat.diagonal())
    #sl,inte,r_v,p_v,std_err = st.linregress(cmm.flatten(),Cmat_full.flatten())
    #print("Full cmm, noisy numerical POLS v.s. analytical, TT included, slope = %.3f, inte = %.3f, r2 = %.3f"%(sl, inte, r_v))
    sl  = np.zeros(64)
    inte  = np.zeros(64)
    r_v  = np.zeros(64)
    p_v  = np.zeros(64)
    std_err  = np.zeros(64)
    for i1 in range(8): 
        for i2 in range(8): 
            cmm1 = cmm[2296*i1:2296*(i1+1),2296*i2:2296*(i2+1)] 
            cmm2 = Cmat_full[2296*i1:2296*(i1+1),2296*i2:2296*(i2+1)]
            ii   = i1*8+i2 
            sl[ii],inte[ii],r_v[ii],p_v[ii],std_err[ii] = st.linregress(cmm1.flatten(),cmm2.flatten()) 
            print("wfs #"+str(i1)+" with wfs #"+str(i2)+", TT included, slope = %.5f, inte = %.5f, r2 = %.5f"%(sl[ii], inte[ii], r_v[ii])) 

    
    #quit

    buffer = supervisor._sim.loopPOLC(100000)
    ran_id = str(int(np.random.random(1)*100000))
    np.save("buffer_"+ran_id+".npy",buffer)
    
    Cmmk = np.array([supervisor._sim.Cmmk(1000,100,k) for k in range(1)])
    cmm0 = Cmmk[0,:,:]
    cmm_diag = cmm0.diagonal()
    plt.figure()
    plt.subplot(2,3,1);plt.plot(cmm_diag); plt.title("All WFS cmm diag"); plt.ylim((0,0.2))
    plt.subplot(2,3,2);plt.plot(cmm_diag[:1148*2]); plt.title("LGS1: 6 pix")
    plt.subplot(2,3,3);plt.plot(cmm_diag[1148*2:1148*4]); plt.title("LGS2: 9 pix")
    plt.subplot(2,3,4);plt.plot(cmm_diag[1148*4:1148*6]); plt.title("LGS3: 12 pix")
    plt.subplot(2,3,5);plt.plot(cmm_diag[1148*6:1148*8]); plt.title("LGS4: 15 pix")
    plt.subplot(2,3,6);plt.plot(cmm_diag[1148*8:1148*10]); plt.title("LGS5: 18 pix")
    plt.subplots_adjust(wspace = 0.5,hspace = 0.4)
    '''
