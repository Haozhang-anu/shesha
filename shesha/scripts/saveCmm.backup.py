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
from astropy.io import fits
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
#    from shesha.util import tao  
#    #from shesha.util import psfMap  
#    tao.VARS["GPUIDS"]="1"   
#    tao.VARS["INPUTPATH"]="./"   
#    tao.VARS["TAOPATH"]="/home/hzhang/moao_dev/chameleon/build_ndoucet/testHalf/install/bin"  
#    tao.VARS["STARPU_FLAGS"]="STARPU_SCHED=dmdas STARPU_SILENT=1"  
#    tao.check()
    M = fits_io.fitsread("./M_mcao.fits")
    sup.setCommandMatrix(M)
    sim.rtc.d_control[0].set_polc(True)
    sim.rtc.d_control[0].set_imat(conf.p_controllers[0]._imat)
    sim.rtc.d_control[0].set_gain(conf.p_controllers[0].gain)
    if sim.config.p_atmos.get_windspeed()[0]>100:
        nniter = 10000
    else:
        nniter = 50000
    nniter = 1000
    imat = np.array(sup._sim.rtc.d_control[0].d_imat)
    np.save("/home/hongy0a/gitrepo/shesha/buffer/imat_hzhang.npy",imat)
    for ii in range(20):
        print("loop ", ii)
        dd=[]
        ss=[]
        save_id = str(nniter)+"_"+str(ii+1)
        dd,ss = sup._sim.loopPOLC(nniter)
        np.save("buffer/dd_"+save_id+".npy",dd)
        np.save("buffer/ss_"+save_id+".npy",ss)
#    dd,ss = tao.run(sup,tao.mcao,nIter=nniter,initialisation=1,nfilt=150,WFS="lgs",DM_TT=False,lgstt=0.0) 


