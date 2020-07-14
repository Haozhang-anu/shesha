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
  --init             Do imat 
"""
import numpy as np
import ATA as ata
from docopt import docopt
import time
import pickle as pkl
import sys

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

    #if arguments["--init"]:
    supervisor.config.p_controllers[0].set_type("ls")

    if arguments["--devices"]:
        supervisor.config.p_loop.set_devices([
                int(device) for device in arguments["--devices"].split(",")
        ])
    if arguments["--generic"]:
        supervisor.config.p_controllers[0].set_type("generic")
        print("Using GENERIC controller...")

    supervisor.initConfig()

    if arguments["--interactive"]:
        from shesha.util.ipython_embed import embed
        from os.path import basename
        embed(basename(__file__), locals())
    
    t1 = time.time() 
    monitoring_freq = 1
    sim = supervisor._sim
    
    param_dict = {}
    if hasattr(sim.config,'NLGS'):
        M_lgs = sim.config.NLGS
        M_ngs = sim.config.NNGS 
        M = M_lgs + M_ngs
    else:
        print("WARNING: NLGS and NNGS should be defined in parameter file!\nUsing flimsy counting method instead.")
        lgs_nxsub = sim.config.p_wfss[0].nxsub
        M_lgs = np.sum([wfs.nxsub==lgs_nxsub for wfs in sim.config.p_wfss])
        M = len(sim.config.p_wfss)
        M_ngs = M-M_lgs

    puppix_size = sim.config.p_geom.pupdiam
    pup_size = sim.config.p_tel.diam
    puppix_cent = sim.config.p_geom.cent
    ndms = len(sim.config.p_dms)
    K_y = []
    K_x = []
    for ni in range(ndms):
        dm = sim.config.p_dms[ni]
        if not (dm.type=="pzt"):
            sys.exit(f"Unsupported DM type ({ni:d} : {dm.type:s})")
        xpos = (dm._xpos-puppix_cent)*pup_size/puppix_size
        ypos = (dm._ypos-puppix_cent)*pup_size/puppix_size
        K_y.append(xpos) # Fix this nightmare
        K_x.append(ypos)
        # ata.fitswrite("dmx%d.fits"%(ni,),xpos,overwrite=True)
        # ata.fitswrite("dmy%d.fits"%(ni,),ypos,overwrite=True)
    
    imat = sim.config.p_controllers[0].get_imat()
    if M_lgs > 0:
        imat_LGS = imat[:sim.config.p_wfss[0].get_nvalid()*2,
                        :sim.config.p_dms[0].get_ntotact()]
    else:
        imat_LGS = None

    if M_ngs > 0:
        imat_NGS = imat[-sim.config.p_wfss[M_lgs].get_nvalid()*2:,
                        :sim.config.p_dms[0].get_ntotact()]
    else:
        imat_NGS = None

    gspos = 4.848e-6*np.array([[
        wfs.get_xpos(),
        wfs.get_ypos()]
        for wfs in sim.config.p_wfss[:M]])
    
    N = ndms
    filter_LGS_TT = sim.config.p_centroiders[0].filter_TT
    whichCsml = ["L" for x in range(M_lgs)] + ["N" for x in range(M_ngs)]
    h_gs = [wfs.gsalt if wfs.gsalt > 0 else np.inf 
            for wfs in sim.config.p_wfss[:M]]

    param_dict["M_lgs"] = M_lgs
    param_dict["M_ngs"] = M_ngs
    param_dict["M"] = M
    param_dict["N"] = N
    param_dict["h_dm"] = [dm.alt for dm in sim.config.p_dms]
    param_dict["zenithangle"] = sim.config.p_geom.zenithangle*np.pi/180
    param_dict["h_gs"] = h_gs
    param_dict["whichCsml"] = whichCsml
    param_dict["filter_LGS_TT"] = filter_LGS_TT
    param_dict["N_meta"] = sim.config.p_atmos.nscreens
    param_dict["weights"] = sim.config.p_atmos.frac
    param_dict["h"] = sim.config.p_atmos.alt
    param_dict["vr"] = sim.config.p_atmos.windspeed
    param_dict["vtheta"] = sim.config.p_atmos.winddir
    param_dict["L0"] = sim.config.p_atmos.L0
    param_dict["r0_global"] = sim.config.p_atmos.r0
    param_dict["ittime"] = sim.config.p_loop.ittime
    param_dict["C_sml_L"] = imat_LGS
    param_dict["C_sml_N"] = imat_NGS
    param_dict["K_y"] = K_y
    param_dict["K_x"] = K_x
    param_dict["gspos"] = gspos
    param_dict["yc"] = gspos[:,0]
    param_dict["xc"] = gspos[:,1]
    param_dict["infl"] = [dm.coupling for dm in sim.config.p_dms]
    param_dict["sim_name"] = param_file.split("/")[-1].split(".")[0] 
    param_dict["D"] = sim.config.p_tel.diam
    if hasattr(sim.config.p_loop,'R_TRUE'):
        param_dict["R_true"] = sim.config.p_loop.R_TRUE
    else:
        param_dict["R_true"] = 1
    pkl.dump(param_dict,open("params_"+param_dict["sim_name"]+".pkl","wb"))


