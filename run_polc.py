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

import numpy as np
from docopt import docopt
import time
import ATA as ata

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
    supervisor.initConfig()

    if arguments["--devices"]:
        supervisor.config.p_loop.set_devices([
                int(device) for device in arguments["--devices"].split(",")
        ])
    supervisor.config.p_controllers[0].set_type("generic")
    print("Using GENERIC controller...")

    if arguments["--interactive"]:
        from shesha.util.ipython_embed import embed
        from os.path import basename
        embed(basename(__file__), locals())
    
    sim = supervisor._sim
    simname = param_file.split("/")[-1].split(".")[0]
    def reloadAndRun(n=10000,monitoring_freq=10,gain=0.4):
        supervisor.reset()
        supervisor.set2MatricesLaw(0)
        cmat = ata.fitsread(simname+"_R.fits").astype(np.float32)
        matE = ata.fitsread(simname+"_E.fits").astype(np.float32)
        size = matE.shape[0]
        decayFactor = np.ones(size, dtype=np.float32)
        mgain = np.ones(size, dtype=np.float32)
        matE = (1.0-gain)*np.eye(size, dtype=np.float32)+gain*matE
        cmat = gain*cmat
        sim.rtc.d_control[0].set_decayFactor(decayFactor)
        sim.rtc.d_control[0].set_mgain(mgain)
        sim.rtc.d_control[0].set_cmat(cmat)
        sim.rtc.d_control[0].set_matE(matE)
        print("""
        Starting %d loops
        """ %(n,))
        sim.loop(n,monitoring_freq=monitoring_freq)
    reloadAndRun(n=1000,monitoring_freq=100)
