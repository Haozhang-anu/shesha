## @package   shesha.sim.simulator
## @brief     Simulator class definition must be instantiated for running a COMPASS simulation script easily
## @author    COMPASS Team <https://github.com/ANR-COMPASS>
## @version   4.4.2
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

import sys
import os

# Binding struct for all initializers - good for subclassing modules
from shesha.init.geom_init import tel_init
from shesha.init.atmos_init import atmos_init
from shesha.init.rtc_init import rtc_init
from shesha.init.dm_init import dm_init
from shesha.init.target_init import target_init
from shesha.init.wfs_init import wfs_init
from shesha.util.utilities import load_config_from_file, load_config_from_module

import shesha.constants as scons
import shesha.util.hdf5_util as h5u

import time
import numpy as np

from typing import Iterable, Any, Dict
from shesha.sutra_wrap import Sensors, Dms, Rtc_FFF as Rtc, Atmos, Telescope, Target, carmaWrap_context


class Simulator:
    """
    The Simulator class is self sufficient for running a COMPASS simulation
    Initializes and run a COMPASS simulation
    """

    def __init__(self, filepath: str = None, use_DB: bool = False) -> None:
        """
        Initializes a Simulator instance

        :parameters:
            filepath: (str): (optional) path to the parameters file
            use_DB: (bool): (optional) flag to use dataBase system
        """
        self.is_init = False  # type: bool
        self.loaded = False  # type: bool
        self.config = None  # type: Any # types.ModuleType ?
        self.iter = 0  # type: int

        self.c = None  # type: carmaWrap_context
        self.atm = None  # type: Atmos
        self.tel = None  # type: Telescope
        self.tar = None  # type: Target
        self.rtc = None  # type: Rtc
        self.wfs = None  # type: Sensors
        self.dms = None  # type: Dms

        self.matricesToLoad = {}  # type: Dict[str,str]
        self.use_DB = use_DB  # type: bool

        if filepath is not None:
            self.load_from_file(filepath)

    def __str__(self) -> str:
        """
        Print the objects created in the Simulator instance
        """
        s = ""
        if self.is_init:
            s += "====================\n"
            s += "Objects initialzed on GPU:\n"
            s += "--------------------------------------------------------\n"

            if self.atm is not None:
                s += self.atm.__str__() + '\n'
            if self.wfs is not None:
                s += self.wfs.__str__() + '\n'
            if self.dms is not None:
                s += self.dms.__str__() + '\n'
            if self.tar is not None:
                s += self.tar.__str__() + '\n'
            if self.rtc is not None:
                s += self.rtc.__str__() + '\n'
        else:
            s += "Simulator is not initialized."

        return s

    def force_context(self) -> None:
        """
        Active all the GPU devices specified in the parameters file
        """
        if self.loaded and self.c is not None:
            current_Id = self.c.activeDevice
            for devIdx in range(len(self.config.p_loop.devices)):
                self.c.set_activeDeviceForce(devIdx)
            self.c.set_activeDevice(current_Id)

    def load_from_file(self, filepath: str) -> None:
        """
        Load the parameters from the parameters file

        :parameters:
            filepath: (str): path to the parameters file

        """
        load_config_from_file(self, filepath)

    def load_from_module(self, filepath: str) -> None:
        """
        Load the parameters from the parameters file

        :parameters:
            filepath: (str): path to the parameters file

        """
        load_config_from_module(self, filepath)

    def clear_init(self) -> None:
        """
        Delete objects initialized in a previous simulation
        """
        if self.loaded and self.is_init:
            self.iter = 0

            del self.atm
            self.atm = None
            del self.tel
            self.tel = None
            del self.tar
            self.tar = None
            del self.rtc
            self.rtc = None
            del self.wfs
            self.wfs = None
            del self.dms
            self.dms = None

            # del self.c  # What is this supposed to do ... ?
            # self.c = None

        self.is_init = False

    def init_sim(self) -> None:
        """
        Initializes the simulation by creating all the sutra objects that will be used
        """
        if not self.loaded:
            raise ValueError("Config must be loaded before call to init_sim")
        if (self.config.simul_name is not None and self.use_DB):
            param_dict = h5u.params_dictionary(self.config)
            self.matricesToLoad = h5u.checkMatricesDataBase(
                    os.environ["SHESHA_ROOT"] + "/data/dataBase/", self.config,
                    param_dict)
        # self.c = carmaWrap_context(devices=self.config.p_loop.devices)
        if (self.config.p_loop.devices.size > 1):
            self.c = carmaWrap_context.get_instance_ngpu(self.config.p_loop.devices.size,
                                                         self.config.p_loop.devices)
        else:
            self.c = carmaWrap_context.get_instance_1gpu(self.config.p_loop.devices[0])
        # self.force_context()

        if self.config.p_tel is None or self.config.p_geom is None:
            raise ValueError("Telescope geometry must be defined (p_geom and p_tel)")

        if self.config.p_atmos is not None:
            r0 = self.config.p_atmos.r0
        else:
            raise ValueError('A r0 value through a Param_atmos is required.')

        if self.config.p_loop is not None:
            ittime = self.config.p_loop.ittime
        else:
            raise ValueError(
                    'An ittime (iteration time in seconds) value through a Param_loop is required.'
            )

        self._tel_init(r0, ittime)

        self._atm_init(ittime)

        self._dms_init()

        self._tar_init()

        self._wfs_init()

        self._rtc_init(ittime)

        self.is_init = True
        if self.use_DB:
            h5u.validDataBase(os.environ["SHESHA_ROOT"] + "/data/dataBase/",
                              self.matricesToLoad)

    def _tel_init(self, r0: float, ittime: float) -> None:
        """
        Initializes the Telescope object in the simulator
        """
        print("->tel")
        self.tel = tel_init(self.c, self.config.p_geom, self.config.p_tel, r0, ittime,
                            self.config.p_wfss)

    def _atm_init(self, ittime: float) -> None:
        """
        Initializes the Atmos object in the simulator
        """
        if self.config.p_atmos is not None:
            #   atmos
            print("->atmos")
            self.atm = atmos_init(self.c, self.config.p_atmos, self.config.p_tel,
                                  self.config.p_geom, ittime, p_wfss=self.config.p_wfss,
                                  p_targets=self.config.p_targets,
                                  dataBase=self.matricesToLoad, use_DB=self.use_DB)
        else:
            self.atm = None

    def _dms_init(self) -> None:
        """
        Initializes the DMs object in the simulator
        """
        if self.config.p_dms is not None:
            #   dm
            print("->dm")
            self.dms = dm_init(self.c, self.config.p_dms, self.config.p_tel,
                               self.config.p_geom, self.config.p_wfss)
        else:
            self.dms = None

    def _tar_init(self) -> None:
        """
        Initializes the Target object in the simulator
        """
        if self.config.p_targets is not None:
            print("->target")
            self.tar = target_init(self.c, self.tel, self.config.p_targets,
                                   self.config.p_atmos, self.config.p_tel,
                                   self.config.p_geom, self.config.p_dms, brahma=False)
        else:
            self.tar = None

    def _wfs_init(self) -> None:
        """
        Initializes the WFS object in the simulator
        """
        if self.config.p_wfss is not None:
            print("->wfs")
            self.wfs = wfs_init(self.c, self.tel, self.config.p_wfss, self.config.p_tel,
                                self.config.p_geom, self.config.p_dms,
                                self.config.p_atmos)
        else:
            self.wfs = None

    def _rtc_init(self, ittime: float) -> None:
        """
        Initializes the Rtc object in the simulator
        """
        if self.config.p_controllers is not None or self.config.p_centroiders is not None:
            print("->rtc")
            #   rtc
            self.rtc = rtc_init(self.c, self.tel, self.wfs, self.dms, self.atm,
                                self.config.p_wfss, self.config.p_tel,
                                self.config.p_geom, self.config.p_atmos, ittime,
                                self.config.p_centroiders, self.config.p_controllers,
                                self.config.p_dms, brahma=False,
                                dataBase=self.matricesToLoad, use_DB=self.use_DB)
        else:
            self.rtc = None


    def next_slopes(self,wfs_trace=None,controllers=None):

        if wfs_trace is None and self.wfs is not None:
            wfs_trace = range(len(self.wfs.d_wfs))

        if controllers is None and self.rtc.d_control is not None:
            controllers = range(len(self.rtc.d_control))
        
        self.moveAtmos()
        for w in wfs_trace:
            self.raytraceWfs(w, ["atmos", "tel", "ncpa"])
            self.compWfsImage(w)
        for ncontrol in controllers:
            self.doCentroids(ncontrol)
        return  np.array(self.rtc.d_control[0].d_centroids)


    def next(self, *, move_atmos: bool = True, see_atmos: bool = True, nControl: int = 0,
             tar_trace: Iterable[int] = None, wfs_trace: Iterable[int] = None,
             do_control: bool = True, apply_control: bool = True,
             compute_tar_psf: bool = True) -> None:
        '''
        Iterates the AO loop, with optional parameters

        :parameters:
             move_atmos: (bool): move the atmosphere for this iteration, default: True

             nControl: (int): Controller number to use, default 0 (single control configurations)

             tar_trace: (None or list[int]): list of targets to trace. None equivalent to all.

             wfs_trace: (None or list[int]): list of WFS to trace. None equivalent to all.

             apply_control: (bool): (optional) if True (default), apply control on DMs
        '''
        if tar_trace is None and self.tar is not None:
            tar_trace = range(len(self.tar.d_targets))
        if wfs_trace is None and self.wfs is not None:
            wfs_trace = range(len(self.wfs.d_wfs))

        if move_atmos and self.atm is not None:
            self.moveAtmos()

        if (
                self.config.p_controllers is not None and
                self.config.p_controllers[nControl].type == scons.ControllerType.GEO):
            if tar_trace is not None:
                for t in tar_trace:
                    if see_atmos:
                        self.raytraceTar(t, ["atmos", "tel"])
                    else:
                        self.raytraceTar(t, "tel")

                    if self.rtc is not None:
                        self.doControl(nControl)
                        self.raytraceTar(t, ["dm", "ncpa"], rst=False)
                        self.applyControl(nControl)
        else:
            if tar_trace is not None:
                for t in tar_trace:
                    if see_atmos:
                        self.raytraceTar(t, "all")
                    else:
                        self.raytraceTar(t, ["tel", "dm", "ncpa"])
            if wfs_trace is not None:
                for w in wfs_trace:
                    if see_atmos:
                        self.raytraceWfs(w, ["atmos", "tel", "ncpa"])
                    else:
                        self.raytraceWfs(w, ["tel", "ncpa"])

                    if not self.config.p_wfss[w].openloop and self.dms is not None:
                        self.raytraceWfs(w, "dm", rst=False)
                    self.compWfsImage(w)
            if do_control and self.rtc is not None:
                for ncontrol in range(len(self.rtc.d_control)):
                    if self.rtc.d_control[ncontrol].type != scons.ControllerType.GEO:
                        if self.rtc.d_control[
                                ncontrol].centro_idx is None:  # RTC standalone case
                            self.doCalibrate_img(ncontrol)
                        self.doCentroids(ncontrol)
                        self.doControl(ncontrol)
                        self.doClipping(ncontrol)

                        if apply_control:
                            self.applyControl(ncontrol)

        if compute_tar_psf:
            for nTar in tar_trace:
                self.compTarImage(nTar)
                self.compStrehl(nTar)

        self.iter += 1

    def print_strehl(self, monitoring_freq: int, t1: float, nCur: int = 0, nTot: int = 0,
                     nTar: int = 0):
        framerate = monitoring_freq / t1
        strehl = self.getStrehl(nTar)
        etr = (nTot - nCur) / framerate
        print("%d \t %.3f \t  %.3f\t     %.1f \t %.1f" % (nCur + 1, strehl[0], strehl[1],
                                                          etr, framerate))

    def loop(self, n: int = 1, monitoring_freq: int = 100, compute_tar_psf: bool = True,
             abortThresh: float = 0. , **kwargs):
        """
        Perform the AO loop for n iterations

        :parameters:
            n: (int): (optional) Number of iteration that will be done
            monitoring_freq: (int): (optional) Monitoring frequency [frames]
        """
        if not compute_tar_psf:
            print("WARNING: Target PSF will be computed (& accumulated) only during monitoring"
                  )

        print("----------------------------------------------------")
        print("iter# | S.E. SR | L.E. SR | ETR (s) | Framerate (Hz)")
        print("----------------------------------------------------")
        # self.next(**kwargs)
        t0 = time.time()
        t1 = time.time()
        if n == -1:
            i = 0
            while (True):
                self.next(compute_tar_psf=compute_tar_psf, **kwargs)
                if ((i + 1) % monitoring_freq == 0):
                    if not compute_tar_psf:
                        self.compTarImage()
                        self.compStrehl()
                    self.print_strehl(monitoring_freq, time.time() - t1, i, i)
                    t1 = time.time()
                    SRmax=max([self.getStrehl(tar)[0] for tar in range(self.tar.ntargets) ])
                    if(SRmax<abortThresh):
                        print("SR SE too low: stopping the loop")
                        break
                i += 1

        for i in range(n):
            self.next(compute_tar_psf=compute_tar_psf, **kwargs)
            if ((i + 1) % monitoring_freq == 0):
                if not compute_tar_psf:
                    self.compTarImage()
                    self.compStrehl()
                self.print_strehl(monitoring_freq, time.time() - t1, i, n)
                t1 = time.time()
                SRmax=max([self.getStrehl(tar)[0] for tar in range(self.tar.ntargets) ])
                if(SRmax<abortThresh):
                    print("SR SE too low: stopping the loop")
                    break
        t1 = time.time()
        print(" loop execution time:", t1 - t0, "  (", n, "iterations), ", (t1 - t0) / n,
              "(mean)  ", n / (t1 - t0), "Hz")


    def Cmm(self, n: int = 1, monitoring_freq: int = 100,wfs_trace=None,controllers=None):

        if controllers is None and self.rtc.d_control is not None:
            controllers = range(len(self.rtc.d_control))



        print("----------------------------------------------------")
        print("iter# | S.E. SR | L.E. SR | ETR (s) | Framerate (Hz)")
        print("----------------------------------------------------")
        # self.next(**kwargs)
        t0 = time.time()
        t1 = time.time()
        nSlopes=np.array(self.rtc.d_control[0].d_centroids).size

        slopes=np.zeros((nSlopes,monitoring_freq))
        Cmm=np.zeros((nSlopes,nSlopes))
        for i in range(n):
            m=i % monitoring_freq
            slopes[:,m]=self.next_slopes(wfs_trace,controllers)
            if(m == monitoring_freq-1 or i==n-1):
                Cmm+=slopes[:,:m+1].dot(slopes[:,:m+1].T)/n
                self.print_strehl(monitoring_freq, time.time() - t1, i, n)
                t1 = time.time()

        return Cmm

    def Cmmk(self, n: int = 1, monitoring_freq: int = 100, delay: int = 0, wfs_trace=None,controllers=None):

        if controllers is None and self.rtc.d_control is not None:
            controllers = range(len(self.rtc.d_control))

        if delay<0:
            raise ValueError("Delay parameter (currently) must be positive")   

        print("----------------------------------------------------")
        print("iter# | S.E. SR | L.E. SR | ETR (s) | Framerate (Hz)")
        print("----------------------------------------------------")
        # self.next(**kwargs)
        t0 = time.time()
        t1 = time.time()
        nSlopes=np.array(self.rtc.d_control[0].d_centroids).size

        slopes=np.zeros((nSlopes,monitoring_freq))
        cmmk=np.zeros((nSlopes,nSlopes))
        lazy_counter = 0
        for i in range(n):
            m=i % monitoring_freq
            slopes[:,m]=self.next_slopes(wfs_trace,controllers)
            lazy_counter += 1
            if(m == monitoring_freq-1 or i==n-1):
                cmmk += slopes[:,:m+1-delay].dot(slopes[:,delay:m+1].T)
                self.print_strehl(monitoring_freq, time.time() - t1, i, n)
                t1 = time.time()
                lazy_counter -= delay
        cmmk /= lazy_counter
        return cmmk

    def loopPOLC(self, n: int = 1, monitoring_freq: int = 100, compute_tar_psf: bool = True,
             abortThresh: float = 0. , **kwargs):
        """
        Perform the AO loop for n iterations, and returns the POL Slope buffer

        :parameters:
            n: (int): (optional) Number of iteration that will be done
            monitoring_freq: (int): (optional) Monitoring frequency [frames]
		:returns:
			slopes: (2d np.array): each column is a sample of pseudo-open
					loop slopes. One column per it.
		###### BIG TODO!!! ######
		Verify that the slopes are aligned with the commands. I think they will
		not be. If they are not, then use a rolling buffer of the commands, and
		apply the correct one inside the "pols = ..." line
        """
        if not compute_tar_psf:
            print("WARNING: Target PSF will be computed (& accumulated) only during monitoring"
                  )

        print("----------------------------------------------------")
        print("iter# | S.E. SR | L.E. SR | ETR (s) | Framerate (Hz)")
        print("----------------------------------------------------")
        # self.next(**kwargs)
        t0 = time.time()
        t1 = time.time()
        dm_hist = np.zeros([self.rtc.d_control[0].nactu,n])
        cls_hist = np.zeros([self.rtc.d_control[0].nslope,n])
        if n == -1:
            i = 0
            while (True):
                self.next(compute_tar_psf=compute_tar_psf, **kwargs)
                if ((i + 1) % monitoring_freq == 0):
                    if not compute_tar_psf:
                        self.compTarImage()
                        self.compStrehl()
                    self.print_strehl(monitoring_freq, time.time() - t1, i, i)
                    t1 = time.time()
                    SRmax=max([self.getStrehl(tar)[0] for tar in range(self.tar.ntargets) ])
                    if(SRmax<abortThresh):
                        print("SR SE too low: stopping the loop")
                        break
                i += 1
        #imat = np.array(self.rtc.d_control[0].d_imat)
        #np.save("imat_"+self.config.simul_name+".npy",imat)
        #print("imat saved!")
        for i in range(n):
            self.next(compute_tar_psf=compute_tar_psf, **kwargs)
            dmcommand = np.array(self.rtc.d_control[0].d_comClipped)
            centroids = np.array(self.rtc.d_control[0].d_centroids)
            '''
            pols = np.array(self.rtc.d_control[0].d_imat) @ \
                    np.array(self.rtc.d_control[0].d_comClipped) + \
                    np.array(self.rtc.d_control[0].d_centroids)
            slopes[:,i] = pols
            '''
            dm_hist[:,i] = dmcommand
            cls_hist[:,i] = centroids

            if ((i + 1) % monitoring_freq == 0):
                if not compute_tar_psf:
                    self.compTarImage()
                    self.compStrehl()
                self.print_strehl(monitoring_freq, time.time() - t1, i, n)
                t1 = time.time()
                SRmax=max([self.getStrehl(tar)[0] for tar in range(self.tar.ntargets) ])
                if(SRmax<abortThresh):
                    print("SR SE too low: stopping the loop")
                    break
        t1 = time.time()
        print(" loop execution time:", t1 - t0, "  (", n, "iterations), ", (t1 - t0) / n,
              "(mean)  ", n / (t1 - t0), "Hz")
        return dm_hist,cls_hist



#  ██╗    ██╗██████╗  █████╗ ██████╗
#  ██║    ██║██╔══██╗██╔══██╗██╔══██╗
#  ██║ █╗ ██║██████╔╝███████║██████╔╝
#  ██║███╗██║██╔══██╗██╔══██║██╔═══╝
#  ╚███╔███╔╝██║  ██║██║  ██║██║
#   ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
#

    def moveAtmos(self):
        """
        Move the turbulent layers according to wind speed and direction for a single iteration
        """
        self.atm.move_atmos()

    def raytraceTar(self, tarNum, layers: list, rst: bool = True):
        """
        Performs the raytracing operation to obtain the phase seen by the tarNum target
        The phase screen is reset before the operations if rst is not set to False

        Parameters
        ------------
        tarNum: (int): target index
        layers: (list): list of string containing the layers to raytrace through.
                        Accepted are : "all" -> raytrace through all layers
                                       "atmos" -> raytrace through turbulent layers only
                                       "dm" -> raytrace through DM shape only
                                       "ncpa" -> raytrace through NCPA only
                                       "tel" -> raytrace through telescope aberrations only
        rst: (bool): reset the phase screen before raytracing (default = True)
        """
        target = self.tar.d_targets[tarNum]
        if (isinstance(layers, str)):
            layers = [layers]
        self._raytrace(target, layers, rst=rst)

    def raytraceWfs(self, wfsNum, layers: list, rst: bool = True):
        """
        Performs the raytracing operation to obtain the phase seen by the wfsNum Wfs
        The phase screen is reset before the operations if rst is not set to False

        Parameters
        ------------
        wfsNum: (int): wfs index
        layers: (list): list of string containing the layers to raytrace through.
                        Accepted are : "all" -> raytrace through all layers
                                       "atmos" -> raytrace through turbulent layers only
                                       "dm" -> raytrace through DM shape only
                                       "ncpa" -> raytrace through NCPA only
                                       "tel" -> raytrace through telescope aberrations only
        rst: (bool): reset the phase screen before raytracing (default = True)
        """
        gs = self.wfs.d_wfs[wfsNum].d_gs
        if (isinstance(layers, str)):
            layers = [layers]
        self._raytrace(gs, layers, rst=rst)

    def _raytrace(self, source, layers: list, rst: bool = True):
        """
        Performs the raytracing operation to obtain the phase screen of the given sutra_source

        Parameters
        ------------
        source : (sutra_source): Source object
        layers: (list): list of string containing the layers to raytrace through.
                Accepted are : "all" -> raytrace through all layers
                                "atmos" -> raytrace through turbulent layers only
                                "dm" -> raytrace through DM shape only
                                "ncpa" -> raytrace through NCPA only
                                "tel" -> raytrace through telescope aberrations only
        rst: (bool): reset the phase screen before raytracing (default = True)
        """
        if (rst):
            source.d_phase.reset()

        for s in layers:
            if (s == "all"):
                source.raytrace(self.tel, self.atm, self.dms)
            elif (s == "atmos"):
                source.raytrace(self.atm)
            elif (s == "dm"):
                source.raytrace(self.dms)
            elif (s == "tel"):
                source.raytrace(self.tel)
            elif (s == "ncpa"):
                source.raytrace()
            else:
                raise ValueError("Unknown layer type : " + str(s) +
                                 ". See help for accepted layers")

    def compWfsImage(self, wfsNum: int = 0, noise: bool = True):
        """
        Computes the image produced by the WFS from its phase screen

        Parameters
        ------------
        wfsNum: (int): wfs index
        """
        self.wfs.d_wfs[wfsNum].comp_image(noise)

    def compTarImage(self, tarNum: int = 0, puponly: int = 0, compLE: bool = True):
        """
        Computes the PSF

        Parameters
        ------------
        tarNum: (int): (optionnal) target index (default=0)
        puponly: (int): (optionnal) if set to 1, computes Airy (default=0)
        compLE: (bool): (optionnal) if True, the computed image is taken into account in long exposure image (default=True)
        """
        self.tar.d_targets[tarNum].comp_image(puponly, compLE)

    def compStrehl(self, tarNum: int = 0, do_fit: bool = True):
        """
        Computes the Strehl ratio

        Parameters
        ------------
        tarNum: (int): (optionnal) target index (default 0)
        """
        self.tar.d_targets[tarNum].comp_strehl(do_fit)

    def doControl(self, nControl: int, n: int = 0, wfs_direction: bool = False):
        '''
        Computes the command from the Wfs slopes

        Parameters
        ------------
        nControl: (int): controller index
        n: (int) : target or wfs index (only used with GEO controller)
        '''
        if (self.rtc.d_control[nControl].type == scons.ControllerType.GEO):
            if (wfs_direction):
                self.rtc.d_control[nControl].comp_dphi(self.wfs.d_wfs[n].d_gs,
                                                       wfs_direction)
            else:
                self.rtc.d_control[nControl].comp_dphi(self.tar.d_targets[n], False)
        self.rtc.do_control(nControl)

    def doCalibrate_img(self, nControl: int):
        '''
        Computes the calibrated image from the Wfs image

        Parameters
        ------------
        nControl: (int): controller index
        '''
        self.rtc.do_calibrate_img(nControl)

    def doCentroids(self, nControl: int):
        '''
        Computes the centroids from the Wfs image

        Parameters
        ------------
        nControl: (int): controller index
        '''
        self.rtc.do_centroids(nControl)

    def doCentroidsGeom(self, nControl: int):
        '''
        Computes the centroids geom from the Wfs image

        Parameters
        ------------
        nControl: (int): controller index
        '''
        self.rtc.do_centroids_geom(nControl)

    def applyControl(self, nControl: int, compVoltage: bool = True):
        """
        Computes the final voltage vector to apply on the DM by taking into account delay and perturbation voltages, and shape the DMs

        Parameters
        ------------
        nControl: (int): controller index
        compVoltage: (bool): If True (default), computes the voltage vector from the command one (delay + perturb). Else, directly applies the current voltage vector
        """
        self.rtc.apply_control(nControl, compVoltage)

    def doClipping(self, nControl: int):
        '''
        Clip the commands between vmin and vmax values set in the RTC

        Parameters
        ------------
        nControl: (int): controller index
        '''
        self.rtc.do_clipping(nControl)

    def getStrehl(self, numTar: int, do_fit: bool = True):
        '''
        Return the Strehl Ratio of target number numTar as [SR short exp., SR long exp., np.var(phiSE), np.var(phiLE)]

        Parameters
        ------------
        numTar: (int): target index
        '''
        src = self.tar.d_targets[numTar]
        src.comp_strehl(do_fit)
        avgVar = 0
        if (src.phase_var_count > 0):
            avgVar = src.phase_var_avg / src.phase_var_count
        return [src.strehl_se, src.strehl_le, src.phase_var, avgVar]
    def loopLQG(self, n: int = 1, monitoring_freq: int = 100, 
                compute_tar_psf: bool = True, abortThresh: float = 0. ,
                A = None,  L = None, K = None , G = None, **kwargs):
        """
        Perform the AO loop using an LQG structure for n iterations

        :parameters:
            n: (int): (optional) Number of iteration that will be done
            monitoring_freq: (int): (optional) Monitoring frequency [frames]
            A: Closed-loop A matrix
            L: Closed-loop L matrix
            K: Closed-loop K matrix
            G: Closed-loop G matrix
        """
        self.x   = None
        self.s   = None
        self.u   = None
        self.um1 = None
        self.srbuffer = []

        if A is None or L is None or \
           K is None or G is None:
            raise TypeError("A, L, G, and K matrices must be passed into the loop function")
        
        self.A = A
        self.L = L
        self.K = K
        self.G = G

        if not compute_tar_psf:
            print("WARNING: Target PSF will be computed (& accumulated) only during monitoring"
                  )

        print("----------------------------------------------------")
        print("iter# | S.E. SR | L.E. SR | ETR (s) | Framerate (Hz)")
        print("----------------------------------------------------")
        # self.next(**kwargs)
        t0 = time.time()
        t1 = time.time()
        for i in range(n):
            self.nextLQG(compute_tar_psf=compute_tar_psf, **kwargs)
            if ((i + 1) % monitoring_freq == 0):
                if not compute_tar_psf:
                    self.compTarImage()
                    self.compStrehl()
                self.print_strehl(monitoring_freq, time.time() - t1, i, n)
                t1 = time.time()
                SRmax=max([self.getStrehl(tar)[0] for tar in range(self.tar.ntargets) ])
                if(SRmax<abortThresh):
                    print("SR SE too low: stopping the loop")
                    break
        t1 = time.time()
        print(" loop execution time:", t1 - t0, "  (", n, "iterations), ", (t1 - t0) / n,
              "(mean)  ", n / (t1 - t0), "Hz")
        return np.array(self.srbuffer)


    def nextLQG(self, *, move_atmos: bool = True, see_atmos: bool = True, nControl: int = 0,
             tar_trace: Iterable[int] = None, wfs_trace: Iterable[int] = None,
             do_control: bool = True, apply_control: bool = True,
             compute_tar_psf: bool = True) -> None:
        '''
        Iterates the AO loop, with optional parameters

        :parameters:
             move_atmos: (bool): move the atmosphere for this iteration, default: True

             nControl: (int): Controller number to use, default 0 (single control configurations)

             tar_trace: (None or list[int]): list of targets to trace. None equivalent to all.

             wfs_trace: (None or list[int]): list of WFS to trace. None equivalent to all.

             apply_control: (bool): (optional) if True (default), apply control on DMs
        '''
        if self.s is None:
            self.old_tmp=None
            self.s   = np.zeros([self.L.shape[1]],dtype=np.float32)
            self.sp1 = np.zeros([self.L.shape[1]],dtype=np.float32)
        
        if do_control and self.rtc is not None:
            for ncontrol in range(len(self.rtc.d_control)):
                if self.rtc.d_control[ncontrol].type != scons.ControllerType.GEO:
                    self.doLQGControl(ncontrol)
                    self.doClipping(ncontrol)
                    if apply_control:
                        self.applyControl(ncontrol,compVoltage=False)
        
        self.s = self.sp1.copy()
        self.sp1 *= 0.0

        if move_atmos and self.atm is not None:
            self.moveAtmos()

        if tar_trace is None and self.tar is not None:
            tar_trace = range(len(self.tar.d_targets))
        if wfs_trace is None and self.wfs is not None:
            wfs_trace = range(len(self.wfs.d_wfs))

        if tar_trace is not None:
            for t in tar_trace:
                if see_atmos:
                    self.raytraceTar(t, "all")
                else:
                    self.raytraceTar(t, ["tel", "dm", "ncpa"])
        if wfs_trace is not None:
            for w in wfs_trace:
                if see_atmos:
                    self.raytraceWfs(w, ["atmos", "tel", "ncpa"])
                else:
                    self.raytraceWfs(w, ["tel", "ncpa"])

                if not self.config.p_wfss[w].openloop and self.dms is not None:
                    self.raytraceWfs(w, "dm", rst=False)
                self.compWfsImage(w)

        if self.rtc.d_control[
                ncontrol].centro_idx is None:  # RTC standalone case
            self.doCalibrate_img(ncontrol)
        self.doCentroids(ncontrol)

        self.sp1 = np.array(self.rtc.d_control[nControl].d_centroids)
        if compute_tar_psf:
            for nTar in tar_trace:
                self.compTarImage(nTar)
                self.compStrehl(nTar)
                self.srbuffer.append(self.getStrehl(0))
        
        self.iter += 1

    def doLQGControl(self, nControl: int, n: int = 0,
            wfs_direction: bool = False, update_state: bool = False,
            command_index: int = 0):
        '''
        Computes the command from the Wfs slopes

        Parameters
        ------------
        nControl: (int): controller index
        n: (int) : target or wfs index (only used with GEO controller)
        '''
        if self.x is None:
            self.x = np.zeros([self.A.shape[0]],dtype=np.float32)

        if self.u is None:
            self.u = np.zeros([self.K.shape[0]],dtype=np.float32)
            self.um1 = np.zeros([self.K.shape[0]],dtype=np.float32)

        self.x = self.A @ self.x + self.L @ self.s - self.G @ self.um1
        self.um2 = self.um1
        self.um1 = self.u
        self.u = self.K @ self.x
        self.rtc.d_control[nControl].set_com(self.u,self.u.size)

