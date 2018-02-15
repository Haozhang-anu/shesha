"""
Class SimulatorCACAO: CACAO overloaded simulator
"""
import sys
import os

from .simulator import Simulator
import shesha_init as init

from pyImageStreamIO import Image, Datatype
import numpy as np

import shesha_constants as scons


class SimulatorCACAO(Simulator):
    """
        Class SimulatorCACAO: CACAO overloaded simulator
        next() to call ....
    """

    def __init__(self, filepath: str=None, use_DB: bool=False) -> None:
        Simulator.__init__(self, filepath=filepath, use_DB=use_DB)
        self.imgs = None
        self.commands = None
        self.config_shm = None
        self.apply_commands = True
        self.n_wfss = 0
        self.n_wfss = 0

    def _dms_init(self) -> None:
        """
        Initializes the DMs object in the simulator
        """
        if self.config.p_dms is not None:
            #   dm
            print("->dm")
            self.dms = init.dm_init(self.c, self.config.p_dms, self.config.p_tel,
                                    self.config.p_geom, self.config.p_wfss,
                                    keepAllActu=True)
        else:
            self.dms = None

    def init_sim(self) -> None:
        Simulator.init_sim(self)

        self.n_wfss = len(self.config.p_wfss)
        self.imgs = np.empty((self.n_wfss), dtype=Image)
        for i, wfs_i in zip(range(self.n_wfss), self.config.p_wfss):
            if wfs_i.type == scons.WFSType.SH:
                data = self.wfs.get_bincube(i)
            elif wfs_i.type == scons.WFSType.PYRHR:
                data = self.wfs.get_pyrimg(i)
            else:
                continue
            self.imgs[i] = Image()
            self.imgs[i].create("compass_wfs%d" % i, data.shape, Datatype.FLOAT, 1, 1)
        self.write_image_in_shm()

        self.n_dms = len(self.config.p_dms)
        self.commands = np.empty((self.n_dms), dtype=Image)
        for i, dm_i in zip(range(self.n_dms), self.config.p_dms):
            n_nx_actu = int(np.sqrt(dm_i._ntotact))
            self.commands[i] = Image()
            self.commands[i].create("compass_dm%d" % i, (n_nx_actu, n_nx_actu),
                                    Datatype.FLOAT, 1, 1)

        # data = np.stack([self.config.p_wfs0._validsubsx, self.config.p_wfs0._validsubsy])
        # self.config_shm = Image()
        # self.config_shm.create("compass_config", data.shape, Datatype.INT32, 1, 3)

    def next(self, **kwargs) -> None:
        """
        Overload of the Simulator.next() function with CACAO publications
        """
        Simulator.next(self, apply_control=True, **kwargs)

        self.write_image_in_shm()

        if self.apply_commands:
            cmd = self.read_commands_from_shm()
            if cmd is not None:
                self.dms.set_full_comm(cmd)

    def read_commands_from_shm(self):
        cmd = np.empty((0), dtype=np.float32)

        for i in range(self.n_dms):
            if self.commands[i] is None:
                tmp_commands = Image()
                if tmp_commands.link("compass_dm%d" % i) == -1:
                    return None
                self.commands[i] = tmp_commands
            cmd = np.append(cmd, np.array(self.commands[i]).flatten())
        # self.commands.semwait(0)
        # self.commands.read("compass_dms")
        return cmd

    def write_image_in_shm(self):
        for i, wfs_i in zip(range(self.n_wfss), self.config.p_wfss):
            if wfs_i.type == scons.WFSType.SH:
                data = self.wfs.binimg(i)
                # sh = data.shape
                # data = data.reshape((sh[0], sh[1]*sh[2]))
            elif wfs_i.type == scons.WFSType.PYRHR:
                data = self.wfs.get_pyrimg(i)
            else:
                continue
            self.imgs[i].write(data)
