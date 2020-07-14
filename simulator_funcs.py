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

