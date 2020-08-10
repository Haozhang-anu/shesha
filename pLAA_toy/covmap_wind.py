#!/usr/bin/env python

import numpy as np
import CovMap_from_Mat as cfm
import matplotlib.pyplot as pp; pp.ion()
from lazy_long_buffer import all_starts,this_ind

prefix="PLAY"
sysconfigfile = "sysconfig_"+prefix+".npz"
locals().update(np.load(sysconfigfile)) # easier than: shnxsub = npfile["shnxsub"], etc.
#imat = np.load("buffer/imat_"+prefix+".npy")
ittime = 0.001
MapMask = np.tile(cfm.MapMask_from_validsubs(validsubs,shnxsub),
            [2*nwfs,2*nwfs])

def CovMap_dT(dT):
    return cfm.CovMap_from_Cn2 (MapMask,telDiam,zenith,
            shnxsub,r0,Cn2,l0,alt,nwfs,gspos,gsalt,
            timedelay=dT*ittime,
            windspeed=windspeed,
            winddir=winddir)

def Cmm_dT(dT): 
    cmm = cfm.Mat_from_CovMap(CovMap_dT(dT),nwfs,validsubs,shnxsub)
    cmm = np.tril(cmm)+np.tril(cmm).T - np.diag(cmm.diagonal())
    return cmm
    """
    # these all return approximately the same thing:
    cmm1 = cfm.Mat_from_CovMap(CovMap_dT(dT),nwfs,validsubs,shnxsub)
    cmm1 = np.tril(cmm1)+np.tril(cmm1).T - np.diag(cmm1.diagonal())
    cmm2 = cfm.CMM_from_npz(prefix=prefix,dT=dT)
    cmm3 = cfm.CMM_from_Cn2(telDiam,zenith,shnxsub,xx,yy,
                            r0,Cn2,l0,alt,nwfs,gspos,gsalt,
                            timedelay=dT*ittime,
                            windspeed=windspeed,winddir=winddir)
    """

def Cmm_num_dT (T,r_deci,dT):
    '''
    dT in frames
    '''
    n_batch = 1
    framerate=0.001
    total_buffer = 1000000
    file_size = 50000
    imat = np.load("../buffer/imat_"+prefix+".npy")
    batch_size = T/framerate
    starts = all_starts(n_batch, batch_size, total_buffer = total_buffer-dT)
    starts_dT = starts + dT
    print(starts)
    print(starts_dT)
    for ii in range(starts.shape[0]):
        f_list,inds = this_ind (starts[ii], batch_size, r_deci, file_size = 50000)
        
        lazy_count  = 0
        if len(f_list) != len(inds):
            raise Exception("number of files and inds are not aligned!")
        for kk in range(len(f_list)):
            this_f = f_list[kk]
            this_dd = np.load("../buffer/dd_"+prefix+"_50000_"+str(this_f)+".npy").astype("float32")
            this_ss = np.load("../buffer/ss_"+prefix+"_50000_"+str(this_f)+".npy").astype("float32")
            print("number of slopes read this time: ",len(inds[kk]))

            if lazy_count == 0 :
                #print(type(inds[ii]))
                dd = this_dd[:,list(inds[kk])]
                ss = this_ss[:,list(inds[kk])]
                lazy_count = lazy_count+1
            else:
                dd = np.append(dd,this_dd[:,list(inds[kk])],axis=1)       
                ss = np.append(ss,this_ss[:,list(inds[kk])],axis=1)      
        pols   = (-imat@dd +ss).astype("float32")
        pols   = pols.T
        print(pols.shape[0])
        pols = pols - pols.mean(axis=0)

        f_list_dT,inds_dT = this_ind (starts_dT[ii], batch_size, r_deci, file_size = 50000)
        lazy_count  = 0
        if len(f_list_dT) != len(inds_dT):
            raise Exception("number of files and inds are not aligned!")
        for kk in range(len(f_list_dT)):
            this_f = f_list_dT[kk]
            this_dd = np.load("../buffer/dd_"+prefix+"_50000_"+str(this_f)+".npy").astype("float32")
            this_ss = np.load("../buffer/ss_"+prefix+"_50000_"+str(this_f)+".npy").astype("float32")
            print("number of slopes read this time, temporal shifted: ",len(inds_dT[kk]))

            if lazy_count == 0 :
                #print(type(inds[ii]))
                dd = this_dd[:,list(inds_dT[kk])]
                ss = this_ss[:,list(inds_dT[kk])]
                lazy_count = lazy_count+1
            else:
                dd = np.append(dd,this_dd[:,list(inds_dT[kk])],axis=1)       
                ss = np.append(ss,this_ss[:,list(inds_dT[kk])],axis=1)      
        pols_dT   = (-imat@dd +ss).astype("float32")
        pols_dT   = pols_dT.T
        print(pols_dT.shape[0])
        pols_dT = pols_dT - pols_dT.mean(axis=0)
        cmm = pols.T@pols_dT / pols.shape[0]
        #time_end = time.time()
        print("average diagonal value for this cmm = %.5f"%(np.average(cmm.diagonal())))
        #print("numerical cmm starting at %06d"%this_start+\
        #       " saved, time taken = %.3f"%(time_end-time_start))
        return cmm





if __name__=="__main__":

    for dT in [0]:
        if 1==0: # Do CovMap (else Cmm)
            CovMap = CovMap_dT(dT)
            pp.figure()
            pp.matshow(CovMap)
            pp.title(f"dT = {dT:d}")
        else: # Do Cmm
            Cmm = Cmm_dT(dT)
            Cmm_num = Cmm_num_dT(50,10,dT) #randomly generated from numerical buffer
            pp.figure()
            pp.subplot(1,3,1)
            pp.imshow(Cmm)
            #pp.colorbar()
            pp.title("a, dT = %03d"%(dT))
            pp.subplot(1,3,2)
            pp.imshow(Cmm_num)
            #pp.colorbar()
            pp.title("n 50/10, dT = %03d"%(dT)) 
            pp.subplot(1,3,3)
            pp.imshow(Cmm-Cmm_num)
            #pp.colorbar()
            #pp.title("numerical 50/10, dT = %03d"%(dT))     
            pp.title(f"diff, sum = {(Cmm-Cmm_num).sum():f}")
            #pp.axis("square")
            pp.subplots_adjust(wspace = 0.4)

            CovMap_num,cmask = cfm.CovMap_from_Mat(Cmm_num,nwfs,validsubs,shnxsub)
            CovMap = CovMap_dT(dT)
            pp.figure()
            pp.subplot(1,3,1)
            pp.imshow(CovMap)
            #pp.colorbar()
            pp.title("a, dT = %03d"%(dT))
            pp.subplot(1,3,2)
            pp.imshow(CovMap_num)
            #pp.colorbar()
            pp.title("n 50/10, dT = %03d"%(dT)) 
            pp.subplot(1,3,3)
            dif = CovMap-CovMap_num
            pp.imshow(dif)
            #pp.colorbar()
            #pp.title("numerical 50/10, dT = %03d"%(dT))     
            pp.title(f"diff, sum = {(CovMap-CovMap_num).sum():f}")
            #pp.axis("square")
            pp.subplots_adjust(wspace = 0.4)

