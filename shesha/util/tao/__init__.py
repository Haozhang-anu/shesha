from shesha.ao import imats 
from shesha.ao import cmats 
from astropy.io import fits 
import numpy as np 
import os
from shutil import copyfile
from shesha.util import write_sysParam 
from shesha.util import fits_io


from . import ltao
from . import mcao
from importlib import reload
#reload(ltao)
reload(mcao)

TILESIZE="1000"

STARPU_FLAGS=""

#variable necessary to run TAO
VARS={"SCHED":"dmdas",
      "STARPU_FLAGS":"",
      "GPUIDS":0,
      "TILESIZE":1000,
      "INPUTPATH":0,
      "TAOPATH":0
      }


def check():
    """Checks that variable are initialized
    """
    stop=0
    try :
        if (not isinstance(VARS["SCHED"], str)):
            print("you must select a scheduler (dmda,dmdas,dmdar...)\n\tex: VARS[\"SCHED\"]=\"dmdas\"")
            stop=1
    except:
        print("you must select a scheduler (dmda,dmdas,dmdar...)\n\tex: VARS[\"SCHED\"]=\"dmdas\"")
        stop=1
    try :
        if( not isinstance(VARS["GPUIDS"], str)):
            print("you must define the GPUs to use as a string \n\tex:VARS[\"GPUIDS\"]=\"1,2\"")
            stop=1
    except:
        print("you must define the GPUs to use as a string \n\tex:VARS[\"GPUIDS\"]=\"1,2\"")
        stop=1
    try :
        if( not isinstance(VARS["INPUTPATH"], str)):
            print("you must define the location of the system parameters \n\tex: VARS[\"INPUTPATH\"]=\"~/workspace/compass/params\"")
            stop=1
    except:
        print("you must define the location of the system parameters \n\tex: VARS[\"INPUTPATH\"]=\"~/workspace/compass/params\"")
        stop=1
    try :
        if( not isinstance(VARS["TAOPATH"], str)):
            print("you must define the location of the tao executables \n\tex: VARS[\"TAOPATH\"]=\"~/workspace/tao/install/bin\"")
            stop=1
    except:
        print("you must define the location of the tao executables \n\tex: VARS[\"TAOPATH\"]=\"~/workspace/tao/install/bin\"")
        stop=1
    try :
        STARPU_FLAGS
    except:
        STARPU_FLAGS=""

    return stop


def init(sup,mod,nfilt=300,WFS="all",DM_TT=False,lgstt=0.):
    """ Set up the compass loop

    set the interaction matrix, loop gain and write parameter files for TAO

    sup : CompassSupervisor :
    mod : module            : AO mode requested (among: ltao , mcao)
    """
    #easier access to datas
    sim=sup._sim
    conf=sup.config

    #setting open loop
    sim.rtc.d_control[0].set_polc(True)

    #if generic: need to update imat in controller
    sim.rtc.d_control[0].set_imat(conf.p_controllers[0]._imat)
    #update gain
    sim.rtc.d_control[0].set_gain(conf.p_controllers[0].gain)

    mod.init(VARS,sup,nfilt=nfilt,DM_TT=DM_TT,WFS=WFS,lgstt=lgstt)

def reconstructor(mod):
    """ Compute the TAO reconstructor for a given AO mode

    mod : module    : AO mode requested (among: ltao , mcao)
    """
    return mod.reconstructor(VARS)

def updateCmat(sup, cmatFile):
    """ Update the compass command matrix from an input fits file
    
    sup         : CompassSupervisor :
    cmatFile    : str               : name of the cmat fits file
    """
    M=fits_io.fitsread(cmatFile).T
    sup.setCommandMatrix(M)
    return M


def run(sup,mod,nIter=1,initialisation=0,reset=1,nfilt=300,WFS="all",DM_TT=False,lgstt=0.0):
    check()
    if(initialisation):
        init(sup,mod,nfilt=nfilt,WFS=WFS,DM_TT=DM_TT,lgstt=lgstt)
    M=reconstructor(mod)
    if(reset):
        sup.reset()
    cmatShape=sup.getCmat().shape
    if(M.shape[0] != cmatShape[0] or M.shape[1] != cmatShape[1]):
        print("ToR shape is not valid:\n\twaiting for:",cmatShape,"\n\tgot        :",M.shape)
    else:
        sup.setCommandMatrix(M)
        ##### 20200702 night time: running 20*50k buffer to be used later
        
        imat = np.array(sup._sim.rtc.d_control[0].d_imat)
        np.save("buffer/imat_PLAY.npy",imat)
        for ii in range(20):
            dd=[]
            ss=[]
            save_id = str(nIter)+"_"+str(ii+1)
            dd,ss = sup._sim.loopPOLC(nIter)
            np.save("buffer/dd_PLAY_"+save_id+".npy",dd)
            np.save("buffer/ss_PLAY_"+save_id+".npy",ss)
    '''
    rand_id = str(int(np.random.random(1)*100000))
    imat = np.array(sup._sim.rtc.d_control[0].d_imat)
    np.save("imat_"+rand_id+".npy",imat)
    #print("imat saved!")
    np.save("dd_"+rand_id+".npy",dd)
    np.save("ss_"+rand_id+".npy",ss)
    print("imat, dd and ss saved with id = "+rand_id)
    '''
    return dd.astype("float32"),ss.astype("float32")
