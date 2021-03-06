import numpy as np
from docopt import docopt
from matplotlib.colors import Normalize as Norm
from scipy import stats as st
from astropy.io import fits
import math
import time
import shesha
import numpy as np
from matplotlib import pyplot as plt
from shesha.util.CovMap_from_Mat import Map_and_Mat,CovMap_from_Cn2, Mat_from_CovMap, DPHI, get_full_index_compass,MapMask_from_validsubs, CovMap_from_Mat
from scipy import stats as st
import glob
from util import Reporter, norm2, norminf,Dataset,Option
import scipy.linalg.lapack as lapack
import scipy.linalg as linalg
#############
##### Functions to translate starting_ind, T and r into 
##### index 
def this_ind (this_start, batch_size, r_deci, file_size = 50000):
    '''
    get file list and ind list for a single batch

    this_start: global index of starting point for this batch
    batch_size: refers to integration time, equals to T/framerate
    r_deci: 1/decimation rate, equal to buffer_every in previous use
    file_size: buffer size of a single buffer file, default is 50,000

    return
    f_list: list, contains all files to be loaded for this batch
    inds: list of array, same size as f_list with each element being valid local 
          indices for corresponding file in f_list
    '''
    global_ind = np.arange(this_start,this_start+batch_size,r_deci)
    f_list     = list(np.unique(np.floor(global_ind/file_size)).astype("int")+1)
    inds_all   = global_ind % file_size
    # now align inds by the corresponding file name
    ind_temp   = inds_all - np.roll(inds_all,1)
    ind_temp[0]= r_deci
    cut_point  = np.where(ind_temp<0)[0]
    cut_point  = np.insert(cut_point,0,0)
    cut_point  = np.append(cut_point,inds_all.shape[0])
    inds       = []
    for ii in range(cut_point.shape[0]-1):
        inds.append(list(inds_all[cut_point[ii]:cut_point[ii+1]].astype("int")))
    return f_list,inds


def all_starts (n_batch, batch_size, total_buffer = 1000000):
    '''
    return random selected starting indices (global) for given # batch and batch size,
    no overlapping between any of the batches.
    '''
    if n_batch*batch_size > total_buffer :
        raise Exception("Out of boundary!")

    #starts = np.zeros(n_batch)
    block_size = int(total_buffer/n_batch)+1 #floor
    min_starts = np.arange(0,total_buffer,block_size)
    max_starts = block_size - batch_size
    starts     = min_starts + np.floor(np.random.random(n_batch)*max_starts).astype("int")
    return starts.astype("int")

def get_num_cmm (n_batch, T, r_deci, framerate=0.001, total_buffer = 1000000, file_size = 50000, prefix = "long_buffer"):
    '''
    save a series of numerical cmm for a given number of batch,
    integration time and decimation rate
    no overlapping between any of the batches is ensured;
    each batch starts from a *relatively* random point    

    n_batch: number of batch required
    T: integration time in secs
    r_deci: 1/r, same as buffer_every
    
    return
    hard copy of numerical cmms saved 
    '''
    if n_batch*T/framerate > total_buffer:
        raise Exception("Out of boundary!")


    imat = np.load("buffer/imat_"+prefix+".npy")
    batch_size = T/framerate
    starts = all_starts(n_batch, batch_size, total_buffer = total_buffer)
    print("begin calculating numerical cmm!")
    for this_start in starts:
        time_start = time.time()
        f_list,inds = this_ind (this_start, batch_size, r_deci, file_size = 50000)
        lazy_count  = 0
        if len(f_list) != len(inds):
            raise Exception("number of files and inds are not aligned!")
        for ii in range(len(f_list)):
            this_f = f_list[ii]
            this_dd = np.load("buffer/dd_"+prefix+"_50000_"+str(this_f)+".npy").astype("float32")
            this_ss = np.load("buffer/ss_"+prefix+"_50000_"+str(this_f)+".npy").astype("float32")
            print("number of slopes read this time: ",len(inds[ii]))

            if lazy_count == 0 :
                #print(type(inds[ii]))
                dd = this_dd[:,list(inds[ii])]
                ss = this_ss[:,list(inds[ii])]
                lazy_count = lazy_count+1
            else:
                dd = np.append(dd,this_dd[:,list(inds[ii])],axis=1)       
                ss = np.append(ss,this_ss[:,list(inds[ii])],axis=1)       
            
        pols   = (-imat@dd +ss).astype("float32")
        pols   = pols.T
        print(pols.shape[0])
        pols = pols - pols.mean(axis=0)
        cmm = pols.T@pols / pols.shape[0]
        np.save("buffer/cmm_num_"+prefix+"_T_"+str(T)+"_buffer_every_"\
                 +str(r_deci)+"_start_from_%06d.npy"%this_start,cmm)
        time_end = time.time()
        print("average diagonal value for this cmm = %.5f"%(np.average(cmm.diagonal())))
        print("numerical cmm starting at %06d"%this_start+\
               " saved, time taken = %.3f"%(time_end-time_start))
        pols = []
        cmm  = []
        dd   = []
        ss   = []

########## Yuxi's linear reconstruction of Cn2

class Optsovler:
    def __init__(self,nwfs,alt,X0,n_valid):
        m = getAshape(nwfs) * n_valid * 3 # xx xy yy
        m = int(m)
        n = alt.shape[0]
        n = int(n)
        self.A = np.zeros((m,n))
        self.b = np.zeros(m)
        self.x_k = np.random.rand(n)
        self.x_sol = np.zeros(n)
        self.wfsrow = n_valid

        
def getAshape(nwfs):
    return int((nwfs+1)*nwfs/2)


def conf_to_Am1 (telDiam,zenith,validsubs,shnxsub,r0,l0,alt,nwfs,gspos,gsalt,timedelay=0,windspeed=[],winddir=[]):
    dtor   = np.pi/180
    astor  = dtor/60/60
    zenith = zenith*dtor
    gspos  = gspos*astor
    subAperSize = telDiam/shnxsub
    r0     = r0*math.cos(zenith)**0.6
    gsalt  = gsalt/math.cos(zenith)
    alt    = alt/math.cos(zenith)
    #Cn2    = Cn2/np.sum(Cn2)
    l0     = np.where(l0>1000,1000,l0)
    windflag = 0 if timedelay==0 else 1
    if windflag :
        print("Temporal evolution enabled!")
        windspeed   = np.array(windspeed)
        winddir     = np.array(winddir)*dtor
        windspeed_x = windspeed*np.cos(winddir)
        windspeed_y = windspeed*np.sin(winddir)
    #validsubs  = get_full_index_compass()
    CovMapMask = MapMask_from_validsubs(validsubs,shnxsub)
    #CovMapMask = np.tile(CovMapMask,[nwfs*2,nwfs*2])
    #CovMap_ana = np.zeros(CovMapMask.shape)
    validmask  = np.where(CovMapMask.flatten())[0]
    x0 = y0    = np.arange(-shnxsub+1,shnxsub,1)
    X0,Y0      = np.meshgrid(x0,y0) ## 79*79
    X0         = X0*subAperSize
    Y0         = Y0*subAperSize
    #supervisor._sim.config.p_wfs_lgs[ii]
    n_deltaY   = Y0.shape[0]
    k         = 1/subAperSize/subAperSize
    lambda2   = (0.5e-6/2/np.pi/astor)**2
    time_start = time.time()
    print("Begin calculating Cov Map from Cn2")
    optsolver = Optsovler(nwfs, alt, X0,validmask.shape[0])
    #print(Cn2)
    #optsolver.x_sol = Cn2
    optcnt = 0
    # print(optsolver.A.shape)
    for wfs_1 in range(nwfs):
        for wfs_2 in range(wfs_1,nwfs):
            #print("begin calculating for WFS # "+str(wfs_1)+" and WFS #"+str(wfs_2)+" !")
            gsalt_1 = gsalt[wfs_1]
            gsalt_2 = gsalt[wfs_2]
            for li in range(alt.shape[0]):
                tmpoptcnt = optcnt
                # print(tmpoptcnt)
                layer_alt = alt[li]
                l0_i      = l0[li]
                #Cn2h      = r0**(-5/3.)*Cn2[li]
                dx_li_wind= 0
                dy_li_wind= 0
                if windflag :
                    dx_li_wind = windspeed_x[li] * timedelay
                    dy_li_wind = windspeed_y[li] * timedelay
                #print("Processing altitude = "+str(layer_alt)+" m")
                if gsalt_1*gsalt_2 == 0:
                    if gsalt_1 == gsalt_2:
                        # 2 NGS
                        Xi = X0 - layer_alt*(gspos[wfs_2,0]-gspos[wfs_1,0]) - dx_li_wind
                        Yi = Y0 - layer_alt*(gspos[wfs_2,1]-gspos[wfs_1,1]) - dy_li_wind
                        # no need to scale subAperSize
                        subAperSize_i = subAperSize
                    else:
                        # 1 NGS 1 LGS
                        raise Exception('Cov Map between LGS and NGS is not valid!')
                else:
                    # 2 LGS
                    # Average gsalt is used here, which is not precise
                    if gsalt_1<layer_alt or gsalt_2<layer_alt:
                        raise Exception('Turbulence layer is higher than LGS altitude!')
                    else:
                        avg_gsalt = (gsalt_1+gsalt_2)/2
                        Xi = (1-layer_alt/avg_gsalt)*X0 - layer_alt*(gspos[wfs_2,0]-gspos[wfs_1,0]) - dx_li_wind
                        Yi = (1-layer_alt/avg_gsalt)*Y0 - layer_alt*(gspos[wfs_2,1]-gspos[wfs_1,1]) - dy_li_wind
                        subAperSize_i = (1-layer_alt/avg_gsalt)*subAperSize
            
                Cov_XX = (-2 * DPHI(Xi,Yi,l0_i) + DPHI(Xi+subAperSize_i,Yi,l0_i) + DPHI(Xi-subAperSize_i,Yi,l0_i))*0.5
                optxxA = Cov_XX * k * lambda2 * r0 **(-5/3.) # for opt genreate A, xx
                #Cov_XX = Cov_XX *k*lambda2*np.abs(Cn2h)
                optAstart = tmpoptcnt*optsolver.wfsrow
                optAend = (tmpoptcnt+1)*optsolver.wfsrow
                # print(optAstart)
                optxxA = optxxA.flatten()[validmask]
                optsolver.A[optAstart:optAend,li] = optxxA
                
                #optsolver.b[optAstart:optAend] = Cov_XX.flatten()
                tmpoptcnt += 1
                Cov_YY = (-2 * DPHI(Xi,Yi,l0_i) + DPHI(Xi,subAperSize_i+Yi,l0_i) + DPHI(Xi,-subAperSize_i+Yi,l0_i))*0.5
                optyyA = Cov_YY * k * lambda2 * r0 **(-5/3.) # for opt genreate A, yy
                #Cov_YY = Cov_YY *k*lambda2*np.abs(Cn2h)
                optAstart = tmpoptcnt*optsolver.wfsrow
                # print(optAstart)
                optAend = (tmpoptcnt+1)*optsolver.wfsrow
                optyyA = optyyA.flatten()[validmask]
                optsolver.A[optAstart:optAend,li] = optyyA
                #optsolver.b[optAstart:optAend] = Cov_YY.flatten()
                Cov_XY = -DPHI(Xi+np.sqrt(2)*subAperSize_i/2,Yi-np.sqrt(2)*subAperSize_i/2,l0_i) +\
                          DPHI(Xi+np.sqrt(2)*subAperSize_i/2,Yi+np.sqrt(2)*subAperSize_i/2,l0_i) +\
                          DPHI(Xi-np.sqrt(2)*subAperSize_i/2,Yi-np.sqrt(2)*subAperSize_i/2,l0_i) -\
                          DPHI(Xi-np.sqrt(2)*subAperSize_i/2,Yi+np.sqrt(2)*subAperSize_i/2,l0_i)
                Cov_XY = Cov_XY/4
                optxyA = Cov_XY * k * lambda2 * r0 **(-5/3.) # for opt genreate A, xy
                #Cov_XY = Cov_XY *k*lambda2*np.abs(Cn2h)
                tmpoptcnt += 1
                optAstart = tmpoptcnt*optsolver.wfsrow
                # print(optAstart)
                optAend = (tmpoptcnt+1)*optsolver.wfsrow
                optxyA = optxyA.flatten()[validmask]
                optsolver.A[optAstart:optAend,li] = optxyA
                #optsolver.b[optAstart:optAend] = Cov_XY.flatten()

                #CovMap_ana[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] += Cov_XX
                #CovMap_ana[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] += Cov_YY
                #CovMap_ana[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] += Cov_XY
                #CovMap_ana[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] += Cov_XY
            
            '''
            tmpoptcnt = optcnt
            optAstart = tmpoptcnt*optsolver.wfsrow
            optAend = (tmpoptcnt+1)*optsolver.wfsrow
            optsolver.b[optAstart:optAend] = CovMap_ana[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY].flatten()
            tmpoptcnt += 1 
            optAstart = tmpoptcnt*optsolver.wfsrow
            optAend = (tmpoptcnt+1)*optsolver.wfsrow
            optsolver.b[optAstart:optAend] = CovMap_ana[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)].flatten()
            tmpoptcnt += 1
            optAstart = tmpoptcnt*optsolver.wfsrow
            optAend = (tmpoptcnt+1)*optsolver.wfsrow
            optsolver.b[optAstart:optAend] = CovMap_ana[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY].flatten()
            '''
            optcnt += 3
            

    time_end = time.time()
    print("Cov Map generation finished! Time taken = %.1f seconds."%(time_end-time_start))
    #hdu = fits.PrimaryHDU(CovMap_ana)
    #hdu.writeto("CovMap_ana_full.fits",overwrite=1)
    #CovMap_ana = CovMap_ana * CovMapMask
    #Am1 = np.linalg.solve(optsolver.A.T@optsolver.A,optsolver.A.T)
    return optsolver,time_end-time_start



def conf_to_Am1_cmm (telDiam,zenith,shnxsub,xx,yy,r0,Cn2,l0,alt,nwfs,gspos,gsalt,timedelay=0,windspeed=[],winddir=[]):
    '''
    CMM_from_Cn2 (telDiam,zenith,shnxsub,xx,yy,r0,Cn2,l0,alt,nwfs,gspos,gsalt,timedelay=0,windspeed=[],winddir=[])

    get analytical CMM from Cn2 profile and system configuration
    loop over all combinations of WFS and layers, calculating covariance using Rod Conan model
    wavelength is fixed at 500 nm, please check r0 before use

    Update: Temporal difference included.

    
    telDiam: float, telescope diametre in metre
    zenith: float, zenith angle in degree
    shnxsub: int, number of subapertures along pupil diamtre
    xx, yy: np 1d array, all valid subaperture's x/y coordinates [self centred] [in metres]
    r0: float, global r0 at 500nm in metre
    Cn2: np 1D array, same size as alt, fractions of every turbulence layer, will be normalised
    l0: np 1D array, same size as alt, outer scale in metre for every layer
    alt: np 1D array, altitude of every layer in metre
    nwfs: int, number of WFS
    gspos: np 2D array, [nwfs, 2], X/Y position in arcsec for every WFS
    gsalt: np 1D array, length = nwfs, altitude for all WFS in metre, 0 for NGS

    timedelay: float, time difference in second, default = 0, usually calculated by (# frame_delay * it_time)
    windspeed: np 1D array, same size as alt, speed of every layer in m/s
    winddir: np 1D array, same size as alt, wind direction of everylayer with respect to x-axis in degree
    --> if timedelay == 0, wind will not be considered at all
    return:

    CMM_ana: np 2D array, analytical CMM for all valid WFS pairs


    '''
    dtor   = np.pi/180
    astor  = dtor/60/60
    zenith = zenith*dtor
    gspos  = gspos*astor
    #print(gspos)
    subAperSize = telDiam/shnxsub
    r0     = r0*math.cos(zenith)**0.6
    gsalt  = gsalt/math.cos(zenith)
    alt    = alt/math.cos(zenith)
    Cn2    = Cn2/np.sum(Cn2)
    l0     = np.where(l0>1000,1000,l0)
    windflag = 0 if timedelay==0 else 1
    if windflag :
        print("Temporal evolution enabled!")
        windspeed   = np.array(windspeed)
        winddir     = np.array(winddir)*dtor
        windspeed_x = windspeed*np.cos(winddir)
        windspeed_y = windspeed*np.sin(winddir)

    CMM_ana = np.zeros((nwfs*2*xx.shape[0],nwfs*2*xx.shape[0]))
    #x0 = y0    = np.arange(-shnxsub+1,shnxsub,1)
    #X0,Y0      = np.meshgrid(xx,yy)
    #X0         = X0*subAperSize
    #Y0         = Y0*subAperSize
    X0 = xx
    Y0 = yy
    n_deltaY   = Y0.shape[0]
    k         = 1/subAperSize/subAperSize
    lambda2   = (0.5e-6/2/np.pi/astor)**2

    time_start = time.time()
    print("Begin calculating CMM from Cn2")
    
    optsolver = Optsovler(nwfs, alt, X0,X0.shape[0]**2)
    #print(Cn2)
    #optsolver.x_sol = Cn2
    optcnt = 0
    for wfs_1 in range(nwfs):
        for wfs_2 in range(wfs_1,nwfs):
            print("begin calculating for WFS # "+str(wfs_1)+" and WFS #"+str(wfs_2)+" !")
            gsalt_1 = gsalt[wfs_1]
            gsalt_2 = gsalt[wfs_2]

            for li in range(alt.shape[0]):
                tmpoptcnt = optcnt

                layer_alt = alt[li]
                l0_i      = l0[li]
                #Cn2h      = r0**(-5/3)*Cn2[li]
                dx_li_wind= 0
                dy_li_wind= 0

                if windflag :
                    dx_li_wind = windspeed_x[li] * timedelay
                    dy_li_wind = windspeed_y[li] * timedelay
                #print("Processing altitude = "+str(layer_alt)+" m")

                if gsalt_1*gsalt_2 == 0:
                    if gsalt_1 == gsalt_2:
                        # 2 NGS
                        X1i = X0 - layer_alt*(gspos[wfs_2,0]-gspos[wfs_1,0]) - dx_li_wind
                        Y1i = Y0 - layer_alt*(gspos[wfs_2,1]-gspos[wfs_1,1]) - dy_li_wind
                        # no need to scale subAperSize
                        subAperSize_i = subAperSize
                    else:
                        # 1 NGS 1 LGS
                        raise Exception('Cov Map between LGS and NGS is not valid!')
                else:
                    # 2 LGS
                    # Average gsalt is used here, which is not precise
                    if gsalt_1<layer_alt or gsalt_2<layer_alt:
                        raise Exception('Turbulence layer is higher than LGS altitude!')
                    else:
                        #avg_gsalt = (gsalt_1+gsalt_2)/2

                        X1i = (1-layer_alt/gsalt_1)*X0 - layer_alt*gspos[wfs_1,0] - dx_li_wind
                        Y1i = (1-layer_alt/gsalt_1)*Y0 - layer_alt*gspos[wfs_1,1] - dy_li_wind
                        X2i = (1-layer_alt/gsalt_2)*X0 - layer_alt*gspos[wfs_2,0] - dx_li_wind
                        Y2i = (1-layer_alt/gsalt_2)*Y0 - layer_alt*gspos[wfs_2,1] - dy_li_wind
                        #subAperSize_i = (1-layer_alt/avg_gsalt)*subAperSize
                        s1 = (1-layer_alt/gsalt_1)*subAperSize
                        s2 = (1-layer_alt/gsalt_2)*subAperSize

                XX1, XX2 = np.meshgrid (X1i,X2i)
                YY1, YY2 = np.meshgrid (Y1i,Y2i)
                Xi  = XX1 - XX2
                Yi  = YY1 - YY2
                #Xi = X2i - X1i.T
                #Yi = Y2i - Y1i.T
                ac = s1/2-s2/2
                ad = s1/2+s2/2
                bc = -s1/2-s2/2
                bd = -s1/2+s2/2
                #print("%.3f,%.3f,%.3f,%.3f"%(ac,ad,bc,bd))

                Cov_XX = (-DPHI(Xi+ac,Yi,l0_i) + DPHI(Xi+ad,Yi,l0_i) + DPHI(Xi+bc,Yi,l0_i)-DPHI(Xi+bd,Yi,l0_i))*0.5
                optxxA = Cov_XX * k * lambda2 * r0 **(-5/3.) # for opt genreate A, xx
                #Cov_XX = Cov_XX *k*lambda2*np.abs(Cn2h)
                optAstart = tmpoptcnt*optsolver.wfsrow
                optAend = (tmpoptcnt+1)*optsolver.wfsrow
                # print(optAstart)
                optxxA = optxxA.flatten()
                optsolver.A[optAstart:optAend,li] = optxxA
                #optsolver.b[optAstart:optAend] = Cov_XX.flatten()

                tmpoptcnt += 1

                #Cov_XX = Cov_XX *k*lambda2*np.abs(Cn2h)
                Cov_YY = (-DPHI(Xi,Yi+ac,l0_i) + DPHI(Xi,Yi+ad,l0_i) + DPHI(Xi,Yi+bc,l0_i)-DPHI(Xi,Yi+bd,l0_i))*0.5
                #Cov_YY = Cov_YY *k*lambda2*np.abs(Cn2h)

                optyyA = Cov_YY * k * lambda2 * r0 **(-5/3.) # for opt genreate A, yy
                #Cov_YY = Cov_YY *k*lambda2*np.abs(Cn2h)
                optAstart = tmpoptcnt*optsolver.wfsrow
                # print(optAstart)
                optAend = (tmpoptcnt+1)*optsolver.wfsrow
                optyyA = optyyA.flatten()
                optsolver.A[optAstart:optAend,li] = optyyA
                #optsolver.b[optAstart:optAend] = Cov_YY.flatten()

                s0 = np.sqrt(s1**2+s2**2)/2
                #subAperSize_i = s1

                Cov_XY = -DPHI(Xi+s0,Yi-s0,l0_i) +\
                          DPHI(Xi+s0,Yi+s0,l0_i) +\
                          DPHI(Xi-s0,Yi-s0,l0_i) -\
                          DPHI(Xi-s0,Yi+s0,l0_i)

                Cov_XY = Cov_XY/4
                #Cov_XY = Cov_XY *k*lambda2*np.abs(Cn2h)
                optxyA = Cov_XY * k * lambda2 * r0 **(-5/3.) # for opt genreate A, xy
                #Cov_XY = Cov_XY *k*lambda2*np.abs(Cn2h)
                tmpoptcnt += 1
                optAstart = tmpoptcnt*optsolver.wfsrow
                # print(optAstart)
                optAend = (tmpoptcnt+1)*optsolver.wfsrow
                optxyA = optxyA.flatten()
                optsolver.A[optAstart:optAend,li] = optxyA
                '''
                CMM_ana[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] += Cov_XX
                CMM_ana[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] += Cov_YY
                CMM_ana[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] += Cov_XY
                CMM_ana[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] += Cov_XY
                '''
            optcnt += 3

    time_end = time.time()
    print("CMM generation finished! Time taken = %.1f seconds."%(time_end-time_start))
    #hdu = fits.PrimaryHDU(CovMap_ana)
    #hdu.writeto("CovMap_ana_full.fits",overwrite=1)
    #CovMap_ana = CovMap_ana * CovMapMask
    #CMM_ana_full = np.tril(CMM_ana.T)+np.tril(CMM_ana.T).T - np.diag(CMM_ana.diagonal())
    #ts = time.time()
    #Am1 = np.linalg.solve(optsolver.A.T@optsolver.A,optsolver.A.T)
    #te = time.time()
    #print("time taken for inversion: %.3f"%(te-ts))
    return optsolver,time_end-time_start



def b_to_map(nwfs,shnxsub,validsubs,b):

    CovMapMask = MapMask_from_validsubs(validsubs,shnxsub)
    validmask  = np.where(CovMapMask.flatten())[0]
    CovMapMask = np.tile(CovMapMask,[nwfs*2,nwfs*2])
    CovMap_rec = np.zeros(CovMapMask.shape)
    n_deltaY   = 2*shnxsub-1
    n_mappoint = validmask.shape[0]
    
    n_counter  = 0
    xx_block   = np.zeros((n_deltaY,n_deltaY)).flatten()
    yy_block   = np.zeros((n_deltaY,n_deltaY)).flatten()
    xy_block   = np.zeros((n_deltaY,n_deltaY)).flatten()
    for wfs_1 in range(nwfs):
        for wfs_2 in range(wfs_1,nwfs):
            #this_n  = wfs_1*nwfs + wfs_2
            this_xx = b[n_counter*n_mappoint:n_counter*n_mappoint+n_mappoint]
            n_counter+=1
            this_yy = b[n_counter*n_mappoint:n_counter*n_mappoint+n_mappoint]
            n_counter+=1
            this_xy = b[n_counter*n_mappoint:n_counter*n_mappoint+n_mappoint]
            n_counter+=1
            xx_block[validmask] = this_xx
            yy_block[validmask] = this_yy
            xy_block[validmask] = this_xy
            CovMap_rec[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] = xx_block.reshape((n_deltaY,n_deltaY))
            CovMap_rec[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] = yy_block.reshape((n_deltaY,n_deltaY))
            CovMap_rec[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] = xy_block.reshape((n_deltaY,n_deltaY))
            CovMap_rec[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] = xy_block.reshape((n_deltaY,n_deltaY))


    return CovMap_rec

def mat_to_b(CovMat,nwfs,validsubs,shnxsub):

    CovMap,CovMapMask = CovMap_from_Mat(CovMat,nwfs,validsubs,shnxsub)
    CovMapMask = MapMask_from_validsubs(validsubs,shnxsub)
    n_deltaY   = 2*shnxsub-1
    validmask  = np.where(CovMapMask.flatten())[0]
    n_mappoint = validmask.shape[0]
    b          = np.zeros(getAshape(nwfs) * n_mappoint * 3)
    n_counter  = 0
    for wfs_1 in range(nwfs):
        for wfs_2 in range(wfs_1,nwfs):
            b[n_counter*n_mappoint:(n_counter+1)*n_mappoint] = CovMap[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY].flatten()[validmask]
            n_counter+=1
            b[n_counter*n_mappoint:(n_counter+1)*n_mappoint] = CovMap[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)].flatten()[validmask]
            n_counter+=1
            b[n_counter*n_mappoint:(n_counter+1)*n_mappoint] = CovMap[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)].flatten()[validmask]
            n_counter+=1

    return b

def mat_to_b_cmm (CovMat,nwfs,validsubs):
    n_mappoint = np.int(np.sum(validsubs)**2)
    n_deltaY   = np.int(np.sum(validsubs))
    b          = np.zeros(getAshape(nwfs) * n_mappoint * 3)
    n_counter  = 0
    for wfs_1 in range(nwfs):
        for wfs_2 in range(wfs_1,nwfs):
            b[n_counter*n_mappoint:(n_counter+1)*n_mappoint] = CovMat[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY].flatten()
            n_counter+=1
            b[n_counter*n_mappoint:(n_counter+1)*n_mappoint] = CovMat[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)].flatten()
            n_counter+=1
            b[n_counter*n_mappoint:(n_counter+1)*n_mappoint] = CovMat[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)].flatten()
            n_counter+=1

    return b


############# below is Yuxi's LM 

def LM(dataset, rpt, option):
    # load information
    A, b, x_sol, x_init, As = dataset.A.copy(), dataset.b.copy(), dataset.x_sol.copy(), \
                              dataset.x_init.copy(), dataset.As.copy()
    datapass, importance, constrained = option.datapass, option.importance, option.constrained
    batchsize = option.batchsize
    lm = option.lm
    timeout = option.timeout
    # remove equation where all 0
    valididx = np.where(b != 0)
    A = A[valididx]
    b = b[valididx]
    As = As[valididx]
    rows = A.shape[0]
    M = A.shape[0]
    N = x_init.shape[0]
    #print("hhhh")
    if not lm:
        maxiter = datapass * M // batchsize
    else:
        maxiter = datapass
    rpt.registervar('lr')
    rpt.registervar('gnorm')
    rpt.registervar('chi2')
    rpt.registervar('thresvec')
    rpt.registervar('threstime')
    rpt.registervar('epochaccutime')
    rpt.registervar('error')
    rpt.registervar('algosol')
    rpt.error.append(1.)
    rpt.gnorm.append(1.)
    rpt.chi2.append(1.)
    rpt.gdenom = norminf(A.T @ (A @ x_init - b))
    rpt.chi2denom = norm2((A @ x_init - b))
    #rpt.errordenom = norm2(x_init - x_sol)
    rpt.epochaccutime.append(0.0)
    x_new = x_init.copy()
    x_real = x_init.copy()
    eps1 = 1e-16
    eps2 = 1e-6
    threscnt = 0
    itertime = 0.0
    tau = 1.0
    k = 0
    s1 = time.time()
    #print("iiii")
    if lm:
        randidx = np.arange(M)
    else:
        if not importance:
            randidx = np.random.randint(low=0, high=A.shape[0],size=batchsize)
        else:
            randidx = np.random.choice(A.shape[0],batchsize,p=As)

    g =  (A[randidx].T @ (A[randidx]@x_init - b[randidx]))
    h = A[randidx].T @ A[randidx]
    Scur = np.linalg.norm(A[randidx]@x_init-b[randidx])**2
    gnorm = np.linalg.norm(g, ord = np.inf)
    found = gnorm < eps1
    mu = 1 * np.max(np.diag(h))
    x_real = x_init.copy()
    x_new = x_init.copy()
    e1 = time.time()
    v = 2
    itertime += e1-s1
    #print("jjjj")
    while not found and k < maxiter:
        s1 = time.time()
        if not lm:
            if not importance:
                randidx = np.random.randint(low=0, high=A.shape[0], size=batchsize)
            else:
                randidx = np.random.choice(A.shape[0], batchsize, replace=False, p=As)
        k += 1
        tmpA = h + mu * np.identity(x_init.shape[0])
        L, info = lapack.dpotrf(tmpA, lower=1)
        assert(info == 0)
        sol = linalg.solve_triangular(L, -g, lower=1)
        hlm = linalg.solve_triangular(L.T, sol, lower=0)
        if(np.linalg.norm(hlm) < eps2*(np.linalg.norm(x_real)+eps2)):
            found = True
            print("found! hlm norm = %.3f, eps2 = %.3f"%(np.linalg.norm(hlm),eps2*(np.linalg.norm(x_real)+eps2)))
        else:
            for i in range(N):
                x_new[i] += hlm[i]
                if constrained:
                    x_new[i] = max(x_new[i], 0.0001)
            Scurnew = np.linalg.norm(A[randidx]@x_new-b[randidx])**2
            rho = (Scur - Scurnew) / (0.5 * hlm.T @ ( mu * hlm - g))
            if rho > 0:
                x_real = x_new.copy()
                g = (A[randidx].T @ (A[randidx]@x_real - b[randidx]))
                h = A[randidx].T @ A[randidx]
                Scur = Scurnew
                found = np.linalg.norm(g, ord=np.inf) < eps1
                mu = mu * max(1/3., 1 - (2*rho-1)**3)
                v = 2
            else:
                x_new = x_real.copy()
                mu = mu * v
                v = 2*v
        e1 = time.time()
        itertime += e1 - s1
        iitime = e1-s1
        if iitime > 1.0:
            print ("time for this iteration: %.3f"%iitime)
        #print("kkkk")
        #itererr = norm2(x_real - x_sol) / rpt.errordenom
        #if threscnt < len(option.threshold) and itererr < option.threshold[threscnt]:
        #    rpt.thresvec.append(x_real.tolist())
        #    rpt.threstime.append(itertime)
        #    threscnt += 1
        rpt.epochaccutime.append(itertime)
        #rpt.error.append(norm2(x_real - x_sol) / rpt.errordenom)
        tmpgrad = A.T @ (A @ x_real - b)
        rpt.gnorm.append(norminf(tmpgrad) / rpt.gdenom)
        rpt.chi2.append(norm2(A @ x_real - b) / rpt.chi2denom)
        if itertime > timeout:
            print(k)
            break
    print("LM stopped, k = %03d, maxiter = %03d"%(k,maxiter))
    rpt.algosol = list(x_real.copy())

############# more new functions from Yuxi
def builddataset_covmap(optsolver, filename,alt_learn):
    ts = time.time()
    cmmana = np.load(filename)
    dataset = Dataset()
    dataset.A = optsolver.A
    optsolver.b = mat_to_b(cmmana,nwfs,validsubs,shnxsub)
    #optsolver.b = mat_to_b_cmm (cmmana,nwfs,validsubs)
    dataset.b = optsolver.b
    #dataset.x_init = np.array([0.34,0.0200,0.0203,0.060000002,0.003000000,0.0500,0.090000,0.04000,0.05000,0.0500])
    #dataset.x_init = np.random.rand(dataset.A.shape[1])
    #dataset.x_init = np.array([0.59,0.0200,0.04,0.06,0.01,0.0500,0.090000,0.04000,0.05000,0.0500])
    dataset.x_init = np.zeros(alt_learn.shape[0])#,0.0,0.0,0.0,0.000,0.00000,0.0000,0.000,0.000])
    #dataset.x_init = np.ones(10)
    dataset.x_sol = Cn2
    dataset.As = np.linalg.norm(optsolver.A,axis=1)**2
    this_value = (np.average(cmmana.diagonal())/0.17/(500e-9)**2*0.2**(1/3)/(1-1.525*(0.2/25)**(1/3))*2.35044e-11)**(-3/5)
    te = time.time()
    return dataset,this_value,te-ts


def builddataset(optsolver, filename,alt_learn):
    ts     = time.time()
    cmmana = np.load(filename)
    dataset = Dataset()
    dataset.A = optsolver.A
    #optsolver.b = mat_to_b_(cmmana,nwfs,validsubs,shnxsub)
    optsolver.b = mat_to_b_cmm (cmmana,nwfs,validsubs)
    dataset.b = optsolver.b
    #dataset.x_init = np.array([0.34,0.0200,0.0203,0.060000002,0.003000000,0.0500,0.090000,0.04000,0.05000,0.0500])
    #dataset.x_init = np.random.rand(dataset.A.shape[1])
    #dataset.x_init = np.array([0.59,0.0200,0.04,0.06,0.01,0.0500,0.090000,0.04000,0.05000,0.0500])
    dataset.x_init = np.zeros(alt_learn.shape[0])#,0.0,0.0,0.0,0.000,0.00000,0.0000,0.000,0.000])
    #dataset.x_init = np.ones(10)
    dataset.x_sol = Cn2
    dataset.As = np.linalg.norm(optsolver.A,axis=1)**2
    this_value = (np.average(cmmana.diagonal())/0.17/(500e-9)**2*0.2**(1/3)/(1-1.525*(0.2/25)**(1/3))*2.35044e-11)**(-3/5)
    te      = time.time()
    return dataset,this_value,te-ts

def directmethod(dataset, Am1):
    x = Am1@dataset.b
    print("Cn2 calculated from direct method:",x)
    return x

def LMmethod(dataset):
    DATAPASS=100
    TIMEOUT=200
    DATANAME="numerical"
    CONSTRAINED = True
    option = Option()
    option.setattr('datapass', DATAPASS)
    option.setattr('importance',0)
    option.setattr('constrained',CONSTRAINED)
    option.setattr('timeout',TIMEOUT)
    option.setattr('batchsize',0)
    xvalscale = 1.0
    option.setattr('lm',1)
    option.setattr('threshold',[7e-1,2e-1,7e-2,2e-2])
    option.setattr('dataset',DATANAME)
    lmrpt = Reporter('./results/','LM',DATANAME,option)
    #print("gggg")
    LM(dataset, lmrpt, option)
    return lmrpt

##### saving sys config npz file
def save_npz (sup,prefix = "2layer"):
    '''
    saving required parametres from supervisor
    '''
    if type(sup) != shesha.supervisor.compassSupervisor.CompassSupervisor:
        raise Exception('No AO system found!')
    sim  = sup._sim
    tel  = sim.config.p_tel
    geom = sim.config.p_geom
    lgs0 = sim.config.p_wfs_lgs[0]#assume all lgs have same configuration
    atmos= sim.config.p_atmos
    shnxsub = lgs0.get_nxsub()
    x_co = np.round(lgs0.get_validsubsx()/lgs0.get_npix()).astype("int")
    y_co = np.round(lgs0.get_validsubsy()/lgs0.get_npix()).astype("int")
    xx   = (x_co - (np.max(x_co) - np.min(x_co))/2)*lgs0.get_subapd()
    yy   = (y_co - (np.max(y_co) - np.min(y_co))/2)*lgs0.get_subapd()
    validsubs = get_full_index_compass(shnxsub,x_co,y_co)
    nwfs = len(sim.config.p_wfs_lgs)
    CovMapMask = np.tile(MapMask_from_validsubs(validsubs,shnxsub),[2*nwfs,2*nwfs])
    telDiam = tel.get_diam()
    zenith  = geom.get_zenithangle()
    r0      = atmos.get_r0()
    Cn2     = atmos.get_frac()
    l0      = atmos.get_L0()
    alt     = atmos.get_alt()*math.cos(zenith*np.pi/180)
    gspos   = np.zeros([nwfs,2])
    gsalt   = np.zeros([nwfs,1])
    for ii in range(nwfs):
        gspos[ii,0] = sim.config.p_wfs_lgs[ii].get_xpos()
        gspos[ii,1] = sim.config.p_wfs_lgs[ii].get_ypos()
        gsalt[ii] = sim.config.p_wfs_lgs[ii].get_gsalt()

    np.savez("sysconfig_"+prefix+".npz",validsubs=validsubs,xx=xx,yy=yy,telDiam=telDiam,\
              zenith=zenith,shnxsub=shnxsub,r0=r0,Cn2=Cn2,l0=l0,\
              alt=alt,gspos=gspos,gsalt=gsalt,nwfs=nwfs,CovMapMask = CovMapMask)

#############

if __name__ == "__main__":

    sysconfigfile = "sysconfig_2layer.npz"
    npfile = np.load(sysconfigfile)
    print(npfile.files)
    validsubs=npfile['validsubs']
    telDiam = npfile['telDiam']
    zenith = npfile['zenith']
    shnxsub = npfile['shnxsub']
    xx = npfile['xx']
    yy = npfile['yy']
    r0 = npfile['r0']
    Cn2 = npfile['Cn2']
    l0 = npfile['l0']
    alt = npfile['alt']
    nwfs = npfile['nwfs']
    gspos = npfile['gspos']
    gsalt = npfile['gsalt']
    prefix = "2layer"
    #validsubs = npfile['validsubs']

    n_batches = [5,5,5,5]
    Ts        = [5,10,50,100]
    r_decis   = [10,10,10,10]
    #posfact   = [1.0,1.5,2.0,3.0,4.0,8.0]
    #alts      = [200,400,1000,4000]
    that_value = 0.15049503594257221
    cn2list = [np.array(Cn2)*that_value]
    labellist = ["Target"]
    errorlist = []
    cum_errorlist = []
    #
    alt_learn = np.arange(0,20000,1000)
    l0_learn  = np.ones(alt_learn.shape[0])*25.
    #l0_learn  = l0
    #alt_learn = alt
    optsolver_covmap,t_covmap = conf_to_Am1 (telDiam,zenith,validsubs,shnxsub,r0,l0_learn,alt_learn,nwfs,gspos,gsalt)
    print('time to init covmap = %.3f'%t_covmap)   
    #print ("sum of cn2 solution",np.sum(np.array(lmrpt.algosol)))
    markerlist = ['^','*','o','+','x','s','p','d']
    '''
    plt.figure()
    plt.plot(alt, cn2list[0],'^k',markersize = 12,label=labellist[0])
    #cn2sum = np.zeros(alt_learn.shape[0])
    '''
    for kk in range(len(Ts)):
        #CovMapMask = np.tile(MapMask_from_validsubs(validsubs,shnxsub),[2*nwfs,2*nwfs])
        #CovMap_ana = CovMap_from_Cn2(CovMapMask,telDiam,zenith,shnxsub,r0,Cn2,l0,alt,nwfs,gspos*posfact[kk],gsalt)
        #Cmm_ana = Mat_from_CovMap(CovMap_ana,nwfs,validsubs,shnxsub)
        #CMM_ana_full = np.tril(Cmm_ana)+np.tril(Cmm_ana).T - np.diag(Cmm_ana.diagonal())
        #np.save("buffer/cmm_ana_"+prefix+"_%.1f.npy"%(posfact[kk]),CMM_ana_full)


        n_batch = n_batches[kk] #number of cmm to be saved
        T = Ts[kk] # integration time
        r_deci = r_decis[kk] # buffer_every
        
        #for ii in range(n_batch):
        filenames = sorted(glob.glob("buffer/cmm_num_"+prefix+"_T_"+str(T)+"_buffer_every_"+str(r_deci)+"_start_from"+"*.npy"))
        if len(filenames)<n_batch:
            get_num_cmm (n_batch, T, r_deci, framerate=0.001, total_buffer = 1000000, file_size = 50000, prefix = prefix)
            filenames = sorted(glob.glob("buffer/cmm_num_"+prefix+"_T_"+str(T)+"_buffer_every_"+str(r_deci)+"_start_from"+"*.npy"))
        
        #optsolver,t_cmm = conf_to_Am1_cmm (telDiam,zenith,shnxsub,xx,yy,r0,Cn2,l0_learn,alt_learn,nwfs,gspos,gsalt)
        #print('time to init cmm = %.3f'%t_cmm)
        #b = optsolver.A@Cn2
        #CovMap_rec = b_to_map(nwfs,shnxsub,validsubs,b)
        #CovMat = np.load("buffer/Cmat_ana_full.npy")
        #b      = mat_to_b(CovMat,nwfs,validsubs,shnxsub)
        #x      = Am1@b

        
        #idlist  = [filename[-15:-4] for filename in filenames]
        #filenames.append('buffer/Cmat_ana_full_2layer.npy')

        '''
        dataset,this_value,t_load = builddataset(optsolver, 'buffer/Cmat_ana_full_2layer.npy',alt_learn)
        print("t_load = %.3f"%(t_load))

        lmrpt_covmap = LMmethod(dataset_covmap)
        print("gnorm: ",lmrpt_covmap.gnorm[-5:])
        print("chi2: ",lmrpt_covmap.chi2[-5:])
        print ("sum of covmap solusion",np.sum(np.array(lmrpt_covmap.algosol)))
        if len(lmrpt.algosol)>=10:
            plt.plot(alt_learn, cn2list[kk+1],markerlist[kk+1]+'-',markersize = 6,label=labellist[kk+1])
        else:
            plt.plot(alt_learn, cn2list[kk+1],markerlist[kk+1],markersize = 6,label=labellist[kk+1])
        
        plt.figure()
        plt.plot(np.array(lmrpt_covmap.epochaccutime),np.array(lmrpt_covmap.gnorm),"o-",label = "covmap")
        plt.plot(np.array(lmrpt.epochaccutime),np.array(lmrpt.gnorm),"^-",label = "CMM")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(-0.001,1e4)
        plt.ylabel("||g||",fontsize=14) 
        plt.xlabel("time / s",fontsize=14)
        plt.title("Computation time comparison, CovMap v.s. CMM")
        plt.savefig("fig/time_to_solution_map_mat_LM_only_1.png")
        
        gnorm_covmap = np.array([1.0,1.0,1.0])#init and load
        gnorm_covmap = np.append(gnorm_covmap,np.array(lmrpt_covmap.gnorm))
        gnorm        = np.array([1.0,1.0,1.0])
        gnorm        = np.append(gnorm,np.array(lmrpt.gnorm))

        time_covmap  = np.array([0.0,t_covmap,t_load_covmap+t_covmap])
        time_covmap  = np.append(time_covmap,t_load_covmap+t_covmap+np.array(lmrpt_covmap.epochaccutime))
        #time_covmap  = np.cumsum(time_covmap)
        time_cmm  = np.array([0.0,t_cmm,t_cmm+t_load])
        time_cmm  = np.append(time_cmm,t_cmm+t_load+np.array(lmrpt.epochaccutime))
        #time_cmm  = np.cumsum(time_cmm)      
        plt.figure()
        plt.plot(time_covmap,gnorm_covmap,"o-",label = "covmap")
        plt.plot(time_cmm,gnorm,"^-",label = "CMM")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("||g||",fontsize=14) 
        plt.xlabel("time / s",fontsize=14)
        plt.title("Computation time comparison, CovMap v.s. CMM")
        plt.savefig("fig/time_to_solution_map_mat_all.png")
        '''

    
        cnt = 0
        tmp_cn2list = []
        cn2sum = np.zeros(alt_learn.shape[0])
        
        for filename in filenames:

            ts = time.time()

            dataset,this_value,t_covmap = builddataset_covmap(optsolver_covmap, filename,alt_learn)
            print("begin calculating LM for file:",filename[-15:])
            #directcn2 = directmethod(dataset, Am1)
            #cn2list.append(directcn2)
            #labellist.append("direct-T-"+str(T)+"-r-"+str(r_deci)+"_"+idlist[cnt])
            lmrpt = LMmethod(dataset)
            te    = time.time()
            print("LM finished! time taken = %.3f s!"%(te-ts))
            print("gnorm: ",lmrpt.gnorm[-5:])
            print("chi2: ",lmrpt.chi2[-5:])
            print ("sum of cn2 solution: ",np.sum(np.array(lmrpt.algosol)))
            #sumtemp = np.array([np.sum(lmrpt.algosol[0:ii+1]) for ii in range(len(lmrpt.algosol))])
            
            tmp_cn2list.append(np.array(lmrpt.algosol)*this_value)
            cn2sum += np.array(lmrpt.algosol)*this_value
            #labellist.append("LM-T-"+str(T)+"-r-"+str(r_deci)+"_"+idlist[cnt])
            cnt += 1
        
        cn2avg = cn2sum/cnt
        cn2avg_cum = np.cumsum(cn2avg)
        cn2err = np.std(np.array(tmp_cn2list),axis=0)   
        cn2err_cum = np.std(np.cumsum(np.array(tmp_cn2list),axis=1),axis=0)
        print("std of estimation results [last 5 term]:",cn2err[-5:])
        cn2list.append(cn2avg)
        errorlist.append(cn2err)
        cum_errorlist.append(cn2err_cum)
        labellist.append("LM-T-"+str(T)+"-r-"+str(r_deci)+"_nbatch"+str(n_batch))
        
                  
        
    # analytical comes at last  
    dataset_covmap,this_value_covmap,t_load_covmap = builddataset_covmap(optsolver_covmap, "buffer/Cmat_ana_full_2layer.npy",alt_learn)
    print("t_load_covmap = %.3f"%(t_load_covmap))    
    lmrpt = LMmethod(dataset_covmap)
    print("gnorm: ",lmrpt.gnorm[-5:])
    print("chi2: ",lmrpt.chi2[-5:])
    print ("sum of covmap solusion",np.sum(np.array(lmrpt.algosol)))   
    cn2list.append(np.array(lmrpt.algosol)*this_value_covmap)
    labellist.append("analytical, %d layers"%(len(alt_learn)))

    plt.figure()

    for ii in range(len(cn2list)-1):
        if ii == 0:
            #plt.plot(alt, cn2list[ii],'*k',markersize = 12-ii*8,label=labellist[ii])
            plt.step(np.concatenate([[0],np.array(alt),[19000]]),np.cumsum(np.concatenate([[0],np.array(cn2list[ii]),[0]])),linewidth=3,where="post",label="Target")
        else: 
            #plt.plot(alt_learn, np.cumsum(cn2list[ii]),markerlist[ii]+'-',markersize = 6,label=labellist[ii])
            plt.errorbar(alt_learn,np.cumsum(cn2list[ii]),yerr=cum_errorlist[ii-1],fmt=markerlist[ii]+'-',marker =markerlist[ii], ms= 6, mew = 0.5,uplims=True, lolims=True,label=labellist[ii])
    #analytical
    plt.plot(alt_learn, np.cumsum(cn2list[-1]),markerlist[-1]+'-',markersize = 6,label=labellist[-1])
        
        #plt.plot(alt_learn, cn2avg,'*',markersize = 12,label="LM averaged")
        #plt.errorbar(alt_learn,cn2avg_cum,yerr=cn2err,fmt='x',marker ='x', ms= 4, mew =2,uplims=True, lolims=True,label="LM averaged")
        
    plt.legend()
    #plt.ylim(-0.003,0.11)
    plt.grid(b=True, which='major', color='#666666', linestyle='-') 
    plt.title("LM with init = 0.0, T = variable, r = 10, nbatch = "+str(cnt)+", cumulative, alt",fontsize=14)
    #plt.title("LM with init = 0.0, constellation",fontsize=14)
    plt.ylabel(r"$C_n^2$ / m",fontsize=14) 
    plt.xlabel("Layer alt / m",fontsize=14)
    plt.xticks(np.arange(-24000,60000,12000))
        #plt.savefig("fig/LMresult_ana_init_0_learn_100layer.png")
    plt.savefig("fig/LMresult_"+prefix+"_T_"+str(T)+"_buffer_every_v_nbatch_"+str(cnt)+"_init_0_cum_new.png")
    #plt.savefig("fig/LMresult_constellation_init_0_cum.png")
    print("fig saved to "+"fig/LMresult_T_v_buffer_every_"+str(r_deci)+"_nbatch_"+str(cnt)+"_init_0_cum_new.png!")
    np.save("buffer/LMresult_"+prefix+"_T_v_buffer_every_"+str(r_deci)+"_nbatch_"+str(cnt)+"_init_0_new.npy",np.array(cn2list))
    np.save("buffer/LMresult_errorlist"+prefix+"_T_v_buffer_every_"+str(r_deci)+"_nbatch_"+str(cnt)+"_init_0_new.npy",np.array(errorlist))
    np.save("buffer/LMresult_cum_errorlist"+prefix+"_T_v_buffer_every_"+str(r_deci)+"_nbatch_"+str(cnt)+"_init_0_cum_new.npy",np.array(cum_errorlist))
        #np.save("buffer/LMresult_ana_init_0_learn_100layer.npy",np.array(cn2list))
        
        
        
    plt.figure()
        #cn2sum = np.zeros(alt_learn.shape[0])
    for ii in range(len(cn2list)-1):
        if ii == 0:
            plt.plot(alt, cn2list[ii],'^k',markersize = 12-ii*8,label=labellist[ii])
                #plt.step(np.concatenate([[0],np.array(alt),[19000]]),np.concatenate([[0],np.array(cn2list[ii]),[0]]),where="post",label="Target")
        else:
            #plt.plot(alt_learn, cn2list[ii],markerlist[ii]+'-',markersize = 6,label=labellist[ii])
            plt.errorbar(alt_learn,cn2list[ii],yerr=errorlist[ii-1],fmt=markerlist[ii]+'-',marker =markerlist[ii], ms= 6, mew = 0.5,uplims=True, lolims=True,label=labellist[ii])
    plt.plot(alt_learn, cn2list[-1],markerlist[-1]+'-',markersize = 6,label=labellist[-1])
        #cn2avg = cn2sum/cnt
        #cn2LM  = np.array(cn2list[1:])
        #cn2err = np.zeros((2,alt_learn.shape[0]))
        #cn2err[0,:]= np.std(cn2LM,axis=0)
        #cn2err[1,:]= np.std(cn2LM,axis=0)
        #plt.errorbar(alt_learn,cn2avg,yerr=cn2err,fmt='X',marker ='X', ms= 6, mew = 0.5,uplims=True, lolims=True,label="LM averaged")
    plt.legend()
    #plt.xlim(-500,30000)
    plt.grid(b=True, which='major', color='#666666', linestyle='-') 
    plt.title("LM with init = 0.0, T = variable, r = 10, nbatch = "+str(cnt)+",alt",fontsize=14)
    #plt.title("LM with init = 0.0, constellation",fontsize=14)
    plt.ylabel(r"$C_n^2$ / m",fontsize=14) 
    plt.xlabel("Layer alt / m",fontsize=14)
    plt.xticks(np.arange(-24000,60000,12000))
        #plt.savefig("fig/LMresult_ana_init_0_learn_100layer.png")
    plt.savefig("fig/LMresult_"+prefix+"_T_v_buffer_every_"+str(r_deci)+"_nbatch_"+str(cnt)+"_init_0_new.png")
    #plt.savefig("fig/LMresult_constellation_init_0.png")
    #print("fig saved to "+"fig/LMresult_T_"+str(T)+"_buffer_every_v_nbatch_"+str(cnt)+"_init_0.png!")
    #np.save("buffer/LMresult_"+prefix+"_T_v_buffer_every_"+str(r_deci)+"_nbatch_"+str(cnt)+"_init_0_not_cum.npy",np.array(cn2list))
    '''
    plt.legend(loc="upper right")
    plt.ylim(-0.003,0.11)
    plt.grid(b=True, which='major', color='#666666', linestyle='-') 
    plt.title("LM with init = 0.0, analytical covmap")
    #plt.title("LM with init = 0.0, analytical CovMap",fontsize=14)
    plt.ylabel(r"$C_n^2$ / m",fontsize=14) 
    plt.xlabel("Layer alt / m",fontsize=14)
    plt.xticks(np.arange(0,24000,4000))
    #plt.savefig("fig/LMresult_ana_init_0_learn_100layer.png")
    plt.savefig("fig/LMresult_"+prefix+"_analytical_layers.png")
    altlist = [alt] 
    for kk in range(len(alts)): 
        altlist.append(np.arange(0,20000,alts[kk])) 
    plt.figure()
    plt.step(np.concatenate([[0],np.array(alt),[19000]]),np.cumsum(np.concatenate([[0],np.array(cn2list[0]),[0]])),linewidth = 2.5,where="post",label="Target")
    
    for kk in range(1,len(altlist)):
        if altlist[kk].shape[0]>=10:
            plt.plot(altlist[kk],np.cumsum(cn2list[kk]),markerlist[kk]+'-',markersize = 4,label = labellist[kk])
        else:
            plt.plot(altlist[kk],np.cumsum(cn2list[kk]),markerlist[kk],markersize = 6,label = labellist[kk])

    plt.legend(loc="lower right")
    plt.grid(b=True, which='major', color='#666666', linestyle='-') 
    plt.title("LM with init = 0.0, analytical covmap, cumsum")
    plt.ylabel(r"$C_n^2$ / m",fontsize=14) 
    plt.xlabel("Layer alt / m",fontsize=14)
    plt.xticks(np.arange(0,24000,4000))
    plt.savefig("fig/LMresult_"+prefix+"_analytical_layers_cum.png")
    

    '''

    '''
    #below is doing block-wise linear fit for numerical cmm v.s. anlytical cmm

    nfiles = 20
    buffer_every = 100
    nniter = int(nfiles*50000/buffer_every) # floor
    #counter
    rema = 50000%buffer_every
    ndelay = 0
    rand_id = "long_buffer"
    
    #dd = np.load("buffer/dd_"+rand_id+".npy").astype("float32")
    imat = np.load("buffer/imat_"+rand_id+".npy")
    #ss = np.load("buffer/ss_"+rand_id+".npy").astype("float32")
    lgs0_x = np.load("buffer/lgs0x_"+rand_id+".npy")
    lgs0_y = np.load("buffer/lgs0y_"+rand_id+".npy")
    
    
    dd = np.zeros((imat.shape[1],nniter))
    ss = np.zeros((imat.shape[0],nniter))
    
    #global_index_dd = np.where((np.arange(1e6) - ndelay) % buffer_every == 0)[0]
    #global_index_ss = np.where(np.arange(1e6) % buffer_every == 0)[0]
    
    
    
    lazy_counter = 0
    
    for fi in range(nfiles):
        start = time.time()
        this_dd = np.load("buffer/dd_50000_"+str(fi+1)+".npy")
        this_ss = np.load("buffer/ss_50000_"+str(fi+1)+".npy")
        print("processing buffer number "+ str(fi+1))
        this_ind = np.arange(50000)+fi*rema
        this_ind = np.where(this_ind % buffer_every == 0)[0]
        for ind in this_ind:
            dd[:,lazy_counter] = this_dd[:,ind+ndelay]
            ss[:,lazy_counter] = this_ss[:,ind]
            lazy_counter +=1
        end  = time.time()
        print("time taken to process this buffer: %.2f"%(end-start))
    
    print(lazy_counter)
    #np.save("buffer/dd_buffer_every_"+str(buffer_every)+"_ndelay_"+str(ndelay)+".npy",dd)
    #np.save("buffer/ss_buffer_every_"+str(buffer_every)+"_ndelay_"+str(ndelay)+".npy",ss)
    
    dd  = dd.astype("float32")
    ss  = ss.astype("float32")
    #ss = ss[:,:-ndelay]
    #dd = dd[:,ndelay:]
    pols   = (-imat@dd +ss).astype("float32")
    pols   = pols.T 
    pols = pols - pols.mean(axis=0)
    
    cmm = pols.T@pols / (nniter-ndelay)
    dd = []
    ss = []
    pols = []
    cmm_diag = cmm.diagonal()
    
    fig=plt.figure();
    for ii in range(9):
        plt.subplot(3,3,ii+1)
        if ii == 0 :
            plt.plot(cmm_diag);plt.title("All WFS cmm diag");
        else:
            npix = 6
            plt.plot(cmm_diag[1148*2*(ii-1):1148*2*ii]);plt.title("LGS"+str(ii)+": "+str(npix)+" pix");plt.ylim((0.04,0.1));
    plt.subplots_adjust(wspace = 0.5,hspace = 0.4)
    plt.savefig("fig/all_wfs_cmm_diag_"+rand_id+"_delay_"+str(ndelay)+"_every_"+str(buffer_every)+"_nfiles_"+str(nfiles)+".png")
    
    nn  = Norm(0.04,0.08)
    pos_sub = np.array([15,9,3,7,11,17,23,19]).astype(int)
    fig=plt.figure()
    for ii in range(8):
        plt.subplot(5,5,int(pos_sub[ii]))
        plt.scatter(lgs0_x,lgs0_y,c=cmm_diag[1148*(2*ii):1148*(2*ii+1)],marker='x',s=0.7,norm=nn)
        plt.colorbar();plt.title("LGS # "+str(ii+1))
    plt.subplots_adjust(wspace = 0.6,hspace = 0.4)
    plt.savefig("fig/cmm_xx_diag_colored_aligned"+rand_id+"_delay_"+str(ndelay)+"_every_"+str(buffer_every)+"_nfiles_"+str(nfiles)+".png");
    
    fig=plt.figure()
    for ii in range(8):
        plt.subplot(5,5,int(pos_sub[ii]))
        plt.scatter(lgs0_x,lgs0_y,c=cmm_diag[1148*(2*ii)+1148:1148*(2*ii+1)+1148],marker='x',s=0.7,norm=nn)
        plt.colorbar();plt.title("LGS # "+str(ii+1))
    plt.subplots_adjust(wspace = 0.6,hspace = 0.4)
    plt.savefig("fig/cmm_yy_diag_colored_aligned"+rand_id+"_delay_"+str(ndelay)+"_every_"+str(buffer_every)+"_nfiles_"+str(nfiles)+".png");
    
    Cmat_full = np.load("buffer/Cmat_ana_full.npy")
    #Cmat_full = Cmat_full - np.diag(Cmat_full.digonal())
    #cmm       = cmm - np.diag(cmm.diagonal())
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
            print(rand_id+": delay_"+str(ndelay)+"_every_"+str(buffer_every)+"_nfiles_"+str(nfiles)+",\n wfs #"+str(i1)+" with wfs #"+str(i2)+", TT included, slope = %.5f, inte = %.5f, r2 = %.5f"%(sl[ii], inte[ii], r_v[ii]))
    
    
    print("----------------------------")
    print("average slope = %.5f, inte = %.5f, r2 = %.5f"%(np.average(sl),np.average(inte),np.average(r_v)))
    '''
