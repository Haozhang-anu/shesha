import matplotlib
#matplotlib.use('Agg')
import numpy as np
#from docopt import docopt
from matplotlib import pyplot as plt
plt.ion()
from matplotlib.colors import Normalize as Norm
#from shesha.util.CovMap_from_Mat import Map_and_Mat,MapMask_from_validsubs
from shesha.util.CovMap_from_Mat import Map_and_Mat,CovMap_from_Cn2, Mat_from_CovMap, DPHI, get_full_index_compass,MapMask_from_validsubs, CovMap_from_Mat
from scipy import stats as st
import time
import math
import glob
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
            this_dd = np.load("buffer/dd_50000_"+str(this_f)+".npy").astype("float32")
            this_ss = np.load("buffer/ss_50000_"+str(this_f)+".npy").astype("float32")
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
    #print("Cov Map generation finished! Time taken = %.1f seconds."%(time_end-time_start))
    #hdu = fits.PrimaryHDU(CovMap_ana)
    #hdu.writeto("CovMap_ana_full.fits",overwrite=1)
    #CovMap_ana = CovMap_ana * CovMapMask
    Am1 = np.linalg.solve(optsolver.A.T@optsolver.A,optsolver.A.T)
    return Am1,optsolver


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

#############



if __name__ == "__main__":

    sysconfigfile = "sysconfig.npz"
    npfile = np.load(sysconfigfile)
    print(npfile.files)
    validsubs=npfile['validsubs']
    telDiam = npfile['telDiam']
    zenith = npfile['zenith']
    shnxsub = npfile['shnxsub']
    r0 = npfile['r0']
    Cn2 = npfile['Cn2']
    l0 = npfile['l0']
    alt = npfile['alt']
    nwfs = npfile['nwfs']
    gspos = npfile['gspos']
    gsalt = npfile['gsalt']
    validsubs = npfile['validsubs']

    n_batch = 3 #number of cmm to be saved
    T = 250 # integration time
    r_deci = 20 # buffer_every

    get_num_cmm (n_batch, T, r_deci, framerate=0.001, total_buffer = 1000000, file_size = 50000, prefix = "long_buffer")
    Am1,optsolver = conf_to_Am1 (telDiam,zenith,validsubs,shnxsub,r0,l0,alt,nwfs,gspos,gsalt)
    print(Am1.max())
    #b = optsolver.A@Cn2
    #CovMap_rec = b_to_map(nwfs,shnxsub,validsubs,b)
    CovMat = np.load("buffer/Cmat_ana_full.npy")
    b      = mat_to_b(CovMat,nwfs,validsubs,shnxsub)
    x      = Am1@b
    #print("Cn2 calculated from analytical Cmat:",x)
    plt.figure()
    plt.plot(x,'-k',linewidth=4,label='Analytical')

    filenames = sorted(glob.glob("buffer/cmm_num_long_buffer_T_"+str(T)+"_buffer_every_"+str(r_deci)+"*.npy"))
    for filename in filenames:
        CovMat = np.load(filename)
        b      = mat_to_b(CovMat,nwfs,validsubs,shnxsub)
        x      = Am1@b
        #print("Cn2 calculated from file name: "+filename+" is: ",x)
        plt.plot(x,linewidth=3,label=filename[-15:-4])
    plt.legend(loc= "upper right")
    plt.xlabel("layer number")
    plt.ylabel("Cn2 fraction")
    plt.title("Least Square fitting")



    '''
    n_batch = 10 #number of cmm to be saved
    T = 50 # integration time
    r_deci = 40 # buffer_every

    get_num_cmm (n_batch, T, r_deci, framerate=0.001, total_buffer = 1000000, file_size = 50000, prefix = "long_buffer")
    


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
