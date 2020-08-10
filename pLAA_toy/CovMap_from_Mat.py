
'''
This file contains tools to generate CovMap from a full CovMat,
and also tools to generate CovMap from Cn2 profile, layer, altitude and constellation.

The first part is adapted from D. J. Laidlaw's PhD Thesis work, which uses a set
of geometric transformation to align all covariance data points.
The second part is adapted from E. Gendron's code, originally written in YORICK.


NOTE: At this stage, this code works only when geometry of all WFSs are the same.
      Anyway, if WFS has different dimensions, CovMap will be more confusing.
      [Specifically, when 2 WFS has different # of subapertures,
      the "steplength" of CovMap will become messy.]

Structure of this file is explained below:
        part 1: Getting proper index from YAO, counterpart for COMPASS will be updated
        part 2: Cov Map from Cov Mat
        part 3: Cov Mat from Cov Map
        part 4: Cov Map from Cn2 + WFS constellation

All required inputs for CovMap from Mat:
        CovMat: np 2D array, full matrix;
        nwfs: int, # of wfs;
        shnxsub: int, # of subapertures along pupil;
        validsubs: np 1D 0-1 array, size (shnxsub[i]**2,),
                   in which all "1"s represents a valid subaperture in a square pupil space;

For MAVIS configuration from YAO:
        CovMat = Cmm;
        nwfs = 8;
        shnxsub = np.array([40,]*8)
        validsubs = get_full_index(prefix="mavis_TLR")
        ##### This is because yao_wfs._validsubs is not in full square pupil.

All required inputs for CovMap from Cn2:
        CovMapMask: np 2D array, same size as CovMap in X or Y direction, 0-1 valued,
		    marks all valid XY displacements in CovMap
        telDiam: float, diametre of telescope in metre
        zenith: float, zenith angle in degree
        shnxsub: int, # of subaperture along pupil
        r0: float, overall r0 in metre
        Cn2: np 1D array, same size as alt, Cn2 fraction of every layer in percentage
        l0: float, outer scale in metre
        alt: np 1D array, altitudes for every layer, alt.shape[0] == n_layers
        nwfs: int, # of WFSs
        gspos: np 2D array, X/Y position of every WFS in arcsec, size [nwfs,2]
        gsalt: np 1D array, altitude of every WFS in metres, 0 for NGS, size [nwfs,1]

####### Please Note that Cov Map trick is precise only when 2 WFS has the same altitude
####### Because cone effect will bring a difference to X/Y displacement in altitude layer
####### At this stage this is not expected to be a big issue because all LGS are assumed to
####### have the same altitude, and NGS/TS is not considered in CovMap
'''
import time
import numpy as np
from astropy.io import fits
from os import path
import os
import itertools
import math
import sys
#import pycuda.autoinit
#import skcuda.linalg as linalg
#from pycuda import gpuarray
#linalg.init()
try:
    import shesha
except ImportError:
    print("SHESHA not installed, skipping!")
    pass

#####=======
#####Part 1: transfer parameters from YAO or COMPASS to required format

def get_full_index (prefix):
    '''
    Yorick profile only: get valid subapertures from wfs._x, wfs._y, wfs._validsubs, shnxsub
    '''
    if not path.exists(prefix+"_coord_x.fits"):
        os.system("yorick -i save_coords_CovMap.i "+prefix)
    x_co = fits.getdata(prefix+"_coord_x.fits")
    y_co = fits.getdata(prefix+"_coord_y.fits")
    vldsb = fits.getdata(prefix+"_validsubs.fits")
    shnxsub = int(fits.getdata(prefix+"_shnxsub.fits")[0])
    telDiam = float(fits.getdata(prefix+"_shnxsub.fits")[1])
    subAperSize = telDiam/shnxsub
    validsubs = np.zeros((shnxsub,shnxsub))
    '''
    xi=yi = np.arange((-telDiam+subAperSize)/2,(telDiam+subAperSize)/2,subAperSize)
    Xi,Yi = np.meshgrid(xi,yi)
    '''
    Xmin=Ymin = (-telDiam+subAperSize)/2
    for i in range(vldsb.shape[0]):
        if vldsb[i] == 1:
            xi = (x_co[i]-Xmin)/subAperSize
            yi = (y_co[i]-Ymin)/subAperSize
            #print(xi,yi)
            xi = np.round(xi).astype('int')
            yi = np.round(yi).astype('int')
            validsubs[xi,yi] = 1

    return validsubs.flatten()

def get_full_index_compass (shnxsub,x_co,y_co):
    '''
    Compass profile only, must initialise configuration file first
    loads configs directly from supervisor._sim
    
    return
    
    validsubs: np 1D array, size (shnxsub**2,1), valid subapertures index array
    '''
    vall = y_co*shnxsub+x_co
    validsubs = np.zeros((shnxsub,shnxsub)).flatten()
    validsubs[list(vall)] = 1
    return validsubs

def MapMask_from_validsubs (validsubs,shnxsub):
    '''
    calculate valid CovMapMask directly from validsubs and shnxsub, i.e. no need to go through a matrix
    there must be cleverer ways to do so but here I just make use of trans matrix

    return
    CovMapMaskBlock: a block of CovMapMask, need to be tiled before use 
    '''

    IndexMatrix = get_IndexMatrix(validsubs)
    Trans_matrix_XY = shift_along_X(shift_along_Y(IndexMatrix,shnxsub),shnxsub)
    Trans_avg_denom = np.sum(Trans_matrix_XY,0)
    Trans_avg_denom[np.where(Trans_avg_denom > 1)] = 1
    return Trans_avg_denom.reshape((2*shnxsub-1,2*shnxsub-1))



#####======
##### Part 2: Cov Mat to Cov Map

def get_IndexMatrix (validsubs):
    '''
    get_IndexMatrix (validsubs)

    getting square Index Matrix

    validsubs: np 1D array, must be the length (shnxsub**2,1)
    '''
    IdxM1 = np.zeros((validsubs.shape[0],validsubs.shape[0]))
    IndexMatrix = np.zeros((validsubs.shape[0],validsubs.shape[0]))
    for i in range(validsubs.shape[0]):
        if validsubs[i] == 1:
            IdxM1[i,] = 1

    for i in range(validsubs.shape[0]):
        if validsubs[i] == 1:
            IndexMatrix[:,i] = IdxM1[:,i]

    return IndexMatrix


def shift_along_Y (IndexMatrix,shnxsub):
    '''
    shift_along_Y (IndexMatrix,shnxsub)

    Shift a square Index Matrix according to Y displacement
    IndexMatrix: square matrix marking all valid covariance points
    shnxsub: # of subapertures along pupil

    Note: IndexMatrix.shape[0] == shnxsub**2 => True
    '''
    if not IndexMatrix.shape[0] == shnxsub**2:
        raise Exception('dimensions not aligned!')
    n_deltaY = 2*shnxsub - 1
    dimY = n_deltaY*shnxsub

    Trans_matrix_Y = np.zeros((IndexMatrix.shape[0],dimY))
    Trans_matrix_Y[:,IndexMatrix.shape[0]-shnxsub:] = IndexMatrix

    for i in range(shnxsub):
        Trans_matrix_Y[i*shnxsub:(i+1)*shnxsub] = np.roll(Trans_matrix_Y[i*shnxsub:(i+1)*shnxsub], -i*shnxsub)

    return Trans_matrix_Y


def shift_along_X (Trans_matrix_Y,shnxsub):
    '''
    shift_along_X (Trans_matrix_Y,shnxsub)

    Shift a Y-shifted index matrix according to X displacement

    Trans_matrix_Y: Y-shifted index matrix, dims (shnxsub**2,(2*shnxsub - 1)*shnxsub)
    shnxsub: # of subapertures along pupil
    '''
    if not Trans_matrix_Y.shape[0] == shnxsub ** 2 or not Trans_matrix_Y.shape[1] == (2*shnxsub - 1) * shnxsub:
        raise Exception('dimensions not aligned!')

    n_deltaY = 2*shnxsub - 1
    n_deltaXY = n_deltaY ** 2
    T_mat_XY1 = np.zeros((Trans_matrix_Y.shape[0],n_deltaXY))
    Trans_matrix_XY = np.zeros((Trans_matrix_Y.shape[0],n_deltaXY))
    for i in range(n_deltaY):
        T_mat_XY1[:,i*n_deltaY+shnxsub-1:i*n_deltaY+2*shnxsub-1] = Trans_matrix_Y[:,i*shnxsub:(i+1)*shnxsub]

    for Yi in range(shnxsub):
        for Xi in range(shnxsub):
            Trans_matrix_XY[Xi + Yi*shnxsub] = np.roll(T_mat_XY1[Xi + Yi*shnxsub], -Xi)

    return Trans_matrix_XY


def get_CovMap_matrices(validsubs,shnxsub):
    '''
    get_CovMap_matrices(validsubs,shnxsub)

    get everything required for Cov Map calculation: Trans_matrix_XY, Trans_index, Trans_avg_denom

    validsubs: np 1D array, must be the length (shnxsub**2,1)
    shnxsub: int, # of subapertures along pupil
    '''
    IndexMatrix = get_IndexMatrix(validsubs)
    Trans_matrix_XY = shift_along_X(shift_along_Y(IndexMatrix,shnxsub),shnxsub)

    Trans_index = np.where(Trans_matrix_XY == 1)

    Trans_avg_denom = np.sum(Trans_matrix_XY,0)
    Trans_avg_denom[np.where(Trans_avg_denom == 0)] = 1
    return Trans_matrix_XY.astype('float64'), Trans_index, Trans_avg_denom.astype('float64')


def CovMap_block(CovBlock,validsubs,shnxsub,Trans_matrix_XY,Trans_index,Trans_avg_denom):
    '''
    CovMap_block(CovBlock,validsubs,shnxsub)

    get CovMap from a Covariance block [either XX block or YY block]

    CovBlock: np 2D array, must be of size [int(np.sum(validsubs)),int(np.sum(validsubs))]
    validsubs: np 1D array, must be the length (shnxsub**2,1)
    shnxsub: int, # of subapertures along pupil
    Trans_matrix_XY,Trans_index,Trans_avg_denom: Transformation matrices, calculated only once

    return:
    CovMap: square matrix of size [2*shnxsub - 1,2*shnxsub - 1]
    '''
    n_deltaY = 2*shnxsub - 1

    Trans_Cov = Trans_matrix_XY.copy()
    Trans_Cov[Trans_index[0],Trans_index[1]] = CovBlock.flatten()

    CovMap_array = np.sum(Trans_Cov, 0)/Trans_avg_denom

    CovMap = CovMap_array.reshape(n_deltaY,n_deltaY)

    return CovMap


def CovMap_from_Mat(CovMat,nwfs,validsubs,shnxsub):
    '''
    CovMap_from_Mat(CovMat,nwfs,validsubs,shnxsub)

    get CovMaps for all combinations of 2 WFS, given the number of toal WFS

    CovMat: np 2D array, full CovMat, square, dimension[2*nwfs*int(np.sum(validsubs)),2*nwfs*int(np.sum(validsubs))]
    nwfs: int, # of WFS
    validsubs: np 1D array, must be the length (shnxsub**2,1)
    shnxsub: int, # of subapertures along pupil

    return
    CovMaps: np 2D array, concatenated CovMaps for XX,YY covariance and all combinations,
                 dimension [n_deltaY*nwfs*2,n_deltaY*nwfs*2]
    CovMapMask: np 2D array, same size as CovMaps_all, 0-1 valued indexing array for CovMaps_all
    '''

    time_start = time.time()
    #selector_range = np.array((range(nwfs)))
    #selector_all = np.array((list(itertools.combinations(selector_range,2))))
    #n_combs = selector_all.shape[0]
    n_validsubs = int(np.sum(validsubs))
    n_deltaY = 2*shnxsub - 1

    Trans_matrix_XY,Trans_index,Trans_avg_denom = get_CovMap_matrices(validsubs,shnxsub)
    time_mid = time.time()
    print("computation of Transformation Matrices took %.1f seconds!"%(time_mid-time_start))
    #CovMaps = np.zeros([n_deltaY,3*n_deltaY,n_combs+nwfs]) ### +nwfs for main diagonal, but are usually identical to each other
    CovMaps = np.zeros([n_deltaY*nwfs*2,n_deltaY*nwfs*2])

    for wfs_1 in range(nwfs):
        for wfs_2 in range(wfs_1,nwfs):
            CovBlockXX = CovMat[2*n_validsubs*wfs_2:2*n_validsubs*wfs_2+n_validsubs,2*n_validsubs*wfs_1:2*n_validsubs*wfs_1+n_validsubs]
            CovBlockYY = CovMat[2*n_validsubs*wfs_2+n_validsubs:2*n_validsubs*(wfs_2+1),2*n_validsubs*wfs_1+n_validsubs:2*n_validsubs*(wfs_1+1)]

            CovBlockXY = (CovMat[2*n_validsubs*wfs_2:2*n_validsubs*wfs_2+n_validsubs,2*n_validsubs*wfs_1+n_validsubs:2*n_validsubs*(wfs_1+1)] +\
                          CovMat[2*n_validsubs*wfs_2+n_validsubs:2*n_validsubs*(wfs_2+1),2*n_validsubs*wfs_1:2*n_validsubs*wfs_1+n_validsubs])/2


            CovMapXX = CovMap_block(CovBlockXX,validsubs,shnxsub,Trans_matrix_XY,Trans_index,Trans_avg_denom)
            CovMapYY = CovMap_block(CovBlockYY,validsubs,shnxsub,Trans_matrix_XY,Trans_index,Trans_avg_denom)

            CovMapXY = CovMap_block(CovBlockXY,validsubs,shnxsub,Trans_matrix_XY,Trans_index,Trans_avg_denom)


            CovMaps[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] = CovMapXX
            CovMaps[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] = CovMapYY
            CovMaps[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] = \
            CovMaps[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] = CovMapXY
    
    Trans_avg = np.sum(Trans_matrix_XY,0)
    CovMapMask = Trans_avg.reshape(n_deltaY,n_deltaY)
    CovMapMask[np.where(CovMapMask>0)] = 1
    CovMapMask = np.tile(CovMapMask,(nwfs*2,nwfs*2))
    time_end = time.time()
    print("computation of Cov Maps took %.1f seconds!"%(time_end-time_start))
    return CovMaps.astype('float64'), CovMapMask.astype('float64')

#####======
##### Part 3: Cov Map back to Cov Mat

def CovBlock_from_Map(CovMap,n_validsubs,shnxsub,Trans_index):
    '''
    CovBlock_from_Map(CovMap,n_validsubs,shnxsub,Trans_index)

    get covariance matrix block for a given Cov Map, Map is required to be square

    CovMap: np 2d array, a sqare CovMap of either XX/XY/YY covariance
    n_validsubs: int, the value of np.sum(validsubs)
    shnxsub: int, # of subapertures along pupil
    Trans_index: transformation matrix valid index calculated from WFS configuration

    return
    CovBlock: np 2d array, square, a block in the full covariance matrix
    '''
    CovMapFlatten = CovMap.flatten()
    Trans_Cov     = np.array([CovMapFlatten,]*(shnxsub**2))

    return Trans_Cov[Trans_index].reshape((n_validsubs,n_validsubs))


def Mat_from_CovMap(CovMaps,nwfs,validsubs,shnxsub):
    '''
    Mat_from_CovMap(CovMaps,nwfs,validsubs,shnxsub)

    get Covariance Matrix Triangle for CovMaps,

    CovMaps: np 2D array, concatenated CovMaps for XX,YY covariance and all combinations,
             dimension [n_deltaY*nwfs*2,n_deltaY*nwfs*2]
    nwfs: int, # of WFS
    validsubs: np 1D array, must be the length (shnxsub**2,1)
    shnxsub: int, # of subapertures along pupil

    return

    CovMat: np 2D array, top right triangle of full covariance matrix, square
    '''
    time_start = time.time()

    n_validsubs = int(np.sum(validsubs))
    n_deltaY = 2*shnxsub - 1

    Trans_matrix_XY,Trans_index,Trans_avg_denom = get_CovMap_matrices(validsubs,shnxsub)
    time_mid = time.time()
    print("computation of Transformation Matrices took %.1f seconds!"%(time_mid-time_start))

    CovMat = np.zeros((n_validsubs*nwfs*2,n_validsubs*nwfs*2))


    for wfs_1 in range(nwfs):
        for wfs_2 in range(wfs_1,nwfs):
            this_CovMapXX = CovMaps[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY]
            this_CovMapYY = CovMaps[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)]
            this_CovMapXY = CovMaps[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)]

            CovMatXX = CovBlock_from_Map(this_CovMapXX,n_validsubs,shnxsub,Trans_index)
            CovMatYY = CovBlock_from_Map(this_CovMapYY,n_validsubs,shnxsub,Trans_index)
            CovMatXY = CovBlock_from_Map(this_CovMapXY,n_validsubs,shnxsub,Trans_index)

            CovMat[2*n_validsubs*wfs_2:2*n_validsubs*wfs_2+n_validsubs,2*n_validsubs*wfs_1:2*n_validsubs*wfs_1+n_validsubs] = CovMatXX
            CovMat[2*n_validsubs*wfs_2+n_validsubs:2*n_validsubs*(wfs_2+1),2*n_validsubs*wfs_1+n_validsubs:2*n_validsubs*(wfs_1+1)] = CovMatYY
            CovMat[2*n_validsubs*wfs_2:2*n_validsubs*wfs_2+n_validsubs,2*n_validsubs*wfs_1+n_validsubs:2*n_validsubs*(wfs_1+1)] =\
            CovMat[2*n_validsubs*wfs_2+n_validsubs:2*n_validsubs*(wfs_2+1),2*n_validsubs*wfs_1:2*n_validsubs*wfs_1+n_validsubs] = CovMatXY

    time_end = time.time()
    print("computation of Cov Mat took %.1f seconds!"%(time_end-time_start))
    return CovMat.astype('float64')


#####======
##### Part 4: Cn2 to Cov Map


def macdo(r,k = 10):
    '''
    macdo(r,k = 10)

    MacDonald function:
    f(x) = x**(5/6)*K_{5/6}(x), using a series for the estimation of K_{5/6},
    originally from Rod Conan Thesis
    K_a(x) = 0.5*sum_{n=0}^{infi}frac{(-1)**n}{n!}
             [(Gamma(-n-a) (x/2)^{2n+a} + Gamma(-n+a) (x/2)^{2n-a}]
    a      = 5/6

    '''
    a  = 5/6
    fn = 1
    r2a= r**(2*a)
    r22= r*r/4
    r2n= 0.5
    Ga = 2.01126983599717856777
    Gma = -3.74878707653729348337
    s = np.zeros(r.shape)

    for n in range(k+1):
        dd = Gma * r2a
        if n > 0:
            dd += Ga
        dd = dd*r2n
        dd = dd/fn

        if n%2 :
            s = s - dd
        else:
            s = s + dd

        if n<k:
            fn  = fn*(n+1)
            Gma = Gma/(-a-n-1)
            Ga  = Ga/(a-n-1)
            r2n = r2n*r22

    return s

def asymp(r):
    '''
    asymp(r)

    MacDonald function when x -> infi
    x must be > 0

    '''
    # k2 = gamma(5./6)*2**(-1./6)
    k2 = 1.00563491799858928388289314170833
    k3 = 1.25331413731550012081
    a1 = 0.22222222222222222222
    a2 = -0.08641975308641974829
    a3 = 0.08001828989483310284
    rm1= 1/r
    res= k2 - k3*np.exp(-r)*(r**(1/3.))*(1 + rm1*(a1 + rm1*(a2 + rm1*a3)))
    return res



def RodConan (RR,l0):
    '''
    RodConan (RR,l0)

    calculates phase structure function for different r
    for small r, use macdo()
    for large r, use asymp()

    '''
    # k1 = 2*gamma_R(11./6)*2^(-5./6)*pi^(-8./3)*(24*gamma_R(6./5)/5.)^(5./6)
    k1 = 0.1716613621245709486
    dprf0 = (2*np.pi/l0)*RR
    res = RR
    xlim = 0.75*2*np.pi
    isLarge = np.where(dprf0>xlim)
    isSmall = np.where(dprf0<=xlim)
    if len(isLarge[0])>0:
        res[isLarge] = asymp(dprf0[isLarge])
    if len(isSmall[0])>0:
        res[isSmall] = -macdo(dprf0[isSmall])

    return res*k1*l0**(5/3)




def DPHI (XX,YY,l0):
    '''
    DPHI (XX,YY,l0)

    calculate phase structure function for coordinates XX and YY
    r0 is not considered here, so result shall be scaled by r0**(-5/3)

    '''
    RR = np.sqrt(XX**2+YY**2)

    return RodConan(RR,l0)

def CovMap_from_Cn2 (CovMapMask,telDiam,zenith,shnxsub,r0,Cn2,l0,alt,nwfs,gspos,gsalt,timedelay=0,windspeed=[],winddir=[]):
    '''
    CovMap_from_Cn2 (CovMapMask,telDiam,zenith,shnxsub,r0,Cn2,l0,alt,nwfs,gspos,gsalt,timedelay=0,windspeed=[],winddir=[])

    get analytical Cov Map from Cn2 profile and system configuration
    loop over all combinations of WFS and layers with the help of Von Karman and Rod Conan
    wavelength is fixed at 500 nm, please check r0 before use

    Update: Temporal difference included.

    CovMapMask: np 3D array, same size as CovMaps_all, indexing array
    telDiam: float, telescope diametre in metre
    zenith: float, zenith angle in degree
    shnxsub: int, number of subapertures along pupil diamtre
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

    CovMap_ana: np 3D array, analytical covariance map for all valid WFS pairs and X/Y displacement pairs


    '''
    dtor   = np.pi/180
    astor  = dtor/60/60
    zenith = zenith*dtor
    gspos  = gspos*astor
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

    CovMap_ana = np.zeros(CovMapMask.shape)
    x0 = y0    = np.arange(-shnxsub+1,shnxsub,1)
    X0,Y0      = np.meshgrid(x0,y0)
    X0         = X0*subAperSize
    Y0         = Y0*subAperSize
    n_deltaY   = Y0.shape[0]
    k         = 1/subAperSize/subAperSize
    lambda2   = (0.5e-6/2/np.pi/astor)**2

    time_start = time.time()
    print("Begin calculating Cov Map from Cn2")
    for wfs_1 in range(nwfs):
        for wfs_2 in range(wfs_1,nwfs):
            #print("begin calculating for WFS # "+str(wfs_1)+" and WFS #"+str(wfs_2)+" !")
            gsalt_1 = gsalt[wfs_1]
            gsalt_2 = gsalt[wfs_2]

            for li in range(alt.shape[0]):

                layer_alt = alt[li]
                l0_i      = l0[li]
                Cn2h      = r0**(-5/3)*Cn2[li]
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
                Cov_XX = Cov_XX *k*lambda2*np.abs(Cn2h)
                Cov_YY = (-2 * DPHI(Xi,Yi,l0_i) + DPHI(Xi,subAperSize_i+Yi,l0_i) + DPHI(Xi,-subAperSize_i+Yi,l0_i))*0.5
                Cov_YY = Cov_YY *k*lambda2*np.abs(Cn2h)

                Cov_XY = -DPHI(Xi+np.sqrt(2)*subAperSize_i/2,Yi-np.sqrt(2)*subAperSize_i/2,l0_i) +\
                          DPHI(Xi+np.sqrt(2)*subAperSize_i/2,Yi+np.sqrt(2)*subAperSize_i/2,l0_i) +\
                          DPHI(Xi-np.sqrt(2)*subAperSize_i/2,Yi-np.sqrt(2)*subAperSize_i/2,l0_i) -\
                          DPHI(Xi-np.sqrt(2)*subAperSize_i/2,Yi+np.sqrt(2)*subAperSize_i/2,l0_i)

                Cov_XY = Cov_XY/4
                Cov_XY = Cov_XY *k*lambda2*np.abs(Cn2h)

                CovMap_ana[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] += Cov_XX
                CovMap_ana[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] += Cov_YY
                CovMap_ana[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] += Cov_XY
                CovMap_ana[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] += Cov_XY

    time_end = time.time()
    print("Cov Map generation finished! Time taken = %.1f seconds."%(time_end-time_start))
    hdu = fits.PrimaryHDU(CovMap_ana)
    hdu.writeto("CovMap_ana_full.fits",overwrite=1)
    CovMap_ana = CovMap_ana * CovMapMask
    return CovMap_ana.astype('float64')



def CMM_from_Cn2 (telDiam,zenith,shnxsub,xx,yy,r0,Cn2,l0,alt,nwfs,gspos,gsalt,timedelay=0,windspeed=[],winddir=[]):
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
    for wfs_1 in range(nwfs):
        for wfs_2 in range(wfs_1,nwfs):
            #print("begin calculating for WFS # "+str(wfs_1)+" and WFS #"+str(wfs_2)+" !")
            gsalt_1 = gsalt[wfs_1]
            gsalt_2 = gsalt[wfs_2]

            for li in range(alt.shape[0]):

                layer_alt = alt[li]
                l0_i      = l0[li]
                Cn2h      = r0**(-5/3)*Cn2[li]
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

                        X1i = (1-layer_alt/gsalt_1)*X0 - layer_alt*gspos[wfs_1,0] + dx_li_wind
                        Y1i = (1-layer_alt/gsalt_1)*Y0 - layer_alt*gspos[wfs_1,1] + dy_li_wind
                        X2i = (1-layer_alt/gsalt_2)*X0 - layer_alt*gspos[wfs_2,0] #- dx_li_wind
                        Y2i = (1-layer_alt/gsalt_2)*Y0 - layer_alt*gspos[wfs_2,1] #- dy_li_wind
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
                
                Cov_XX = (-DPHI(Xi+ac,Yi,l0_i) + DPHI(Xi+ad,Yi,l0_i) + DPHI(Xi+bc,Yi,l0_i)-DPHI(Xi+bd,Yi,l0_i))*0.5
                Cov_XX = Cov_XX *k*lambda2*np.abs(Cn2h)
                Cov_YY = (-DPHI(Xi,Yi+ac,l0_i) + DPHI(Xi,Yi+ad,l0_i) + DPHI(Xi,Yi+bc,l0_i)-DPHI(Xi,Yi+bd,l0_i))*0.5
                Cov_YY = Cov_YY *k*lambda2*np.abs(Cn2h)
                s0 = np.sqrt(s1**2+s2**2)/2
                #subAperSize_i = s1

                Cov_XY = -DPHI(Xi+s0,Yi-s0,l0_i) +\
                          DPHI(Xi+s0,Yi+s0,l0_i) +\
                          DPHI(Xi-s0,Yi-s0,l0_i) -\
                          DPHI(Xi-s0,Yi+s0,l0_i)

                Cov_XY = Cov_XY/4
                Cov_XY = Cov_XY *k*lambda2*np.abs(Cn2h)


                CMM_ana[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] += Cov_XX
                CMM_ana[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] += Cov_YY
                CMM_ana[n_deltaY*2*wfs_1:n_deltaY*2*wfs_1+n_deltaY,n_deltaY*2*wfs_2+n_deltaY:n_deltaY*2*(wfs_2+1)] += Cov_XY
                CMM_ana[n_deltaY*2*wfs_1+n_deltaY:n_deltaY*2*(wfs_1+1),n_deltaY*2*wfs_2:n_deltaY*2*wfs_2+n_deltaY] += Cov_XY

    time_end = time.time()
    print("CMM generation finished! Time taken = %.1f seconds."%(time_end-time_start))
    #hdu = fits.PrimaryHDU(CovMap_ana)
    #hdu.writeto("CovMap_ana_full.fits",overwrite=1)
    #CovMap_ana = CovMap_ana * CovMapMask
    CMM_ana_full = np.tril(CMM_ana.T)+np.tril(CMM_ana.T).T - np.diag(CMM_ana.diagonal())
    return CMM_ana_full.astype('float32')

def CMM_from_npz (prefix = 'mavis',dT=0):
    '''
    load config npz file and prepares for CMM_from_Cn2 ()
    telDiam,zenith,shnxsub,xx,yy,r0,Cn2,l0,alt,nwfs,gspos,gsalt,timedelay=0,windspeed=[],winddir=[]
    
    dT is the number of frames to delay, so the time delay in seconds will be dT*ittime

    '''
    sysconfigfile = "sysconfig_"+prefix+".npz"
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
    windspeed = npfile['windspeed']
    winddir = npfile['winddir']
    ittime = npfile['ittime']
    return CMM_from_Cn2 (telDiam,zenith,shnxsub,xx,yy,r0,Cn2,l0,alt,nwfs,gspos,gsalt,
                            timedelay=dT*ittime,windspeed=windspeed,winddir=winddir)



#####=====
##### Part 5: All in one function for Compass users
#####         after initialisation, run Map_and_Mat()
#####         to get CovMap and Cmm Matrix
def Map_and_Mat (sup):
    '''
    generate CovMap and Cmm Matrix with current initialisation,
    prepares and calls CovMap_from_Cn2();

    return
    CovMap_ana: np 2D array, CovMap for all LGS
    Cmm_ana: np 2D array, Cmm matrix recovered from CovMap_ana
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

    # all parametres ready!
    CovMap_ana = CovMap_from_Cn2(CovMapMask,telDiam,zenith,shnxsub,r0,Cn2,l0,alt,nwfs,gspos,gsalt)
    Cmm_ana = Mat_from_CovMap(CovMap_ana,nwfs,validsubs,shnxsub)

    return CovMap_ana.astype('float64'),Cmm_ana.astype('float64')


##### saving sys config npz file
def save_npz (sup,prefix = "mavis"):
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
    # x_co, y_co all in subapertures, shall be centred and scaled
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
    npfile = np.load("sysconfig_"+prefix+".npz")
    print(npfile.files)
#####======
##### Part 6: Merging test for numerical covariance
#####         Projection mat: taking par file and return square matrix
#####         Mat from Buffer: getting cov matrix from cbmes for a series of frame delay


##### function for generating quadratic modes projection matrix
def gen_quad_mat (prefix,wfs_type="MNGS"):
    ### this function should take the altitude of highest DM, coords for all wfs
    ### and return a 6-column mat, rotation included.
    as2r = 4.84814e-6
    if not os.path.exists("mngsx_"+prefix+".fits"):
        os.system("yorick -i save_coords_merge.i "+prefix)
    gsposfile = "gspos_"+prefix+".fits"
    dmaltfile = "dmalt_"+prefix+".fits"
    nwfsfile  = "nwfs_"+prefix+".fits"
    gspos = fits.getdata(gsposfile) ## pos for LGS+NGS, last 3 WFSs are NGS
    dmalt = fits.getdata(dmaltfile)
    nlgs = fits.getdata(nwfsfile)[0]
    nngs = fits.getdata(nwfsfile)[1]
    alt = 15000 # consider quad modes on highest DM altitude
    #alt = 40000 # consider quad modes on highest DM altitude
    if (wfs_type == "LGS"):
        ### LGS coords, here we start from a 0 altitude
        wfsxfile = "lgsx_"+prefix+".fits"
        wfsyfile = "lgsy_"+prefix+".fits"
        wfs_x = fits.getdata(wfsxfile)
        wfs_y = fits.getdata(wfsyfile)
        # number of LGS?
        nwfs = nlgs
        gsaltfile = "gsalt_"+prefix+".fits"
        gsalts = fits.getdata(gsaltfile) # alt for all LGSs
        gspos = gspos * as2r
        for ii in range(nwfs):
            # for LGSs, get shifted and scaled x/y coord
            gsalt = gsalts[ii]
            if (gsalt == 0):
                #x_coord = wfs_x  + alt * gspos[ii,0]
                #y_coord = wfs_y  + alt * gspos[ii,1]
                cone_eff = 1
            else :
                #x_coord = wfs_x *(gsalt-alt)/gsalt + alt * gspos[ii,0]
                #y_coord = wfs_y *(gsalt-alt)/gsalt + alt * gspos[ii,1]
                cone_eff = (gsalt-alt)/gsalt
            #print(cone_eff)
            cone_var1 = 1 - cone_eff**2
            cone_var0 = cone_eff
            tt_term1 = np.ones(wfs_x.shape)
            tt_term0 = np.zeros(wfs_x.shape)
            # get proj arrays, for LGS, cone effect shall be considered
            # now we are just dealing with 0 altitude situation
            ###### Plate Scale projection
            cone_var1 = cone_eff
            proj_x = np.array([tt_term1*2,tt_term0,\
                              4*math.sqrt(3)*(cone_var1**2 *wfs_y*0+cone_var1*alt*gspos[ii,0]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_y*0+cone_var1*alt*gspos[ii,1]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_x*0+cone_var1*alt*gspos[ii,0])])
            proj_y = np.array([tt_term0,tt_term1*2,\
                              4*math.sqrt(3)*(cone_var1**2 *wfs_y*0+cone_var1*alt*gspos[ii,1]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_x*0+cone_var1*alt*gspos[ii,0]),\
                              -2*math.sqrt(6)*(cone_var1**2 *wfs_y*0+cone_var1*alt*gspos[ii,1])])

            if (ii == 0):
                #initialise proj_mat
                proj_mat = np.concatenate((proj_x,proj_y),axis=1)
                #print(proj_mat.shape)
            else :
                #proj_mat = np.array([proj_mat,proj_x,proj_y])
                proj_mat = np.concatenate((proj_mat,proj_x),axis=1)
                proj_mat = np.concatenate((proj_mat,proj_y),axis=1)
                #print(proj_mat.shape)
        print(proj_mat.shape)
        return proj_mat

    if (wfs_type == "MNGS"):
        ### MNGS coords, here we save NGS XY
        wfsxfile = "mngsx_"+prefix+".fits"
        wfsyfile = "mngsy_"+prefix+".fits"
        wfs_x = fits.getdata(wfsxfile)
        wfs_y = fits.getdata(wfsyfile)
        # number of MNGS?
        nwfs = nngs
        #gsaltfile = "gsalt_"+prefix+".fits"
        gsalts = 0 # NGS
        gspos = gspos * as2r
        for ii in range(nwfs):
            # for LGSs, get shifted and scaled x/y coord
            gsalt = gsalts
            if (gsalt == 0):
                #x_coord = wfs_x  + alt * gspos[ii,0]
                #y_coord = wfs_y  + alt * gspos[ii,1]
                cone_eff = 1
            else :
                #x_coord = wfs_x *(gsalt-alt)/gsalt + alt * gspos[ii,0]
                #y_coord = wfs_y *(gsalt-alt)/gsalt + alt * gspos[ii,1]
                cone_eff = (gsalt-alt)/gsalt
            #print(cone_eff)
            cone_var1 = 1 - cone_eff**2
            cone_var0 = cone_eff
            tt_term1 = np.ones(wfs_x.shape)
            tt_term0 = np.zeros(wfs_x.shape)
            # get proj arrays, for MNGS
            ###### Direct projection
            cone_var1 = cone_eff
            proj_x = np.array([tt_term1*2,tt_term0,\
                              4*math.sqrt(3)*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii+nlgs,0]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii+nlgs,1]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii+nlgs,0]),\
                              -1*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii+nlgs,1])])
            proj_y = np.array([tt_term0,tt_term1*2,\
                              4*math.sqrt(3)*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii+nlgs,1]),\
                              2*math.sqrt(6)*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii+nlgs,0]),\
                              -2*math.sqrt(6)*(cone_var1**2 *wfs_y+cone_var1*alt*gspos[ii+nlgs,1]),\
                              1*(cone_var1**2 *wfs_x+cone_var1*alt*gspos[ii+nlgs,0])])

            if (ii == 0):
                #initialise proj_mat
                proj_mat = np.concatenate((proj_x,proj_y),axis=1)
                #print(proj_mat.shape)
            else :
                #proj_mat = np.array([proj_mat,proj_x,proj_y])
                proj_mat = np.concatenate((proj_mat,proj_x),axis=1)
                proj_mat = np.concatenate((proj_mat,proj_y),axis=1)
                #print(proj_mat.shape)

        print(proj_mat.shape)
        return proj_mat


##### function for generating full square block-structured merging matrix
def gen_merging_mat (prefix,nwfs):
    t1 = time.time()
    NGS_proj = gen_quad_mat (prefix, wfs_type="MNGS")
    LGS_proj = gen_quad_mat (prefix, wfs_type="LGS")
    NGS_m1 = np.linalg.solve(NGS_proj@NGS_proj.T,NGS_proj) # check if transpose needed
    #LGS_m1 = np.linalg.solve(LGS_proj@LGS_proj.T,LGS_proj)
    LGS_proj_1 = LGS_proj[:2,:int(LGS_proj.shape[1]/nwfs)]
    LGS_m1_1 = np.linalg.solve(LGS_proj_1@LGS_proj_1.T,LGS_proj_1)
    #LGS_m1 = np.kron(np.eye(nwfs),LGS_m1_1)

    Rot_mat = np.zeros([5,6])
    #Rot_mat[:5,:5] = np.eye(5) # how much quad to be projected to LGS, here we assume 0
    Rot_mat[:2,:2] = np.eye(2) # how much TT to be projected

    TT_mat = np.zeros([6,6]) # how many terms to be kept in NGS measurement, here we need none
    #TT_mat[:2,:2] = tt_fact*np.eye(2)

    LGS_proj_block = LGS_proj.T@Rot_mat@NGS_m1
    NGS_proj_block = NGS_proj.T@TT_mat@NGS_m1
    LGS_TTF_1 = np.eye(LGS_m1_1.shape[1]) - LGS_proj_1.T@LGS_m1_1
    LGS_TTF = np.kron(np.eye(nwfs), LGS_TTF_1)

    all_proj = np.zeros([LGS_TTF.shape[1]+NGS_m1.shape[1],LGS_TTF.shape[1]+NGS_m1.shape[1]])
    all_proj[:LGS_TTF.shape[1],:LGS_TTF.shape[1]] = LGS_TTF # top left block
    all_proj[LGS_TTF.shape[1]:,LGS_TTF.shape[1]:] = NGS_proj_block # bottom right block, 0 in this case
    all_proj[:LGS_TTF.shape[1],LGS_TTF.shape[1]:] = LGS_proj_block # bottom left block, merging part

    t2 = time.time()
    print("merging matrix got! Time taken = %.1f seconds!"%(t2-t1))
    print("dims of All_proj matrix:",all_proj.shape)

    # normally, if we want to build a merging LAA cMat, we simply do cMat_old @ all proj
    # since we are looking for a projection matrix for M @ cmm_all @ M.T, it might be helpful to keep things square
    return all_proj.astype('float64')



def Cov_from_Buffer(cbmes,prefix,f_delay,nwfs,validsubs,shnxsub,MergeFlag=0):
    if MergeFlag :
        All_proj = gen_merging_mat(prefix,nwfs)
        cbmes    = All_proj @ cbmes
        print("merged!")
    
    cbmes = cbmes[:-6,:] # assume 3 NGS case
    CMMk = np.zeros((cbmes.shape[0],cbmes.shape[0],f_delay.shape[0]))
    for i in range(f_delay.shape[0]):
        t_start = time.time()
        if f_delay[i]>0:
            CMMk[:,:,i] = (cbmes[:,:-f_delay[i]].T - cbmes[:,:-f_delay[i]].mean(axis=1)).T @ \
                          (cbmes[:,f_delay[i]:].T - cbmes[:,f_delay[i]:].mean(axis=1)) / cbmes[:,f_delay[i]:].shape[1]
        elif f_delay[i]==0:
            CMMk[:,:,i] = (cbmes.T - cbmes.mean(axis=1)).T @ \
                          (cbmes.T - cbmes.mean(axis=1)) / cbmes.shape[1]
        elif f_delay[i]<0:
            CMMk[:,:,i] = (cbmes[:,-f_delay[i]:].T - cbmes[:,-f_delay[i]:].mean(axis=1)).T @ \
                          (cbmes[:,:f_delay[i]].T - cbmes[:,:f_delay[i]].mean(axis=1)) / cbmes[:,-f_delay[i]:].shape[1]
        t_end = time.time()
        print("calculation of 1 CMM from buffer takes %.1f seconds!"%(t_end-t_start))
        if i == 0:
            CM_0,CovMapMask = CovMap_from_Mat(CMMk[:,:,i],nwfs,validsubs,shnxsub)
            CM_all = np.zeros((CM_0.shape[0],CM_0.shape[1],f_delay.shape[0]))
            CM_all[:,:,i] = CM_0
        else:
            CM_all[:,:,i],CovMapMask = CovMap_from_Mat(CMMk[:,:,i],nwfs,validsubs,shnxsub)

    return CMMk,CM_all,CovMapMask
'''
###### testing, too lazy to "if name == main"
from matplotlib import pyplot as plt
plt.ion()
from astropy.io import fits

prefix = "CovMap_mavis"

validsubs = get_full_index(prefix)
nwfs = 2
#if not path.exists("cmm_"+prefix+".fits"):
#    os.system("yorick -i gen_covariance.i "+prefix)

#cmm = fits.getdata("cmm_"+prefix+".fits")
cbmes = fits.getdata("cbmes_CovMap_mavis_2lgs.fits").T
print("shape of cbmes:",cbmes.shape)
shnxsub = 40
zenith = 30
r0 = 0.1289
telDiam = 8.
Cn2 = fits.getdata("Cn2_"+prefix+".fits")
alt = fits.getdata("layer_"+prefix+".fits")
gsalt = fits.getdata("GSalt_"+prefix+".fits")
gspos = fits.getdata("GSpos_"+prefix+".fits") # no need to transpose
layer_speed = fits.getdata("layerspeed_"+prefix+".fits")
wind_dir = fits.getdata("winddir_"+prefix+".fits")
l0 = np.array([25.,]*alt.shape[0])
f_delay = np.array([0])
t_delay = f_delay/1000

CMMmerged,CM_all_m,CovMapMask = Cov_from_Buffer(cbmes,prefix,f_delay,nwfs,validsubs,shnxsub,MergeFlag=1)
CMMorigin,CM_all_o,CovMapMask = Cov_from_Buffer(cbmes,prefix,f_delay,nwfs,validsubs,shnxsub,MergeFlag=0)
#CMMk = CovMat_from_Buffer(cbmes,f_delay)
#CM_all,CovMapMask = CovMap_from_Mat(cmm,nwfs,validsubs,shnxsub)

CMAk = np.zeros((CM_all_m.shape[0],CM_all_m.shape[1],t_delay.shape[0]))
#CMAo = np.zeros((CM_all_o.shape[0],CM_all_o.shape[1],t_delay.shape[0]))

for i in range(t_delay.shape[0]):
    ti = t_delay[i]
    CMAk[:,:,i] = CovMap_from_Cn2(CovMapMask,telDiam,zenith,shnxsub,r0,Cn2,l0,alt,nwfs,gspos,gsalt,timedelay=ti,windspeed=layer_speed,winddir=wind_dir)

#CMA = CovMap_from_Cn2(CovMapMask,telDiam,zenith,shnxsub,r0,Cn2,l0,alt,nwfs,gspos,gsalt)

CovMat_reco = Mat_from_CovMap(CMAk[:,:,0],nwfs,validsubs,shnxsub)


cdiag_merge = np.diag(CMMmerged[:,:,0]) 
cdiag_origin = np.diag(CMMorigin[:,:,0])
cdiag_reco  = np.diag(CovMat_reco)

LGS_proj = gen_quad_mat (prefix, wfs_type="LGS") [:2,]
LGS_proj_1 = LGS_proj[:,:int(LGS_proj.shape[1]/nwfs)]
LGS_m1_1 = np.linalg.solve(LGS_proj_1@LGS_proj_1.T,LGS_proj_1)
LGS_TTF_1 = np.eye(LGS_m1_1.shape[1]) - LGS_proj_1.T@LGS_m1_1
LGS_TTF = np.kron(np.eye(nwfs), LGS_TTF_1)
CovMat_reco_full = np.tril(CovMat_reco)+np.tril(CovMat_reco).T-np.diag(CovMat_reco.diagonal())
CovMat_TTF = LGS_TTF @ CovMat_reco_full @ LGS_TTF.T
cdiag_TTF = np.diag(CovMat_TTF)


plt.figure();plt.plot(cdiag_merge,'-r',label="merged") ;
plt.plot(cdiag_origin,'-b',label="origin") ;
plt.plot(cdiag_reco,'-k',label="analytical");
plt.plot(cdiag_TTF,'-c',label="analytical-TTF");

plt.title("Diagonal values of Covariance matrices");
plt.legend(loc="upper right");


major_ticks = np.arange(0,CM_all_m.shape[0],(2*shnxsub-1)*2);
fig1=plt.figure();ax=fig1.add_subplot(1,1,1);plt.imshow(CM_all_m[:,:,0]-CMAk[:,:,0]);
plt.colorbar();plt.title("Merged v.s. Analytical");
ax.set_xticks(major_ticks);ax.set_yticks(major_ticks);ax.grid(which='major', alpha=4.5);
fig2=plt.figure();ax=fig2.add_subplot(1,1,1);plt.imshow(CM_all_o[:,:,0]-CMAk[:,:,0]);
plt.colorbar();plt.title("Original v.s. Analytical");
ax.set_xticks(major_ticks);ax.set_yticks(major_ticks);ax.grid(which='major', alpha=4.5);
fig3=plt.figure();ax=fig3.add_subplot(1,1,1);plt.imshow(CM_all_m[:,:,0]-CM_all_o[:,:,0]);
plt.colorbar();plt.title("Merged v.s. Original");
ax.set_xticks(major_ticks);ax.set_yticks(major_ticks);ax.grid(which='major', alpha=4.5);

'''
