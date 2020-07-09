import numpy as np
from scipy.special import gamma, kv
from astropy.io.fits import open as fitsopen
from astropy.io.fits import writeto as fitswrite
from scipy.linalg import block_diag as blkdiag
from numpy.linalg import inv, solve
from scipy import stats

def fitsread(filename,ext=0):
    return (fitsopen(filename)[ext].data)

def cell2mat(X):
    n = np.sum([x.shape[1] for x in X[0]])
    A = np.zeros([0,n])
    for mi in range(len(X)):
        tmp = np.zeros([X[mi][0].shape[0],0])
        for ni in range(len(X[0])):
            tmp = np.concatenate([tmp,X[mi][ni]],axis=1)
        A = np.concatenate([A,tmp],axis=0)
    return A          

def covfromcoords(lcoords,rcoords,r0,L0):
    r = np.sqrt((np.array([lcoords[:,0]])-np.array([rcoords[:,0]]).T)**2 +
                (np.array([lcoords[:,1]])-np.array([rcoords[:,1]]).T)**2)
    return dist2vkcov(r,r0,L0)

def dist2vkcov(r,r0,L0):
    """
    Takes an N-D array, r, and returns the von-karman
    covariance of two points in space separated by the
    value of each element of r, with r0 and L0 given.
    """
    r = r+1e-10
    A = (L0/r0)**(5/3)
    B1 = 2**(-5/6)*gamma(11/6)/(np.pi**(8/3));
    B2 = (24/5*gamma(6/5))**(5/6);
    C  = (2*np.pi*r/L0)**(5/6)*kv(5/6,2*np.pi*r/L0);
    cov = A*B1*B2*C
    return cov

def influenceMatrix(r,coupling):
    """ 
    Takes an N-D array, r, and returns the influence value
    of a point which is "r" dm-pitch units away.
    """
    infl = coupling**r
    return infl

def getDMPitch(actCoords):
    dm_pitch = stats.mode([x for x in abs(actCoords[:,1]-np.roll(actCoords[:,1],1)) if x > 0])[0]
    return dm_pitch

def cmmTTFilter(nsub,nwfs):
    X = np.zeros([2*nwfs*nsub,2*nwfs])
    for ni in range(2*nwfs):
        X[ni*nsub:(ni+1)*nsub,ni] = 1
    filt = np.eye(2*nwfs*nsub) - X @ np.linalg.solve(X.T @ X, X.T)
    return filt
