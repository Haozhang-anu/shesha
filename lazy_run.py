import matplotlib
#matplotlib.use('Agg')
import numpy as np
from docopt import docopt
from matplotlib import pyplot as plt
plt.ion()
from matplotlib.colors import Normalize as Norm
from shesha.util.CovMap_from_Mat import Map_and_Mat
from scipy import stats as st


nniter = 50000


rand_id = "10237"

dd = np.load("buffer/dd_"+rand_id+".npy").astype("float32")
imat = np.load("buffer/imat_"+rand_id+".npy")
ss = np.load("buffer/ss_"+rand_id+".npy").astype("float32")
lgs0_x = np.load("buffer/lgs0_x"+rand_id+".npy")
lgs0_y = np.load("buffer/lgs0_y"+rand_id+".npy")

ndelay = 2

ss = ss[:,:-ndelay]
dd = dd[:,ndelay:]
pols   = (-imat@dd +ss).astype("float32")
pols   = pols.T 
pols = pols - pols.mean(axis=0)

cmm = pols.T@pols / (nniter-ndelay)

cmm_diag = cmm.diagonal()

fig=plt.figure();
for ii in range(9):
    plt.subplot(3,3,ii+1)
    if ii == 0 :
        plt.plot(cmm_diag);plt.title("All WFS cmm diag");
    else:
        npix = 12
        plt.plot(cmm_diag[1148*2*(ii-1):1148*2*ii]);plt.title("LGS"+str(ii)+": "+str(npix)+" pix");plt.ylim((0.04,0.1));
plt.subplots_adjust(wspace = 0.5,hspace = 0.4)
plt.savefig("fig/all_wfs_cmm_diag_"+rand_id+"_delay_"+str(ndelay)+".png")

nn  = Norm(0.04,0.08)
pos_sub = np.array([15,9,3,7,11,17,23,19]).astype(int)
fig=plt.figure()
for ii in range(8):
    plt.subplot(5,5,int(pos_sub[ii]))
    plt.scatter(lgs0_x,lgs0_y,c=cmm_diag[1148*(2*ii):1148*(2*ii+1)],marker='x',s=0.7,norm=nn)
    plt.colorbar();plt.title("LGS # "+str(ii+1))
plt.subplots_adjust(wspace = 0.6,hspace = 0.4)
plt.savefig("fig/cmm_xx_diag_colored_aligned"+rand_id+"_delay_"+str(ndelay)+".png");

fig=plt.figure()
for ii in range(8):
    plt.subplot(5,5,int(pos_sub[ii]))
    plt.scatter(lgs0_x,lgs0_y,c=cmm_diag[1148*(2*ii)+1148:1148*(2*ii+1)+1148],marker='x',s=0.7,norm=nn)
    plt.colorbar();plt.title("LGS # "+str(ii+1))
plt.subplots_adjust(wspace = 0.6,hspace = 0.4)
plt.savefig("fig/cmm_yy_diag_colored_aligned"+rand_id+"_delay_"+str(ndelay)+".png");

Cmat_full = np.load("buffer/Cmat_ana_full.npy")
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
        print("wfs #"+str(i1)+" with wfs #"+str(i2)+", TT included, slope = %.5f, inte = %.5f, r2 = %.5f"%(sl[ii], inte[ii], r_v[ii]))
