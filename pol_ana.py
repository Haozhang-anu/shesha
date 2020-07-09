import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize as Norm
plt.ion()
imat = np.load("imat_22904.npy")
dd = np.load("dd_22904.npy")
ss = np.load("ss_22904.npy")
#imat = np.array(sim.rtc.d_control[0].d_imat)
dd_cov = dd@dd.T / 50000
ss_cov = ss@ss.T / 50000
pols   = imat@dd +ss
pols_mean = pols.T - np.average(pols,axis=1)
cmm = pols_mean.T@pols_mean / 50000
cmm_diag = cmm.diagonal()
dd_diag = dd_cov.diagonal()
ss_diag = ss_cov.diagonal()
plt.figure();plt.imshow(dd_cov);plt.colorbar()
plt.figure();plt.imshow(ss_cov);plt.colorbar()
#plt.figure();

plt.figure();
for ii in range(8):
    plt.subplot(3,3,ii+1)
    if ii == 0 :
        plt.plot(cmm_diag);plt.title("All WFS cmm diag");
    else:
        plt.plot(cmm_diag[1148*(ii-1):1148*(ii+1)]);plt.title("LGS"+str(ii)+": 6 pix");plt.ylim((0.0,0.1));
plt.subplots_adjust(wspace = 0.5,hspace = 0.4)


sim = supervisor._sim
lgs0_y = sim.config.p_wfs_lgs[0].get_validpuppixy()
lgs0_x = sim.config.p_wfs_lgs[0].get_validpuppixx()
nn  = Norm(0.05,0.1)
pos_sub = np.array([15,9,3,7,11,17,23,19]).astype(int)
plt.figure()
for ii in range(8):
    plt.subplot(5,5,int(pos_sub[ii]))
    plt.scatter(lgs0_x,lgs0_y,c=cmm_diag[1148*(2*ii):1148*(2*ii+1)],marker='x',s=0.7,norm=nn)
    plt.colorbar();plt.title("LGS # "+str(ii+1))
plt.subplots_adjust(wspace = 0.6,hspace = 0.4)


 
'''
plt.subplot(3,3,1);plt.plot(cmm_diag); plt.title("All WFS cmm diag"); plt.ylim((0,0.2));
plt.subplot(3,3,2);plt.plot(cmm_diag[:1148*2]); plt.title("LGS1: 6 pix"); plt.ylim((0,0.2));
plt.subplot(3,3,3);plt.plot(cmm_diag[1148*2:1148*4]); plt.title("LGS2: 6 pix"); plt.ylim((0,0.2));
plt.subplot(3,3,4);plt.plot(cmm_diag[1148*4:1148*6]); plt.title("LGS3: 6 pix"); plt.ylim((0,0.2));
plt.subplot(3,3,5);plt.plot(cmm_diag[1148*6:1148*8]); plt.title("LGS4: 6 pix"); plt.ylim((0,0.2));
plt.subplot(3,3,6);plt.plot(cmm_diag[1148*8:1148*10]); plt.title("LGS5: 6 pix"); plt.ylim((0,0.2));
'''

plt.subplots_adjust(wspace = 0.5,hspace = 0.4)
