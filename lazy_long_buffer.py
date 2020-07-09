import matplotlib
matplotlib.use('Agg')
import numpy as np
#from docopt import docopt
from matplotlib import pyplot as plt
#plt.ion()
from matplotlib.colors import Normalize as Norm
from shesha.util.CovMap_from_Mat import Map_and_Mat
from scipy import stats as st
import time

nfiles = 1
buffer_every = 100
nniter = int(nfiles*50000/buffer_every) # floor
#counter
rema = 50000%buffer_every
ndelay = 0
rand_id = "10237"

#dd = np.load("buffer/dd_"+rand_id+".npy").astype("float32")
imat = np.load("buffer/imat_"+rand_id+".npy")
#ss = np.load("buffer/ss_"+rand_id+".npy").astype("float32")
lgs0_x = np.load("buffer/lgs0_x"+rand_id+".npy")
lgs0_y = np.load("buffer/lgs0_y"+rand_id+".npy")


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
        npix = 12
        plt.plot(cmm_diag[1148*2*(ii-1):1148*2*ii]);plt.title("LGS"+str(ii)+": "+str(npix)+" pix");plt.ylim((0.04,0.1));
plt.subplots_adjust(wspace = 0.5,hspace = 0.4)
plt.savefig("fig/all_wfs_cmm_diag_"+rand_id+"_delay_"+str(ndelay)+"_every_"+str(buffer_every)+"_nfiles_"+str(nfiles)+".png")

nn  = Norm(0.06,0.08)
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
