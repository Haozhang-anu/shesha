import numpy as np
from matplotlib import pyplot as plt
import math
import time


x = y = np.arange(-100,101,1)
XX,YY = np.meshgrid(x,y) 
fwhm = 15
x0 = 0
y0 = 0
zz = np.exp(-4*np.log(2)*((XX-x0)**2+(YY-y0)**2)/fwhm**2)
plt.figure()

for ii in range(3000): 
    xy = np.random.normal(0,20,(1,2))[0] 
    x0 = xy[0] 
    y0 = xy[1] 
    this_zz = np.exp(-4*np.log(2)*((XX-x0)**2+(YY-y0)**2)/fwhm**2) 
    zz += this_zz
    zz_mean = zz 
    plt.cla() 
    plt.imshow(zz_mean*0.75+this_zz)    
    if ii%50 == 0:
        plt.savefig("fake_psf/fig_%02d.png"%(np.int(ii/50)),bbox_inches='tight')
