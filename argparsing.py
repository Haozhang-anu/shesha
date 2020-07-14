# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 18:35:35 2018

@author: jcranney
"""
import numpy as np
import pickle as pkl
import argparse

parser = argparse.ArgumentParser(description='Build reconstructor for AO')

parser.add_argument("simname", nargs="?",
                   help="filename of the simulated system")

parser.add_argument('--rate','-R',dest="R",nargs="?",default=1,type=int,
                   help='multirate ratio (default: 1, single-rate)')

parser.add_argument('--dareit', nargs="?",default=50,type=int,
                   help='max number of iterations for the DARE (default: 50)')

parser.add_argument('--framedelay','-d', nargs="?",default=0,type=int,
                   help='framedelay in AO loop (default: 0, non-predictive)')

parser.add_argument('--metares',dest='n_meta',nargs="?",default=41,type=int,
                   help='metapupil sampling resolution (default: 41)')

parser.add_argument('--boil','-b', nargs="?",default=0.0,type=float,
                   help='boiling coefficient (default: 0.0, pure frozen-flow)')

parser.add_argument('--sigwlgs','--sl', nargs="?",default=0.1,type=float,
                   help='lgs noise standard deviation (default: 0.1 arcsec)')

parser.add_argument('--sigwngs','--sn', nargs="?",default=0.01,type=float,
                   help='ngs noise standard deviation (default: 0.01 arcsec)')

parser.add_argument('--projres','-p',dest="proj_res",nargs="?",default=50,type=int,
                   help='support x-resolution for metapupils (default: 50)')

parser.add_argument('--projreg',dest="proj_reg",nargs="?",default=0.01,type=float,
                   help='regularising parameter for DM projection (default: 0.01)')

parser.add_argument("--mode","-m",nargs="?",default="mcao",
                   choices=["scao","mcao","ltao"], 
                   help='AO Mode (default: mcao)')

parser.add_argument('--numtargets',dest="num_points",nargs="?",type=int,default=32,
                   help='number of targets for computing MCAO projector (default: 32)')

parser.add_argument('--fovrad',dest="max_rad",nargs="?",type=float,default=10.0,
                   help='MCAO FoV radius (default: 10.0 arcsec)')

parser.add_argument('--fovx',nargs="?",type=float,default=0.0,
                   help='LTAO/MCAO FoV x-offset (default: 0.0 arcsec)')

parser.add_argument('--fovy',nargs="?",type=float,default=0.0,
                   help='LTAO/MCAO FoV y-offset (default: 0.0 arcsec)')

parser.add_argument('--metafov',nargs="?",type=float,default=6.0,
                   help='fov to be overlapped by metapupils (default: 6.0 arcsec)')

locals().update(vars(parser.parse_args()))

simname = simname.split("/")[-1].split(".")[0]

locals().update(pkl.load(open("params_" + simname + ".pkl","rb")))

ittime = ittime*R_true

h_gs = [x/np.cos(zenithangle) for x in h_gs]

n_meta =  [n_meta for x in range(N_meta)]

# This is already done in compass it seems!
# h  =      [x/np.cos(zenithangle) for x in h[:N_meta]]

r0 = [(np.cos(zenithangle)**0.6)*r0_global/(x**(3.0/5.0)) for x in weights[:N_meta]]   # r0 at 0.5 micron

boil = [boil for x in range(N_meta)]
vr = [v*np.cos(zenithangle) for v in vr] # shouldnt be physically the case I think, but this is to match compass
vy = [-vr[ni]*np.sin(np.pi/180*vtheta[ni]) for ni in range(N_meta)]
vx = [-vr[ni]*np.cos(np.pi/180*vtheta[ni]) for ni in range(N_meta)]

metafov = metafov*4.848e-6

if mode == "scao":
    dirs = np.array([[0,0,1]])
elif mode == "ltao":
    dirs = np.array([[fovx*4.848e-6,fovy*4.848e-6,1]])
elif mode == "mcao":
    gr = (3-np.sqrt(5))*np.pi # golden angle
    dirs = np.array([[(max_rad*np.sqrt((ni+1)/num_points)*np.cos((ni+1)*gr)+fovx)*4.848e-6,
                      (max_rad*np.sqrt((ni+1)/num_points)*np.sin((ni+1)*gr)+fovy)*4.848e-6,
                      1] for ni in range(num_points)])
else:
    throw("invalid AO mode selected: " + mode)

h_ts = [np.inf for x in range(len(dirs))]
