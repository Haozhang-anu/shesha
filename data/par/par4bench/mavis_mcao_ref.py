import shesha.config as conf
import numpy as np
simul_name = "mavis"
AOtype="mcao" 

###################
# LOOP
p_loop = conf.Param_loop()
p_loop.set_niter(1000)
p_loop.set_ittime(0.001)  # =1/1500

###################
# GEOM
p_geom = conf.Param_geom()
p_geom.set_zenithangle(30.)

###################
# TEL
p_tel = conf.Param_tel()
p_tel.set_diam(8.0)
p_tel.set_cobs(0.16)

###################
# ATMOS
p_atmos = conf.Param_atmos()
p_atmos.set_r0(0.1289)

p_atmos.set_nscreens(4)
p_atmos.set_frac([0.25, 0.25,  0.25,  0.25])
p_atmos.set_alt([0, 4000, 8000, 16000])
#p_atmos.set_windspeed([ 8000., 9000,  10000,  11000.])
p_atmos.set_windspeed([ 8., 9,  10,  11.])
p_atmos.set_winddir([18.,  70.,    126.,   160.])
p_atmos.set_L0([25., 25., 25., 25.])


'''
p_atmos.set_nscreens(10)
p_atmos.set_frac([0.59, 0.02, 0.04, 0.06, 0.01, 0.05, 0.09, 0.04,  0.05,  0.05])
p_atmos.set_alt([30,  140,  281,  562, 1125, 2250, 4500, 7750, 11000, 14000])
p_atmos.set_windspeed([ 6.6,  5.9,  5.1,  4.5,  5.1,  8.3, 16.3, 30.2,  34.3,  17.5])
p_atmos.set_winddir([0.,  10.,  20.,  25.,   0.,  10.,  20.,  25.,    0.,   10.])
p_atmos.set_L0([25., 25., 25., 25., 25., 25., 25., 25., 25., 25.])
'''
###################
# TARGET
p_targets=[]
RADIUS_TAR = 10
NTAR_side  = 2
NTAR       = NTAR_side * NTAR_side
tar_xpos   =np.array([0.])
tar_ypos   =np.array([0.])
#NTAR=49
if (NTAR > 1):
    tar_xpos, tar_ypos = np.meshgrid(np.linspace(-RADIUS_TAR, RADIUS_TAR, NTAR_side), np.linspace(-RADIUS_TAR, RADIUS_TAR, NTAR_side))
for t in np.arange(NTAR):
    p_targets.append(conf.Param_target())
    p_targets[t].set_xpos(tar_xpos.flatten()[t])
    p_targets[t].set_ypos(tar_ypos.flatten()[t])
    #IR 1.6 / visible 0.65
    p_targets[t].set_Lambda( 0.55)
    p_targets[t].set_mag(10.)
    #1 DM per target: pzt
    p_targets[t].set_dms_seen([0,1,2,3])
    #2 DM per target: pzt + tt
    #p_targets[t].set_dms_seen(np.array([t,NTAR]))


###################
# WFS
RADIUS      = 0.
FRACSUB     = 0.99
NXSUB_LGS   = 40
NXSUB_NGS   = 40
NXSUB_TAR   = max(NXSUB_LGS,NXSUB_NGS)
NLGS        = 5
NNGS        = 1
NTS_side    = 1
NTS         = NTS_side*NTS_side

p_wfs_lgs = []
p_wfs_ngs = []
p_wfs_ts  = []

#lgs position
x = np.linspace(0, 2 * np.pi, NLGS+1)[:-1]
lgs_xpos = RADIUS * np.cos(x)
lgs_ypos = RADIUS * np.sin(x)

#NGS asterism
# closest star from asterism F2 to axis
#asterism_x = [ -1.213]
#asterism_y = [ 24.719 ]
x = np.linspace(np.pi/6.,2 * np.pi+np.pi/6., NNGS+1)[:-1]
asterism_x = RADIUS * np.cos(x)
asterism_y = RADIUS * np.sin(x)

#Truth Sensors position
radius = 15
lspace=np.linspace(-radius+1,radius-1,NTS_side)
mesh=np.meshgrid(lspace,lspace)
TS_xpos = mesh[0].flatten() #radius * np.cos(x)
TS_ypos = mesh[1].flatten() #radius * np.sin(x)

# add 1 position for target
asterism_x = np.append(asterism_x,[0.])
asterism_y = np.append(asterism_y,[0.])

#create wfs lists
#LGS
for i in range(NLGS):
    p_wfs_lgs.append(conf.Param_wfs())
#NGS
for i in range(NNGS+1):
    p_wfs_ngs.append(conf.Param_wfs())
#TS
for i in range(NTS):
    p_wfs_ts.append(conf.Param_wfs())
#concatenate LGS and NGS
p_wfss = p_wfs_lgs + p_wfs_ngs
nnlgs = 0
#setting LGS
for p_wfs in p_wfs_lgs:
    k=p_wfs_lgs.index(p_wfs)
    p_wfs.set_type("sh")
    p_wfs.set_nxsub(NXSUB_LGS)
    p_wfs.set_npix(6+nnlgs*2)
    nnlgs = nnlgs + 1
    p_wfs.set_pixsize(0.5)
    p_wfs.set_fracsub(FRACSUB)
    p_wfs.set_xpos(lgs_xpos[k])
    p_wfs.set_ypos(lgs_ypos[k])

    p_wfs.set_Lambda(0.589)
    #p_wfs.set_gsmag(8.)
    p_wfs.set_optthroughput(0.5)
    p_wfs.set_zerop(1.e11)
    p_wfs.set_noise(0.5)
    p_wfs.set_atmos_seen(1)

    p_wfs.set_gsalt(90 * 1.e3)
    p_wfs.set_lltx(0.)
    p_wfs.set_llty(0.)
    p_wfs.set_laserpower(20)
    p_wfs.set_lgsreturnperwatt(22.)
    p_wfs.set_proftype("Gauss1")
    p_wfs.set_beamsize(0.8)

#setting NGS / target
for p_wfs in p_wfs_ngs:
    k=p_wfs_ngs.index(p_wfs)
    if( k >= NNGS):
        nxsub=NXSUB_TAR
        p_wfs.set_fracsub(FRACSUB)
        p_wfs.set_npix(8)
    else:
        nxsub=NXSUB_NGS
        #p_wfs.set_fracsub(FRACSUB)
        p_wfs.set_fracsub(0.)
        p_wfs.set_npix(20)
        #p_wfs.set_is_low_order(True)

    p_wfs.set_type("sh")
    p_wfs.set_nxsub(nxsub)
    p_wfs.set_pixsize(0.5)

    p_wfs.set_Lambda(0.589)
    p_wfs.set_gsmag(8.)
    p_wfs.set_optthroughput(0.5)
    p_wfs.set_zerop(1.e11)
    p_wfs.set_noise(0.3)
    p_wfs.set_atmos_seen(1)
    p_wfs.set_xpos(asterism_x[k])
    p_wfs.set_ypos(asterism_y[k])

#setting TS
for p_wfs in p_wfs_ts:
    k=p_wfs_ts.index(p_wfs)
    p_wfs.set_xpos(TS_xpos[k])
    p_wfs.set_ypos(TS_ypos[k])

###################
# DM
p_dm0 = conf.Param_dm()
p_dm1 = conf.Param_dm()
p_dm2 = conf.Param_dm()
p_dm3 = conf.Param_dm()
p_dms = [p_dm0,p_dm1,p_dm2, p_dm3]
#adding target DMs
p_dm0.set_type("pzt")
p_dm0.set_nact(41)
p_dm0.set_alt(0.)
p_dm0.set_thresh(0.3) # change actuator threshold
p_dm0.set_coupling(0.3)
p_dm0.set_unitpervolt(1)
p_dm0.set_push4imat(1)

p_dm1.set_type("pzt")
p_dm1.set_nact(49)
p_dm1.set_alt(6000.)
p_dm1.set_thresh(0.3)
p_dm1.set_coupling(0.3)
p_dm1.set_unitpervolt(0.1)
p_dm1.set_push4imat(1)

p_dm2.set_type("pzt")
p_dm2.set_nact(49)
p_dm2.set_alt(13500.)
p_dm2.set_thresh(0.3)
p_dm2.set_coupling(0.3)
p_dm2.set_unitpervolt(1)
p_dm2.set_push4imat(1)

p_dm3.set_type("tt")
p_dm3.set_alt(0.)
p_dm3.set_unitpervolt(0.0005)
p_dm3.set_push4imat(10.)


###################
# CENTROIDERS
p_centroiders = []
for i in range( NNGS+NLGS):
    p_centroiders.append(conf.Param_centroider())

for p_centroider in p_centroiders:
    k=p_centroiders.index(p_centroider)
    p_centroider.set_nwfs(k)
    p_centroider.set_type("cog")
    if (p_wfss[k].get_gsalt() > 0):
        p_centroider.set_filter_TT(False)


###################
# CONTROLLERS
p_controller0 = conf.Param_controller()
p_controller1 = conf.Param_controller()
p_controllers = [p_controller0, p_controller1]

p_controller0.set_type("ls")
p_controller0.set_nwfs(np.arange(NLGS))
#p_controller0.set_ndm(list(range(len(p_dms)-1)))
p_controller0.set_ndm([0,1,2])
p_controller0.set_maxcond(150.)
p_controller0.set_delay(1.)
p_controller0.set_gain(0.2)

p_controller1.set_type("ls")
p_controller1.set_nwfs(np.arange(NNGS)+NLGS)
#p_controller1.set_ndm([len(p_dms)-1])
p_controller1.set_ndm([3])
p_controller1.set_maxcond(15000.)
p_controller1.set_delay(1.)
p_controller1.set_gain(0.2)
