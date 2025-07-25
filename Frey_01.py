import flopy     ; import numpy as np    ; import matplotlib.pyplot as plt   ; from pathlib import Path 
import warnings  ; warnings.filterwarnings("ignore", category=DeprecationWarning) 

sim_ws=Path("./data/Freyberg")  ;sim_ws2=Path("./01a")  ;sim_ws2.mkdir(exist_ok=True)  ;sim_name='Freyberg'  

bottom = np.loadtxt(sim_ws / 'bottom.txt')  ; k11 = np.loadtxt(sim_ws / 'hydraulic_conductivity.txt')    ; idomain = np.loadtxt(sim_ws / 'idomain.txt', dtype=np.int32)
length_units="meters"                       ; nlay=1;nrow=40;ncol=20        ; delr=250.0;delc=250.0                                    ; top=35.0;icelltype=1;strt=45.0
time_units="seconds"                        ; recharge=1.60000000e-09       ; nouter=100;ninner=25;hclose=1e-9;rclose=1e-3             ; nper=1; tdis_ds=((1.0,1.0,1),)

wel_spd = {0: [[0,  8, 15, -0.00820000]     ,[0, 10, 12, -0.00410000] ,[0, 19, 13, -0.00390000] ,[0, 25, 9, -8.30000000e-04],
               [0, 28,  5, -7.20000000e-04] ,[0, 33, 11, -0.00430000]]}
chd_spd = {0: [[0, 39,  5, 16.90000000]     ,[0, 39,  6, 16.40000000] ,[0, 39,  7, 16.10000000] ,[0, 39,  8, 15.60000000]   ,[0, 39,  9, 15.10000000],
               [0, 39, 10, 14.00000000]     ,[0, 39, 11, 13.00000000] ,[0, 39, 12, 12.50000000] ,[0, 39, 13, 12.00000000]   ,[0, 39, 14, 11.40000000]]}
rbot = np.linspace(20.0, 10.25, num=nrow)   ; rstage = np.linspace(20.1, 11.25, num=nrow)   ; riv_spd = []
for idx, (s, b) in enumerate(zip(rstage, rbot)): riv_spd.append([0, idx, 14, s, 0.05, b]) 
riv_spd = {0: riv_spd}

sim = flopy.mf6.MFSimulation   (sim_name=sim_name,sim_ws=sim_ws2,exe_name="mf6",)
flopy.mf6.ModflowTdis    (sim, nper=nper, perioddata=tdis_ds, time_units=time_units)
flopy.mf6.ModflowIms     (sim,linear_acceleration="BICGSTAB",outer_maximum=nouter,outer_dvclose=hclose * 10.0,inner_maximum=ninner,
                          inner_dvclose=hclose,rcloserecord=f"{rclose} strict")
gwf = flopy.mf6.ModflowGwf     (sim,modelname=sim_name,newtonoptions="NEWTON UNDER_RELAXATION")
flopy.mf6.ModflowGwfdis  (gwf,length_units=length_units,nlay=nlay,nrow=nrow,ncol=ncol,delr=delr,delc=delc,top=top,botm=bottom,idomain=idomain)
flopy.mf6.ModflowGwfnpf  (gwf,icelltype=icelltype,k=k11,)
flopy.mf6.ModflowGwfic   (gwf, strt=strt)
flopy.mf6.ModflowGwfriv  (gwf, stress_period_data = riv_spd, pname="RIV-1")
flopy.mf6.ModflowGwfwel  (gwf, stress_period_data = wel_spd, pname="WEL-1")
flopy.mf6.ModflowGwfrcha (gwf, recharge=recharge)
flopy.mf6.ModflowGwfchd  (gwf, stress_period_data = chd_spd)                                                     # ; bf=f"{sim_name}.cbc"
flopy.mf6.ModflowGwfwel  (gwf,maxbound=1,pname="CF-1",filename=f"{sim_name}.cf.wel")        ; hf=f"{sim_name}.hds" ; bf=f"{sim_name}.bud"
flopy.mf6.ModflowGwfoc   (gwf ,head_filerecord=hf,budget_filerecord=bf,headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
                         saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")] ,printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])  
sim.write_simulation(silent=False)  
sim.run_simulation(silent=False) 