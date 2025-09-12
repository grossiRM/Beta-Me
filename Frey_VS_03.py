import flopy     ; import numpy as np    ; import matplotlib.pyplot as plt   ; from pathlib import Path 
import warnings  ; warnings.filterwarnings("ignore", category=DeprecationWarning) 

ref_data=Path("./data/Freyberg")
FA = Path("_RES/folder_01c") ; exe_name = 'E:\\15_REPOS\\00_BETAMI\\bin\\mf6.exe' ; sim_name='Freyberg'  
bottom = np.loadtxt(ref_data / 'bottom.txt')   

sim = flopy.mf6.MFSimulation.load(sim_ws = FA, verbosity_level=0, exe_name=exe_name) 

gwf = sim.get_model(sim_name)       ; heads = gwf.output.head()  
hds = gwf.output.head()             ; head = hds.get_alldata()[0] 

wel=gwf.get_package("WEL-1")      ; sat_thk=gwf.modelgrid.saturated_thickness(head,mask=[1e30]) 
thk=gwf.modelgrid.cell_thickness  ; modelgrid=gwf.modelgrid

Hk = gwf.npf.k.get_data() ; print(Hk.shape)
fig = plt.figure(figsize=(19, 4))                             
ax1 = fig.add_subplot(151) ; mm = flopy.plot.PlotMapView(model=gwf, layer=0, extent=gwf.modelgrid.extent)                     ; mm.plot_grid()                    
mm.plot_ibound(); mm.plot_bc(package=wel)             ; mm.plot_bc("CHD" , color="white" );  mm.plot_bc("RIV" , color="cyan") ; ax1.set_title("Top (35m) cte", fontweight='bold')          
ax2 = fig.add_subplot(162) ; pm = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid) ; pc= pm.plot_array([bottom])               ; ax2.set_title("Bottom")                                                  ;ax2.yaxis.set_visible(False) ; plt.colorbar(pc) 
ax3 = fig.add_subplot(163) ; pm = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid) ; pc= pm.plot_array(thk,vmin=5,vmax=35)     ; ax3.set_title("Grid thickness" )   ; pm.plot_ibound()                    ;ax3.yaxis.set_visible(False) ; plt.colorbar(pc) 
ax4 = fig.add_subplot(166) ; pm = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid) ; pc= pm.plot_array(head, cmap='RdYlBu')    ; ax4.set_title("Heads_H", color = 'blue', fontweight='bold')              ;ax4.yaxis.set_visible(False) ; plt.colorbar(pc)
ax5 = fig.add_subplot(165) ; pm = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid) ; pc= pm.plot_array(sat_thk,cmap='Blues')  ; ax5.set_title("Sat.thickness_H", color = 'red', fontweight='bold')  ;ax5.yaxis.set_visible(False) ; plt.colorbar(pc)
ax6 = fig.add_subplot(164) ; pm = flopy.plot.PlotMapView(modelgrid=gwf.modelgrid) ; pc= pm.plot_array(Hk,cmap='RdYlBu')  ; ax6.set_title("Hk", color = 'red', fontweight='bold')  ;ax6.yaxis.set_visible(False) ; plt.colorbar(pc)

pm.plot_ibound(); plt.show()