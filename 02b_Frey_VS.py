from _thon import *
from flopy.utils.postprocessing import get_water_table  

gwf = sim.get_model(ID)       ; heads = gwf.output.head()  
hds = gwf.output.head()       ; head = hds.get_alldata()[0]          ; wt = get_water_table(head)

irow, icol = gwf.modelgrid.intersect(1200, 100)  
def beta_plot(ax):
    pmv=flopy.plot.PlotCrossSection(model=gwf,ax=ax1, line={"column": icol}) ; pmv.plot_grid(linewidth=0.5) ; 
    pmv.plot_inactive() ; pmv.plot_surface(wt, masked_values=[1e30], color="blue", lw=1) 
    
    pmv=flopy.plot.PlotCrossSection(model=gwf,ax=ax2,line={"row": irow})     ; pmv.plot_grid(linewidth=0.5) ; 
    pmv.plot_inactive() ; pmv.plot_surface(wt, masked_values=[1e30], color="blue", lw=1)

plt.figure(figsize=(16,4))    ;ax1 = plt.subplot2grid((2,2),(0,0),colspan=1,rowspan=1); 
ax2 = plt.subplot2grid((2,2),(0,1),colspan=2,rowspan=1)    

beta_plot(ax1)  ; beta_plot(ax2)  ; plt.show()