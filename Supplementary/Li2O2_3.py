import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Li2O2_3_functions import bucket, Index_start, residual
import os
import matplotlib.pyplot as plt
import matplotlib as mP
from datetime import datetime

save_picture = 1 # saves the data to a folder if this is a 1
start_time = datetime.now()

'''
Constants
''' 
F = 96485.34 #Faraday's number [C/mol_electron]
R = 8.3145 #Universal gas constant [J/mol-K]

'''
USER INPUTS
'''
## Operating Conditions
t_sim_max = [13.1] # the maximum time the battery will be held at each current [hr]
T = 298.15 # standard temperature [K]

## Material Properties
rho_Li2O2 = 2.14e3 # density [kg/m^3]
MW_Li2O2 = 48.88e-3 # molecular weight [kg/mol]
mv_Li2O2 = MW_Li2O2/rho_Li2O2 # constant molar volume [m^3/mol]
sc_Li2O2 = 1165e3 # specific capacity [mAh/kg]

'''
Parameters
'''
area_carbon_0 =  9e-3  # initial area of carbon (this is the area where nucleation happens) [m^2]
# Each of the buckets are the same size, with the exception of the final bucket which will extend to infinity
n_bucket_Li2O2 = 100 # number of buckets [-]
t_bucket_Li2O2 = 1e-9 # the radius range (aka thickness) of each bucket [m]

bucket_Li2O2 = bucket(n_bucket_Li2O2,t_bucket_Li2O2,mv_Li2O2,"Li_2O_2")
SV_index = Index_start(n_bucket_Li2O2) # Holds the pointers for the SV vector

'''
Initialize the State Variable vector
'''
sim_inputs = np.zeros(n_bucket_Li2O2 + 1)
sim_inputs[:SV_index.Li2O2] = np.zeros(n_bucket_Li2O2)
bin_nuc = 50
sim_inputs[0] = 2.1e10
sim_inputs[SV_index.bm_Li2O2_front] = t_bucket_Li2O2*bin_nuc  
atol = np.multiply(np.ones(n_bucket_Li2O2+1),1e6)
atol[-1] = 1e-8

time_start = 0 # Initial time [s]
time_end = t_sim_max[0] #Final time [s]
times = np.linspace(time_start,time_end,1001)

'''
Integration
'''
grow_rate_per_area = 5.18e-7 # i = 1 A/m^2 ; i/2F
nuc_rate_per_area = 2.46e11 
params = [nuc_rate_per_area,grow_rate_per_area , SV_index, bucket_Li2O2,area_carbon_0]

t_span = [time_start,time_end]
min_time_intervals = 1000
max_t_step = max(time_end/min_time_intervals, 1e-8)
solution = (solve_ivp(residual,t_span,sim_inputs,method='BDF',
            args=[params], rtol = 1e-6,atol = atol))#, max_step = max_t_step))
sim_outputs =np.stack((*(solution.y), solution.t))

# print the end of the 
end_time = datetime.now()
print(f"Duration: {end_time - start_time}") 

'''
Post Processing
'''

N_Li2O2  = sim_outputs[:SV_index.Li2O2]
bm_Li2O2_front = sim_outputs[SV_index.bm_Li2O2_front]
time = sim_outputs[-1]
nLi2O2 = [0]*bucket_Li2O2.n

# here is where I remap the data for graphing. I flip the data so the biggest bin is now at 
# the largest index instead of zero, and shift the bins based on where the bookmarks are
N_Li2O2_empty = N_Li2O2.copy()*0
for j in range(len(N_Li2O2_empty[0])):
    front_indx = int(bm_Li2O2_front[j]/bucket_Li2O2.thickness)
    for indx, ele in enumerate(N_Li2O2):
        val = ele[j]
        if val != 0:
            destination_row = np.copy(N_Li2O2_empty[(front_indx)-indx])
            destination_row[j] = val
            N_Li2O2_empty[(front_indx)-indx] = np.copy(destination_row)
N_Li2O2 = N_Li2O2_empty


cmap = mP.colormaps['plasma']
plt.rcParams['font.family'] = 'Times' 
plt.rcParams['mathtext.fontset']='cm'
mP.rcParams['font.family'] = 'serif'
mP.rcParams['font.serif'] = 'Times New Roman'
plot_percs = np.array([0.125,0.25,0.5,0.9])
time_snapshots = np.multiply(plot_percs,max(time))
plt_clrs = [cmap(0.1),cmap(0.35),cmap(0.55),cmap(0.75)]

CS_area_phase = [0]*bucket_Li2O2.n
V_phase = [0]*bucket_Li2O2.n
for ind, ele in enumerate(N_Li2O2):
    nLi2O2[ind] = ele[-1]
    CS_area_phase[ind] = np.pi*((bucket_Li2O2.r_avg_graph[ind]+bucket_Li2O2.r_nuc)**2)*nLi2O2[ind]
    V_phase[ind] = 2/3*np.pi*((bucket_Li2O2.r_avg_graph[ind]+bucket_Li2O2.r_nuc)**3)*nLi2O2[ind]
theta= 1 - np.exp(-sum(CS_area_phase)/area_carbon_0)
Capacity = (theta*sum(V_phase))*rho_Li2O2*sc_Li2O2
fig1 = plt.figure(num=1,figsize=(3,2.25),dpi=400)
#plt.bar(np.multiply(bucket_Li2O2.r_avg_graph,2), np.multiply(np.divide(V_phase,2.54),theta*rho_Li2O2*sc_Li2O2), 
#        color=plt_clrs[3], edgecolor='k', linewidth=0.5,alpha=0.5, width=t_bucket_Li2O2*2)
plt.plot(np.multiply(bucket_Li2O2.r_avg_graph,2), np.multiply(np.divide(V_phase,Capacity),theta*rho_Li2O2*sc_Li2O2*100), linestyle='-', color=plt_clrs[3],linewidth=2)

ax = plt.gca() 
plt.yticks(fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylim([0,0.08*100])
plt.xlim([0,1e-7])
plt.xlabel(r"Particle diameter [nm]",fontsize = 12)
plt.xticks([0,50e-9,100e-9,150e-9,200e-9],['0','50','100','150','200'],fontsize = 10)
plt.ylabel("Capacity Fraction [%]",fontsize = 12)
#plt.text(0.05, 0.95, f"{Capacity:.2f} mAh", transform=plt.gca().transAxes,ha='left', va='top', fontsize=8)
plt.tight_layout()
print(f"Total capacity {Capacity} [mAh]")

def save_fig(pic_name,folder_name):
    if folder_name != None:
        fp_pic = f"{folder_name}/{pic_name}" + ".svg"        
        plt.savefig(fp_pic,transparent=True, format="svg")

if save_picture == 1:
    folder_name = "Supplementary"
    os.makedirs(folder_name, exist_ok=True)
else:
    folder_name = None

save_fig('PSD_3_CapFrac',folder_name)

fig2 = plt.figure(num=2,figsize=(3,2.25),dpi=400)
plt.plot(np.multiply(bucket_Li2O2.r_avg_graph,2), np.divide(nLi2O2,sum(nLi2O2))*100, linestyle='-', color=plt_clrs[3],linewidth=2)
plt.xlim([0,1e-7])
plt.ylim([0,10])
plt.xlabel(r"Particle diameter [nm]",fontsize = 12)
plt.xticks([0,50e-9,100e-9,150e-9,200e-9],['0','50','100','150','200'],fontsize = 10)
plt.ylabel("Percent of Particles",fontsize = 12)
plt.tight_layout()

save_fig('PSD_3',folder_name)

'''
# plots the progress of r0
fig6, (ax10) = plt.subplots(1)
ax10.set_title('Leading Bookmark')    
for i in range(1, int(max(bm_Li2O2_front)/bucket_Li2O2.thickness)): # plot every bin threshold
    ax10.axhline(y=bucket_Li2O2.thickness*i,linestyle='dashed',color='silver')
ax10.plot(time,bm_Li2O2_front,'.')
ax10.set_ylabel(r'Distance [m]',fontsize=12)
fig6.tight_layout()
'''

# Save the profile at the end of discharge
import pandas as pd
fp = "Supplementary/Num_Particles_3.csv"
df  = pd.DataFrame()
df[0] = nLi2O2
df.to_csv(fp, index=False)
plt.show()
