# LiS_main.py
#
# This file serves as the main model file.  
#   It is called by the user to run the model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mP
from matplotlib import cm
from scikits.odes import dae
from scipy.integrate import solve_ivp
from LiS_functions import bucket, Index_start, residual_ivp, area_carbon, cs_area_phase, volume_fraction
import datetime
import os
import pandas as pd

save_data = 0
# Anode is on the left at x=0 and Cathode is on the right
# Li -> Li+ + e- (reaction at the anode)
# 1/2S_8 + e- -> 1/2S_8^2- (reaction at the cathode)
# Currently, the model begins with all Li2S and S8 dissolved in the electroltye, 
# then tracks the deposition of Li2S an S8 on a planer carbon assuming constant 
# rates for both nucleation and growth 

'''
Constants
''' 
F = 96485.34 #Faraday's number [C/mol_electron]
R = 8.3145 #Universal gas constant [J/mol-K]

'''
USER INPUTS
'''
## Simulation parameters
# I will add to theses later, for now the only termination checks
# are if one of the buckets has a negative value for the number of particles, or if the final bucket gets too full
# maybe add a cuttoff for the concetration of a species in the elyte
S8_limit = 2e40#1e-3 # The maximum number of particles that can be in the final bucket for S_8
Li2S_limit = 2e40#1e-3 # The maximum number of particles that can be in the final bucket for Li_2S

## Operating Conditions
t_sim_max = [10] # the maximum time the battery will be held at each current [s]
T = 298.15 # standard temperature [K]

## Material Properties
rho_C = 2260 # density of carbon [kg/m^3]
rho_S8 = 2070 # density of Sulfur (S8) [kg/m^3]
rho_Li2S = 1660 # density of Li_2S [kg/m^3]

MW_S8 = 0.25652  # molecular weight [kg/mol]
MW_Li2S = 0.045947 # molecular weight [kg/mol]

mv_S8 = MW_S8/rho_S8 # constant molar volume S_8 [m^3/mol]
mv_Li2S = MW_Li2S/rho_Li2S # constant molar volume Li_2S [m^3/mol]

## Initial Values
C_std = 1000 # Standard Concentration [mol/m^3] (same as 1 M)
mol_S8_elyt_0 = 1e-5 # Initial moles of S_8 in the electrolyte [mol/m^3]
mol_Li2S_elyt_0 = 1e-5 # Initial moles of Li_2S in the electrolyte [mol/m^3]

## Material parameters: (Replaced by Cantera?)

'''
Parameters
'''
########################################################################### Danger!!!!!!!!!!!!! delete later
# used so I can compare the two plots directly
mv_S8 = mv_Li2S

## Geometry
# I do not track the volume fraction for carbon because it is planer and inert
# Later I will track the volume fraction of the anode since Lithum disolves
Epsilon_S8_0 = 0 # Initial volume fraction of S_8 in the cathode [-]
Epsilon_Li2S_0 = 0 # Initial volume fraction of Li_2S in the cathode [-]
Epsilon_eltye_0 = 1 - Epsilon_S8_0 - Epsilon_Li2S_0 # Initial volume fraction of electrolyte [-]
area_carbon_0 = 10**2 # inital area of carbon (this is the area where nucleation happens) [m^2]

h = 1e-8 # height of the tank [m] (stand in for eletrolyte thickness)
V_elyte_0 = area_carbon_0*h # initial volume of electrolyte

# Each of the buckets are the same size, with the exception of the final bucket which will extend to infinity
n_bucket_S8 = 100 # number of buckets for S8 [-]
n_bucket_Li2S = 100 # number of buckets for Li2S [-]

t_bucket_S8 = 9e-6 # the radius range (aka thickness) of each bucket for S8 [m]
t_bucket_Li2S = 9e-6 # the radius range (aka thickness) of each bucket for Li2S [m]

bucket_S8 = bucket(n_bucket_S8,t_bucket_S8,mv_S8,"S_8")

bucket_Li2S = bucket(n_bucket_Li2S,t_bucket_Li2S,mv_Li2S,"Li_2S")

SV_index = Index_start(n_bucket_S8,n_bucket_Li2S) # Holds the pointers for the SV vector

'''
Initialize the SV vector
'''
sim_inputs = np.zeros(n_bucket_S8 + n_bucket_Li2S + 2 + 2 + 4)

# I put S8 on top of Li2S. All buckets everything start with zero particles
sim_inputs[:SV_index.S8] = np.zeros(n_bucket_S8)
sim_inputs[SV_index.S8:SV_index.Li2S] = np.zeros(n_bucket_Li2S)
sim_inputs[SV_index.mol_S8_ca] = 0
sim_inputs[SV_index.mol_Li2S_ca] = 0
sim_inputs[SV_index.mol_S8_elyte] = mol_S8_elyt_0 
sim_inputs[SV_index.mol_Li2S_elyte] = mol_Li2S_elyt_0 
sim_inputs[SV_index.bm_S8_front] = 0 
sim_inputs[SV_index.bm_S8_back] = 0 
sim_inputs[SV_index.bm_Li2S_front] = 0 
sim_inputs[SV_index.bm_Li2S_back] = 0

time_start = 0 # Initial time [s]
time_end = t_sim_max[0] #Final time [s]
times = np.linspace(time_start,time_end,1001)
t_span = [time_start,time_end]
'''
Integration
'''
# Integration Limits 
# will add concentration and volatage checks later on
num_roots = 3 # number of termination checks
def terminate_check(t,SV,SV_dot,return_val,user_data):
    #checks the number of particles in the largest boxes
    return_val[0] = SV[SV_index.S8-1] - S8_limit
    return_val[1] = SV[-1]- Li2S_limit
    neg_check = 1
    for ele in SV:
        if ele < 0:
            neg_check = 1
    return_val[2] = neg_check 
    
    
# I am using a DAE solver, but for now there are no algebraic equations
algvars = []

# I am not sure if these will be params or if the residual can call cantera directly
#[s_k_nuc_S8,s_k_grow_S8,s_k_nuc_Li2S,s_k_grow_Li2S] are the first 4 terms in params [mol/m^3]
grow_rate_per_area = 2
nuc_rate_per_area = 2

params = [nuc_rate_per_area,grow_rate_per_area,nuc_rate_per_area,grow_rate_per_area , SV_index, bucket_S8, bucket_Li2S,area_carbon_0]
options =  {'user_data':params, 'rtol':1e-12,
        'atol':1e-12, 'algebraic_vars_idx':algvars, 'first_step_size':1e-10,'rootfn':terminate_check,'nr_rootfns':num_roots}
            # , 'compute_initcond':'yp0', 'max_steps':10000}

SV_0 = sim_inputs
SV_dot_0  = np.zeros_like(SV_0)
solution = solve_ivp(residual_ivp,t_span,sim_inputs,method='BDF',
            args=(nuc_rate_per_area,grow_rate_per_area,nuc_rate_per_area,grow_rate_per_area , SV_index, bucket_S8, bucket_Li2S,area_carbon_0), rtol = 1e-8,atol = 1e-10)

sim_outputs =np.stack((*(solution.y), solution.t))
print(solution)

'''
Post Processing          
'''
N_S8  = sim_outputs[:SV_index.S8]
N_Li2S  = sim_outputs[SV_index.S8:SV_index.Li2S] 
mol_S8_ca = sim_outputs[SV_index.mol_S8_ca]
mol_Li2S_ca = sim_outputs[SV_index.mol_Li2S_ca]
mol_S8_elyt = sim_outputs[SV_index.mol_S8_elyte]
mol_Li2S_elyt = sim_outputs[SV_index.mol_Li2S_elyte]
bm_S8_front = sim_outputs[SV_index.bm_S8_front]
bm_S8_back = sim_outputs[SV_index.bm_S8_back]
bm_Li2S_front = sim_outputs[SV_index.bm_Li2S_front]
bm_Li2S_back = sim_outputs[SV_index.bm_Li2S_back]
time = sim_outputs[-1]

a_c = [0]*len(time)
a_S8 = [0]*len(time)
a_Li2S = [0]*len(time)
Epsilon_eltye = [0]*len(time)
Epsilon_S8 = [0]*len(time)
Epsilon_Li2S = [0]*len(time)
nS8 = [0]*bucket_S8.n
nLi2S = [0]*bucket_Li2S.n

for i in range(len(time)):
    for ind, ele in enumerate(N_S8): # extracts the number of particles in each bucket at the curren timestep
        nS8[ind] = ele[i]
    a_S8[i] = cs_area_phase(bucket_S8,nS8) 
    for ind, ele in enumerate(N_Li2S):
        nLi2S[ind] = ele[i]
    a_Li2S[i] = cs_area_phase(bucket_Li2S,nLi2S) 
    a_c[i] = area_carbon(bucket_S8,nS8,bucket_Li2S,nLi2S,area_carbon_0)
    
    [Epsilon_eltye[i], Epsilon_S8[i] ,Epsilon_Li2S[i]] = volume_fraction(V_elyte_0,bucket_S8,nS8,bucket_Li2S,nLi2S)

'''
plot the results 
'''
# pick what plots to display (1 yes, anything else no)
num_particles_bin = 1
cs_area = 1
total_particles = 1
conc_and_moles = 0
vol_frac = 0

# Number of particles
if num_particles_bin == 1:
    fig1, (ax1, ax2) = plt.subplots(2)
    for ind, ele in enumerate(N_S8):
        ax1.plot(time,ele,label=str(ind))
    for ind, ele in enumerate(N_Li2S):
        ax2.plot(time,ele,label=str(ind))
    #ax1.legend(ncol=1, bbox_to_anchor=(1, 0.5),loc = 'center left')
    ax1.set_title(r"S$_8$, "+str(bucket_S8.n)+" bins")
    ax1.set_ylabel("Number of Particles [-]")
    #ax2.legend(ncol=1, bbox_to_anchor=(1, 0.5),loc = 'center left')
    ax2.set_title(r"Li$_2$S, "+str(bucket_Li2S.n)+" bins")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("Number of Particles [-]")
    fig1.tight_layout()

# Cross sectional area on cathode surface
if cs_area == 1:
    fig2, (ax3, ax4) = plt.subplots(2)
    ax3.plot(time,a_c)
    ax3.set_title("Area of Carbon [m$^2$]")
    ax4.plot(time,a_S8, label = r'$S_8$')
    ax4.plot(time,a_Li2S, label = r'$Li_2S$')
    ax4.legend(ncol=1, loc = 'upper left')
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel(r"Cross Sectional Area [m$^2$]")
    fig2.tight_layout()

# Number of deposited particles
if total_particles == 1:
    fig3 = plt.figure()
    plt.title("Total number of particles")
    plt.plot(time,sum(N_S8))
    plt.plot(time,sum(N_Li2S))
    plt.legend([r"S$_8$",r"Li$_2$S"])

# moles of species in the electrolyte and cathode
# plus concentration of species in the electrolyte
if conc_and_moles == 1:
    fig4, (ax5, ax6, ax7) = plt.subplots(3)
    ax5.set_title("Electrolyte Concentrations [mol/m]")
    ax5.plot(time,mol_S8_elyt/h,label=r"S$_8$")
    ax5.plot(time,mol_Li2S_elyt/h,label=r"Li$_2$S")
    ax5.legend()
    ax6.set_title("Moles of species")
    ax6.plot(time,mol_S8_elyt,label=r"$\rm S_{8(elyte)}$")
    ax6.plot(time,mol_Li2S_elyt,label=r"$\rm Li_2S_{(elyte)}$")
    ax6.legend()
    ax7.plot(time,mol_S8_ca,label=r"$\rm S_{8(ca)}$")
    ax7.plot(time,mol_Li2S_ca,label=r"$\rm Li_2S_{(ca)}$")
    ax7.set_xlim([0.5,3])
    ax7.set_ylim([5e-6-1e-18,5e-6+1e-18])
    ax7.legend()
    ax7.set_xlabel("time [s]")
    
    fig4.tight_layout()

if vol_frac == 1:
    fig5 = plt.figure()
    plt.plot(time,Epsilon_S8, label=r"S$_8$")
    plt.plot(time,Epsilon_Li2S, label=r"Li$_2$S")
    #plt.plot(time,Epsilon_eltye, label="elyte")
    plt.xlabel("time [s]")
    plt.title(r"Volume fraction")
    plt.ylabel(r"$\varepsilon$ [-]")
    plt.legend()
    fig5.tight_layout()
    
#fig6 = plt.figure()
#plt.plot(time,np.multiply(1e-3*2,a_Li2S))

#mP.rcParams['mathtext.fontset'] = 'cm'

cmap = cm.get_cmap('plasma')
figg = plt.figure()
plot_percs = np.linspace(0.1,0.99,4)
plot_percs = np.array([0.2,0.4,0.6,0.8])
#plot_percs = np.array([0.125,0.25,0.5,1])
time_frac = np.multiply(plot_percs,time_end)
time_ind = np.multiply(plot_percs,len(time))
particles_ind = [0]*len(time_ind)
plt_clrs = [cmap(0.1),cmap(0.35),cmap(0.55),cmap(0.75)]
plt_counter = 0
for el in enumerate(time_ind):
    for i in range(len(time)):
        if i == int(el[1]-1):# or i == int(time_ind[1]) or i == int(time_ind[2]) or i == int(time_ind[3]) or i == int(time_ind[4]):
            for ind, ele in enumerate(N_S8):
                nS8[ind] = ele[i]
            #print(ele[i])
            #print(time[i])
            plt.plot(bucket_S8.r_avg,np.divide(nS8,sum(nS8)),'-', color=plt_clrs[plt_counter],linewidth=3)#,label=str(round(time[i]/time_end,2)))
            plt_counter = plt_counter + 1
        
   # particles_ind[i]= N_Li2S[int(ele)-1]
plt.rcParams['font.family'] = 'Arial' 
xTicks = np.array([0,1,2,3,4])*1e-7
#plt.xlim([0,4e-7])
#xTicklabels = np.array([r'$0$', r'$0.1$', r'$0.2$', r'$0.3$',r'$0.4$'])
#yTicks = np.array([0,0.1,0.2,0.3,0.4,0.5])
#yTicklabels = np.array([r'$0$', r'$10$', r'$20$', r'$30$',r'$40$',r'$50$'])
# yTicklabels = np.array()
#plt.yticks(yTicks, yTicklabels, fontsize=12)
#plt.xticks(xTicks, xTicklabels, fontsize=12)
#plt.ylim([0,.40])
#plt.legend([r'$0.125$', r'$0.25$', r'$0.5$', r'$1$'],title=r'$\frac{t}{t_\mathrm{max}}$=',fontsize=12,title_fontsize=16) 
plt.legend([str(time[int(time_ind[0])]), str(time[int(time_ind[1])]), str(time[int(time_ind[2])]), str(time[int(time_ind[3]-1)])],title=r'$\frac{t}{t_\mathrm{max}}$=',fontsize=12,title_fontsize=16) 
#plt.ylabel(r'Percent of Particles',fontsize=16)
#plt.xlabel(r'Particle radius [$\mu$m]',fontsize=16)
figg.tight_layout()
#print(plt_clrs)

plt_counter = 0
figgg = plt.figure()
for el in enumerate(time_ind):
    for i in range(len(time)):
        if i == int(el[1]-1):# or i == int(time_ind[1]) or i == int(time_ind[2]) or i == int(time_ind[3]) or i == int(time_ind[4]):
            for ind, ele in enumerate(N_S8):
                nS8[ind] = ele[i]
            plt.plot(bucket_Li2S.r_avg,np.divide(nS8,sum(nS8)), color=plt_clrs[plt_counter],linewidth=3)#,label=str(round(time[i]/time_end,2)))
            plt_counter = plt_counter + 1
xTicks = np.array([0,1,2,3,4])*1e-7
yTicks = np.array([0,0.1,0.2,0.3,0.4,0.5])
yTicklabels = np.array([r'$0$', r'$10$', r'$20$', r'$30$',r'$40$',r'$50$'])
plt.yticks(yTicks, yTicklabels, fontsize=12) 
plt.title("S8")
figgg.tight_layout()
'''
figgggg = plt.figure()
for ind, ele in enumerate(N_S8):
    plt.plot(time,ele-N_Li2S[ind],label=str(ind))
#plt.plot(time,N_Li2S[-1])
'''

fif = plt.figure()
plt.title(str(int(grow_rate_per_area*bucket_S8.mv/bucket_S8.thickness*100) )+"%")
plt.plot(time,bm_S8_back,'.')
plt.plot(time,bm_S8_front,'.')
for i in range(15):
    plt.axhline(y=bucket_S8.thickness*i)
plt.legend(["back","front"])

fiff = plt.figure()
first_bin = np.zeros_like(time)
second_bin = np.zeros_like(time)
third_bin = np.zeros_like(time)
fourth_bin = np.zeros_like(time)
last_bin = np.zeros_like(time)
next_last_bin = np.zeros_like(time)

for i, ele in enumerate(np.transpose(N_S8)):
    #for ind in range(int(bm_S8_back[i]/bucket_S8.thickness),int(bm_S8_front[i]/bucket_S8.thickness)):
    first_bin[i] = ele[int(bm_S8_back[i]/bucket_S8.thickness)]
    second_bin[i] = ele[int(bm_S8_back[i]/bucket_S8.thickness)+2]
    third_bin[i] = ele[int(bm_S8_back[i]/bucket_S8.thickness)+4]
    fourth_bin[i] = ele[int(bm_S8_back[i]/bucket_S8.thickness)+5]

    last_bin[i] = ele[int(bm_S8_front[i]/bucket_S8.thickness)]
    next_last_bin[i] = ele[int(bm_S8_front[i]/bucket_S8.thickness)-1]
    fourth_bin[i] = ele[int(bm_S8_front[i]/bucket_S8.thickness)-2]
plt.plot(time,first_bin,'.')
plt.plot(time,second_bin,'.')
plt.plot(time,third_bin,'.')
plt.plot(time,fourth_bin,'.')
#plt.plot(time,next_last_bin,'.')
#plt.plot(time,last_bin,'.')
#plt.legend(["1",'3','5','7','end-1','end'])
plt.title(int(bm_S8_front[i]/bucket_S8.thickness)-int(bm_S8_back[i]/bucket_S8.thickness))


# Save data
if save_data == 1:
    now = datetime.datetime.now()
    # Format "YYYY-MM-DD_HH-MM-SS"
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(folder_name, exist_ok=True)
    fn_bookmarks = "bookmarks.csv"
    fn_S8 = "Num_Particles_S8.csv"
    fn_Li2S = "Num_Particles_Li2S.csv"
    fn_inputs = "inputs.csv"
    fn_moles = "moles.csv"
    fp_bookmarks = f"{folder_name}/{fn_bookmarks}"
    fp_S8 = f"{folder_name}/{fn_S8}"
    fp_Li2S = f"{folder_name}/{fn_Li2S}"
    fp_inputs = f"{folder_name}/{fn_inputs}"
    fp_moles = f"{folder_name}/{fn_moles}"
    
    df_bookmarks = ( pd.DataFrame({'time': time,
        'Bookmark front S8' : bm_S8_front, 'Bookmark back S8' : bm_S8_back,
        'Bookmark front Li2S' : bm_Li2S_front, 'Bookmark back Li2S' : bm_Li2S_back}))
    df_bookmarks.to_csv(fp_bookmarks, index=False)
    df_S8  = pd.DataFrame()
    for column, ele in enumerate(N_S8):
        df_S8[column+1] = ele
    df_S8.to_csv(fp_S8, index=False)
    df_Li2S  = pd.DataFrame()
    for column, ele in enumerate(N_Li2S):
        df_Li2S[column+1] = ele
    df_Li2S.to_csv(fp_Li2S, index=False)
    df_moles = ( pd.DataFrame({'mol_S8_ca' : mol_S8_ca, 'mol_Li2S_ca' : mol_Li2S_ca,
        'mol_S8_elyt' : mol_S8_elyt, 'mol_Li2S_elyt' : mol_Li2S_elyt}))
    df_moles.to_csv(fp_moles, index=False)
    
    inputs= ([S8_limit, Li2S_limit,Epsilon_S8_0, Epsilon_Li2S_0, Epsilon_eltye_0, area_carbon_0,
        h, V_elyte_0, t_bucket_S8, t_bucket_Li2S, mol_S8_elyt_0, mol_Li2S_elyt_0, mv_S8, mv_Li2S])
    inputs_names = (['S8_limit', 'Li2S_limit','Epsilon_S8_0', 'Epsilon_Li2S_0', 'Epsilon_eltye_0', 'area_carbon_0',
        'h', 'V_elyte_0', 't_bucket_S8', 't_bucket_Li2S', 'mol_S8_elyt_0', 'mol_Li2S_elyt_0',  'mv_S8', 'mv_Li2S'])
    df_inputs = pd.DataFrame([inputs], columns=[inputs_names])
    df_inputs.to_csv(fp_inputs, index=False)
else:
    folder_name = None



plt.show()