# LiS_main.py
#
# This file serves as the main model file.  
#   It is called by the user to run the model
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib as mP
from scikits.odes import dae
from LiS_functions import bucket, Index_start, residual_ivp, residual, plot_results
import datetime
import os
import pandas as pd

save_data = 0 # saves the data to a folder if this is a 1

ivp = 1 #True # cnages if I use solve ivp or dae 

# Anode is on the left at x=0 and Cathode is on the right
# Li -> Li+ + e- (reaction at the anode)
# 1/2S_8 + e- -> 1/2S_8^2- (reaction at the cathode)
# Currently, the model begins with all Li2S and S8 dissolved in the electrolyte, 
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
# maybe add a cutoff for the consecration of a species in the elyte
S8_limit = 2e10 # The maximum number of particles that can be in the final bucket for S_8
Li2S_limit = 2e10 # The maximum number of particles that can be in the final bucket for Li_2S

## Operating Conditions
t_sim_max = [4] # the maximum time the battery will be held at each current [s]
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
mol_S8_elyt_0 = 5e-5 # Initial moles of S_8 in the electrolyte [mol/m^3]
mol_Li2S_elyt_0 = 5e-5 # Initial moles of Li_2S in the electrolyte [mol/m^3]

## Material parameters: (Replaced by Cantera?)

#################################### Danger!
mv_Li2S = mv_S8

'''
Parameters
'''

## Geometry
# I do not track the volume fraction for carbon because it is planer and inert
# Later I will track the volume fraction of the anode since Lithium dissolves
Epsilon_S8_0 = 0 # Initial volume fraction of S_8 in the cathode [-]
Epsilon_Li2S_0 = 0 # Initial volume fraction of Li_2S in the cathode [-]
Epsilon_eltye_0 = 1 - Epsilon_S8_0 - Epsilon_Li2S_0 # Initial volume fraction of electrolyte [-]
# I set the initial area of carbon as 1 m so everything becomes per unit area
area_carbon_0 = 1 # initial area of carbon (this is the area where nucleation happens) [m^2]

h = 1e-8 # height of the tank [m] (stand in for electrolyte thickness)
V_elyte_0 = area_carbon_0*h # initial volume of electrolyte

# Each of the buckets are the same size, with the exception of the final bucket which will extend to infinity
n_bucket_S8 = 150 # number of buckets for S8 [-]
n_bucket_Li2S = 150 # number of buckets for Li2S [-]

t_bucket_S8 = 8e-10 # the radius range (aka thickness) of each bucket for S8 [m]
t_bucket_Li2S = 8e-10 # the radius range (aka thickness) of each bucket for Li2S [m]

bucket_S8 = bucket(n_bucket_S8,t_bucket_S8,mv_S8,"S_8")
bucket_Li2S = bucket(n_bucket_Li2S,t_bucket_Li2S,mv_Li2S,"Li_2S")

SV_index = Index_start(n_bucket_S8,n_bucket_Li2S) # Holds the pointers for the SV vector

'''
Initialize the SV vector
'''
sim_inputs = np.zeros(n_bucket_S8 + n_bucket_Li2S + 2 + 2 + 2)

# I put S8 on top of Li2S. All buckets everything start with zero particles
sim_inputs[:SV_index.S8] = np.zeros(n_bucket_S8)
sim_inputs[SV_index.S8:SV_index.Li2S] = np.zeros(n_bucket_Li2S)
sim_inputs[SV_index.mol_S8_ca] = 0
sim_inputs[SV_index.mol_Li2S_ca] = 0
sim_inputs[SV_index.mol_S8_elyte] = mol_S8_elyt_0 
sim_inputs[SV_index.mol_Li2S_elyte] = mol_Li2S_elyt_0 
sim_inputs[SV_index.bm_S8_front] = 0 
sim_inputs[SV_index.bm_Li2S_front] = 0 

time_start = 0 # Initial time [s]
time_end = t_sim_max[0] #Final time [s]

'''
Integration
'''
# Integration Limits 
# will add concentration and voltage checks later on
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
    
# I am not sure if these will be params or if the residual can call Cantera directly
#[s_k_nuc_S8,s_k_grow_S8,s_k_nuc_Li2S,s_k_grow_Li2S] are the first 4 terms in params [mol/m^3]
grow_rate_per_area = 1e-4
nuc_rate_per_area = 10e-1 
params = [nuc_rate_per_area,grow_rate_per_area,nuc_rate_per_area,grow_rate_per_area , SV_index, bucket_S8, bucket_Li2S,area_carbon_0]


if ivp == True:
    t_span = [time_start,time_end]
    min_time_intervals = 100
    max_t_step = time_end/min_time_intervals
    solution = (solve_ivp(residual_ivp,t_span,sim_inputs,method='BDF',
                args=[params], rtol = 1e-8,atol = 1e-10, max_step = max_t_step))
    sim_outputs =np.stack((*(solution.y), solution.t))
else:
    times = np.linspace(time_start,time_end,1001)
    algvars = []
    options =  {'user_data':params, 'rtol':1e-11,'atol':1e-11, 
                'algebraic_vars_idx':algvars, 'first_step_size':1e-15,'rootfn':terminate_check,'nr_rootfns':num_roots}
                # , 'compute_initcond':'yp0', 'max_steps':10000}
    solver = dae('ida', residual, **options)
    SV_0 = sim_inputs
    SV_dot_0  = np.zeros_like(SV_0)
    solution = solver.solve(times, SV_0, SV_dot_0)
    sim_outputs =np.stack((*np.transpose(solution.values.y), solution.values.t))


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
bm_Li2S_front = sim_outputs[SV_index.bm_Li2S_front]
time = sim_outputs[-1]

# Save data
# Creates a new folder based on the time and saves the inputs for the simulation 
# along with the values of the state variables from the solution
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
        'Bookmark front S8' : bm_S8_front, 'Bookmark front Li2S' : bm_Li2S_front,}))
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
        h, V_elyte_0, t_bucket_S8, t_bucket_Li2S, mol_S8_elyt_0, mol_Li2S_elyt_0, mv_S8, mv_Li2S, bucket_S8.r_min, bucket_Li2S.r_min])
    inputs_names = (['S8_limit', 'Li2S_limit','Epsilon_S8_0', 'Epsilon_Li2S_0', 'Epsilon_eltye_0', 'area_carbon_0',
        'h', 'V_elyte_0', 't_bucket_S8', 't_bucket_Li2S', 'mol_S8_elyt_0', 'mol_Li2S_elyt_0',  'mv_S8', 'mv_Li2S','r_min_S8','r_min_Li2S'])
    df_inputs = pd.DataFrame([inputs], columns=[inputs_names])
    df_inputs.to_csv(fp_inputs, index=False)
else:
    folder_name = None

'''
plot the results 
'''
# pick what plots to display (1 yes, anything else no)
num_particles_bin = 1
cs_area = 0
total_particles = 1
conc_and_moles = 1
vol_frac = 0
time_stamps_bins = 1
bookmark_movement = 1

plot_flags = [num_particles_bin, cs_area, total_particles, conc_and_moles, vol_frac, time_stamps_bins, bookmark_movement]

plot_results(plot_flags, time, N_S8, N_Li2S, bucket_S8, bucket_Li2S, 
        mol_S8_elyt, mol_Li2S_elyt, mol_S8_ca, mol_Li2S_ca,
        bm_S8_front, bm_Li2S_front, h, area_carbon_0, V_elyte_0, time_end, folder_name)

plt.show()

