# LiS_main.py
#
# This file serves as the main model file.  
#   It is called by the user to run the model
# The model begins with all S8 dissolved in the electroltye, 
# then tracks the deposition of S8 on a planer carbon assuming constant 
# rates for both nucleation and growth 
import numpy as np
import matplotlib.pyplot as plt
from scikits.odes import dae
from LiS_functions import bucket, Index_start, residual, plot_results
import datetime
import os
import pandas as pd

save_data = 0 # saves the data to a folder if this is a 1

'''
Constants
''' 
F = 96485.34 #Faraday's number [C/mol_electron]
R = 8.3145 #Universal gas constant [J/mol-K]

'''
USER INPUTS
'''
## Operating Conditions
t_sim_max = [4] # the maximum time the battery will be held at each current [s]
T = 298.15 # standard temperature [K]

## Material Properties
rho_S8 = 2070 # density of Sulfur (S8) [kg/m^3]
MW_S8 = 0.25652  # molecular weight [kg/mol]
mv_S8 = MW_S8/rho_S8 # constant molar volume S_8 [m^3/mol]

'''
Parameters
'''
area_carbon_0 = 1 # inital area of carbon (this is the area where nucleation happens) [m^2]
# Each of the buckets are the same size, with the exception of the final bucket which will extend to infinity
n_bucket_S8 = 100 # number of buckets for S8 [-]
t_bucket_S8 = 1e-8 # the radius range (aka thickness) of each bucket for S8 [m]

bucket_S8 = bucket(n_bucket_S8,t_bucket_S8,mv_S8,"S_8")
SV_index = Index_start(n_bucket_S8) # Holds the pointers for the SV vector

'''
Initialize the State Variable vector
'''
sim_inputs = np.zeros(n_bucket_S8 + 2)
sim_inputs[:SV_index.S8] = np.zeros(n_bucket_S8)
sim_inputs[SV_index.bm_S8_front] = 0 
sim_inputs[SV_index.bm_S8_back] = 0 

time_start = 0 # Initial time [s]
time_end = t_sim_max[0] #Final time [s]
times = np.linspace(time_start,time_end,1001)

'''
Integration
'''

# I am using a DAE solver, but for now there are no algebraic equations
algvars = []

#[s_k_nuc_S8,s_k_grow_S8] are the first 4 terms in params [mol/m^3]
grow_rate_per_area = 1e-3#1e-4
nuc_rate_per_area = 2e-3#10e-1 
params = [nuc_rate_per_area,grow_rate_per_area , SV_index, bucket_S8,area_carbon_0]
options =  {'user_data':params, 'rtol':1e-11,'atol':1e-11, 
            'algebraic_vars_idx':algvars, 'first_step_size':1e-15}
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
bm_S8_front = sim_outputs[SV_index.bm_S8_front]
bm_S8_back = sim_outputs[SV_index.bm_S8_back]
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
    fn_inputs = "inputs.csv"
    fp_bookmarks = f"{folder_name}/{fn_bookmarks}"
    fp_S8 = f"{folder_name}/{fn_S8}"
    fp_inputs = f"{folder_name}/{fn_inputs}"

    
    df_bookmarks = ( pd.DataFrame({'time': time,
        'Bookmark front S8' : bm_S8_front, 'Bookmark back S8' : bm_S8_back}))
    df_bookmarks.to_csv(fp_bookmarks, index=False)
    df_S8  = pd.DataFrame()
    for column, ele in enumerate(N_S8):
        df_S8[column+1] = ele
    df_S8.to_csv(fp_S8, index=False)
    
    inputs= ([area_carbon_0,t_bucket_S8, mv_S8])
    inputs_names = (['area_carbon_0', 't_bucket_S8', 'mv_S8'])
    df_inputs = pd.DataFrame([inputs], columns=[inputs_names])
    df_inputs.to_csv(fp_inputs, index=False)
else:
    folder_name = None

'''
plot the results 
'''
# pick what plots to display (1 yes, anything else no)
time_stamps_bins = 1
bookmark_movement = 0

plot_flags = [time_stamps_bins, bookmark_movement]

plot_results(plot_flags, time, N_S8, bucket_S8, 
        bm_S8_front, bm_S8_back, time_end, folder_name)

plt.show()