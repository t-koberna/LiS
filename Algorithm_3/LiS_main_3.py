# LiS_main.py
#
# This file serves as the main model file.  
#   It is called by the user to run the model
# The model begins with all S8 dissolved in the electrolyte, 
# then tracks the deposition of S8 on a planer carbon assuming constant 
# rates for both nucleation and growth 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from LiS_functions_3 import bucket, Index_start, residual, plot_results
import os

save_picture = 1 # saves the data to a folder if this is a 1

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
mv_S8 = 1.25e-4 # changed from 1.239e-4 so the numbers are cleaner

'''
Parameters
'''
area_carbon_0 = 1 # initial area of carbon (this is the area where nucleation happens) [m^2]
# Each of the buckets are the same size, with the exception of the final bucket which will extend to infinity
n_bucket_S8 = 200 # number of buckets for S8 [-]
t_bucket_S8 = 1e-8 # the radius range (aka thickness) of each bucket for S8 [m]

bucket_S8 = bucket(n_bucket_S8,t_bucket_S8,mv_S8,"S_8")
SV_index = Index_start(n_bucket_S8) # Holds the pointers for the SV vector

'''
Initialize the State Variable vector
'''
sim_inputs = np.zeros(n_bucket_S8 + 1)
sim_inputs[:SV_index.S8] = np.zeros(n_bucket_S8)
sim_inputs[SV_index.bm_S8_front] = 0 + bucket_S8.thickness/2

time_start = 0 # Initial time [s]
time_end = t_sim_max[0] #Final time [s]
times = np.linspace(time_start,time_end,1001)

'''
Integration
'''
grow_rate_per_area = 2e-3
nuc_rate_per_area = 2e-4
params = [nuc_rate_per_area,grow_rate_per_area , SV_index, bucket_S8,area_carbon_0]

t_span = [time_start,time_end]
min_time_intervals = 100
max_t_step = time_end/min_time_intervals
solution = (solve_ivp(residual,t_span,sim_inputs,method='BDF',
            args=[params], rtol = 1e-8,atol = 1e-10, max_step = max_t_step))
sim_outputs =np.stack((*(solution.y), solution.t))

'''
Post Processing          
'''
N_S8  = sim_outputs[:SV_index.S8]
bm_S8_front = sim_outputs[SV_index.bm_S8_front]
time = sim_outputs[-1]

if save_picture == 1:
    folder_name = "Algorithm_3"
    os.makedirs(folder_name, exist_ok=True)
else:
    folder_name = None

'''
plot the results 
'''
plot_flags = save_picture

plot_results(plot_flags, time, N_S8, bucket_S8, 
        bm_S8_front, time_end, folder_name)

plt.show()