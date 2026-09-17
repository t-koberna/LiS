import cantera as ct
import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup_tank import SV_pointer, Solution_Vector_0, SV_0_from_data, Tank, Solid, Parameters
from residual_tank_log import residual
from post_process_tank_log import create_plots
from datetime import datetime
import os
import pandas as pd

save = 0 # 1 for save, this saves all of the state variables in a csv file
data_start = []

# Read in the yaml input file
path = Path("Li_Sulfur_tank_no_ions_log.yaml")
yaml = YAML(typ='rt')
inputs = yaml.load(path)

# The Cantera objects are initialized during the creation of the following
params = Parameters(inputs)
tank = Tank(path, inputs, params)
solids = Solid(path, inputs, tank, params)

# Set the Pointers
SV_idx = SV_pointer(tank)
SV_0 = Solution_Vector_0(SV_idx, tank, solids)
# Change to the log concentrations
SV_0[SV_idx.ptr['C_k_elyte']] = np.log(SV_0[SV_idx.ptr['C_k_elyte']])
elyte_volume = 1e-5 # m^3 = 10 ml
# The total volume of the electrolyte and the solids, this is the baseline I use to calculate current electrolyte volume
total_volume = elyte_volume + SV_0[SV_idx.ptr['mass_S8']]/solids.rho_S8 + SV_0[SV_idx.ptr['mass_Li2S']]/solids.rho_Li2S

## if the solver is starting from data, it overwrites the SV using the final values from the data set
if data_start != []:
    SV_0 = SV_0_from_data(SV_idx, np.size(SV_0), data_start)
    print("starting from data")  
else:
    print("starting from inputfile")

# I track the reaction and species production rates during the simulation, in retrospect, this could be done in post processing
rxn_rates = [tank.elyte_obj.net_rates_of_progress]
S_8_disolve_rate = [solids.surf_S8_obj.get_net_production_rates(solids.elyte_obj)[SV_idx.elyte_species.index('S8(e)')]]
Li2S_disolve_rate = [solids.surf_Li2S_obj.get_net_production_rates(solids.elyte_obj)[SV_idx.elyte_species.index('Li2S(e)')]]

R = tank.elyte_obj.reactant_stoich_coeffs()
P = tank.elyte_obj.product_stoich_coeffs()
nu = P - R
nu = nu.T
q_dot = []
q_dot.append(np.dot(nu.T,tank.elyte_obj.net_rates_of_progress))

#####################################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# User inputs
method = 'Adams'                                                 # Method used by the solver
method= 'BDF'
rtol = 1e-8 #1e-6                                                # Relative tolerance
atol = 1e-8#1e-12                                                # Absolute tolerance
first_step = 1e-20                                               # Size of the initial time step
time_start = 0                                                   # Initial time [s]
time_end = 1e4                                                  # Final time [s]      
min_steps = 1e02                                                 # the minimum number of steps the solver will take 
max_step = time_end/min_steps

tspan = [time_start,time_end]

name_elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]

# allows me to set different absolute tolerances for different state variables 
custom_atol = np.full(np.size(SV_0), atol)   
custom_atol[SV_idx.ptr['volume_Li2S']] = 1e-11#atol#*1e3
custom_atol[SV_idx.ptr['volume_S8']] = 1e-11 #atol#*1e3    
custom_atol[SV_idx.ptr['mass_S8']] = 1e-12 #atol#*1e3
custom_atol[SV_idx.ptr['mass_Li2S']] = 1e-12#atol#*1e3
#custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('TEGDME(e)')]] = atol

start_time = datetime.now() 
# Constrain masses and volumes to >0
const_idx = [int(SV_idx.ptr['volume_S8'][-1]), int(SV_idx.ptr['volume_Li2S'][-1]),int(SV_idx.ptr['mass_S8'][-1]) , int(SV_idx.ptr['mass_Li2S'][-1]) ]
const_type = np.array([1]*len(const_idx))

options =  {'userdata':(SV_idx, tank, solids, params, total_volume),
                'rtol':rtol,'atol':custom_atol, 'first_step':first_step, 'method': method, 
                'constraints_idx': const_idx,'constraints_type': const_type, 'max_step':max_step}

SV_dot_0  = np.zeros_like(SV_0)
solver = sun.cvode.CVODE(residual, **options)
solver.init_step(t0=time_start, y0=SV_0)

t_current = time_start
y_current = SV_0.copy()
time_list = [time_start]
SV_list = [SV_0.copy()]

while t_current < time_end:
     
    # Take one valid internal step (ignores rejected steps)
    sol = solver.step(time_end, method='onestep')
    t_current, y_current = sol.t, sol.y
    
    # Update all of the lists with the values for that step
    time_list.append(t_current)
    SV_list.append(np.copy(y_current))
    tank.elyte_obj = ct.Solution(path, tank.inputs['electrolyte-phase'])
    C_k = y_current[SV_idx.ptr['C_k_elyte']]
    C_total = np.sum(C_k)
    X_k = C_k/C_total
    tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
    solids.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
    rxn_rates.append(np.array(tank.elyte_obj.net_rates_of_progress))
    q_dot.append(np.array(np.dot(nu.T,tank.elyte_obj.net_rates_of_progress)))
    s_dot_S8 = solids.surf_S8_obj.get_net_production_rates(solids.elyte_obj)
    s_dot_S8 = s_dot_S8[SV_idx.elyte_species.index('S8(e)')]
    S_8_disolve_rate.append(np.array(s_dot_S8))
    s_dot_Li2S = solids.surf_Li2S_obj.get_net_production_rates(solids.elyte_obj)
    s_dot_Li2S = s_dot_Li2S[SV_idx.elyte_species.index('Li2S(e)')]
    Li2S_disolve_rate.append(np.array(s_dot_Li2S))

# change them to arrays for plotting  
S_8_disolve_rate = np.array(S_8_disolve_rate)
Li2S_disolve_rate = np.array(Li2S_disolve_rate)
rxn_rates = np.array(rxn_rates)
q_dot = np.array(q_dot)

time = np.array(time_list)
SV = np.transpose(np.vstack(SV_list))
sim_outputs = np.stack((*SV, time))

# print how long the simulation took
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()
print(duration)

# save the date
if save == 1:
    folder_name = "Data/"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(folder_name, exist_ok=True)
    fn = "Outputs.csv"
    fp = f"{folder_name}/{fn}"
    df = pd.DataFrame()
    df[1] = time

    headers = ["time","volume_S8","volume_Li2S","mass_S8","mass_Li2S"] + name_elyte_species

    for column, ele in enumerate(SV):
        df[column+2] = ele
    df.columns = headers 
    df.to_csv(fp,index=False)

#plot results, 1 means the plot is created
solid_and_disolved = 1
species_C_k = 1
C_k_bar = 0
conservation_check = 1
volumes = 1
g_f_rxn = 1
solid_disolution_vs_rxn = 1
rxn_rates_pf = 1
q_dots = 1
gibbs_mixture = 0

plot_flags = [solid_and_disolved, species_C_k, C_k_bar, conservation_check, volumes, g_f_rxn, solid_disolution_vs_rxn, rxn_rates_pf, q_dots, gibbs_mixture]
create_plots(SV_idx, sim_outputs, tank, solids, params, total_volume, time_end, rxn_rates,q_dot,S_8_disolve_rate,Li2S_disolve_rate,plot_flags)
