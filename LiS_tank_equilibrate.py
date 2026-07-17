import cantera as ct
import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup_tank import SV_pointer, Solution_Vector_0, SV_0_from_outputs, SV_0_from_data, Tank, Solid, Parameters
from residual_tank import residual
from post_process_tank import create_plots
from datetime import datetime
import os
import pandas as pd

save = 0 # 1 for save, this saves all of the state variables in a csv file
data_start = []
#data_start = "stage_1"

# Read in the yaml input file
path = Path("Li_Sulfur_tank_rates_1em20.yaml")

yaml = YAML(typ='safe')
inputs = yaml.load(path)

# The Cantera objects are initialized during the creation of the following
params = Parameters(inputs)
tank = Tank(path, inputs, params)
solids = Solid(path, inputs, tank, params)

SV_idx = SV_pointer(tank)
SV_0 = Solution_Vector_0(SV_idx, tank, solids)
elyte_volume = 0.0005 # m^3 = 500 ml
# The total volume of the electrolyte and the solids, this is the baseline I use to calculate current electrolyte volume
total_volume = elyte_volume + SV_0[SV_idx.ptr['mass_S8']]/solids.rho_S8 + SV_0[SV_idx.ptr['mass_Li2S']]/solids.rho_Li2S

## if the solver is starting from data, it overwrites the SV using the final values from the data set
if data_start != []:
    SV_0 = SV_0_from_data(SV_idx, np.size(SV_0), data_start)
    print("starting from data")  
else:
    print("starting from inputfile")

# User inputs
method = 'Adams'                                                 # Method used by the solver
#method= 'BDF'
rtol = 1e-9 #1e-6                                                # Relative tolerance
atol = 1e-9#1e-12    # max 1e-8                                 # Absolute tolerance
first_step = 1e-8                                                # Size of the initial time step
time_start = 0                                                   # Initial time [s]
time_end = 1e5#6e-8                                            # Final time [s]                                                      # no algebraic variables
max_rate = 1e12                                                  # Maximum rate of progress for a reaction 
min_rate = 0                                                     # Rates below this threshold are set to zero
min_steps = 1000                                                 # the minimum number of steps the solver will take                                             

tspan = [time_start,time_end]

name_elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]

# allows me to set different absolute tolerances for different state variables 
custom_atol = np.full(np.size(SV_0), atol)       
custom_atol[SV_idx.ptr['mass_S8']] = atol
custom_atol[SV_idx.ptr['mass_Li2S']] = atol
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('TEGDME(e)')]] = 1e-2
#custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('Li+(e)')]] = atol
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('S8(e)')]] = atol
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('S8(e)')+1:name_elyte_species.index('Li2S8(e)')]] = atol

start_time = datetime.now() 
# Constrain masses to >0 , concentrations and volumes to >=0
const_idx = list(range(len(SV_0)))
const_type = np.array([1]*len(SV_0))  
const_type[SV_idx.ptr['mass_S8']] = 2
const_type[SV_idx.ptr['mass_Li2S']] = 2
const_type[SV_idx.ptr['volume_S8']] = 1
const_type[SV_idx.ptr['volume_Li2S']] = 1
options =  {'userdata':(SV_idx, tank, solids, params, total_volume, max_rate, min_rate),
                'rtol':rtol,'atol':custom_atol, 'first_step':first_step, 'method': method, 
                'constraints_idx': const_idx,'constraints_type': const_type, 'max_step':time_end/min_steps}

SV_dot_0  = np.zeros_like(SV_0)
solver = sun.cvode.CVODE(residual, **options)
solution = solver.solve(tspan, SV_0)
time = solution.t
SV = np.transpose(solution.y)
sim_outputs = np.stack((*SV,time ))

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

#plot results
create_plots(SV_idx, sim_outputs, tank, solids, params, total_volume, time_end)
