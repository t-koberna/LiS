import cantera as ct
import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup_tank import SV_pointer, Solution_Vector_0, SV_0_from_outputs, SV_0_from_data, Tank, Solid, Parameters, RateTracker
from residual_tank import residual
from post_process_tank import create_plots
from datetime import datetime
import os
import pandas as pd
import re

save = 1 # 1 for save, this saves all of the state variables in a csv file
data_start = "disolve"
data_start = []
#data_start = 'almost'
#data_start = 'Sulfur working great'
#data_start = '2026-08-21_14-44-51'
data_start = "2026-08-21_15-59-08 fghy"


# Read in the yaml input file
path = Path("Li_Sulfur_tank_test_no_ions_in.yaml")
path = Path("Li_Sulfur_tank_test_no_ions_in_reduced.yaml")
path = Path("Li_Sulfur_tank_no_ions_in.yaml")
path = Path("Li_Sulfur_tank_no_ions_train.yaml")
#path = Path("Li_Sulfur_tank_no_ions_train_eq_Cs.yaml")

#path = Path("Li_Sulfur_tank_no_ions_in.yaml")

path_out = Path("Li_Sulfur_tank_no_ions.yaml")

yaml = YAML(typ='rt')
inputs = yaml.load(path)

# The Cantera objects are initialized during the creation of the following
params = Parameters(inputs)
tank = Tank(path, inputs, params)
solids = Solid(path, inputs, tank, params)
#rxn_rates = RateTracker(tank)


SV_idx = SV_pointer(tank)
SV_0 = Solution_Vector_0(SV_idx, tank, solids)
elyte_volume = 1e-5 # m^3 = 10 ml
# The total volume of the electrolyte and the solids, this is the baseline I use to calculate current electrolyte volume
total_volume = elyte_volume + SV_0[SV_idx.ptr['mass_S8']]/solids.rho_S8 + SV_0[SV_idx.ptr['mass_Li2S']]/solids.rho_Li2S

## if the solver is starting from data, it overwrites the SV using the final values from the data set
if data_start != []:
    SV_0 = SV_0_from_data(SV_idx, np.size(SV_0), data_start)
    print("starting from data")  
else:
    print("starting from inputfile")

eff_C_zero = 1e-32
C_min = 1e-80
max_rate = 1e-30 #20
#C_k = np.copy(SV_0[SV_idx.ptr['C_k_elyte']])
C_k = SV_0[SV_idx.ptr['C_k_elyte']].copy()
#C_k[2:8] = [3e-8,1e-8,2e-9,5e-11,2e-9,2e-10]

# set initial amounts of dissolved solids at equilibrium
R = ct.gas_constant
T = tank.elyte_obj.T
g_f = tank.elyte_obj.standard_gibbs_RT * R * T
g_f_solid_Li2S = solids.solid_Li2S_obj.standard_gibbs_RT * R * T
g_f_solid_S8 = solids.solid_S8_obj.standard_gibbs_RT * R * T

g_f_S8 = g_f[SV_idx.elyte_species.index('S8(e)')] - g_f_solid_S8[-1]
g_f_Li2S = g_f[SV_idx.elyte_species.index('Li2S(e)')] - g_f_solid_Li2S[-1]

for l in range(4):
    X_S8 = np.exp(-g_f_S8/R/T)
    C_S8_eq = X_S8*np.sum(C_k)/(1-X_S8)
    #C_k[SV_idx.elyte_species.index('S8(e)')] = C_S8_eq

    X_Li2S = np.exp(-g_f_Li2S/R/T)
    C_Li2S_eq = X_Li2S*np.sum(C_k)/(1-X_Li2S)
    #C_k[SV_idx.elyte_species.index('Li2S(e)')] = C_Li2S_eq

#SV_0[SV_idx.ptr['C_k_elyte'][SV_idx.elyte_species.index('S8(e)')]] = C_S8_eq*.95
#SV_0[SV_idx.ptr['C_k_elyte'][SV_idx.elyte_species.index('Li2S(e)')]] = C_Li2S_eq*.95


for i, el in enumerate(C_k):
    if el < eff_C_zero:
        C_k[i] = eff_C_zero
    if el < C_min:
        print(el)
        SV_0[SV_idx.ptr['C_k_elyte'][i]] = C_min
#SV_0[SV_idx.ptr['C_k_elyte']] = C_k

C_total = np.sum(C_k)
X_k = C_k/C_total
tank_train = Tank(path, inputs, params)
SV_0_train = Solution_Vector_0(SV_idx, tank_train, solids)
C_k_train = SV_0_train[SV_idx.ptr['C_k_elyte']].copy()
C_total_train = np.sum(C_k_train)
X_k_train = C_k_train/C_total_train 

tank_train.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k_train
solids.elyte_obj.TPX = solids.elyte_obj.T, solids.elyte_obj.P, X_k


net_rates = tank_train.elyte_obj.net_rates_of_progress
write_yaml = 0
for i, rxn in enumerate(inputs['bulk-elyte-reactions']):
    reaction_rate = np.abs(net_rates[i])
    pre_exp_factor = tank.elyte_obj.reaction(i).rate.pre_exponential_factor
    
    if reaction_rate > max_rate * 10:
        print("rate")
        new_A = pre_exp_factor* (max_rate / reaction_rate)
        
        # 1. Format explicitly to 2 decimal places in scientific notation
        # e.g., 2.97948e+30 becomes "2.98e+30"
        formatted_str = f"{new_A:.2e}"
        
        # 2. Store it directly as a standard string. 
        # This keeps ruamel's internal layout properties from crashing.
        rxn['rate-constant']['A'] = formatted_str
#############################################################################################
write_yaml = 1
if write_yaml == 1:
    # Step 1: Write file out using ruamel 
    # (This completely preserves all your inline comments and spacing)
    with open(path_out, 'w') as file:
        yaml.dump(inputs, file)

    # Step 2: Open file as a plain text string and strip the quotes from 'A' fields
    with open(path_out, 'r', newline="", encoding='utf-8') as file:
        text_content = file.read()

    # This regular expression scans the file for patterns like: A: '2.98e+30' or A: "2.98e+30"
    # and safely converts them to bare scalar numbers: A: 2.98e+30
    cleaned_content = re.sub(r"(A:\s*)['\"]([+-]?\d+\.?\d*e[+-]?\d+)['\"]", r"\1\2", text_content)

    # Overwrite the file with the cleaned plain text string
    with open(path_out, 'w', newline="", encoding='utf-8') as file:
        file.write(cleaned_content)

    # 3. Reload your Cantera solution matrix cleanly
    tank.elyte_obj = ct.Solution(path_out, tank.inputs['electrolyte-phase'])

#SV_0 = Solution_Vector_0(SV_idx, tank, solids)
C_k = SV_0[SV_idx.ptr['C_k_elyte']].copy()
C_total = np.sum(C_k)
X_k = C_k/C_total
tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
solids.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
print(tank.elyte_obj.net_rates_of_progress)
#rxn_rates.rates = [tank.elyte_obj.net_rates_of_progress]
rxn_rates = [tank.elyte_obj.net_rates_of_progress]
S_8_disolve_rate = [solids.surf_S8_obj.get_net_production_rates(solids.elyte_obj)[SV_idx.elyte_species.index('S8(e)')]]
Li2S_disolve_rate = [solids.surf_Li2S_obj.get_net_production_rates(solids.elyte_obj)[SV_idx.elyte_species.index('Li2S(e)')]]

name_elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]
surf = tank.elyte_obj
R = surf.reactant_stoich_coeffs()
P = surf.product_stoich_coeffs()
nu = P - R
nu = nu.T
q_dot = []
q_dot.append(np.dot(nu.T,tank.elyte_obj.net_rates_of_progress))

#####################################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#print(tank.inputs['bulk-elyte-reactions'])
                                 

# User inputs
method = 'Adams'                                                 # Method used by the solver
method= 'BDF'
rtol = 1e-4 #1e-6                                                # Relative tolerance
atol = 1e-17#1e-12    # max 1e-8                                 # Absolute tolerance
first_step = 1e-30                                                # Size of the initial time step
time_start = 0                                                   # Initial time [s]
time_end = 2.6e5#17500#6e-8                                            # Final time [s]      
#time_end = 1.1e10#6e-8                                                  
max_rate = 1e-13 #max_rate*1e6                                          # Maximum rate of progress for a reaction 
min_rate = 0 #1e20                                                    # Rates below this threshold are set to zero
min_steps = 1e02                                               # the minimum number of steps the solver will take 
max_step = time_end/min_steps

tspan = [time_start,time_end]

name_elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]

# allows me to set different absolute tolerances for different state variables 
custom_atol = np.full(np.size(SV_0), atol)   
custom_atol[SV_idx.ptr['volume_Li2S']] = 1e-14 #atol#*1e3
custom_atol[SV_idx.ptr['volume_S8']] = 1e-14 #atol#*1e3    
custom_atol[SV_idx.ptr['mass_S8']] = 1e-12 #atol#*1e3
custom_atol[SV_idx.ptr['mass_Li2S']] = 1e-12#atol#*1e3
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('TEGDME(e)')]] = atol
#custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('Li+(e)')]] = atol
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('S8(e)')]] = 1e-25#1e-6#atol#*1e3
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('S6(e)')]] = 1e-25#1e3
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('S7(e)')]] = 1e-25#-4
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('S2(e)')]] = 1e-25#21
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('S3(e)')]] = 1e-25#15
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('S4(e)')]] = 1e-25# -21
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('S5(e)')]] = 1e-25#atol#*1e3

custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('Li2S(e)')]] = 5e-10
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('Li2S2(e)')]] = 1e-25#18
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('Li2S3(e)')]] = 5e-25#1e-13
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('Li2S4(e)')]] = 1e-25
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('Li2S5(e)')]] = 1e-25
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('Li2S6(e)')]] = 1e-25
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('Li2S7(e)')]] = 1e-25
custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('Li2S8(e)')]] = 1e-25

#custom_atol[SV_idx.ptr['C_k_elyte'][name_elyte_species.index('S8(e)')+1:name_elyte_species.index('Li2S8(e)')]] = atol*1e-1
ind_sulfides_start = name_elyte_species.index('S8(e)')
ind_sulfides_stop = name_elyte_species.index('Li2S8(e)')
#custom_atol[SV_idx.ptr['C_k_elyte'][ind_sulfides_start:ind_sulfides_stop]]  = C_k[ind_sulfides_start:ind_sulfides_stop].copy()*1e-1

start_time = datetime.now() 
# Constrain masses to >0 , concentrations and volumes to >=0
const_idx = list(range(len(SV_0)))
const_type = np.array([1]*len(SV_0))  
const_type[SV_idx.ptr['mass_S8']] = 1
const_type[SV_idx.ptr['mass_Li2S']] = 1
const_type[SV_idx.ptr['volume_S8']] = 1
const_type[SV_idx.ptr['volume_Li2S']] = 1
options =  {'userdata':(SV_idx, tank, solids, params, total_volume, max_rate, min_rate, path_out),
                'rtol':rtol,'atol':custom_atol, 'first_step':first_step, 'method': method, 
                'constraints_idx': const_idx,'constraints_type': const_type, 'max_step':max_step}

SV_dot_0  = np.zeros_like(SV_0)
solver = sun.cvode.CVODE(residual, **options)
solver.init_step(t0=time_start, y0=SV_0)

t_current = time_start
y_current = SV_0.copy()
time_list = [time_start]
SV_list = [SV_0.copy()]
pre_exponential_factors = []
for i, rxn in enumerate(tank.elyte_obj.reactions()):
    A_val = rxn.rate.input_data['rate-constant'].get('A', None)
    pre_exponential_factors.append(A_val)
pre_exponential_factors = [pre_exponential_factors]

while t_current < time_end:
     
    # 1. Take exactly one valid internal step (ignores rollbacks/rejected steps)
    sol = solver.step(time_end, method='onestep')
    t_current, y_current = sol.t, sol.y
    
    # 2. Append the individual 1D row array to your temporary history lists
    time_list.append(t_current)
    SV_list.append(np.copy(y_current))
    tank.elyte_obj = ct.Solution(path_out, tank.inputs['electrolyte-phase'])
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

    pre_exponential_factors_current = []
    for i, rxn in enumerate(tank.elyte_obj.reactions()):
        A_val = rxn.rate.input_data['rate-constant'].get('A', None)
        pre_exponential_factors_current.append(A_val)

    pre_exponential_factors.append(pre_exponential_factors_current)

    
S_8_disolve_rate = np.array(S_8_disolve_rate)
Li2S_disolve_rate = np.array(Li2S_disolve_rate)
rxn_rates = np.array(rxn_rates)
q_dot = np.array(q_dot)
pre_exponential_factors = np.array(pre_exponential_factors)

time = np.array(time_list)
SV = np.transpose(np.vstack(SV_list))
sim_outputs = np.stack((*SV, time))

#solution = solver.solve(tspan, SV_0)
#time = solution.t
#SV = np.transpose(solution.y)
#sim_outputs = np.stack((*SV,time ))

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
solid_and_disolved = 1
species_C_k = 1
C_k_bar = 1
conservation_check = 0
volumes = 1
g_f_rxn = 1
solid_disolution_vs_rxn = 1
rxn_rates_and_A = 1
q_dots = 1
gibbs_mixture = 0
#print(tank.elyte_obj.concentrations)
#print([i[-1] for i in sim_outputs[SV_idx.ptr['C_k_elyte']]])
plot_flags = [solid_and_disolved, species_C_k, C_k_bar, conservation_check, volumes, g_f_rxn, solid_disolution_vs_rxn, rxn_rates_and_A, q_dots, gibbs_mixture]
create_plots(SV_idx, sim_outputs, tank, solids, params, total_volume, time_end, rxn_rates, pre_exponential_factors,q_dot,S_8_disolve_rate,Li2S_disolve_rate,plot_flags)
