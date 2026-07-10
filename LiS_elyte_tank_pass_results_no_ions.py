# elyte_concentration.py
#
# Determine what species will exist in the eltye at equilibrium

import cantera as ct
import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup_tank import SV_pointer, Solution_Vector_0, Tank, Solid, Parameters
from residual_tank import residual, residual_ivp
from post_process_tank import create_plots, stack_results
from scipy.integrate import solve_ivp 

# create an interface and replace the equilibrate function with one for a multiphase object

# Read in the yaml input file
path = Path("Li_Sulfur_tank_test.yaml")
path = Path("Li_Sulfur_tank_test_no_ions.yaml")
yaml = YAML(typ='safe')
inputs = yaml.load(path)

params = Parameters(inputs)
# The Cantera objects are initialized during the creation of the following
tank = Tank(path, inputs, params)
solids = Solid(path, inputs, tank, params)

SV_idx = SV_pointer(path, tank, params)
SV_0 = Solution_Vector_0(SV_idx, tank, solids)
# assume a constant volume (maybe add in change with mass dissolving later)
elyte_volume = 0.0005 # m^3 = 500 ml 

rtol = 1e-8                                                         # Relative tolerance
atol = 1e-10    # max 1e-8                                                   # Absolute tolerance
first_step = 1e-10                                                 # Size of the initial time step
# set up and run the solver
time_start = 0                                                      # Initial time [s]
time_end = 1.2e-10    
time_end_2 = 1e-10                                        # Final time [s]
tspan = [time_start,time_end]
max_rate = 0.1e1
min_rate = 1e-10
max_rate_2 = 0.1e3
min_rate_2 = 1e-10
rtol_2 = 1e-8                                                         # Relative tolerance
atol_2 = 1e-10 


solver = 0 # 0 for sundials, anything else for solve_ivp

const_idx = list(range(len(SV_0)))
const_type = np.array([1]*len(SV_0))
const_type[SV_idx.ptr['mass_S8']] = 2
const_type[SV_idx.ptr['mass_Li2S']] = 2
options =  {'userdata':(SV_idx, tank, solids, params, elyte_volume,max_rate, min_rate),
            'rtol':rtol,'atol':atol, 'first_step':first_step, 'method': 'Adams', 'constraints_idx': const_idx,'constraints_type': const_type}
solver = sun.cvode.CVODE(residual, **options)
solution_1 = solver.solve(tspan, SV_0)
sim_outputs = np.stack((*np.transpose(solution_1.y), solution_1.t))

mass_S8, = sim_outputs[SV_idx.ptr['mass_S8']]
mass_Li2S, = sim_outputs[SV_idx.ptr['mass_Li2S']]
C_k_elyte = sim_outputs[SV_idx.ptr['C_k_elyte']]
C_k_elyte = np.concatenate((C_k_elyte,C_k_elyte),axis=1)
c_k_end = [i[-1] for i in C_k_elyte]
c_k_end = np.abs(c_k_end)

SV_0_2 = np.zeros_like(SV_0)
SV_0_2[SV_idx.ptr['mass_S8']] = mass_S8[-1]
SV_0_2[SV_idx.ptr['mass_Li2S']] = mass_Li2S[-1]
SV_0_2[SV_idx.ptr['C_k_elyte']] =  c_k_end
C_total = np.sum(c_k_end)
X_k = c_k_end/C_total
tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
solids.elyte_obj.TPX = solids.elyte_obj.T, solids.elyte_obj.P, X_k

#rtol = 1e-8                                                         # Relative tolerance
#atol = 1e-12
options_2 =  {'userdata':(SV_idx, tank, solids, params, elyte_volume,max_rate_2, min_rate_2),
            'rtol':rtol_2,'atol':atol_2, 'first_step':first_step, 'method': 'Adams', 'constraints_idx': const_idx,'constraints_type': const_type}
#time_end = 9.625e-10                                             # Final time [s]
tspan_2 = [time_start,time_end_2]
solver_2 = sun.cvode.CVODE(residual, **options_2)

solution_2 = solver_2.solve(tspan_2, SV_0_2)
y_total, time_total = stack_results(solution_1,solution_2)
sim_outputs_total = np.stack((*y_total, time_total))

###########################
max_rate_2 = 1e5
min_rate_2 = 1e1
rtol_2 = 1e-7                                                         # Relative tolerance
atol_2 = 1e-8 
time_end_2 = 1e-10                                        # Final time [s]

mass_S8, = sim_outputs_total[SV_idx.ptr['mass_S8']]
mass_Li2S, = sim_outputs_total[SV_idx.ptr['mass_Li2S']]
C_k_elyte = sim_outputs_total[SV_idx.ptr['C_k_elyte']]
C_k_elyte = np.concatenate((C_k_elyte,C_k_elyte),axis=1)
c_k_end = [i[-1] for i in C_k_elyte]
c_k_end = np.abs(c_k_end)

SV_0_2 = np.zeros_like(SV_0)
SV_0_2[SV_idx.ptr['mass_S8']] = mass_S8[-1]
SV_0_2[SV_idx.ptr['mass_Li2S']] = mass_Li2S[-1]
SV_0_2[SV_idx.ptr['C_k_elyte']] =  c_k_end
C_total = np.sum(c_k_end)
X_k = c_k_end/C_total
tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
solids.elyte_obj.TPX = solids.elyte_obj.T, solids.elyte_obj.P, X_k

#rtol = 1e-8                                                         # Relative tolerance
#atol = 1e-12
options_2 =  {'userdata':(SV_idx, tank, solids, params, elyte_volume,max_rate_2, min_rate_2),
            'rtol':rtol_2,'atol':atol_2, 'first_step':first_step, 'method': 'Adams', 'constraints_idx': const_idx,'constraints_type': const_type}
#time_end = 9.625e-10                                             # Final time [s]
tspan_2 = [time_start,time_end_2]
solver_2 = sun.cvode.CVODE(residual, **options_2)
solution_2 = solver_2.solve(tspan_2, SV_0_2)

time_1 = time_total
time_2 = solution_2.t + time_1[-1]
time_total = np.concatenate((time_1,time_2))
y_total = np.concatenate((y_total,np.transpose(solution_2.y)),axis=1)
sim_outputs_total = np.stack((*y_total, time_total))





create_plots(SV_idx, sim_outputs_total, tank, solids, params, elyte_volume,[time_end,time_1[-1]])



'''
print(solids.surf_S8_obj.get_net_production_rates(solids.elyte_obj))
print(name_elyte_species[3])
print(solids.surf_Li2S_obj.get_net_production_rates(solids.elyte_obj))
(name_elyte_species[-1])

print(solids.solid_S8_obj.partial_molar_volumes)
print(solids.solid_Li2S_obj.partial_molar_volumes)
print(solids.solid_S8_obj.density)
print(solids.solid_Li2S_obj.density)
# This is a check to make sure molar volume times density yields molar mass
print(solids.mv_Li2S*solids.rho_Li2S)
print(solids.mv_S8*solids.rho_S8)
'''
