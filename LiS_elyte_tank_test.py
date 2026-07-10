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
from post_process_tank import create_plots
from scipy.integrate import solve_ivp 

# create an interface and replace the equilibrate function with one for a multiphase object

# Read in the yaml input file
path = Path("Li_Sulfur_tank_test.yaml")
#path = Path("Li_Sulfur_tank_test_no_ions.yaml")
yaml = YAML(typ='safe')
inputs = yaml.load(path)

params = Parameters(inputs)
# The Cantera objects are initialized during the creation of the following
tank = Tank(path, inputs, params)
solids = Solid(path, inputs, tank, params)

'''
tank.elyte_obj.X= 0.1*np.ones_like([species['C_k'] for species in tank.inputs['transport']['diffusion-coefficients']])
name_elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]
equations = tank.elyte_obj.reaction_equations()
num_reactions = np.size(equations)
max_rate = 1e2
reactants_nu = tank.elyte_obj.reactant_stoich_coeffs()
products_nu  = tank.elyte_obj.product_stoich_coeffs()
nu = products_nu - reactants_nu
reaction_rates = tank.elyte_obj.net_rates_of_progress
reaction_rates_capped = max_rate * np.tanh(reaction_rates / max_rate)
print(reaction_rates_capped)
low_rate_indx = np.abs(reaction_rates_capped) < 1e-3
reaction_rates_capped[low_rate_indx] = 0.0
print(reaction_rates_capped)
q_dot = np.dot(nu, reaction_rates_capped)
z = np.array([0,1,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0,0,0,0,0,0])
print(np.dot(q_dot,z))
print(reaction_rates)
print(q_dot)
#print(products_nu-reactants_nu)
#print(equations[0])
#print(name_elyte_species)
'''


SV_idx = SV_pointer(path, tank, params)
SV_0 = Solution_Vector_0(SV_idx, tank, solids)
# assume a constant volume (maybe add in change with mass dissolving later)
elyte_volume = 0.0005 # m^3 = 500 ml 

rtol = 1e-8                                                         # Relative tolerance
atol = 1e-11    # max 1e-8                                                   # Absolute tolerance
first_step = 1e-10                                                 # Size of the initial time step
# set up and run the solver
time_start = 3e-10                                                      # Initial time [s]
time_end = 5e-10                                              # Final time [s]
algvars = []                                                        # no algebraic variables
tspan = [time_start,time_end]
max_rate = 0.1e3
min_rate = 1e-10

solver = 1 # 0 for sundials, anything else for solve_ivp
# Adams
if solver == 0:
    const_idx = list(range(len(SV_0)))
    const_type = np.array([1]*len(SV_0))
    const_type[SV_idx.ptr['mass_S8']] = 2
    const_type[SV_idx.ptr['mass_Li2S']] = 2
    options =  {'userdata':(SV_idx, tank, solids, params, elyte_volume,max_rate, min_rate),
                'rtol':rtol,'atol':atol, 'first_step':first_step, 'method': 'BDF', 'constraints_idx': const_idx,'constraints_type': const_type}
    SV_dot_0  = np.zeros_like(SV_0)
    solver = sun.cvode.CVODE(residual, **options)
    solution = solver.solve(tspan, SV_0)
    sim_outputs = np.stack((*np.transpose(solution.y), solution.t))

else:
#SV_dot_0  = np.zeros_like(SV_0)
#BDF Radau LSODA
    solution = solve_ivp(residual_ivp,tspan,SV_0 ,method='Radau',
                args=(SV_idx, tank, solids, params, elyte_volume,max_rate, min_rate ), rtol = 1e-8,atol = 1e-10)

    concentrations = solution.y
    #concentrations = concentrations.T
    time = solution.t
    sim_outputs = np.stack((*concentrations, time))

create_plots(SV_idx, sim_outputs, tank, solids, params, elyte_volume,time_end)


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
