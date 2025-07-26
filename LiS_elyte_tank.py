# elyte_concentration.py
#
# Determine what species will exisit in the eltye at equalibrium

import cantera as ct
import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup import Seperator, Parameters
import matplotlib.pyplot as plt


# Read in the yaml input file
path = Path("Li_Sulfur_tank.yaml")
yaml = YAML(typ='safe')
inputs = yaml.load(path)

params = Parameters(inputs)
elyte = Seperator(path, inputs, params)

#production rates are in kmol/m^3-s
#print(elyte.elyte_obj.kinetics_species_names)
#print(elyte.elyte_obj.net_production_rates)
#print(elyte.elyte_obj.concentrations*0)
#print(elyte.elyte_obj.concentrations.tolist())
print(elyte.elyte_obj.concentrations)
X = elyte.elyte_obj.X
print(X)
C_k = np.array([1.024e-6, 1.943e-6, 1.943e-6, 1.821e-6, 1.0e-6, 3.314e-6, 1.0e-6, 2.046e-6, 1.0e-6, 5.348e-6, 1.456e-6])
print(C_k/sum(C_k))
print(C_k/elyte.elyte_obj.concentrations)
molar_volumes = elyte.elyte_obj.partial_molar_volumes
print(molar_volumes/X)

SV_0 = elyte.elyte_obj.concentrations

time_start = 0 # Initial time [s]
time_end = params.inputs['simulations']['time_max'] #Final time [s]
tspan = [time_start,time_end]
algvars = []

#num_roots = len(SV_0) # number of termination checks
#def terminate_check(t,SV,SV_dot,return_val,user_data):
#    return_val =  SV

options =  {'userdata':(elyte, params), 
            'rtol':1e-5,'atol':1e-12, 'algebraic_idx':algvars, 'first_step':1e-10 } 
                #,'eventsfn':terminate_check,'num_events':num_roots}

def residual(t,SV,SV_dot,resid,user_data):
    elyte, params  = user_data

    elyte.elyte_obj.X = SV[:]
    indx = 3
    
    resid[:] = SV_dot[:] - elyte.elyte_obj.net_production_rates
    #resid[indx] = SV_dot[indx] - elyte.elyte_obj.net_production_rates[indx]
    #print(resid)
    #print(SV_dot)

solver = sun.ida.IDA(residual, **options)
SV_dot_0  = np.zeros_like(SV_0)
solution = solver.solve(tspan, SV_0, SV_dot_0)
concentrations = solution.y
time = solution.t

elyte_species_names = [species['name-plot'] for species in elyte.inputs['transport']['diffusion-coefficients']]

plt.figure()
plt.plot(time, concentrations,'.-')
plt.title("Elyte Concentration")
plt.ylabel("Concentration [kmol/m^3]")
plt.xlabel("Time [s]")
plt.legend(elyte_species_names,loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

elyte.elyte_obj.equilibrate('TP', solver='gibbs', rtol = 1e-10)
X = elyte.elyte_obj.X
#print(elyte.elyte_obj.X)
#print(np.sum(concentrations))

plt.figure()
plt.plot([0]*len(elyte_species_names), X,'.-')
plt.title("Elyte Concentration")
plt.ylabel("Concentration [kmol/m^3]")
plt.xlabel("Time [s]")
plt.legend(elyte_species_names,loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

#print(elyte.elyte_obj.partial_molar_volumes)
molar_volumes = elyte.elyte_obj.partial_molar_volumes

print(elyte.elyte_obj.concentrations)
print(molar_volumes)
print(X)
print(X/molar_volumes)
hjk
plt.show()


