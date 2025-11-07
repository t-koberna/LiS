# elyte_concentration.py
#
# Determine what species will exist in the eltye at equilibrium

import cantera as ct
import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup import Separator, Parameters
import matplotlib.pyplot as plt

# create an interface and replace the equilibrate function with one for a multiphase object

# Read in the yaml input file
path = Path("Li_Sulfur_tank.yaml")
yaml = YAML(typ='safe')
inputs = yaml.load(path)

params = Parameters(inputs)
elyte = Separator(path, inputs, params)

#production rates are in kmol/m^3-s
#print(elyte.elyte_obj.kinetics_species_names)
#print(elyte.elyte_obj.net_production_rates)
#print(elyte.elyte_obj.concentrations*0)
#print(elyte.elyte_obj.concentrations.tolist())
#print(elyte.elyte_obj.concentrations)
X = elyte.elyte_obj.X
#print(X)
C_k_0 = np.array([1.024e-1, 0.512e-1, 0.512e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#C_k_0 = np.array([2.04888853e-06, 1.39567034e-06, 2.45544426e-06, 1.93407535e-06, 1.00000000e-06, 3.31400000e-06, 1.00000000e-06, 2.04600000e-06, 1.69530332e-07, 7.00893934e-06, 1.07184344e-11])
#print(sum(elyte.elyte_obj.concentrations))
#print(C_k/elyte.elyte_obj.concentrations)
#factor = (C_k_0/elyte.elyte_obj.concentrations)[0]
#print(C_k_0/factor)
molar_volumes = elyte.elyte_obj.partial_molar_volumes
#print(molar_volumes)
#print(elyte.elyte_obj.concentrations*factor)

elyte.elyte_obj.equilibrate('TP', solver='gibbs', rtol = 1e-10)
X = elyte.elyte_obj.X
C_k = elyte.elyte_obj.concentrations

#random = np.random.randint(1, 1001, size=len(C_k))

#print(C_k)
#SV_0 =  C_k_0/factor #C_k*random
#print(SV_0)
#ghjk
SV_0 = C_k_0
#SV_0 = [2.16202597e-04, 6.93498021e-02, 1.08101298e-04, 7.58731999e-03, 7.58731999e-03, 7.58731999e-03, 7.58731999e-03, 7.58731999e-03, 9.60657574e-03, 3.54872300e-03, 3.84256869e-11]

time_start = 0 # Initial time [s]
time_end = params.inputs['simulations']['time_max'] #Final time [s]
time_end = 100
tspan = [time_start,time_end]
algvars = []

#num_roots = len(SV_0) # number of termination checks
#def terminate_check(t,SV,SV_dot,return_val,user_data):
#    return_val =  SV

options =  {'userdata':(elyte, params), 
            'rtol':1e-6,'atol':1e-8, 'algebraic_idx':algvars, 'first_step':1e-3 } 
                #,'eventsfn':terminate_check,'num_events':num_roots}

def residual(t,SV,SV_dot,resid,user_data):
    print(t)
    elyte, params  = user_data

    elyte.elyte_obj.X = SV[:]
    #indx = 3
    
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

#elyte.elyte_obj.equilibrate('TP', solver='gibbs', rtol = 1e-10)
#X = elyte.elyte_obj.X
#C_k = elyte.elyte_obj.concentrations
#print(sum(C_k*factor))
#print(sum(C_k_0))

#print(elyte.elyte_obj.X)
#print(np.sum(concentrations))
'''
plt.figure()
plt.plot([0]*len(elyte_species_names), X,'.-')
plt.title("Elyte Concentration")
plt.ylabel("Concentration [kmol/m^3]")
plt.xlabel("Time [s]")
plt.legend(elyte_species_names,loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
'''
c_k_simulation = concentrations[-1:][0]
print(c_k_simulation)
x_k_simulation = c_k_simulation/sum(c_k_simulation)
plt.figure()
plt.bar(elyte_species_names,x_k_simulation,color='red', label='sim',width = -0.4,align='edge')
plt.bar(elyte_species_names,X,color='k', label='equal',width = 0.4,align='edge')
plt.legend()

#print(c_k_simulation[0]/c_k_simulation[2])

plt.show()


