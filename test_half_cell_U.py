import cantera as ct
import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup import Anode, Seperator, Parameters, Cathode
import matplotlib.pyplot as plt


path = Path("Li_Sulfur.yaml")
yaml = YAML(typ='safe')
inputs = yaml.load(path)

i_ext = 0.0
SV_0 = [0.1]

params = Parameters(inputs)
sep = Seperator(path, inputs, params)
cathode = Cathode(path, inputs, sep, params)
anode = Anode(path, inputs, sep, params)

electrode = cathode

def residual(t,SV,SV_dot,resid,user_data):
    ed, i_ext  = user_data

    
    #ca.elyte_obj.X = SV[SV_idx.ptr['C_k_elyte'][n_elyte_species*(n_elyte_nodes-1):]] # the concetrations in the final elyte node
    if isinstance(electrode, Cathode):
        ed.elyte_obj.electric_potential = 0
        ed.host_obj.electric_potential = SV[0]
        sdot_electron_ca = ed.surf_obj.get_net_production_rates(ed.host_obj) # rate electrons, positive if produced
        i_far = ct.faraday*sdot_electron_ca # [C/kmol]*[kmol/m^2-s] = [A/m^2]

    else:
        ed.elyte_obj.electric_potential = SV[0]
        sdot_electron_ca = ed.surf_obj.get_net_production_rates(ed.conductor_obj)
        i_far = -ct.faraday*sdot_electron_ca # [C/kmol]*[kmol/m^2-s] = [A/m^2]
    i_dl = i_ext - i_far # [A/m^2]
    c_dl = ed.inputs['C_dl'] # [F/m^2]

    resid[0] = SV_dot[0]  - i_dl/c_dl

time_start = 0 # Initial time [s]
time_end = params.inputs['simulations']['time_max'] #Final time [s]
tspan = [time_start,time_end]

options =  {'userdata':(electrode, i_ext), 
            'rtol':1e-4,'atol':1e-12, 
            'algebraic_idx':[], 'first_step':1e-15,}

solver = sun.ida.IDA(residual, **options)
SV_dot_0  = np.zeros_like(SV_0)
solution = solver.solve(tspan, SV_0, SV_dot_0)

time = solution.t

phi_dl = solution.y

R = ct.gas_constant
F = ct.faraday
T = params.T

if isinstance(electrode, Cathode):
    n = 16
    G_Li_ion = cathode.elyte_obj.standard_gibbs_RT[0]*R*T + R*T*np.log(cathode.elyte_obj.X[0])
    G_S8 = cathode.elyte_obj.standard_gibbs_RT[1]*R*T + R*T*np.log(cathode.elyte_obj.X[1])
    G_Li2S = cathode.elyte_obj.standard_gibbs_RT[3]*R*T + R*T*np.log(cathode.elyte_obj.X[3])
    Delta_G_rxn = 8*G_Li2S - 16*G_Li_ion - G_S8
    U = -Delta_G_rxn/(n*F)/2
else:
    n = 1
    G_Li_ion = anode.elyte_obj.standard_gibbs_RT[0]*R*T + R*T*np.log(anode.elyte_obj.X[0])
    G_Li = anode.bulk_obj['Li(b)'].gibbs_mole
    Delta_G_rxn = G_Li_ion - G_Li
    U = -Delta_G_rxn/(n*F)


plt.figure()
plt.hlines(y=U, xmin=0, xmax=time[-1], linewidth=0.5, color='k',label="hand calc eq")
plt.plot(time, phi_dl,'.',label="model")
plt.title("Cathode Double Layer Potential")
plt.ylabel("Voltage [V]")
plt.xlabel("Time [s]")
plt.show()