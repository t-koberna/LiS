import cantera as ct
import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup import Anode, Separator, Parameters, Cathode
import matplotlib.pyplot as plt


path = Path("Li_Sulfur.yaml")
yaml = YAML(typ='safe')
inputs = yaml.load(path)

i_ext = 0.0
SV_0 = [0.1]

params = Parameters(inputs)
sep = Separator(path, inputs, params)
cathode = Cathode(path, inputs, sep, params)
anode = Anode(path, inputs, sep, params)

electrode = cathode

def residual(t,SV,SV_dot,resid,user_data):
    ed, i_ext  = user_data

    
    #ca.elyte_obj.X = SV[SV_idx.ptr['C_k_elyte'][n_elyte_species*(n_elyte_nodes-1):]] # the concentrations in the final elyte node
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

def residual_Li_Li(t,SV,SV_dot,resid,user_data):
    '''
    The residual used for a symmetric cell, in the main model the cathode is set up differently than the anode so there
    needed to be slight adaptations
    '''
    SV_idx, i_ext, an, sep, ca, params, algebraic  = user_data

    n_elyte_nodes = sep.inputs['n_points']
    n_elyte_species = sep.elyte_obj.n_species

    #### Set the state of the Cantera objects based on SV
    # the potential of the eleyte near the electrode is the same as the potential of
    #  the double layer plus the first node potential of the separator (The separator
    #  only has potentials relative to itself)
    an.elyte_obj.electric_potential = SV[SV_idx.ptr['phi_dl_an']] + SV[SV_idx.ptr['phi_elyte'][0]]

    # Concentrations in the first elyte node
    an.elyte_obj.X = SV[SV_idx.ptr['C_k_elyte'][0:n_elyte_species]]

    # The potential at the cathode is the last node in the elyte relative to the
    #  del_phi_dl_ca for now I am using the anode double layer since I do not have the
    #  potentials in the separator working
    ca.elyte_obj.electric_potential = SV[SV_idx.ptr['phi_elyte'][-1]]
    ca.conductor_obj.electric_potential = (SV[SV_idx.ptr['phi_elyte'][-1]]
                                           + SV[SV_idx.ptr['phi_dl_ca']])
    # Concentrations in the final elyte node:
    ca.elyte_obj.X = SV[SV_idx.ptr['C_k_elyte'][n_elyte_species*(n_elyte_nodes-1):]]

    """ Anode: """
    #  Electron production rate (kmol/m2/s)
    sdot_electron_an = an.surf_obj.get_net_production_rates(an.conductor_obj) #
    #  Li metal production (kmol/m2/s)
    sdot_Li_an = an.surf_obj.get_net_production_rates(an.bulk_obj)

    # Faradic current density (A/m2) is positive when electrons are consumed (positive
    #  charge transferred to the electrode):
    i_far_an = -ct.faraday*sdot_electron_an
    # Double layer curren (A/m2):
    i_dl_an = i_ext - i_far_an

    # Read out the double layer capacitance (F/m2):
    c_dl_an = an.inputs['C_dl']

    # Derivative of the double layer potential difference (V/s):
    resid[SV_idx.ptr['phi_dl_an']] = SV_dot[SV_idx.ptr['phi_dl_an']] - i_dl_an/c_dl_an

    # Derivative of the anode thickness (m/s):
    resid[SV_idx.ptr['thickness_an']] = (SV_dot[SV_idx.ptr['thickness_an']]
                                         - sdot_Li_an*an.bulk_obj.partial_molar_volumes)

    """ Cathode """
    #  Electron production rate (kmol/m2/s)
    sdot_electron_ca = ca.surf_obj.get_net_production_rates(ca.conductor_obj)

    # Faradic current density (A/m2) is positive when electrons are produced (positive
    #  charge transferred to the electrolyte):
    i_far_ca = ct.faraday*sdot_electron_ca

    # Double layer current (A/m2)
    i_dl_ca = i_ext - i_far_ca

    # Read out the double layer capacitance (F/m2):
    c_dl_ca = ca.inputs['C_dl']

    # Derivative of the double layer potential difference (V/s):
    resid[SV_idx.ptr['phi_dl_ca']] = SV_dot[SV_idx.ptr['phi_dl_ca']]  - i_dl_ca/c_dl_ca

    """Separator"""
    # This function calculates the derivative of the separator species concentrations
    #  (kmol/m3/s) and the ionic current (A/m2) at every node boundary (n+1 currents
    #  for n nodes)
    dC_k_elyte_dt, i_io = elyte_rates(SV_idx, an, ca, sep, params, i_dl_an, i_dl_ca, SV)
    # Define electrolyte species residual
    resid[SV_idx.ptr['C_k_elyte']] = SV_dot[SV_idx.ptr['C_k_elyte']]  - dC_k_elyte_dt

    if algebraic == True:
        # Charge conservation in the electrolyte arises from enforcing the divergence
        #  of the ionic current equals zero:
        resid[SV_idx.ptr['phi_elyte'][1:]] = i_io[1:] - i_io[:-1]

        # At the first electrolyte node, the ionic current equals the external current:
        resid[SV_idx.ptr['phi_elyte'][0]] = i_io[0] - i_ext
    else:
        # Calculate the differential of the local potential as dPhi/dt= Sigma z_k*C_k.
        #   This approach enforces charge neutrality, in a transient manner (if the node
        #   accumulates positive charge, this implies the 2nd derivative of the
        #   potential, w/r/t space is positive.  Increasing the potential will reduce
        #   the 2nd derivative).

        # A large multiplier moves the system more quickly toward charge neutrality:
        multiplier = 1e4

        # I reshape the dC_k_elyte_dt so the dot product with the species charges
        #   yields the sum of the change in charge of the ions in each node
        dC_k_elyte_dt_r = np.reshape(dC_k_elyte_dt, (n_elyte_nodes,n_elyte_species))
        resid[SV_idx.ptr['phi_elyte']] = (SV_dot[SV_idx.ptr['phi_elyte']]
                        - multiplier*np.dot(dC_k_elyte_dt_r,   sep.elyte_obj.charges))

    # If my solution is failing at t=0, I use this so I can trouble shoot the rest
    #resid[SV_idx.ptr['phi_elyte']] = SV_dot[SV_idx.ptr['phi_elyte']]

    # Not addressed yet
    resid[SV_idx.ptr['Li2S']] = SV_dot[SV_idx.ptr['Li2S']]
    resid[SV_idx.ptr['S8']] = SV_dot[SV_idx.ptr['S8']]
    resid[SV_idx.ptr['bm_Li2S']] = SV_dot[SV_idx.ptr['bm_Li2S']]
    resid[SV_idx.ptr['bm_S8']] = SV_dot[SV_idx.ptr['bm_S8']]

    #print(f"anode elec {sdot_electron_an}, cathode elec {sdot_electron_ca}")
    #print(f"anode i_far {i_far_an}, cathode i_far {i_far_ca}")
    #print(f"anode i_dl {i_dl_an}, cathode i_dl {i_dl_ca}")

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