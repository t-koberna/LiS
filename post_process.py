# post_process.py
#
# This file saves the solution and creates plots
import matplotlib.pyplot as plt
import numpy as np 

def create_plots(SV_idx, sim_outputs, sep):
    thickness_an, = sim_outputs[SV_idx.ptr['thickness_an']]
    phi_dl_an, = sim_outputs[SV_idx.ptr['phi_dl_an']]
    phi_elyte = sim_outputs[SV_idx.ptr['phi_elyte']]
    C_k_elyte = sim_outputs[SV_idx.ptr['C_k_elyte']]
    phi_dl_ca, = sim_outputs[SV_idx.ptr['phi_dl_ca']] 
    phi_ca, = sim_outputs[SV_idx.ptr['phi_ca']]
    Li2S = sim_outputs[SV_idx.ptr['Li2S']]
    S8 = sim_outputs[SV_idx.ptr['S8']]
    bm_Li2S, = sim_outputs[SV_idx.ptr['bm_Li2S']]
    bm_S8, = sim_outputs[SV_idx.ptr['bm_S8']]
    time = sim_outputs[-1]

    C_k_elyte_species = [species['name-plot'] for species in sep.inputs['transport']['diffusion-coefficients']]

    plt.figure()
    plt.plot(time, phi_dl_an,'.',label="model")
    plt.hlines(y=2.791, xmin=0, xmax=time[-1], linewidth=0.5, color='k',label="hand calc eq")
    plt.title("Anode Double Layer Potential")
    plt.ylabel("Voltage [V]")
    plt.xlabel("Time [s]")
    plt.legend()

    plt.figure()
    plt.plot(time, phi_dl_ca,'.',label="model")
    plt.title("Cathode Double Layer Potential")
    plt.ylabel("Voltage [V]")
    plt.xlabel("Time [s]")

    plt.figure()
    plt.plot(time, thickness_an,'.')
    plt.title("Anode Thickness")
    plt.ylabel("Thickness [m]")
    plt.xlabel("Time [s]")

    plt.figure()
    plot_species = 'Li+(e)'
    species_idx = SV_idx.elyte_species.index(plot_species)
    for i in range(sep.inputs['n_points']):
        plt.plot(time, np.transpose(C_k_elyte[species_idx+i*sep.elyte_obj.n_species,:]),'.',label= f"node {i}")
    plt.title("Li Concentration in the Electrolyte")
    plt.ylabel("Concentration []")
    plt.xlabel("Time [s]")
    plt.legend()

    
    plt.figure()
    for i in range(sep.inputs['n_points']):
        plt.plot(time, np.transpose(phi_elyte[i,:]),'.',label= f"node {i}")
    #plt.plot(time, np.transpose(C_k_elyte[0,:]),'.')
    plt.title("Electrolyte Potential")
    plt.ylabel("Potential [V]")
    plt.xlabel("Time [s]")
    plt.legend()
    
    plt.show()

def save_data(SV_idx, sim_outputs):
    d = 4