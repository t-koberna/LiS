# post_process.py
#
# This file saves the solution and creates plots
import matplotlib.pyplot as plt
import numpy as np 

def create_plots(SV_idx, sim_outputs, sep_inputs):
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

    C_k_elyte_species = [species['name'] for species in sep_inputs['transport']['diffusion-coefficients']]

    plt.figure()
    plt.plot(time, phi_dl_an)
    plt.title("Anode Double Layer Potential")
    plt.ylabel("Voltage [V]")
    plt.xlabel("Time [s]")

    plt.figure()
    plt.plot(time, thickness_an)
    plt.title("Anode Thickness")
    plt.ylabel("Thickness [m]")
    plt.xlabel("Time [s]")

    plt.show()

def save_data(SV_idx, sim_outputs):
    d = 4