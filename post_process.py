# post_process.py
#
# This file saves the solution and creates plots
import matplotlib.pyplot as plt
import numpy as np 

def create_plots(SV_idx, sim_outputs, sep, params):
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

    plot_name_elyte_species = [species['name-plot'] for species in sep.inputs['transport']['diffusion-coefficients']]
    name_elyte_species = [species['name'] for species in sep.inputs['transport']['diffusion-coefficients']]
    n_elyte_nodes = sep.inputs['n_points']
    n_elyte_species = sep.elyte_obj.n_species

    # Anode Double layer potential
    plt.figure()
    plt.plot(time, phi_dl_an,'.',label="model")
    plt.hlines(y=2.791, xmin=0, xmax=time[-1], linewidth=0.5, color='k',label="hand calc eq")
    plt.title("Anode Double Layer Potential")
    plt.ylabel("Voltage [V]")
    plt.xlabel("Time [s]")
    plt.legend()

    # Cathode double layer potential
    '''
    plt.figure()
    plt.plot(time, phi_dl_ca,'.',label="model")
    plt.title("Cathode Double Layer Potential")
    plt.ylabel("Voltage [V]")
    plt.xlabel("Time [s]")
    '''

    # Anode thickness
    
    plt.figure()
    plt.plot(time, thickness_an,'.')
    plt.title("Anode Thickness")
    plt.ylabel("Thickness [m]")
    plt.xlabel("Time [s]")

    ## Elyte species concentrations
    # Individual plots for species listed to tbe plotted in the input file
    species_to_plot = [species['name'] for species in params.inputs['plot-species']]
    
    for x, plot_species in enumerate(species_to_plot):
        plt.figure()
        species_idx = SV_idx.elyte_species.index(plot_species)
        for i in range(sep.inputs['n_points']):
            plt.plot(time, np.transpose(C_k_elyte[species_idx+i*sep.elyte_obj.n_species,:]),'.',label= f"node {i}")
        plt.title(f"{plot_species} Concentration in the Electrolyte")
        plt.ylabel("Concentration [kmol/m^3]")
        plt.xlabel("Time [s]")
        plt.legend()

    # One plot for all of the elyte species concentrations (each get thier own subplot)
    num_species = len(name_elyte_species)
    a = int(np.sqrt(num_species))
    b = a + 1
    if a*b<num_species:
        a = a + 1
    fig, axes = plt.subplots(a, b, figsize=(8,8*a/b))
    axes = axes.flatten()
    for j in range(num_species):
        ax = axes[j]
        plot_species = name_elyte_species[j]
        species_idx = SV_idx.elyte_species.index(plot_species)
        for i in range(sep.inputs['n_points']):
            ax.plot(time, np.transpose(C_k_elyte[species_idx+i*sep.elyte_obj.n_species,:]),'.',label= f"node {i}")
        ax.set_title(plot_name_elyte_species[j])
    if a*b > num_species:
        ax = axes[-1]
        for i in range(sep.inputs['n_points']):
            ax.plot(time, np.transpose(C_k_elyte[species_idx+i*sep.elyte_obj.n_species,:]),'.',label= f"node {i}")
            ax.plot(time, np.transpose(C_k_elyte[species_idx+i*sep.elyte_obj.n_species,:]),'wo')
            ax.legend(loc='center') #, fontsize= 8)
    else:
        ax.legend(ncol=int(sep.inputs['n_points']/2), loc='best', fontsize = 6)
    plt.tight_layout()
    
    # Check charge neutrality
    plt.figure()
    for node_idx in range(sep.inputs['n_points']):
        net_charge = np.zeros_like(time)
        for spec_idx, species in enumerate(SV_idx.elyte_species[:num_species]):
            concentration_species = np.transpose(C_k_elyte[spec_idx+node_idx*sep.elyte_obj.n_species,:])
            charge_species = float(sep.elyte_obj[species].charges)
            total_charge_species = np.multiply(concentration_species,charge_species)
            net_charge = np.add(net_charge, total_charge_species)
        plt.plot(time, net_charge, '.-', label = f"node {node_idx}")
        plt.legend() 
        plt.title("Conservation of charge")
    
    # Potential in the electrolyte
    plt.figure()
    for i in range(sep.inputs['n_points']):
        plt.plot(time, np.transpose(phi_elyte[i,:]),'.',label= f"node {i}")
    #plt.plot(time, np.transpose(C_k_elyte[0,:]),'.')
    plt.title("Electrolyte Potential")
    plt.ylabel("Potential [V]")
    plt.xlabel("Time [s]")
    plt.legend()

    # Li+ vs TFSI- concentration graph, right now they are the only two charged species so I can use this
    #   as another way to judge charge neutrality
    plt.figure()
    plot_species_1 = 'Li+(e)'
    species_idx_1 = SV_idx.elyte_species.index(plot_species_1)
    plot_species_2 = 'TFSI-(e)'
    species_idx_2 = SV_idx.elyte_species.index(plot_species_2)
    for i in range(sep.inputs['n_points']):
        plt.plot(time, np.transpose(C_k_elyte[species_idx_2+i*sep.elyte_obj.n_species,:]),'.',label= f"node {i}")
        plt.plot(time, np.transpose(C_k_elyte[species_idx_1+i*sep.elyte_obj.n_species,:]),'x',label= f"node {i}", markersize=3)

        #plt.plot(time, np.transpose(C_k_elyte[species_idx_1+i*sep.elyte_obj.n_species,:] - C_k_elyte[species_idx_2+i*sep.elyte_obj.n_species,:]),'.',label= f"node {i}")
    plt.title("Li Concentration in the Electrolyte")
    plt.ylabel("Concentration [kmol/m^3]")
    plt.xlabel("Time [s]")
    plt.legend()
    
    plt.show()

def save_data(SV_idx, sim_outputs):
    d = 4