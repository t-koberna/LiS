# post_process.py
#
# This file saves the solution and creates plots
import matplotlib.pyplot as plt
import numpy as np 
import cantera as ct

def create_plots(SV_idx, sim_outputs, sep, anode, cathode, params):
    #==================================================================================================
    #
    #   Read in the outputs from the simulation  
    #
    #==================================================================================================

    thickness_an, = sim_outputs[SV_idx.ptr['thickness_an']]
    phi_dl_an, = sim_outputs[SV_idx.ptr['phi_dl_an']]
    phi_elyte = sim_outputs[SV_idx.ptr['phi_elyte']]
    C_k_elyte = sim_outputs[SV_idx.ptr['C_k_elyte']]
    phi_dl_ca, = sim_outputs[SV_idx.ptr['phi_dl_ca']] 
    Li2S = sim_outputs[SV_idx.ptr['Li2S']]
    S8 = sim_outputs[SV_idx.ptr['S8']]
    bm_Li2S, = sim_outputs[SV_idx.ptr['bm_Li2S']]
    bm_S8, = sim_outputs[SV_idx.ptr['bm_S8']]
    time = sim_outputs[-1]


    #==================================================================================================
    #
    #   Set plotting varibles and run post processing calculations (in the future I will also set plot flags here)
    #
    #==================================================================================================

    phi_cell = phi_dl_an + (phi_elyte[-1,:] - phi_elyte[0,:]) + phi_dl_ca               # Total Cell potential, [V]

    plot_name_elyte_species = [species['name-plot'] for species in sep.inputs['transport']['diffusion-coefficients']]
    name_elyte_species = [species['name'] for species in sep.inputs['transport']['diffusion-coefficients']]
    n_elyte_nodes = sep.inputs['n_points']
    n_elyte_species = sep.elyte_obj.n_species

    # Standard Anode Double layer potential
    R = ct.gas_constant
    F = ct.faraday
    T = params.T
    n = 1
    G_Li_ion = anode.elyte_obj.standard_gibbs_RT[0]*R*T + R*T*np.log(anode.elyte_obj.X[0])
    G_Li = anode.bulk_obj['Li(b)'].gibbs_mole
    Delta_G_rxn = G_Li_ion - G_Li
    U_an = -Delta_G_rxn/(n*F)                                                           # The half cell equilibrium potential

    # Cathode double layer potential
    n = 16
    G_Li_ion = cathode.elyte_obj.standard_gibbs_RT[0]*R*T + R*T*np.log(cathode.elyte_obj.X[0])
    G_S8 = cathode.elyte_obj.standard_gibbs_RT[1]*R*T + R*T*np.log(cathode.elyte_obj.X[1])
    G_Li2S = cathode.elyte_obj.standard_gibbs_RT[3]*R*T + R*T*np.log(cathode.elyte_obj.X[3])
    Delta_G_rxn = 8*G_Li2S - 16*G_Li_ion - G_S8
    U_ca = -Delta_G_rxn/(n*F)


    #==================================================================================================
    #
    #   Create the plots
    #
    #==================================================================================================

    # Double layer potentials
    fig1, [ax1, ax2] = plt.subplots(1,2)
    # Anode
    ax1.hlines(y=U_an, xmin=0, xmax=time[-1], linewidth=0.5, color='k',label="hand calc eq")
    ax1.plot(time, phi_dl_an,'.',label="model")
    ax1.set_title("Anode Double Layer Potential")
    ax1.set_ylabel("Voltage [V]")
    ax1.set_xlabel("Time [s]")
    # Cathode
    ax2.hlines(y=U_ca, xmin=0, xmax=time[-1], linewidth=0.5, color='k',label="hand calc eq")
    ax2.plot(time, phi_dl_ca,'.',label="model")
    ax2.set_title("Cathode Double Layer Potential")
    ax2.set_ylabel("Voltage [V]")
    ax2.set_xlabel("Time [s]")
    ax2.legend()
    plt.tight_layout()

    # Anode thickness
    plt.figure()
    plt.plot(time, thickness_an,'.')
    plt.title("Anode Thickness")
    plt.ylabel("Thickness [m]")
    plt.xlabel("Time [s]")

    ## Elyte species concentrations
    # The list of species that have idividual plots
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
    fig2, axes = plt.subplots(a, b, figsize=(8,8*a/b))
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
        
    # Potential at each node in the electrolyte (relative to each other)
    plt.figure()
    for i in range(sep.inputs['n_points']):
        plt.plot(time, np.transpose(phi_elyte[i,:]),'.',label= f"node {i}")
    plt.title("Electrolyte Potential")
    plt.ylabel("Potential [V]")
    plt.xlabel("Time [s]")
    plt.legend()

    # Check charge neutrality at each node in the elyte
    fig3, [ax1, ax2] = plt.subplots(1,2)
    # plot the total negative charges from S82- and TSFI- against the total positive charges
    #   If the solution is charge nuetral they should overlap
    plot_species_1 = 'Li+(e)'
    species_idx_1 = SV_idx.elyte_species.index(plot_species_1)
    plot_species_2 = 'TFSI-(e)'
    species_idx_2 = SV_idx.elyte_species.index(plot_species_2)
    charge_species_2 = abs(float(sep.elyte_obj[plot_species_2].charges))
    plot_species_3 = 'S82-(e)'
    species_idx_3 = SV_idx.elyte_species.index(plot_species_3)
    charge_species_3 = abs(float(sep.elyte_obj[plot_species_3].charges))
    for i in range(sep.inputs['n_points']):
        total_charge_neg = np.multiply(charge_species_2,np.transpose(C_k_elyte[species_idx_2+i*sep.elyte_obj.n_species,:])) + np.multiply(charge_species_3,np.transpose(C_k_elyte[species_idx_3+i*sep.elyte_obj.n_species,:]))
        total_charge_pos = np.transpose(C_k_elyte[species_idx_1+i*sep.elyte_obj.n_species,:])
        ax1.plot(time, total_charge_neg,'.',label= f"node {i} pos")
        ax1.plot(time, total_charge_pos,'x',label= f"node {i} neg", markersize=3)
    ax1.set_ylabel("Concentration [kmol/m^3]")
    ax1.set_xlabel("Time [s]")
    ax1.legend()
    
    # plots the difference between the points above
    for node_idx in range(sep.inputs['n_points']):
        net_charge = np.zeros_like(time)
        for spec_idx, species in enumerate(SV_idx.elyte_species[:num_species]):
            concentration_species = np.transpose(C_k_elyte[spec_idx+node_idx*sep.elyte_obj.n_species,:])
            charge_species = float(sep.elyte_obj[species].charges)
            total_charge_species = np.multiply(concentration_species,charge_species)
            net_charge = np.add(net_charge, total_charge_species)
        ax2.plot(time, net_charge, '.-', label = f"node {node_idx}")
        ax2.legend() 
    fig3.suptitle("Conservation of charge")
    plt.tight_layout()
    
    # Total cell potential
    plt.figure()
    plt.plot(time, phi_cell)
    plt.title("Cell potential")

    plt.show()

def save_data(SV_idx, sim_outputs):
    d = 4