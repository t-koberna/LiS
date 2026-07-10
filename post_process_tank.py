import matplotlib.pyplot as plt
import numpy as np 
import cantera as ct

def create_plots(SV_idx, sim_outputs, tank, solids, params, elyte_volume,time_end):

    mass_S8, = sim_outputs[SV_idx.ptr['mass_S8']]
    mass_Li2S, = sim_outputs[SV_idx.ptr['mass_Li2S']]
    C_k_elyte = sim_outputs[SV_idx.ptr['C_k_elyte']]
    time = sim_outputs[-1]

    #species_to_plot = [species['name'] for species in params.inputs['plot-species']]
    plot_name_elyte_species = [species['name-plot'] for species in tank.inputs['transport']['diffusion-coefficients']]
    name_elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]

    fig1, [ax1, ax2] = plt.subplots(1,2)
    ax1.plot(time, mass_S8)
    ax1.set_title("")
    ax1.set_ylabel(r"Mass of $S_8$ [kg]")
    ax1.set_xlabel("Time [s]")

    ax2.plot(time, mass_Li2S)
    ax2.set_title("")
    ax2.set_ylabel(r"Mass of $Li_2S$ [kg]")
    ax2.set_xlabel("Time [s]")
    plt.tight_layout()

    fig2, [ax3, ax4] = plt.subplots(1,2)
    ax3.plot(time, mass_S8/solids.mv_S8/solids.rho_S8, label='solid')
    ax3.plot(time, C_k_elyte[SV_idx.elyte_species.index('S8(e)')]*elyte_volume, label='aqueous')
    ax3.plot(time, mass_S8/solids.mv_S8/solids.rho_S8+ C_k_elyte[SV_idx.elyte_species.index('S8(e)')]*elyte_volume, label='total')
    ax3.set_title("")
    ax3.set_ylabel(r"Moles of $S_8$ [kmol]")
    ax3.set_xlabel("Time [s]")
    ax3.legend()

    ax4.plot(time, mass_Li2S/solids.mv_Li2S/solids.rho_Li2S, label='solid')
    ax4.plot(time, C_k_elyte[SV_idx.elyte_species.index('Li2S(e)')]*elyte_volume, label='aqueous')
    ax4.plot(time, mass_Li2S/solids.mv_Li2S/solids.rho_Li2S+ C_k_elyte[SV_idx.elyte_species.index('Li2S(e)')]*elyte_volume, label='total')
    ax4.set_title("")
    ax4.set_ylabel(r"Moles of $Li_2S$ [kmol]")
    ax4.set_xlabel("Time [s]")
    ax4.legend()
    plt.tight_layout()

    # One plot for all of the elyte species concentrations (each get their own subplot)
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
        for i in range(tank.inputs['n_points']):
            y = np.transpose(C_k_elyte[species_idx+i*num_species,:])
            ax.plot(time, y ,'-',label= f"node {i}")
            ax.vlines(x=time_end,ymin=min(y), ymax = max(y), linewidth=0.5,color='blue', linestyle='--') # =10, ymax=50, c
        ax.set_title(plot_name_elyte_species[j])
    plt.tight_layout()
    
    '''
    if a*b > num_species:
        ax = axes[-1]
        for i in range(tank.inputs['n_points']):
            ax.plot(time, np.transpose(C_k_elyte[species_idx+i*num_species,:]),'.',label= f"node {i}")
            ax.plot(time, np.transpose(C_k_elyte[species_idx+i*num_species,:]),'wo')
            ax.legend(loc='center') #, fontsize= 8)
    '''
    c_k_end = [i[-1] for i in C_k_elyte]
    c_k_end = c_k_end[1:]
    x_k_simulation = c_k_end/sum(c_k_end)
    plt.figure()
    plt.bar(name_elyte_species[1:],x_k_simulation,color='red', label='sim',width = -0.4,align='edge')
    plt.legend()
    plt.tight_layout()

    plt.figure()
    z = np.array([0,1,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0,0,0,0,0,0])
    charge = np.zeros_like(time)
    for ind, el in enumerate(C_k_elyte):
        if z[ind] != 0:
            charge = np.copy(charge) + el*z[ind]
            plt.plot(time,el*z[ind],label=plot_name_elyte_species[ind])
    plt.plot(time,charge,label=r'Total')
    plt.xlabel('time [s]')
    plt.ylabel(r'charge [$\rm kmol_e/m^3$]')
    plt.legend()
    plt.tight_layout()

    plt.show()

def stack_results(solution1, solution2):
    time_1 = solution1.t
    time_2 = solution2.t + solution1.t[-1]
    time_total = np.concatenate((time_1,time_2))
    y_total = np.concatenate((np.transpose(solution1.y),np.transpose(solution2.y)),axis=1)
    
    
    return y_total, time_total

def stack_results2(time_1, time_2, y1, y2):
    time_2 = time_2 + time_1[-1]
    time_total = np.concatenate((time_1,time_2))
    y_total = np.concatenate((y1,y2),axis=1)
    sim_outputs = np.stack((*y_total, time_total))
    
    return y_total, time_total