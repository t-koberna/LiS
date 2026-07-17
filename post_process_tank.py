import matplotlib.pyplot as plt
import numpy as np 
import cantera as ct

def create_plots(SV_idx, sim_outputs, tank, solids, params, total_volume,time_end):
    '''
    Create all of the plots from the solution 
    '''
    # set the variable values from the simulation solution 
    volume_S8, = sim_outputs[SV_idx.ptr['volume_S8']]
    volume_Li2S, = sim_outputs[SV_idx.ptr['volume_Li2S']]
    mass_S8, = sim_outputs[SV_idx.ptr['mass_S8']]
    mass_Li2S, = sim_outputs[SV_idx.ptr['mass_Li2S']]
    C_k_elyte = sim_outputs[SV_idx.ptr['C_k_elyte']]
    time = sim_outputs[-1]
    elyte_volume = total_volume*np.ones_like(time) - volume_S8 - volume_Li2S

    # create a list of the names of the electrolyte species
    plot_name_elyte_species = [species['name-plot'] for species in tank.inputs['transport']['diffusion-coefficients']]
    name_elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]

    # plot the moles of S_8 and Li_2S, both the solid and dissolved phases
    fig2, [ax3, ax4] = plt.subplots(1,2)
    plt.suptitle(r"The solid and aqueous phases of $S_8$ and $Li_2S$")
    moles_solid_S8 = mass_S8/solids.mv_S8/solids.rho_S8
    moles_disolved_S8 = np.multiply(C_k_elyte[SV_idx.elyte_species.index('S8(e)')],elyte_volume)
    ax3.plot(time, moles_solid_S8, label='solid')
    ax3.plot(time, moles_disolved_S8, label='aqueous')
    ax3.plot(time, moles_solid_S8 + moles_disolved_S8, label='total')
    #ax3.set_title(r"S_8")
    ax3.set_ylabel(r"Moles of $S_8$ [kmol]")
    ax3.set_xlabel("Time [s]")
    ax3.legend()

    moles_solid_Li2S = mass_Li2S/solids.mv_Li2S/solids.rho_Li2S
    moles_disolved_Li2S = np.multiply(C_k_elyte[SV_idx.elyte_species.index('Li2S(e)')],elyte_volume)
    ax4.plot(time, moles_solid_Li2S , label='solid')
    ax4.plot(time, C_k_elyte[SV_idx.elyte_species.index('Li2S(e)')]*elyte_volume, label='aqueous')
    ax4.plot(time, moles_solid_Li2S + moles_disolved_Li2S, label='total')
    #ax4.set_title("")
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
            #############################
            # If I split up the simulation into multiple runs and stich the data together blue lines separate each section
            #ax.vlines(x=time_end,ymin=min(y), ymax = max(y), linewidth=0.5,color='blue', linestyle='--') # =10, ymax=50, c
        ax.set_title(plot_name_elyte_species[j])
    plt.tight_layout()
    
    # Plot the species distribution at the end of the simulation as a bar plot
    c_k_end = [i[-1] for i in C_k_elyte]
    c_k_end = c_k_end[1:]
    x_k_simulation = c_k_end/sum(c_k_end)
    plt.figure()
    plt.bar(name_elyte_species[1:],x_k_simulation,color='red', label='sim',width = -0.4,align='edge')
    plt.legend()
    plt.tight_layout()

    # check conservation of charge (only use this if I include ionic species)
    '''
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    '''

    # plot the total moles of Li and S to check and see if they are being conserved
    fig4, [ax5, ax6] = plt.subplots(1,2)
    # number of atoms of Li and S in each electrolyte species 
    nLi = np.array([tank.elyte_obj.n_atoms(name, 'Li') for name in SV_idx.elyte_species])
    nS = np.array([tank.elyte_obj.n_atoms(name, 'S') for name in SV_idx.elyte_species])

    mole_Li_solid = 2*mass_Li2S/solids.mv_Li2S/solids.rho_Li2S
    mole_S_solid = 8*mass_S8/solids.mv_S8/solids.rho_S8 + 1*mass_Li2S/solids.mv_Li2S/solids.rho_Li2S
    mole_S_aq = np.zeros_like(time)
    mole_Li_aq = np.zeros_like(time)

    for ind, el in enumerate(C_k_elyte):
        mole_Li_aq = np.copy(mole_Li_aq) + el*nLi[ind]*elyte_volume
        mole_S_aq = np.copy(mole_S_aq) + el*nS[ind]*elyte_volume
    #ax5.plot(time,mole_Li_aq,label='Li aq')
    #ax5.plot(time,mole_Li_solid,label='Li solid')
    ax5.plot(time,mole_Li_solid+mole_Li_aq,label='total')
    #ax4.set_title("")
    ax5.set_ylabel(r'moles of Li [$\rm kmol$]')
    ax5.set_xlabel("Time [s]")
    #ax5.legend()
    #ax6.plot(time,mole_S_aq,label='S aq')
    #ax6.plot(time,mole_S_solid,label='S solid')
    ax6.plot(time,mole_S_solid+mole_S_aq,label='total')
    ax6.set_ylabel(r'moles of S [$\rm kmol$]')
    ax6.set_xlabel("Time [s]")
    #ax6.legend()
    plt.tight_layout()

    # the volumes of the solid S_8 and Li_2S
    #plt.figure()
    #plt.plot(time,volume_S8,label='S8' )
    #plt.plot(time,volume_Li2S,label='Li2S' )
    #plt.tight_layout()

    plt.show()

def stack_results(solution1, solution2):
    '''
    stack two solution together
    '''
    time_1 = solution1.t
    time_2 = solution2.t + solution1.t[-1]
    time_total = np.concatenate((time_1,time_2))
    y_total = np.concatenate((np.transpose(solution1.y),np.transpose(solution2.y)),axis=1)
    
    return y_total, time_total

def stack_results2(time_1, time_2, y1, y2):
    '''
    stack two solutions together
    '''
    time_2 = time_2 + time_1[-1]
    time_total = np.concatenate((time_1,time_2))
    y_total = np.concatenate((y1,y2),axis=1)
    sim_outputs = np.stack((*y_total, time_total))
    
    return y_total, time_total