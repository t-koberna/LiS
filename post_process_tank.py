import matplotlib.pyplot as plt
import numpy as np 
import cantera as ct
import math
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import copy

def create_plots(SV_idx, sim_outputs, tank, solids, params, total_volume,time_end,rxn_rates, pre_exponential_factors,q_dot,S_8_disolve_rate,Li2S_disolve_rate, plot_flags):
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

    c_k_end = [i[-1] for i in C_k_elyte]
    c_k_end = c_k_end[0:]
    x_k_simulation = c_k_end/sum(c_k_end)
    tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, x_k_simulation
    solids.elyte_obj.TPX = solids.elyte_obj.T, solids.elyte_obj.P, x_k_simulation
    #print(tank.elyte_obj.forward_rate_constants)
    #print(tank.elyte_obj.reverse_rate_constants)
    #print(tank.elyte_obj.net_rates_of_progress)

    # create a list of the names of the electrolyte species
    plot_name_elyte_species = [species['name-plot'] for species in tank.inputs['transport']['diffusion-coefficients']]
    name_elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]

    if plot_flags[0] == 1:
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

    if plot_flags[1] == 1:
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
                ax.vlines(x=time_end,ymin=min(y), ymax = max(y), linewidth=0.5,color='blue', linestyle='--') # =10, ymax=50, c
            ax.set_title(plot_name_elyte_species[j])
        plt.tight_layout()

# Plot the species distribution at the end of the simulation as a bar plot

    c_k_end = [i[-1] for i in C_k_elyte]
    c_k_end = c_k_end[1:]
    x_k_simulation = c_k_end/sum(c_k_end)
    if plot_flags[2] == 1:
        elyte = tank.elyte_obj
        elyte.equilibrate('TP', solver='gibbs', rtol = 1e-10)
        C_k_eq = elyte.concentrations
        C_k_eq = C_k_eq[1:]
        X_k_eq = C_k_eq/sum(C_k_eq)
        #X_eq = elyte.elyte_obj.X
        plt.figure()
        plt.bar(name_elyte_species[1:],x_k_simulation,color='red', label='sim',width = -0.4,align='edge')
        plt.bar(name_elyte_species[1:],X_k_eq,color='k', label='eq',width = 0.4,align='edge')
        plt.legend()
        plt.tight_layout()
        plt.figure()

        plt.bar(name_elyte_species[-8:],x_k_simulation[-8:],color='red', label='sim',width = -0.4,align='edge')
        plt.bar(name_elyte_species[-8:],X_k_eq[-8:],color='k', label='eq',width = 0.4,align='edge')
        plt.legend()
        plt.tight_layout()
        c_k_end = [i[-1] for i in C_k_elyte]
        x_k_simulation = c_k_end/sum(c_k_end)
        tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, x_k_simulation



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
    if plot_flags[3] == 1:
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

    if plot_flags[4] == 1:
        # the volumes of the solid S_8 and Li_2S
        fig5, [ax7, ax8, ax9] = plt.subplots(1,3)
        ax7.plot(time,volume_S8,label='S8' )
        ax7.set_ylabel(r'Volume solid $\rm S_8$ [$\rm m^3$]')
        ax7.set_xlabel("Time [s]")
        ax8.plot(time,volume_Li2S,label='Li2S' )
        ax8.set_ylabel(r'Volume solid $\rm Li_2S$ [$m^3$]')
        ax8.set_xlabel("Time [s]")
        ax9.plot(time,total_volume*np.ones_like(time), label= 'total')
        ax9.plot(time,elyte_volume, label= 'elyte')
        ax9.set_ylabel(r'Volume [$\rm m^3$]')
        ax9.set_xlabel("Time [s]")
        ax9.legend()
        plt.tight_layout()

    # Calculate chemical potential
    surf = tank.elyte_obj
    R = surf.reactant_stoich_coeffs()
    P = surf.product_stoich_coeffs()
    nu = P - R
    nu = nu.T
    g_f = tank.elyte_obj.standard_gibbs_RT * ct.gas_constant * tank.elyte_obj.T
    g_f = g_f*1e-6 # convert J/kmol to kJ/mol
    g_f_rxn =np.dot(nu,g_f)
    
    R = ct.gas_constant/1e3 # J/mol-K
    T = tank.elyte_obj.T
    conc_adjustment = np.zeros((np.size(C_k_elyte,1),np.size(nu,0)))
    g_f_rxn_adjusted = np.zeros((np.size(C_k_elyte,1),np.size(nu,0)))

    g_f_solid_Li2S = (solids.solid_Li2S_obj.standard_gibbs_RT * ct.gas_constant * tank.elyte_obj.T)/1e6
    g_f_solid_S8 = (solids.solid_S8_obj.standard_gibbs_RT * ct.gas_constant * tank.elyte_obj.T)/1e6
    g_f_S8 = g_f[SV_idx.elyte_species.index('S8(e)')] - g_f_solid_S8[-1]
    g_f_Li2S = g_f[SV_idx.elyte_species.index('Li2S(e)')] - g_f_solid_Li2S[-1]
    g_f_solid_S8_adjusted = np.zeros((np.size(C_k_elyte,1)))
    g_f_solid_Li2S_adjusted = np.zeros((np.size(C_k_elyte,1)))
    
    

        #if t == 4:
            #print(C_k_elyte.T[t,:])
            #print(g_f_rxn_adjusted[t,:])
            #njko


    num_rxn = tank.elyte_obj.n_reactions
    a = int(np.round(num_rxn/6,0))
    b = 6
    if a*b<num_rxn:
        a = a + 1

    num_rxn = tank.elyte_obj.n_reactions
    num_figs = int(np.ceil(num_rxn/20))
    #num_figs =0 ##############################################################################$RFGHJHHHGFDRTYUJBGFRT^U&JBVFRTYUJHG
 
        
    
    if plot_flags[5] == 1:
        for t in range(np.size(C_k_elyte,1)):
                for i, el in enumerate(nu):
                    C_k_timestep = C_k_elyte.T[t,:]
                    X_k_timestep = C_k_timestep/np.sum(C_k_timestep)
                    conc_adjustment[t,i] = R*T/(1000)*np.log(np.prod(np.power(X_k_timestep,el)))
                    min_C = 1e-60
                    C_k_timestep_capped = C_k_timestep.copy()
                    while math.isnan(conc_adjustment[t,i]) or math.isinf(conc_adjustment[t,i]) or conc_adjustment[t,i]==0:
                        low_C_indx = C_k_timestep_capped < min_C
                        C_k_timestep_capped[low_C_indx] = min_C
                        X_k_timestep_capped = C_k_timestep_capped/np.sum(C_k_timestep_capped)
                        conc_adjustment[t,i] = R*T/(1000)*np.log(np.prod(np.power(X_k_timestep_capped,el)))
                        min_C = min_C*2
        
                g_f_rxn_adjusted[t,:]= conc_adjustment[t,:]+g_f_rxn
        
                X_k_timestep_S8 = X_k_timestep[SV_idx.elyte_species.index('S8(e)')]
                X_k_timestep_Li2S = X_k_timestep[SV_idx.elyte_species.index('Li2S(e)')]
                g_f_solid_S8_adjusted[t] = R*T/(1000)*np.log(X_k_timestep_S8) + g_f_S8
                g_f_solid_Li2S_adjusted[t] = R*T/(1000)*np.log(X_k_timestep_Li2S) + g_f_Li2S
        fig , (ax1,ax2)= plt.subplots(2, 1)#,figsize=(20,8*a/b))
        ax1.plot(time,g_f_solid_S8_adjusted)
        ax1.set_title(r"Dissolving $S_8$")
        ax2.plot(time,g_f_solid_Li2S_adjusted)
        ax2.set_title(r"Dissolving $Li_2S$")
        for i in range(num_figs):
            fig, axes = plt.subplots(4,5,num=50+i)
            axes = axes.flatten()
            lower = i*20
            upper = 20*i+20
            for j in range(lower,upper):
                if j<num_rxn:
                    rxn = tank.elyte_obj.reaction(j) 
                    ax = axes[j-i*20]
                    ax.plot(time,g_f_rxn[j]+conc_adjustment[:,j],color='k',linewidth = 0.75)#,linestyle='--')
                    ax.axhline(y=0, color='b', linestyle='--', linewidth=.5)
                    ax.xaxis.set_ticklabels([])     
                    ax.set_xlabel('')
                    ax.tick_params(axis='y', labelsize=7)
                    #ax.set_ylim([-.1,.1]) ################EDFYGHUJIKOJHUGRFYGHUJIUHGYRDFGYHUJIUHYGRD
                    ax.set_title(rxn.equation,fontsize=6)
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()
            plt.tight_layout(pad=0.1, h_pad=0.1)
            plt.subplots_adjust(wspace=0.2) 

    ##########################################################################
    if plot_flags[6] == 1:
        fig, (ax1,ax2) = plt.subplots(2, 1)
        A_surf_S8 = 2 * np.pi * ((3 * mass_S8) / (2 * np.pi * solids.rho_S8))**(2/3)
        A_surf_Li2S = 2 * np.pi * ((3 * mass_Li2S) / (2 * np.pi * solids.rho_Li2S))**(2/3)
        ax1.plot(time, S_8_disolve_rate*A_surf_S8/elyte_volume, label = 'disolve' )
        ax1.plot(time, q_dot[:,SV_idx.elyte_species.index('S8(e)')],label = 'react')
        ax1.set_yscale('symlog')
        ax1.set_xscale('symlog')
        #ax1.yaxis.set_major_locator(FixedLocator(log_ticks(ax1)))
        #ax1.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax1.set_title(r'$S_8$')
        ax2.plot(time, Li2S_disolve_rate*A_surf_Li2S/elyte_volume, label = 'disolve' )
        ax2.plot(time, q_dot[:,SV_idx.elyte_species.index('Li2S(e)')],label = 'react')
        ax2.set_yscale('symlog')
        ax2.set_xscale('symlog')
        #ax2.yaxis.set_major_locator(FixedLocator(log_ticks(ax2)))
        #ax2.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax2.set_title(r'$Li_2S$')
        plt.legend()
        plt.tight_layout()


    num_rxn = tank.elyte_obj.n_reactions
    num_figs = int(np.ceil(num_rxn/8))
    #num_figs =0 ##############################################################################$RFGHJHHHGFDRTYUJBGFRT^U&JBVFRTYUJHG
    if plot_flags[7] == 1:    
        for i in range(num_figs):
            fig, axes = plt.subplots(4,4,num=20+i)
            axes = axes.flatten()
            lower = i*8
            upper = 8*i+8

            for j in range(lower,upper):
                if j<num_rxn:
                    rxn = tank.elyte_obj.reaction(j) 
                    ax_rate = axes[(j-i*8)*2]
                    ax_A = axes[(j-i*8)*2+1]
                    ax_rate.plot(time,rxn_rates[:,j],color='k',linewidth = 0.5)#,linestyle='--')
                    ax_rate.set_title(rxn.equation,fontsize=6)
                    ax_rate.set_yscale('symlog')
                    ax_rate.yaxis.set_major_locator(MaxNLocator(nbins=5)) 
                    ax_A.plot(time,pre_exponential_factors[:,j],color='k',linewidth = 0.5)#,linestyle='--')
                    ax_A.set_title(rxn.equation,fontsize=6)
                    ax_A.set_yscale('symlog')
                    ax_A.set_xscale('symlog')
                    ax_rate.axhline(y=0, color='b', linestyle='--', linewidth=1.5)
                

                #if j<num_rxn:
                    #ax_rate.yaxis.set_major_locator(FixedLocator(log_ticks(ax_rate)))
                    #ax_rate.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
                    #ax_A.yaxis.set_major_locator(FixedLocator(log_ticks(ax_A)))
                    #ax_A.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()
            plt.tight_layout(pad=0.1, h_pad=0.1)
            plt.subplots_adjust(wspace=0.2)
    if plot_flags[8] == 1:     
        a = int(np.sqrt(num_species))
        b = a + 1
        if a*b<num_species:
            a = a + 1
        fig121, axes = plt.subplots(a, b, figsize=(8,8*a/b))
        axes = axes.flatten()
        for j in range(num_species):
            ax = axes[j]
            plot_species = name_elyte_species[j]
            species_idx = SV_idx.elyte_species.index(plot_species)
            ax.plot(time, q_dot[:,j] ,'-')
            ax.set_title(plot_name_elyte_species[j])
        plt.tight_layout()
    '''
    a = int(np.sqrt(num_species))
    b = a + 1
    if a*b<num_species:
        a = a + 1
    fig122, axes = plt.subplots(a, b, figsize=(8,8*a/b))
    axes = axes.flatten()
    for j in range(num_species):
        ax = axes[j]
        plot_species = name_elyte_species[j]
        species_idx = SV_idx.elyte_species.index(plot_species)
        ax.plot(time[-10:], q_dot[-10:,j] ,'-')
        ax.set_title(plot_name_elyte_species[j])
    plt.tight_layout()
    '''

    print(tank.elyte_obj.net_rates_of_progress)
    last_data_point = np.size(time)
    gibbs_mixture = np.zeros(last_data_point)
    for i in range(np.size(gibbs_mixture)):
        X_k_row = C_k_elyte.T[i,:]/np.sum(C_k_elyte.T[i,:])
        C_k_timestep = C_k_elyte.T[i,:]
        moles_timestep = elyte_volume[i]*C_k_timestep
        gibbs_mixture[i] = np.dot(moles_timestep,g_f)

    if plot_flags[9] == 1:
        plt.figure()
        plt.plot(time[:last_data_point],gibbs_mixture)

    plt.show()

def log_ticks(ax):
    ymin, ymax = ax.get_ylim()

    if np.abs(ymin) > np.abs(ymax) and ymax < 0:
        log_max = int(np.floor(np.log10(np.abs(ymin))))
        log_min = int(np.ceil(np.log10(np.abs(ymax))))
        log_step = int((log_max - log_min)/4)
        exponents = [log_min+log_step*i for i in range(4)]
        log_ticks = [-1*10**el for el in exponents]
    else:
        log_min = np.sign(ymin)*int(np.floor(np.log10(np.abs(ymin))))
        log_max = int(np.ceil(np.log10(ymax)))
        log_step = int((log_max - log_min)/4)
        log_ticks =np.zeros(5)
        for k in range(5):
            log_ticks[k] = np.sign(log_min+(log_step*k))*10**(np.abs(log_min+log_step*k))
    log_ticks = np.array(log_ticks, dtype=float)

    return log_ticks
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