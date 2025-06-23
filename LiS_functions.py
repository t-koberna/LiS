# LiS_functions.py
#
#  This file holds utility functions called by the LiS model.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mP
import os



class seperator:
    '''
    Holds all of the information for the seperator
    '''

class cathode:
    '''
    Holds all of the information for the cathode
    '''

class bucket:
    '''
    Holds all of the information for each bucket
    '''
    def __init__(self,number_buckets,bucket_thickness,molecular_volume,species_name):
        self.n = number_buckets               # number of buckets
        self.species = species_name           # holds a string with the species name (maybe don't need this)
        self.mv = molecular_volume            # constant molecular volume [m^3/mol]
        self.thickness = bucket_thickness     # r_max - r_min for the bucket [m]
        self.r_nuc = 0                        # holds the radius of nucleating particles. This is set to zero because it will be updated once nucleation begins
        self.r_avg = np.zeros(self.n)         # this will be set each loop depending on the nucleation radius and bookmark locations
        
        # The average radius of each bucket will change depending on the nucleation particle size and and
        # where nucleation is occurring since the bins are stagnant now, the average radius of each bin will change
        # as the particles grow
        # The way I post process right now maps the current data points to where they would fit on the r_avg_graph array
        self.r_avg_graph = [None]*self.n            # the average radius of particles in each bucket [m]
        # I start at one so it is updated immediately in the first pass of the solver
        self.r_min = 0                              # I don't want to start graphing from zero, so I need to keep track of what the smallest bin is


class Index_start:
    '''
    contains all of the index boundaries for the State Variable (SV) vector
    I only track the starts because python indexes in the form {this index}:{one before this index}
    so the starting index of the next section works as the ending index of the previous section
    
    The order of the definitions needs to match the order the species are listed in the SV
    '''
    def __init__(self,n_var_an, n_var_sep, n_var_ca, n_buckets_S8,n_buckets_Li2S):
        self.an = n_var_an
        self.sep = self.an + n_var_sep
        self.ca = self.sep + n_var_ca
        self.S8 = self.ca +  n_buckets_S8                    
        self.Li2S = self.S8 + n_buckets_Li2S
        self.bm_S8_front = self.Li2S
        self.bm_Li2S_front = self.bm_S8_front + 1

def cs_area_phase(bucket_phase,n_particles_phase):
    '''
    Find the total cross sectional area of a phase deposited on the Cathode
    This is for S_8 and Li_2S. The cross sectional area is half the surface area
    '''
    CS_area_phase = [0]*bucket_phase.n
    for i in range(bucket_phase.n):
        CS_area_phase[i] = np.pi*((bucket_phase.r_avg[i])**2)*n_particles_phase[i]
    CS_area_phase = sum(CS_area_phase) 
    return CS_area_phase

def volume_phase(bucket_phase,n_particles_phase):
    '''
    Find the total volume of a phase deposited on the Cathode
    This is for S_8 and Li_2S
    '''
    Vol_phase = [0]*bucket_phase.n
    for i in range(bucket_phase.n):
        Vol_phase[i] = 2/3*np.pi*((bucket_phase.r_avg[i])**3)*n_particles_phase[i]
    Vol_phase = sum(Vol_phase) 
    return Vol_phase
    
def area_carbon(bucket_S8,n_particles_S8,bucket_Li2S,n_particles_Li2S,area_carbon_0):
    '''
    Finds the cross sectional area occupied by the phases, then uses that to find the area 
    of carbon that is available for nucleation
    '''
    CS_area_S8 = cs_area_phase(bucket_S8,n_particles_S8)
    CS_area_Li2S = cs_area_phase(bucket_Li2S,n_particles_Li2S)
        
    # I may find a termination check to stop things once this happens later
    area_carbon = area_carbon_0 - CS_area_S8 - CS_area_Li2S
    if area_carbon<0:
        area_carbon = 0

    return area_carbon

def volume_fraction(vol_0,bucket_S8,n_particles_S8,bucket_Li2S,n_particles_Li2S):
    # the volume fractions for S8 and Li2S only take into account the deposited solids on the 
    # cathode surface. The electrolyte encompasses all dissolved species 
    
    vol_S8 = volume_phase(bucket_S8, n_particles_S8)
    epsilon_S8 = vol_S8/vol_0
    
    vol_Li2S = volume_phase(bucket_Li2S, n_particles_Li2S)
    epsilon_Li2S = vol_Li2S/vol_0
    
    epsilon_eltye = 1 - epsilon_S8 - epsilon_Li2S
    return [epsilon_eltye, epsilon_S8, epsilon_Li2S]

def particle_flux(bucket, nuc_rate_per_area, leading_bookmark, area_carbon):
    Np_flux = np.zeros(bucket.n) # the change in the number of particles for each bucket
    nuc_location = np.zeros(bucket.n) # set based on location of bookmarks
    
    # drdt = grow_rate_per_area*bucket.mv # the average change in radius with time for each bucket [m/s]        
    # All growth rates will have the same sign
  
    # locate bookmarks
    # indx_bm = bucket.n - int(leading_bookmark/bucket.thickness) - 1 #this code moves back to front
    # post processing was easier if I started at the first index
    # the leading bookmark tells me where nucleation is occurring. 
    indx_nuc = int((leading_bookmark)/bucket.thickness)
    nuc_location[indx_nuc] = 1

    Np_flux = nuc_rate_per_area*area_carbon*nuc_location

    return Np_flux

def residual(t,SV,SV_dot,resid,user_data):
    # can this file call the cantera directly or will that be in the user data?
    s_k_nuc_S8_per_area = user_data[0]
    s_k_grow_S8_per_area = user_data[1]
    s_k_nuc_Li2S_per_area = user_data[2]
    s_k_grow_Li2S_per_area = user_data[3]
    SV_index = user_data[4]
    bucket_S8 = user_data[5]
    bucket_Li2S = user_data[6]
    area_carbon_0 = user_data[7]
    an = user_data[8]
    sep = user_data[9]
    ca = user_data[10]
      
    # read state variable values    
    Np_S8 = SV[:SV_index.S8]
    Np_Li2S = SV[SV_index.S8:SV_index.Li2S]
    bm_S8_front = SV[SV_index.bm_S8_front]
    bm_Li2S_front = SV[SV_index.bm_Li2S_front]

    # Used to cut off nucleation
    if t>1:
        s_k_nuc_S8_per_area = 0
    else:
        s_k_nuc_S8_per_area = s_k_nuc_S8_per_area*np.exp(-0.85*(t-0.25)**2)*2
    if t>1:
        s_k_nuc_Li2S_per_area = 0 

    a_carbon = area_carbon(bucket_S8,Np_S8,bucket_Li2S,Np_Li2S,area_carbon_0)
    # get the particle deposition rates due to nucleation [particles/m^2]
    Np_flux_S8 = particle_flux(bucket_S8, s_k_nuc_S8_per_area, bm_S8_front, a_carbon)
    Np_flux_Li2S = particle_flux(bucket_Li2S, s_k_nuc_Li2S_per_area, bm_Li2S_front, a_carbon)

    ## Set residuals 
    resid[:SV_index.an] = SV_dot[:SV_index.an]
    resid[SV_index.an:SV_index.sep] = SV_dot[SV_index.an:SV_index.sep]
    resid[SV_index.sep:SV_index.ca] = SV_dot[SV_index.sep:SV_index.ca]
    # (starts at the index of the previous species, ends at one less than the index of the current species)

    # S8
    resid[SV_index.ca:SV_index.S8] = SV_dot[SV_index.ca:SV_index.S8] - Np_flux_S8

    # Li2S   
    resid[SV_index.S8:SV_index.Li2S] = SV_dot[SV_index.S8:SV_index.Li2S] - Np_flux_Li2S

    # Concentrations In the electrolyte. Surface area is twice the cross sectional area
    surface_area_S8 = 2*cs_area_phase(bucket_S8,Np_S8) 
    surface_area_Li2S = 2*cs_area_phase(bucket_Li2S,Np_Li2S)

    # moles of Li2S and S8. 
    #resid[SV_index.mol_S8_ca] = SV_dot[SV_index.mol_S8_ca] - s_k_nuc_S8_per_area*a_carbon - surface_area_S8*s_k_grow_S8_per_area
    #resid[SV_index.mol_Li2S_ca] = SV_dot[SV_index.mol_Li2S_ca] - s_k_nuc_Li2S_per_area*a_carbon - surface_area_Li2S*s_k_grow_Li2S_per_area

    #resid[SV_index.mol_S8_elyte] = SV_dot[SV_index.mol_S8_elyte] + s_k_nuc_S8_per_area*a_carbon + surface_area_S8*s_k_grow_S8_per_area
    #resid[SV_index.mol_Li2S_elyte] = SV_dot[SV_index.mol_Li2S_elyte] + s_k_nuc_Li2S_per_area*a_carbon + surface_area_Li2S*s_k_grow_Li2S_per_area

    # The leading bookmarks always move
    drdt_Li2S = s_k_grow_Li2S_per_area*bucket_Li2S.mv
    drdt_S8 = s_k_grow_S8_per_area*bucket_S8.mv 
    # I assume that the process starts with no particles deposited
    resid[SV_index.bm_S8_front] = SV_dot[SV_index.bm_S8_front] - drdt_S8
    resid[SV_index.bm_Li2S_front] = SV_dot[SV_index.bm_Li2S_front] - drdt_Li2S

def plot_results(plot_flags, time, N_S8, N_Li2S, bucket_S8, bucket_Li2S, 
        mol_S8_elyt, mol_Li2S_elyt, mol_S8_ca, mol_Li2S_ca,
        bm_S8_front, bm_Li2S_front,
        h, area_carbon_0, V_elyte_0, time_end, folder_name):
    
    num_particles_bin = plot_flags[0]
    cs_area = plot_flags[1]
    total_particles = plot_flags[2]
    conc_and_moles = plot_flags[3]
    vol_frac = plot_flags[4]
    time_stamps_bins = plot_flags[5]
    bookmark_movement = plot_flags[6]
    
    cmap = mP.colormaps['plasma']
    plt.rcParams['font.family'] = 'Arial' 
    
    a_c = [0]*len(time)
    a_S8 = [0]*len(time)
    a_Li2S = [0]*len(time)
    Epsilon_eltye = [0]*len(time)
    Epsilon_S8 = [0]*len(time)
    Epsilon_Li2S = [0]*len(time)
    nS8 = [0]*bucket_S8.n
    nLi2S = [0]*bucket_Li2S.n

    # I use the minimum particle radius to determine the smallest bin. Bins have bounds that 
    # are integer multiples of the bin width.
    # bucket_S8.r_avg_graph[0] = bucket_S8.r_min - (bucket_S8.r_min % bucket_S8.thickness) + bucket_S8.thickness/2
    bucket_S8.r_avg_graph[0] = bucket_S8.thickness/2
    for indx in range(1,bucket_S8.n):
            bucket_S8.r_avg_graph[indx] = bucket_S8.r_avg_graph[indx-1] + bucket_S8.thickness

    # bucket_Li2S.r_avg_graph[0] = bucket_Li2S.r_min - (bucket_Li2S.r_min % bucket_Li2S.thickness) + bucket_Li2S.thickness/2
    bucket_Li2S.r_avg_graph[0] = bucket_Li2S.thickness/2
    for indx in range(1,bucket_Li2S.n):
            bucket_Li2S.r_avg_graph[indx] = bucket_Li2S.r_avg_graph[indx-1] + bucket_Li2S.thickness

    # I remap the data for graphing. I flip the data so the biggest bin is now at 
    # the largest index instead of zero, and shift the bins based on where the bookmarks are
    # The front position is based on how man bins ahead of the r_min (right now zero) the leading bookmark is
    N_S8_empty = N_S8.copy()*0
    for j in range(len(N_S8_empty[0])):
        # front = int((bm_S8_front[j] -r_min bucket_S8.)/bucket_S8.thickness)
        front = int((bm_S8_front[j])/bucket_S8.thickness)
        for indx, ele in enumerate(N_S8):
            val = ele[j]
            if val != 0:
                destination_row = np.copy(N_S8_empty[front-indx])
                destination_row[j] = val
                N_S8_empty[front-indx] = np.copy(destination_row)
    N_S8 = N_S8_empty

    N_Li2S_empty = N_Li2S.copy()*0
    for j in range(len(N_Li2S_empty[0])):
        # front = int((bm_Li2S_front[j] - bucket_Li2S.r_min)/bucket_Li2S.thickness)
        front = int((bm_Li2S_front[j])/bucket_Li2S.thickness)
        for indx, ele in enumerate(N_Li2S):
            val = ele[j]
            if val != 0:
                destination_row = np.copy(N_Li2S_empty[(front)-indx])
                destination_row[j] = val
                N_Li2S_empty[(front)-indx] = np.copy(destination_row)
    N_Li2S = N_Li2S_empty

    for i in range(len(time)):
        for ind, ele in enumerate(N_S8): # extracts the number of particles in each bucket at the current time step
            nS8[ind] = ele[i]
        a_S8[i] = cs_area_phase(bucket_S8,nS8) 
        for ind, ele in enumerate(N_Li2S):
            nLi2S[ind] = ele[i]
        a_Li2S[i] = cs_area_phase(bucket_Li2S,nLi2S) 
        a_c[i] = area_carbon(bucket_S8,nS8,bucket_Li2S,nLi2S,area_carbon_0)
        
        [Epsilon_eltye[i], Epsilon_S8[i] ,Epsilon_Li2S[i]] = volume_fraction(V_elyte_0,bucket_S8,nS8,bucket_Li2S,nLi2S)
    
    #mP.rcParams['mathtext.fontset'] = 'cm'
    #plt.rcParams['xtick.top'] = plt.rcParams['ytick.right'] = True

    # Number of particles
    if num_particles_bin == 1:
        fig1, (ax1, ax2) = plt.subplots(2)
        for ind, ele in enumerate(N_S8):
            ax1.plot(time,ele,label=str(ind))
        for ind, ele in enumerate(N_Li2S):
            ax2.plot(time,ele,label=str(ind))
        #ax1.legend(ncol=1, bbox_to_anchor=(1, 0.5),loc = 'center left')
        ax1.set_title(r"S$_8$, "+str(len(N_S8))+" bins")
        ax1.set_ylabel("Number of Particles [-]")
        ax1.plot((2*np.ones(2)), np.array([0,np.max(N_S8)]), color='silver', linewidth=1, linestyle='dashed')
        #ax2.legend(ncol=1, bbox_to_anchor=(1, 0.5),loc = 'center left')
        ax2.set_title(r"Li$_2$S, "+str(len(N_Li2S))+" bins")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("Number of Particles [-]")
        ax2.plot((2*np.ones(2)), np.array([0,np.max(N_Li2S)]), color='silver', linewidth=1, linestyle='dashed')
        fig1.tight_layout()
        save_fig('num_particles_bin',folder_name)
        

    # Cross sectional area on cathode surface
    if cs_area == 1:
        fig2, (ax3, ax4) = plt.subplots(2)
        ax3.plot(time,a_c)
        ax3.set_title("Area of Carbon [m$^2$]")
        ax4.plot(time,a_S8, label = r'$S_8$')
        ax4.plot(time,a_Li2S, label = r'$Li_2S$')
        ax4.legend(ncol=1, loc = 'upper left')
        ax4.set_xlabel("time [s]")
        ax4.set_ylabel(r"Cross Sectional Area [m$^2$]")
        fig2.tight_layout()
        save_fig('cs_area',folder_name)

    # Number of deposited particles
    if total_particles == 1:
        fig3 = plt.figure()
        plt.title("Total number of particles")
        plt.plot(time,sum(N_S8))
        plt.plot(time,sum(N_Li2S))
        plt.legend([r"S$_8$",r"Li$_2$S"])
        fig3.tight_layout()
        save_fig('total_particles',folder_name)

    # moles of species in the electrolyte and cathode
    # plus concentration of species in the electrolyte
    if conc_and_moles == 1:
        fig4, (ax5, ax6, ax7) = plt.subplots(3)
        ax5.set_title("Electrolyte Concentrations [mol/m]")
        ax5.plot(time,mol_S8_elyt/h,label=r"S$_8$")
        ax5.plot(time,mol_Li2S_elyt/h,label=r"Li$_2$S")
        ax5.legend()
        ax6.set_title("Moles of species")
        ax6.plot(time,mol_S8_elyt,label=r"$\rm S_{8(elyte)}$")
        ax6.plot(time,mol_Li2S_elyt,label=r"$\rm Li_2S_{(elyte)}$")
        ax6.legend()
        ax7.plot(time,mol_S8_ca,label=r"$\rm S_{8(ca)}$")
        ax7.plot(time,mol_Li2S_ca,label=r"$\rm Li_2S_{(ca)}$")
        #ax7.set_xlim([0.5,3])
        #ax7.set_ylim([5e-6-1e-18,5e-6+1e-18])
        ax7.legend()
        ax7.set_xlabel("time [s]")
        fig4.tight_layout()
        save_fig('conc_and_moles',folder_name)

    if vol_frac == 1:
        fig5 = plt.figure()
        plt.plot(time,Epsilon_S8, label=r"S$_8$")
        plt.plot(time,Epsilon_Li2S, label=r"Li$_2$S")
        #plt.plot(time,Epsilon_eltye, label="elyte")
        plt.xlabel("time [s]")
        plt.title(r"Volume fraction")
        plt.ylabel(r"$\varepsilon$ [-]")
        plt.legend()
        fig5.tight_layout()
        save_fig('vol_frac',folder_name)
    
    # plots the time evolution of the particle distrobution
    if time_stamps_bins == 1:
        def plot_indexs_for_time(t, t_frac):
            ind_plot_percs = 0
            time_index = np.zeros_like(t_frac)
            for i, ele in enumerate(t):
                if ele >= t_frac[ind_plot_percs]:
                    time_index[ind_plot_percs] = i
                    ind_plot_percs = ind_plot_percs + 1 
            return time_index

        xTicks = np.array([0,1,2,3,4])*1e-7
        xTicklabels = np.array([r'$0$', r'$0.1$', r'$0.2$', r'$0.3$',r'$0.4$'])
        yTicks = np.array([0,0.1,0.2,0.3,0.4,0.5])*100
        yTicklabels = np.array([r'$0$', r'$10$', r'$20$', r'$30$',r'$40$',r'$50$'])
        
        fig5, (ax8, ax9) = plt.subplots(2)
        plot_percs = np.array([0.25,0.5,0.75,1])
        time_frac = np.multiply(plot_percs,time_end)
        time_ind = np.multiply(plot_percs,len(time))

        time_ind = plot_indexs_for_time(time, time_frac)            

        plt_clrs = [cmap(0.1),cmap(0.35),cmap(0.55),cmap(0.75)]
        
        plt_counter = 0
        for el in enumerate(time_ind):
            for i in range(len(time)):
                if i == int(el[1]-1):
                    for ind, ele in enumerate(N_S8):
                        nS8[ind] = ele[i]
                    ax8.plot(bucket_S8.r_avg_graph,np.divide(nS8,sum(nS8))*100,linestyle='-', marker='.', color=plt_clrs[plt_counter],linewidth=2)
                    for ind, ele in enumerate(N_Li2S):
                        nLi2S[ind] = ele[i]
                    ax9.plot(bucket_Li2S.r_avg_graph,np.divide(nLi2S,sum(nLi2S))*100,linestyle='-', marker='.', color=plt_clrs[plt_counter],linewidth=2)
                    plt_counter = plt_counter + 1
        ax8.set_title(r'$\rm S_8$')    
        ax9.set_title(r'$\rm Li_2S$')            
        ax8.set_yticks(yTicks, yTicklabels, fontsize=12)
        #ax8.set_xticks(xTicks, xTicklabels, fontsize=12)
        ax8.set_ylabel(r'Percent of Particles',fontsize=12)
        ax9.set_ylabel(r'Percent of Particles',fontsize=12)
        ax9.set_xlabel(r'Particle radius [m]',fontsize=12)
        #ax9.set_yticks(yTicks, yTicklabels, fontsize=12)
        #ax9.set_xticks(xTicks, xTicklabels, fontsize=12)
        ax8.legend([str(time[int(time_ind[0])]), str(time[int(time_ind[1])]), str(time[int(time_ind[2])]), str(time[int(time_ind[3]-1)])],title=r'$\rm t $=',fontsize=10,title_fontsize=12)
        fig5.tight_layout()
        save_fig('time_stamps_bins',folder_name)

        #mP.rcParams['mathtext.fontset'] = 'sans-serif'
        mP.rcParams['font.family'] = 'sans-serif'
        mP.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['xtick.top'] = plt.rcParams['ytick.right'] = True
        #plt.rcParams['text.usetex'] = True
        #fig8 = plt.figure(num=7)#,dpi=250)
        fig8 = plt.figure(num=7, figsize = (6,4))#,dpi=250)

        #plot_percs = np.array([0.125,0.25,0.5,0.625,0.75,0.875,1])
        
        #plt_clrs = [cmap(0.1),cmap(0.25),cmap(0.35),cmap(0.45),cmap(0.65),cmap(0.85),cmap(1)]

        plot_percs = np.array([0.2,0.4,0.6,0.8])
        plot_percs = np.array([0.125,0.25,0.5,1])
        plot_percs = np.array([0.129,0.25,0.5,1])
        #time_ind = np.multiply(plot_percs,len(time))
        time_frac = np.multiply(plot_percs,time_end)
        time_ind = plot_indexs_for_time(time, time_frac)  
        print(time_ind)
        plt_clrs = [cmap(0.1),cmap(0.35),cmap(0.55),cmap(0.75)]

        plt_counter = 0
        for el in enumerate(time_ind):
            for i in range(len(time)):
                if i == int(el[1]-1):
                    for ind, ele in enumerate(N_S8):
                        nS8[ind] = ele[i]
                    #plt.plot(bucket_S8.r_avg_graph, np.divide(nS8,sum(nS8))*100, linestyle='-', color=plt_clrs[plt_counter],linewidth=2)
                    plt.plot(bucket_S8.r_avg_graph, nS8, linestyle='-', color=plt_clrs[plt_counter],linewidth=2)
                    plt_counter = plt_counter + 1
        
        ax = plt.gca() 
        plt.yticks(fontsize = 12)
        plt.xticks(np.linspace(0,0.6,7)*1e-7,['0','0.1','0.2','0.3','0.4','0.5','0.6'], fontsize = 12 )
        plt.xlim([0,0.505*1e-7])
        plt.ylabel("Particles per unit area",fontsize = 16)
        plt.xlabel(r"Particle radius [$\mu$m]",fontsize = 16)
        #ax.set_box_aspect(1)
        plt.tight_layout()
    
    if bookmark_movement ==1:
        fig6, (ax10, ax11) = plt.subplots(2)
        ax10.set_title(r'$\rm S_8$')    
        ax11.set_title(r'$\rm Li_2S$') 
        ax10.plot(time,bm_S8_front)
        #for i in range(1, int(max(bm_S8_front)/bucket_S8.thickness)): # if I want to plot every bin
            #ax10.axhline(y=bucket_S8.thickness*i,linestyle='dashed',color='silver')
        ax10.axhline(y=bucket_S8.thickness,linestyle='dashed',color='silver')
        ax10.legend(["front","bin"])
        ax11.plot(time,bm_Li2S_front)
        ax11.axhline(y=bucket_Li2S.thickness, linestyle='dashed',color='silver')
        ax10.set_ylabel(r'Distance [m]',fontsize=12)
        ax11.set_ylabel(r'Distance [m]',fontsize=12)
        ax11.set_xlabel(r'time [s]',fontsize=12)
        fig6.tight_layout()
        save_fig('bookmark_movement',folder_name)

def save_fig(pic_name,folder_name):
    if folder_name != None:
        fp_pic = f"{folder_name}/{pic_name}"+".png"
        if os.path.exists(fp_pic):
            fp_pic = f"{folder_name}/{pic_name}"+"_2.png"        
        plt.savefig(fp_pic)
            
        
        
        
        
        
        
   
