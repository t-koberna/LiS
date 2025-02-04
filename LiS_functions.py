# LiS_functions.py
#
#  This file holds utility functions called by the LiS model.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mP
import os

class bucket:
    '''
    Holds all of the information for each bucket
    '''
    def __init__(self,number_buckets,bucket_thickness,molecular_volume,species_name):
        self.n = number_buckets               # number of buckets
        self.r_max = [None]*self.n            # the maximum radius of particles in each bucket [m] (need to decide if it is inclusive or exclusive)
        self.r_avg = [None]*self.n            # the average raidus of particles in each bucket [m]
        for indx in range(self.n):
            self.r_max[indx] = bucket_thickness*(indx + 1)
            self.r_avg[indx] = bucket_thickness*(0.5 + indx)
        self.species = species_name           # holds a string with the species name (maybe don't need this)
        self.mv = molecular_volume            # constant molecular volume [m^3/mol]
        self.thickness = bucket_thickness     # r_max - r_min for the bucket [m]
        
class Index_start:
    '''
    contains all of the index boundries for the State Variable (SV) vector
    I only track the starts because python indexes in the form {this index}:{one before this index}
    so the starting index of the next section works as the ending index of the previous section
    
    The order of the definitions needs to match the order the species are listed in the SV
    '''
    def __init__(self,n_buckets_S8,n_buckets_Li2S):
        self.S8 = n_buckets_S8                    
        self.Li2S = n_buckets_S8  + n_buckets_Li2S
        self.mol_S8_ca = n_buckets_S8  + n_buckets_Li2S
        self.mol_Li2S_ca = n_buckets_S8  + n_buckets_Li2S + 1
        self.mol_S8_elyte = n_buckets_S8  + n_buckets_Li2S + 2
        self.mol_Li2S_elyte = n_buckets_S8  + n_buckets_Li2S + 3
        self.bm_S8_front = n_buckets_S8  + n_buckets_Li2S + 4
        self.bm_S8_back = n_buckets_S8  + n_buckets_Li2S + 5
        self.bm_Li2S_front = n_buckets_S8  + n_buckets_Li2S + 6
        self.bm_Li2S_back = n_buckets_S8  + n_buckets_Li2S + 7

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
    of carbon that is availible for nucleation
    
    How does this function work if I do not define an initial area?
    '''
    CS_area_S8 = cs_area_phase(bucket_S8,n_particles_S8)
    CS_area_Li2S = cs_area_phase(bucket_Li2S,n_particles_Li2S)
        
    # I may find a termination check to stop things once this happens later
    area_carbon = area_carbon_0 - CS_area_S8 - CS_area_Li2S
    if area_carbon<0:
        area_carbon = 0
    
    #print(area_carbon)
    return area_carbon

def volume_fraction(vol_0,bucket_S8,n_particles_S8,bucket_Li2S,n_particles_Li2S):
    # the volume fractions for S8 and Li2S only take into account the deposited solids on the 
    # cathode surface. The electroyle encompasses all disovled species 
    
    vol_S8 = volume_phase(bucket_S8, n_particles_S8)
    epsilon_S8 = vol_S8/vol_0
    
    vol_Li2S = volume_phase(bucket_Li2S, n_particles_Li2S)
    epsilon_Li2S = vol_Li2S/vol_0
    
    epsilon_eltye = 1 - epsilon_S8 - epsilon_Li2S
    return [epsilon_eltye, epsilon_S8, epsilon_Li2S]

def particle_flux(bucket, nuc_rate_per_area, grow_rate_per_area, n_particles, leading_bookmark, trailing_bookmark, area_carbon):
    s_grow_rates = [0]*bucket.n # the growth rate for each bucket [events/s]
    area = [0]*bucket.n # find the area of each bucket assuming all particles have an average radius [m^2]
    Np_flux = np.zeros(bucket.n) # the change in the number of particles for each bucket
    flux_factor = np.zeros(bucket.n) # set based on location of bookmarks
    
    # first I find drdt, then I find dN_pdt
    drdt = grow_rate_per_area*bucket.mv # the average change in radius with time for each bucket [m/s]        
    # All growth rates will have the same sign
    # only include the rate from the bucket above if it adds to current bucket (aka when it is negative)
    # only include the rate from the bucket below if it adds to current bucket (aka when it is positive)
    # I need a seperate statement for the first and last buckets
   
    
    # locate bookmarks
    indx_bm_leading = int(leading_bookmark/bucket.thickness)
    indx_bm_trailing = int(trailing_bookmark/bucket.thickness)
    r_bm_trailing = trailing_bookmark % bucket.thickness # how far into the bin the bookmark is
    # between the bookmarks the particles move normally, in the bin with the leading bookmark they do not leave
    # in the bin with the trailing bookmark a correction factor is needed so the correct amount leave
    flux_factor[indx_bm_trailing:indx_bm_leading] = 1
    if indx_bm_leading > 0: # if the leading bookmark is in the first bin I dont want the trailing bookmark to put a 1 in the first index 
        flux_factor[indx_bm_trailing] = bucket.thickness/(bucket.thickness - r_bm_trailing)
    #if indx_bm_leading == 0:  #redundant code
        #flux_factor = np.zeros(bucket.n)

    if grow_rate_per_area > 0: 
        Np_flux[0] = nuc_rate_per_area*area_carbon - drdt/bucket.thickness*n_particles[0]*flux_factor[0]
        Np_flux[1:-1] = (drdt/bucket.thickness*np.multiply(n_particles[:-2],flux_factor[:-2]) 
                         - drdt/bucket.thickness*np.multiply(n_particles[1:-1],flux_factor[1:-1]) )
        Np_flux[-1] = drdt/bucket.thickness*n_particles[-2]*flux_factor[-2]
    else:
         # Not updated for the bookmarks yet, so far only growth has that 
        Np_flux[0] = nuc_rate_per_area*area_carbon + drdt/bucket.thickness*n_particles[0] - drdt/bucket.thickness*n_particles[1]
        Np_flux[1:-1] =  - drdt/bucket.thickness*n_particles[1:-1] - drdt/bucket.thickness*n_particles[2:]
        Np_flux[-1] = drdt/bucket.thickness*n_particles[-1]

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
      
    # read state variable values    
    Np_S8 = SV[:SV_index.S8]
    Np_Li2S = SV[SV_index.S8:SV_index.Li2S]
    mol_S8_ca = SV[SV_index.mol_S8_ca]
    mol_Li2S_ca = SV[SV_index.mol_Li2S_ca]
    mol_S8_elyt = SV[SV_index.mol_S8_elyte]
    mol_Li2S_elyt = SV[SV_index.mol_Li2S_elyte]
    bm_S8_front = SV[SV_index.bm_S8_front]
    bm_S8_back = SV[SV_index.bm_S8_back]
    bm_Li2S_front = SV[SV_index.bm_Li2S_front]
    bm_Li2S_back = SV[SV_index.bm_Li2S_back]

    # Used to cut off nucleation
    if t>1:
        s_k_nuc_S8_per_area = 0
    #else:
        #s_k_nuc_S8_per_area = s_k_nuc_S8_per_area*np.exp(-0.85*(t-0.25)**2)*2
    if t>1:
        s_k_nuc_Li2S_per_area = 0 
    
    a_carbon = area_carbon(bucket_S8,Np_S8,bucket_Li2S,Np_Li2S,area_carbon_0)
    # get the particle growth rates for each bucket [particles/m^2]
    Np_flux_S8 = particle_flux(bucket_S8, s_k_nuc_S8_per_area, s_k_grow_S8_per_area, Np_S8, bm_S8_front, bm_S8_back, a_carbon)
    Np_flux_Li2S = particle_flux(bucket_Li2S, s_k_nuc_Li2S_per_area, s_k_grow_Li2S_per_area, Np_Li2S, bm_Li2S_front, bm_Li2S_back, a_carbon)
    
    # get the rates of change for each bucket 
    
    ## Set residuals 
    # (starts at the index of the previous species, ends at one less than the index of the current species)
    
    # S8
    resid[:SV_index.S8] = SV_dot[:SV_index.S8] - Np_flux_S8

    # Li2S   
    resid[SV_index.S8:SV_index.Li2S] = SV_dot[SV_index.S8:SV_index.Li2S] - Np_flux_Li2S
    # Concentrations In the electroltye
    surface_area_S8 = 2*cs_area_phase(bucket_S8,Np_S8)
    surface_area_Li2S = 2*cs_area_phase(bucket_Li2S,Np_Li2S)
    
    # moles of Li2S and S8
    resid[SV_index.mol_S8_ca] = SV_dot[SV_index.mol_S8_ca] - s_k_nuc_S8_per_area*a_carbon - surface_area_S8*s_k_grow_S8_per_area
    resid[SV_index.mol_Li2S_ca] = SV_dot[SV_index.mol_Li2S_ca] - s_k_nuc_Li2S_per_area*a_carbon - surface_area_Li2S*s_k_grow_Li2S_per_area
    
    resid[SV_index.mol_S8_elyte] = SV_dot[SV_index.mol_S8_elyte] + s_k_nuc_S8_per_area*a_carbon + surface_area_S8*s_k_grow_S8_per_area
    resid[SV_index.mol_Li2S_elyte] = SV_dot[SV_index.mol_Li2S_elyte] + s_k_nuc_Li2S_per_area*a_carbon + surface_area_Li2S*s_k_grow_Li2S_per_area
    
    drdt_Li2S = s_k_grow_Li2S_per_area*bucket_Li2S.mv
    drdt_S8 = s_k_grow_S8_per_area*bucket_S8.mv
    
    # Move the bookmarks
    nuc_cuttoff = 0 # value nucleation needs to be bellow for me to assume the nucleation stage is over 
    # I assume that the process starts with no particles depsosited
    resid[SV_index.bm_S8_front] = SV_dot[SV_index.bm_S8_front] - drdt_S8
    if sum(Np_S8)> 0 and s_k_nuc_S8_per_area <= nuc_cuttoff:
        resid[SV_index.bm_S8_back] = SV_dot[SV_index.bm_S8_back] - drdt_S8
    else:
        resid[SV_index.bm_S8_back] = SV_dot[SV_index.bm_S8_back]

    resid[SV_index.bm_Li2S_front] = SV_dot[SV_index.bm_Li2S_front] - drdt_Li2S
    if sum(Np_Li2S)> 0 and s_k_nuc_Li2S_per_area <= nuc_cuttoff:
        resid[SV_index.bm_Li2S_back] = SV_dot[SV_index.bm_Li2S_back] - drdt_Li2S
    else:
        resid[SV_index.bm_Li2S_back] = SV_dot[SV_index.bm_Li2S_back]


def plot_results(plot_flags, time, N_S8, N_Li2S, bucket_S8, bucket_Li2S, 
        mol_S8_elyt, mol_Li2S_elyt, mol_S8_ca, mol_Li2S_ca,
        bm_S8_front, bm_S8_back, bm_Li2S_front, bm_Li2S_back,
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

    for i in range(len(time)):
        for ind, ele in enumerate(N_S8): # extracts the number of particles in each bucket at the curren timestep
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
        xTicks = np.array([0,1,2,3,4])*1e-7
        xTicklabels = np.array([r'$0$', r'$0.1$', r'$0.2$', r'$0.3$',r'$0.4$'])
        yTicks = np.array([0,0.1,0.2,0.3,0.4,0.5])*100
        yTicklabels = np.array([r'$0$', r'$10$', r'$20$', r'$30$',r'$40$',r'$50$'])
        
        fig5, (ax8, ax9) = plt.subplots(2)
        plot_percs = np.array([0.25,0.5,0.75,1])
        time_frac = np.multiply(plot_percs,time_end)
        time_ind = np.multiply(plot_percs,len(time))
        plt_clrs = [cmap(0.1),cmap(0.35),cmap(0.55),cmap(0.75)]
        
        plt_counter = 0
        for el in enumerate(time_ind):
            for i in range(len(time)):
                if i == int(el[1]-1):
                    for ind, ele in enumerate(N_S8):
                        nS8[ind] = ele[i]
                    ax8.plot(bucket_S8.r_avg,np.divide(nS8,sum(nS8))*100,linestyle='-', marker='.', color=plt_clrs[plt_counter],linewidth=2)
                    for ind, ele in enumerate(N_Li2S):
                        nLi2S[ind] = ele[i]
                    ax9.plot(bucket_Li2S.r_avg,np.divide(nLi2S,sum(nLi2S))*100,linestyle='-', marker='.', color=plt_clrs[plt_counter],linewidth=2)
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
        fig8 = plt.figure(num=7)#,dpi=250)

        #plot_percs = np.array([0.125,0.25,0.5,0.625,0.75,0.875,1])
        time_frac = np.multiply(plot_percs,time_end)
        #plt_clrs = [cmap(0.1),cmap(0.25),cmap(0.35),cmap(0.45),cmap(0.65),cmap(0.85),cmap(1)]

        plot_percs = np.array([0.2,0.4,0.6,0.8])
        plot_percs = np.array([0.125,0.25,0.5,1])
        time_ind = np.multiply(plot_percs,len(time))
        plt_clrs = [cmap(0.1),cmap(0.35),cmap(0.55),cmap(0.75)]

        plt_counter = 0
        for el in enumerate(time_ind):
            for i in range(len(time)):
                if i == int(el[1]-1):
                    for ind, ele in enumerate(N_S8):
                        nS8[ind] = ele[i]
                    plt.plot(bucket_S8.r_avg, np.divide(nS8,sum(nS8))*100, linestyle='-', color=plt_clrs[plt_counter],linewidth=2)
                    plt_counter = plt_counter + 1
        
        ax = plt.gca() 
        plt.yticks(fontsize = 12)
        plt.xticks(np.linspace(0,0.6,7)*1e-7,['0','0.1','0.2','0.3','0.4','0.5','0.6'], fontsize = 12 )
        plt.xlim([0,0.505*1e-7])
        plt.ylabel("Percent of Particles",fontsize = 16)
        plt.xlabel(r"Particle radius [$\mu$m]",fontsize = 16)
        #ax.set_box_aspect(1)
        plt.tight_layout()
    
    if bookmark_movement ==1:
        fig6, (ax10, ax11) = plt.subplots(2)
        ax10.set_title(r'$\rm S_8$')    
        ax11.set_title(r'$\rm Li_2S$') 
        ax10.plot(time,bm_S8_back)
        ax10.plot(time,bm_S8_front)
        #for i in range(1, int(max(bm_S8_front)/bucket_S8.thickness)): # if I want to plot every bin
            #ax10.axhline(y=bucket_S8.thickness*i,linestyle='dashed',color='silver')
        ax10.axhline(y=bucket_S8.thickness,linestyle='dashed',color='silver')
        ax10.legend(["back","front","bin"])
        ax11.plot(time,bm_Li2S_back)
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
            
        
        
        
        
        
        
   
