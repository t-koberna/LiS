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
        self.species = species_name           # holds a string with the species name (maybe don't need this)
        self.mv = molecular_volume            # constant molecular volume [m^3/mol]
        self.thickness = bucket_thickness     # r_max - r_min for the bucket [m]
        self.r_nuc = 0                        # holds the radius of nucleating particles. This is set to zero because it will be updated once nucleation begins
        self.r_avg = np.zeros(self.n)         # this will be set each loop depending on the nucleation radius and bookmark locations
        
        # The average radius of each bucket will change depending on the nucleation particle size and and
        # where nucleation is occurring since the bins are stagnant now, the average radius of each bin will change
        # as the particles grow
        # The way I post process right now maps the current data points to where they would fit on the r_avg_graph array
        self.r_max_graph = [None]*self.n            # the maximum radius of particles in each bucket [m] (need to decide if it is inclusive or exclusive)
        self.r_avg_graph = [None]*self.n            # the average radius of particles in each bucket [m]
        for indx in range(self.n):
            self.r_max_graph[indx] = bucket_thickness*(indx + 1)
            self.r_avg_graph[indx] = bucket_thickness*(0.5 + indx)

class Index_start:
    '''
    contains all of the index boundaries for the State Variable (SV) vector
    I only track the starts because python indexes in the form {this index}:{one before this index}
    so the starting index of the next section works as the ending index of the previous section
    
    The order of the definitions needs to match the order the species are listed in the SV
    '''
    def __init__(self,n_buckets_S8):
        self.S8 = n_buckets_S8                    
        self.bm_S8_front = self.S8
        self.bm_S8_back = self.bm_S8_front +1


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
    
def area_carbon(bucket_S8,n_particles_S8,area_carbon_0):
    '''
    Finds the cross sectional area occupied by the phases, then uses that to find the area 
    of carbon that is available for nucleation
    '''
    CS_area_S8 = cs_area_phase(bucket_S8,n_particles_S8)
        
    # I may find a termination check to stop things once this happens later
    area_carbon = area_carbon_0 - CS_area_S8
    if area_carbon<0:
        area_carbon = 0

    return area_carbon

def particle_flux_1(bucket, nuc_rate_per_area,grow_rate_per_area, n_particles, area_carbon):
    s_grow_rates = [0]*bucket.n # the growth rate for each bucket [events/s]
    area = [0]*bucket.n # find the area of each bucket assuming all particles have an average radius [m^2]
    drdt = [1]*bucket.n # the average change in radius with time for each bucket [m/s]
    Np_flux = [0]*bucket.n # the change in the number of particles for each bucket
    
    drdt = np.multiply(drdt,grow_rate_per_area*bucket.mv)

    Np_flux[0] = nuc_rate_per_area*area_carbon - drdt[0]/bucket.thickness*n_particles[0]
    Np_flux[1:-1] = drdt[:-2]/bucket.thickness*n_particles[:-2] - drdt[1:-1]/bucket.thickness*n_particles[1:-1]
    Np_flux[-1] = drdt[-2]/bucket.thickness*n_particles[-2]
    #drdt = grow_rate_per_area*bucket.mv # the average change in radius with time for each bucket [m/s]        
    # All growth rates will have the same sign

    return Np_flux

def particle_flux_2(bucket, nuc_rate_per_area, grow_rate_per_area, n_particles, leading_bookmark, trailing_bookmark, area_carbon):
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

    Np_flux[0] = nuc_rate_per_area*area_carbon - drdt/bucket.thickness*n_particles[0]*flux_factor[0]
    Np_flux[1:-1] = (drdt/bucket.thickness*np.multiply(n_particles[:-2],flux_factor[:-2]) 
                        - drdt/bucket.thickness*np.multiply(n_particles[1:-1],flux_factor[1:-1]) )
    Np_flux[-1] = drdt/bucket.thickness*n_particles[-2]*flux_factor[-2]

    return Np_flux

def particle_flux_3(bucket, nuc_rate_per_area, leading_bookmark, area_carbon):
    Np_flux = np.zeros(bucket.n) # the change in the number of particles for each bucket
    nuc_location = np.zeros(bucket.n) # set based on location of bookmarks
    
    #drdt = grow_rate_per_area*bucket.mv # the average change in radius with time for each bucket [m/s]        
    # All growth rates will have the same sign
  
    
    # locate bookmarks
    #indx_bm = bucket.n - int(leading_bookmark/bucket.thickness) - 1 #this code moves back to front
    # post processing was easier if I started at the first index
    # the leading bookmark tells me where nucleation is occurring. 
    indx_nuc = int(leading_bookmark/bucket.thickness)
    nuc_location[indx_nuc] = 1

    Np_flux = nuc_rate_per_area*area_carbon*nuc_location

    return Np_flux

def residual(t,SV,SV_dot,resid,user_data):
    # can this file call the cantera directly or will that be in the user data?
    s_k_nuc_S8_per_area = user_data[0]
    s_k_grow_S8_per_area = user_data[1]
    SV_index = user_data[2]
    bucket_S8 = user_data[3]
    area_carbon_0 = user_data[4]    
      
    # read state variable values    
    Np_S8 = SV[:SV_index.S8]
    bm_S8_front = SV[SV_index.bm_S8_front]
    bm_S8_back = SV[SV_index.bm_S8_back]

    bucket_S8.r_nuc = bucket_S8.thickness

    # Used to cut off nucleation
    if t>0.5:
        s_k_nuc_S8_per_area = 0
    else:
        s_k_nuc_S8_per_area = s_k_nuc_S8_per_area*1#np.exp(-0.85*(t-0.25)**2)*2
    
    a_carbon = area_carbon(bucket_S8,Np_S8,area_carbon_0)
    # get the particle deposition rates due to nucleation [particles/m^2]
    Np_flux_S8 = particle_flux_1(bucket_S8, s_k_nuc_S8_per_area, s_k_grow_S8_per_area, Np_S8, a_carbon)
    Np_flux_S8 = particle_flux_2(bucket_S8, s_k_nuc_S8_per_area, s_k_grow_S8_per_area, Np_S8, bm_S8_front, bm_S8_back, a_carbon)
    #particle_flux(bucket_S8, s_k_nuc_S8_per_area, bm_S8_front, a_carbon)
    
    ## Set residuals 
    resid[:SV_index.S8] = SV_dot[:SV_index.S8] - Np_flux_S8
    
    # The leading bookmarks always move
    drdt_S8 = s_k_grow_S8_per_area*bucket_S8.mv
    
    # Move the trailing bookmarks
    nuc_cuttoff = 0 # value nucleation needs to be bellow for me to assume the nucleation stage is over 
    # I assume that the process starts with no particles deposited
    resid[SV_index.bm_S8_front] = SV_dot[SV_index.bm_S8_front] - drdt_S8
    if sum(Np_S8)> 0 and s_k_nuc_S8_per_area <= nuc_cuttoff:
        resid[SV_index.bm_S8_back] = SV_dot[SV_index.bm_S8_back] - drdt_S8
    else:
        resid[SV_index.bm_S8_back] = SV_dot[SV_index.bm_S8_back]


def plot_results(plot_flags, time, N_S8, bucket_S8,
        bm_S8_front, bm_S8_back, time_end, folder_name):
    
    time_stamps_bins = plot_flags[0]
    bookmark_movement = plot_flags[1]
    
    cmap = mP.colormaps['plasma']
    plt.rcParams['font.family'] = 'Arial' 
    
    nS8 = [0]*bucket_S8.n

    '''
    # here is where I remap the data for graphing. I flip the data so the biggest bin is now at 
    # the largest index instead of zero, and shift the bins based on where the bookmarks are
    N_S8_empty = N_S8.copy()*0
    for j in range(len(N_S8_empty[0])):
        shift = int(bm_S8_back[j]/bucket_S8.thickness)
        clm = int(bm_S8_front[j]/bucket_S8.thickness)
        spread = clm-shift
        for indx, ele in enumerate(N_S8):
            val = ele[j]
            if val != 0:
                destination_row = np.copy(N_S8_empty[(spread+shift)-indx])
                destination_row[j] = val
                N_S8_empty[(spread+shift)-indx] = np.copy(destination_row)
    N_S8 = N_S8_empty
   '''
    #mP.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['xtick.top'] = plt.rcParams['ytick.right'] = True

    for i,b in enumerate(N_S8[:,-1]):
        if b !=0:
            biggest_bin = i

    bucket_S8.r_avg_graph = np.array(bucket_S8.r_avg_graph)/(bucket_S8.r_avg_graph[biggest_bin+1])
    # plots the time evolution of the particle distrobution
    if time_stamps_bins == 1:
        #mP.rcParams['mathtext.fontset'] = 'sans-serif'
        mP.rcParams['font.family'] = 'sans-serif'
        mP.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['xtick.top'] = plt.rcParams['ytick.right'] = True
        #plt.rcParams['text.usetex'] = True
        fig8 = plt.figure(num=7)#,dpi=250)

        plot_percs = np.array([0.129,0.25,0.5,1])
        plot_percs = np.array([0.125,0.25,0.5,1])
        time_ind = np.multiply(plot_percs,len(time))
        print(time_ind)
        plt_clrs = [cmap(0.1),cmap(0.35),cmap(0.55),cmap(0.75)]

        plt_counter = 0
        for el in enumerate(time_ind):
            for i in range(len(time)):
                if i == int(el[1]-1):
                    for ind, ele in enumerate(N_S8):
                        nS8[ind] = ele[i]
                    plt.plot(bucket_S8.r_avg_graph, np.divide(nS8,sum(nS8))*100, linestyle='-', color=plt_clrs[plt_counter],linewidth=2)
                    #plt.plot(bucket_S8.r_avg_graph, nS8, linestyle='-', color=plt_clrs[plt_counter],linewidth=2)
                    plt_counter = plt_counter + 1
        
        ax = plt.gca() 
        plt.yticks(fontsize = 12)
        #plt.xticks(np.linspace(0,0.6,7)*1e-7,['0','0.1','0.2','0.3','0.4','0.5','0.6'], fontsize = 12 )
        plt.xlim([0,1.01])
        plt.ylabel("Percent of Particles",fontsize = 16)
        plt.xlabel(r"Particle radius [-]",fontsize = 16)
        #ax.set_box_aspect(1)
        plt.tight_layout()
        save_fig('Particle_Distribution',folder_name)
    
    if bookmark_movement ==1:
        fig6 = plt.figure()
        plt.title(r'$\rm S_8$')    
        plt.plot(time,bm_S8_back)
        plt.plot(time,bm_S8_front)
        #for i in range(1, int(max(bm_S8_front)/bucket_S8.thickness)): # if I want to plot every bin
            #ax10.axhline(y=bucket_S8.thickness*i,linestyle='dashed',color='silver')
        plt.axhline(y=bucket_S8.thickness,linestyle='dashed',color='silver')
        plt.legend(["back","front","bin"])
        plt.ylabel(r'Distance [m]',fontsize=12)
        plt.xlabel(r'time [s]',fontsize=12)
        fig6.tight_layout()
        save_fig('bookmark_movement',folder_name)

def save_fig(pic_name,folder_name):
    if folder_name != None:
        fp_pic = f"{folder_name}/{pic_name}"+".png"
        if os.path.exists(fp_pic):
            fp_pic = f"{folder_name}/{pic_name}"+"_2.png"        
        plt.savefig(fp_pic)