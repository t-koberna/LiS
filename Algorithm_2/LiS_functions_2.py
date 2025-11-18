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
        self.mv = molecular_volume            # constant molecular volume [m^3/mol]
        self.thickness = bucket_thickness     # r_max - r_min for the bucket [m]
        self.r_avg_graph = [None]*self.n      # the average radius of particles in each bucket [m]
        for indx in range(self.n):
            self.r_avg_graph[indx] = bucket_thickness*(0.5 + indx)
        self.r_avg = self.r_avg_graph         

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

def cs_area_phase(bucket_phase,n_particles_phase):
    '''
    Find the total cross sectional area of a phase deposited on the Cathode
    The cross sectional area is half the surface area
    '''
    CS_area_phase = [0]*bucket_phase.n
    for i in range(bucket_phase.n):
        CS_area_phase[i] = np.pi*((bucket_phase.r_avg[i])**2)*n_particles_phase[i]
    CS_area_phase = sum(CS_area_phase) 
    return CS_area_phase

def area_carbon(bucket_S8,n_particles_S8,area_carbon_0):
    '''
    Finds the cross sectional area occupied by the phases, then uses that to find the area 
    of carbon that is available for nucleation
    '''
    CS_area_S8 = cs_area_phase(bucket_S8,n_particles_S8)
    area_carbon = area_carbon_0 - CS_area_S8

    return area_carbon

def particle_flux_2(bucket, nuc_rate_per_area, grow_rate_per_area, n_particles, leading_bookmark, trailing_bookmark, area_carbon):
    Np_flux = np.zeros(bucket.n) # the change in the number of particles for each bucket
    flux_factor = np.zeros(bucket.n) # set based on location of bookmarks

    # locate bookmarks
    indx_bm_leading = int(leading_bookmark/bucket.thickness)
    indx_bm_trailing = int(trailing_bookmark/bucket.thickness)
    r_bm_trailing = trailing_bookmark % bucket.thickness # how far into the bin the bookmark is
    # between the bookmarks the particles move normally, in the bin with the leading bookmark they do not leave
    # I need a separate statement for the first and last buckets

    # first I find drdt, then I find dN_pdt
    drdt = grow_rate_per_area*bucket.mv # the average change in radius with time for each bucket [m/s]        
    
    # in the bin with the trailing bookmark a correction factor is needed so the correct amount leave
    flux_factor[indx_bm_trailing:indx_bm_leading] = 1
    if indx_bm_leading > 0: # if the leading bookmark is in the first bin I dont want the trailing bookmark to put a 1 in the first index 
        flux_factor[indx_bm_trailing] = bucket.thickness/(bucket.thickness - r_bm_trailing)

    Np_flux[0] = nuc_rate_per_area*area_carbon - drdt/bucket.thickness*n_particles[0]*flux_factor[0]
    Np_flux[1:-1] = (drdt/bucket.thickness*np.multiply(n_particles[:-2],flux_factor[:-2]) 
                        - drdt/bucket.thickness*np.multiply(n_particles[1:-1],flux_factor[1:-1]) )
    Np_flux[-1] = drdt/bucket.thickness*n_particles[-2]*flux_factor[-2]

    return Np_flux

def residual(t,SV,user_data):
    # can this file call the cantera directly or will that be in the user data?
    s_k_nuc_S8_per_area = user_data[0]
    s_k_grow_S8_per_area = user_data[1]
    SV_index = user_data[2]
    bucket_S8 = user_data[3]
    area_carbon_0 = user_data[4]    
    variable_nucleation_rate = user_data[5]

    resid = np.zeros_like(SV)
      
    # read state variable values    
    Np_S8 = SV[:SV_index.S8]
    bm_S8_front = SV[SV_index.bm_S8_front]
    bm_S8_back = SV[SV_index.bm_S8_back]

    # Used to cut off nucleation
    if t>0.5:
        s_k_nuc_S8_per_area = 0
    else:
        if variable_nucleation_rate == 1:
            s_k_nuc_S8_per_area = s_k_nuc_S8_per_area*np.exp(-0.85*(t*2-0.25)**2)*2
        else:
            s_k_nuc_S8_per_area = s_k_nuc_S8_per_area

    
    a_carbon = area_carbon(bucket_S8,Np_S8,area_carbon_0)
    # get the particle deposition rates due to nucleation [particles/m^2]
    Np_flux_S8 = particle_flux_2(bucket_S8, s_k_nuc_S8_per_area, s_k_grow_S8_per_area, Np_S8, bm_S8_front, bm_S8_back, a_carbon)
    
    ## Set residuals 
    resid[:SV_index.S8] = Np_flux_S8
    
    # The leading bookmarks always move
    drdt_S8 = s_k_grow_S8_per_area*bucket_S8.mv
    
    # Move the trailing bookmarks
    nuc_cuttoff = 0 # value nucleation needs to be bellow for me to assume the nucleation stage is over 
    # I assume that the process starts with no particles deposited
    resid[SV_index.bm_S8_front] = drdt_S8
    if sum(Np_S8)> 0 and s_k_nuc_S8_per_area <= nuc_cuttoff:
        resid[SV_index.bm_S8_back] = drdt_S8
    else:
        resid[SV_index.bm_S8_back] = 0

    return resid


def plot_results(plot_flags, time, N_S8, bucket_S8,
        bm_S8_front, bm_S8_back, time_end, folder_name, variable_nucleation_rate):
    
    time_stamps_bins = plot_flags
    
    cmap = mP.colormaps['plasma']
    plt.rcParams['font.family'] = 'Times' 
    nS8 = [0]*bucket_S8.n

    for i,b in enumerate(N_S8[:,-1]):
        if b !=0:
            biggest_bin = i

    if biggest_bin == i:
        biggest_bin = biggest_bin -1

    # plots the time evolution of the particle distribution
    if time_stamps_bins == 1:
        plt.rcParams['mathtext.fontset']='cm'
        mP.rcParams['font.family'] = 'serif'
        mP.rcParams['font.serif'] = 'Times New Roman'
        fig8 = plt.figure(num=7,figsize=(3,2.25),dpi=400)

        plot_percs = np.array([0.125,0.25,0.5,1])
        time_snapshots = np.multiply(plot_percs,max(time))
        plt_clrs = [cmap(0.1),cmap(0.35),cmap(0.55),cmap(0.75)]

        plt_counter = 0
        for el in enumerate(time_snapshots):
            for i in range(len(time)):
                if time[i] > el[1] and  time[i-1] < el[1]:
                    for ind, ele in enumerate(N_S8):
                        nS8[ind] = ele[i]
                    plt.plot(bucket_S8.r_avg_graph, np.divide(nS8,sum(nS8))*100, linestyle='-', color=plt_clrs[plt_counter],linewidth=2)
                    plt_counter = plt_counter + 1
        for ind, ele in enumerate(N_S8):
            nS8[ind] = ele[i]
        plt.plot(bucket_S8.r_avg_graph, np.divide(nS8,sum(nS8))*100, linestyle='-', color=plt_clrs[plt_counter],linewidth=2)

        ax = plt.gca() 
        plt.yticks(fontsize = 8)
        plt.xticks(fontsize = 8)
        plt.ylim([-.4,10.5])
        plt.xlim([0,1e-6+bucket_S8.thickness/2*3])
        plt.xlabel(r"Particle radius [$\mu$m]",fontsize = 10)
        plt.xticks([0,0.25e-6,0.5e-6,0.75e-6,1e-6],['0.00','0.25','0.50','0.75','1.00'],fontsize = 8)
        plt.ylabel("Percent of Particles",fontsize = 10)
        plt.tight_layout()
        if variable_nucleation_rate == 1:
            save_fig('Particle_Distribution_var',folder_name)
        else:
            save_fig('Particle_Distribution_cons',folder_name)
        
        # Plots confirming the solver takes small steps when the bookmarks cross the thresholds
        fig6, (ax10, ax11) = plt.subplots(2)
        ax10.set_title('Leading Bookmark')    
        ax11.set_title('Trailing Bookmark') 
        for i in range(1, int(max(bm_S8_front)/bucket_S8.thickness)): # plot every bin threshold
            ax10.axhline(y=bucket_S8.thickness*i,linestyle='dashed',color='silver')
            ax11.axhline(y=bucket_S8.thickness*i,linestyle='dashed',color='silver')
        ax10.plot(time,bm_S8_front,'.')
        ax11.plot(time,bm_S8_back,'.')
        ax10.set_ylabel(r'Distance [m]',fontsize=12)
        ax11.set_ylabel(r'Distance [m]',fontsize=12)
        ax11.set_xlabel(r'time [s]',fontsize=12)

        ax10.set_xlim([0.5,0.6])
        ax11.set_xlim([0.5,0.6])
        ax10.set_ylim([1.2e-7,1.6e-7])
        ax11.set_ylim([0,3e-8])

        fig6.tight_layout()
        save_fig('bookmark_movement',folder_name)
    
def save_fig(pic_name,folder_name):
    if folder_name != None:
        fp_pic = f"{folder_name}/{pic_name}" + ".svg"        
        plt.savefig(fp_pic, format="svg")