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
        self.r_avg = self.r_avg_graph         # this will be set each loop depending on the nucleation radius and bookmark locations
        
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

def set_r_avg(bucket_phase, leading_bookmark):
    bucket_phase.r_avg = [0]*bucket_phase.n
    number_occupied_bins = int(leading_bookmark/bucket_phase.thickness)
    for i in range(number_occupied_bins):
        bucket_phase.r_avg[i] = (number_occupied_bins-i+0.5)*bucket_phase.thickness
    return bucket_phase

def cs_area_phase(bucket_phase,n_particles_phase):
    '''
    Find the total cross sectional area of a phase deposited on the Cathode
    The cross sectional area is half the surface area
    '''
    CS_area_phase = [0]*bucket_phase.n
    for i in range(bucket_phase.n):
        CS_area_phase[i] = np.pi*((bucket_phase.r_avg[i])**2)*n_particles_phase[i]
        #print(bucket_phase.r_avg[i])
        
    CS_area_phase = sum(CS_area_phase)
    #print(CS_area_phase) 
    return CS_area_phase

def area_carbon(bucket_S8,n_particles_S8,area_carbon_0):
    '''
    Finds the cross sectional area occupied by the phases, then uses that to find the area 
    of carbon that is available for nucleation
    '''
    CS_area_S8 = cs_area_phase(bucket_S8,n_particles_S8)
    #print(CS_area_S8/area_carbon_0)
    #theta = 1 - np.exp(-CS_area_S8/area_carbon_0)
    #area_carbon = (1- theta)*area_carbon_0 #area_carbon_0 - CS_area_S8
    area_carbon = area_carbon_0 - CS_area_S8

    return area_carbon

def particle_flux_3(bucket, nuc_rate_per_area, leading_bookmark, area_carbon):
    Np_flux = np.zeros(bucket.n) # the change in the number of particles for each bucket
    nuc_location = np.zeros(bucket.n) # set based on location of bookmarks
    
    # the leading bookmark tells me where nucleation is occurring. 
    indx_nuc = int(leading_bookmark/bucket.thickness)
    nuc_location[indx_nuc] = 1

    Np_flux = nuc_rate_per_area*area_carbon*nuc_location

    return Np_flux

def residual(t,SV,user_data):
    # can this file call the cantera directly or will that be in the user data?
    s_k_nuc_S8_per_area = user_data[0]
    s_k_grow_S8_per_area = user_data[1]
    SV_index = user_data[2]
    bucket_S8 = user_data[3]
    area_carbon_0 = user_data[4]    
    
    resid = np.zeros_like(SV)

    # read state variable values    
    Np_S8 = SV[:SV_index.S8]
    bm_S8_front = SV[SV_index.bm_S8_front]

    bucket_S8 = set_r_avg(bucket_S8, bm_S8_front)

    # Used to cut off nucleation
    if t>0.5:#0.5: 500
        s_k_nuc_S8_per_area = 0
    else:
        s_k_nuc_S8_per_area = s_k_nuc_S8_per_area*np.exp(-0.85*(t*2-0.25)**2)*2 #make flat for other approach
    
    a_carbon = area_carbon(bucket_S8,Np_S8,area_carbon_0)
    # get the particle deposition rates due to nucleation [particles/m^2]
    Np_flux_S8 = particle_flux_3(bucket_S8, s_k_nuc_S8_per_area, bm_S8_front, a_carbon)
    
    ## Set residuals 
    resid[:SV_index.S8] = Np_flux_S8
    
    # The leading bookmarks always move
    #CS_area_S8 = cs_area_phase(bucket_S8,Np_S8)
    #theta = 1 - np.exp(-CS_area_S8/area_carbon_0)
    #drdt_S8 = s_k_grow_S8_per_area*bucket_S8.mv/(1e-12+CS_area_S8)#/(1- theta)
    drdt_S8 = s_k_grow_S8_per_area*bucket_S8.mv
    #print(CS_area_S8)
    #print(drdt_S8)

    # I assume that the process starts with no particles deposited
    resid[SV_index.bm_S8_front] = drdt_S8

    return resid


def plot_results(plot_flags, time, N_S8, bucket_S8,
        bm_S8_front, time_end, folder_name):
    
    time_stamps_bins = plot_flags
    
    cmap = mP.colormaps['plasma']
    plt.rcParams['font.family'] = 'Times' 
    
    nS8 = [0]*bucket_S8.n

    # here is where I remap the data for graphing. I flip the data so the biggest bin is now at 
    # the largest index instead of zero, and shift the bins based on where the bookmarks are
    N_S8_empty = N_S8.copy()*0
    for j in range(len(N_S8_empty[0])):
        front_indx = int(bm_S8_front[j]/bucket_S8.thickness)
        for indx, ele in enumerate(N_S8):
            val = ele[j]
            if val != 0:
                destination_row = np.copy(N_S8_empty[(front_indx)-indx])
                destination_row[j] = val
                N_S8_empty[(front_indx)-indx] = np.copy(destination_row)
    N_S8 = N_S8_empty
    
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
        save_fig('Particle_Distribution',folder_name)


        #fig8 = plt.figure(num=8,figsize=(.9,.675),dpi=300)
        #fig8 = plt.figure(num=8,figsize=(1,.75),dpi=300)
        fig8 = plt.figure(num=8,figsize=(1.2,.9),dpi=300)
        #fig8 = plt.figure(num=8,figsize=(3,2.25),dpi=300)
        plt_counter = 0
        for el in enumerate(time_snapshots):
            for i in range(len(time)):
                if time[i] > el[1] and  time[i-1] < el[1]:
                    for ind, ele in enumerate(N_S8):
                        nS8[ind] = ele[i]
                    plt.plot(bucket_S8.r_avg_graph, np.divide(nS8,sum(nS8))*100, linestyle='-', color=plt_clrs[plt_counter],linewidth=1)
                    plt_counter = plt_counter + 1
        for ind, ele in enumerate(N_S8):
            nS8[ind] = ele[i]
        plt.plot(bucket_S8.r_avg_graph, np.divide(nS8,sum(nS8))*100, linestyle='-', color=plt_clrs[plt_counter],linewidth=1)
        #print(nS8)
        ax = plt.gca() 
        plt.yticks(fontsize = 6)
        plt.xticks(fontsize = 6)
        plt.ylim([0,10])
        plt.xlim([0,1e-6+bucket_S8.thickness/2*4])
        plt.xlabel(r"r [$\mu$m]",fontsize = 8, labelpad=-5)
        plt.xticks([0,0.25e-6,0.5e-6,0.75e-6,1e-6],['0','','','','1'])
        plt.yticks([0,5,10],['1','','10'])
        plt.ylabel(r"[$\%$]",fontsize = 8,labelpad=-5)
        ax.tick_params(axis='x', which='major', length=2, pad = 1)
        ax.tick_params(axis='y', which='major', length=2, pad = 1)
        plt.tight_layout()
        save_fig('Particle_Distribution_TOC',folder_name)

# Plots confirming the solver takes small steps when the bookmarks cross the thresholds
        fig6, (ax10) = plt.subplots(1)
        ax10.set_title('Leading Bookmark')    
        for i in range(1, int(max(bm_S8_front)/bucket_S8.thickness)): # plot every bin threshold
            ax10.axhline(y=bucket_S8.thickness*i,linestyle='dashed',color='silver')
        ax10.plot(time,bm_S8_front,'.')
        ax10.set_ylabel(r'Distance [m]',fontsize=12)

        #ax10.set_xlim([0.5,0.6])
        #ax10.set_ylim([1.2e-7,1.6e-7])

        fig6.tight_layout()

    

    
def save_fig(pic_name,folder_name):
    if folder_name != None:
        fp_pic = f"{folder_name}/{pic_name}" + ".svg"        
        plt.savefig(fp_pic,transparent=True, format="svg")