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

def particle_flux_1(bucket, nuc_rate_per_area,grow_rate_per_area, n_particles, area_carbon):
    Np_flux = [0]*bucket.n # the change in the number of particles for each bucket
    drdt = np.multiply([1]*bucket.n,grow_rate_per_area*bucket.mv) # the average change in radius with time for each bucket [m/s]

    Np_flux[0] = nuc_rate_per_area*area_carbon - drdt[0]/bucket.thickness*n_particles[0]
    Np_flux[1:-1] = drdt[:-2]/bucket.thickness*n_particles[:-2] - drdt[1:-1]/bucket.thickness*n_particles[1:-1]
    Np_flux[-1] = drdt[-2]/bucket.thickness*n_particles[-2]

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

    # Used to cut off nucleation
    if t>0.5:
        s_k_nuc_S8_per_area = 0
    else:
        s_k_nuc_S8_per_area = s_k_nuc_S8_per_area
    
    a_carbon = area_carbon(bucket_S8,Np_S8,area_carbon_0)
    # get the particle deposition rates due to nucleation [particles/m^2]
    Np_flux_S8 = particle_flux_1(bucket_S8, s_k_nuc_S8_per_area, s_k_grow_S8_per_area, Np_S8, a_carbon)
    
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
        bm_S8_front, time_end, folder_name):
    
    time_stamps_bins = plot_flags
    
    cmap = mP.colormaps['plasma']
    plt.rcParams['font.family'] = 'Times' 
    #plt.rcParams['xtick.top'] = plt.rcParams['ytick.right'] = True
    nS8 = [0]*bucket_S8.n

    for i,b in enumerate(N_S8[:,-1]):
        if b !=0:
            biggest_bin = i

    if biggest_bin == i:
        biggest_bin = biggest_bin -1

    #bucket_S8.r_avg_graph = np.array(bucket_S8.r_avg_graph)/(bucket_S8.r_avg_graph[biggest_bin+1])
    # plots the time evolution of the particle distribution
    if time_stamps_bins == 1:
        plt.rcParams['mathtext.fontset']='cm'
        mP.rcParams['font.family'] = 'serif'
        mP.rcParams['font.serif'] = 'Times New Roman'
        #plt.rcParams['xtick.top'] = plt.rcParams['ytick.right'] = True
        fig8 = plt.figure(num=7,figsize=(3,2.25),dpi=400)#,dpi=250)

        #plot_percs = np.array([0.25,0.5,0.75,1])
        plot_percs = np.array([0.125,0.25,0.5,1])
        time_ind = np.multiply(plot_percs,len(time))
        print(time_ind)
        plt_clrs = [cmap(0.1),cmap(0.35),cmap(0.55),cmap(0.75)]

        plt_counter = 0
        for el in enumerate(time_ind):
            for i in range(len(time)):
                if i == int(el[1]):
                    for ind, ele in enumerate(N_S8):
                        nS8[ind] = ele[i]
                    plt.plot(bucket_S8.r_avg_graph, np.divide(nS8,sum(nS8))*100, linestyle='-', color=plt_clrs[plt_counter],linewidth=2)
                    #plt.plot(bucket_S8.r_avg_graph, nS8, linestyle='-', color=plt_clrs[plt_counter],linewidth=2)
                    plt_counter = plt_counter + 1
        for ind, ele in enumerate(N_S8):
            nS8[ind] = ele[i]
        plt.plot(bucket_S8.r_avg_graph, np.divide(nS8,sum(nS8))*100, linestyle='-', color=plt_clrs[plt_counter],linewidth=2)
        
        #plt.axvline(x=bucket_S8.r_avg_graph[biggest_bin+1], linestyle='-',linewidth=0.5)
        #plt.axvline(x=1.005e-7, linewidth=0.5, linestyle='--', color='silver')
        #plt.axvline(x=1.25e-7, linewidth=0.5, linestyle='--')
        #plt.title(bucket_S8.r_avg_graph[biggest_bin+1])

        ax = plt.gca() 
        plt.yticks(fontsize = 8)
        plt.xticks(fontsize = 8)
        #plt.xlim([0,1.01])
        #plt.xlabel(r"Particle radius [-]",fontsize = 10)
        plt.xlim([0,bucket_S8.r_avg_graph[-1]+bucket_S8.thickness/2])
        plt.xlabel(r"Particle radius [$\mu$m]",fontsize = 10)
        #plt.xticks([0,0.5e-7,1e-7,1.5e-7,2.0e-7],['0','0.05','0.10','0.15','0.20'],fontsize = 8)
        plt.xticks([0,0.5e-6,1e-6,1.5e-6,2.0e-6],['0.0','0.5','1.0','1.5','2.0'],fontsize = 8)
        #ax.set_xticks([1.25e-6],[''], minor=True)
        plt.ylabel("Percent of Particles",fontsize = 10)
        plt.tight_layout()
        save_fig('Particle_Distribution',folder_name)

        fig_zoom = plt.figure(num=10,figsize=(1.5,1),dpi=400)
        plt.plot(bucket_S8.r_avg_graph[biggest_bin+1:], nS8[biggest_bin+1:])#np.divide(nS8[biggest_bin+1:],sum(nS8[biggest_bin+1:]))*100)
        
        plt.plot(bucket_S8.r_avg_graph, np.divide(nS8,sum(nS8))*100,color=plt_clrs[plt_counter],linewidth=2)
        #plt.xlim([1.005e-7,bucket_S8.r_avg_graph[-1]])
        plt.xlim([1.5e-6,bucket_S8.r_avg_graph[-1]+bucket_S8.thickness/2])
        plt.ylim([1e-17,1e1]) #1e-3 for 150 bins
        plt.yscale('log')
        ax = plt.gca()
        plt.yticks(fontsize = 8)
        plt.xticks([1.5e-6,2.0e-6],['1.5','2.0'],fontsize = 8)
        ax.set_yticks([1e0,1e-16])#, fontname='Times New Roman')[r'$10^0$','','','',r'$10^{-16}$']
        #minor_locator = mP.ticker.LogLocator(subs=(4)) 
        ax.set_yticks([1e0,1e-4,1e-8,1e-12,1e-16],['','','','',''], minor=True)
        #ax.set_xticks([1.25e-6],[''], minor=True)
        #ax.yaxis.set_minor_locator(minor_locator)
        plt.tight_layout()
        save_fig('Particle_Distribution_inset',folder_name)
        
    
def save_fig(pic_name,folder_name):
    if folder_name != None:
        fp_pic = f"{folder_name}/{pic_name}" #"+".png"        
        plt.savefig(fp_pic,transparent=True, format="svg")