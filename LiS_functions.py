# LiS_functions.py
#
#  This file holds utility functions called by the LiS model.
import numpy as np

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

def area_carbon(bucket_S8,n_particles_S8,bucket_Li2S,n_particles_Li2S,area_carbon_0):
    '''
    Finds the cross sectional area occupied by the phases, uses that to find the area 
    of carbon that is availible for nucleation
    '''
    CS_area_S8 = [0]*bucket_S8.n
    for i in range(bucket_S8.n):
        CS_area_S8[i] = np.pi*((bucket_S8.r_avg[i])**2)*n_particles_S8[i]
    CS_area_S8 = sum(CS_area_S8) 
    
    CS_area_Li2S = [0]*bucket_Li2S.n
    for i in range(bucket_Li2S.n):
        CS_area_Li2S[i] = np.pi*((bucket_Li2S.r_avg[i])**2)*n_particles_Li2S[i]
    CS_area_Li2S = sum(CS_area_Li2S) 
    
    # I may find a termination check to stop things once this happens later
    area_carbon = area_carbon_0 - CS_area_S8 - CS_area_Li2S
    if area_carbon<0:
        area_carbon = 0
    
    #print(area_carbon)
    return area_carbon

def particle_flux(bucket, nuc_rate_per_area, grow_rate_per_area, n_particles, area_carbon):
    s_grow_rates = [0]*bucket.n # the growth rate for each bucket [events/s]
    area = [0]*bucket.n # find the area of each bucket assuming all particles have an average radius [m^2]
    drdt = [1]*bucket.n # the average change in radius with time for each bucket [m/s]
    Np_flux = [0]*bucket.n # the change in the number of particles for each bucket
    
    '''
    # may not need this
    for i in range(bucket.n):
        area[i] = 2*np.pi*((bucket.r_avg[i])**2)*n_particles[i] # calculates the surface area for growth
        s_grow_rates[i] = grow_rate_per_area*area[i]  # takes the full per area growth rate and spreads it over each bucket based on area
    '''    
    #print(bucket.species)
    #print(s_grow_rates)
    #area_grow_total = sum(area) # the total area of the phase 
    
    # first I find drdt, then I find dN_pdt
    drdt = np.multiply(drdt,grow_rate_per_area*bucket.mv)
    # All growth rates will have the same sign
    # only include the rate from the bucket above if it adds to current bucket (aka when it is negative)
    # only include the rate from the bucket below if it adds to current bucket (aka when it is positive)
    # I need a seperate statement for the first and last buckets
    
    if grow_rate_per_area > 0:
        Np_flux[0] = nuc_rate_per_area*area_carbon - drdt[0]/bucket.thickness*n_particles[0]
        Np_flux[1:-1] = drdt[:-2]/bucket.thickness*n_particles[:-2] - drdt[1:-1]/bucket.thickness*n_particles[1:-1]
        Np_flux[-1] = drdt[-2]/bucket.thickness*n_particles[-2]
    else:
         # does nucleation take care of the particles disovling back into solution, or is the growth term that does that?  
        Np_flux[0] = nuc_rate_per_area*area_carbon + drdt[0]/bucket.thickness*n_particles[0] - drdt[1]/bucket.thickness*n_particles[1] 
        Np_flux[1:-1] =  - drdt[1:-1]/bucket.thickness*n_particles[1:-1] - drdt[2:]/bucket.thickness*n_particles[2:]
        Np_flux[-1] = drdt[-1]/bucket.thickness*n_particles[-1]
    #print(n_flux)
    
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
    Epsilon_C = user_data[7]
    area_carbon_0 = user_data[8]    
      
    # read state variable values    
    Np_S8 = SV[:SV_index.S8]
    Np_Li2S = SV[SV_index.S8:-2]
    Epsilon_S8 = SV[-2]
    Epsilon_Li2S = SV[-1]
    
    a_carbon = area_carbon(bucket_S8,Np_S8,bucket_Li2S,Np_Li2S,area_carbon_0)
    
    # get the particle growth rates for each bucket [particles/m^2]
    Np_flux_S8 = particle_flux(bucket_S8, s_k_nuc_S8_per_area, s_k_grow_S8_per_area, Np_S8, a_carbon)
    Np_flux_Li2S = particle_flux(bucket_Li2S, s_k_nuc_Li2S_per_area, s_k_grow_Li2S_per_area, Np_Li2S, a_carbon)
    
    # get the rates of change for each bucket 
    
    ## Set residuals 
    # (starts at the index of the previous species, ends at one less than the index of the current species)
    
    # S8
    resid[:SV_index.S8] = SV_dot[:SV_index.S8] - Np_flux_S8

    # Li2S   
    resid[SV_index.S8:SV_index.Li2S] = SV_dot[SV_index.S8:SV_index.Li2S] - Np_flux_Li2S
    
    # Volume Fractions
    # right now I am not updating the interfacial areas using the a_m formula (21) so I also cannot use equation (1)
    # to write these residials so they are zero for now
    resid[-2] = SV_dot[-2]
    resid[-1] = SV_dot[-1]
    
