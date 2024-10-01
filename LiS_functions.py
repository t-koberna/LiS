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
        self.range = bucket_thickness         # the range of radii held in each bucket (aka the thickness) [m]
        self.r_max = [None]*self.n            # the maximum radius of particles in each bucket [m] (need to decide if it is inclusive or exclusive)
        self.r_avg = [None]*self.n            # the average raidus of particles in each bucket [m]
        for indx in range(self.n):
            self.r_max[indx] = self.range*(indx + 1)
            self.r_avg[indx] = self.range*(0.5 + indx)
        self.species = species_name           # holds a string with the species name (maybe don't need this)
        self.mv = molecular_volume            # constant molecular volume [m^3/mol]
        

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

def particle_flux(bucket, nuc_rate, grow_rate, n_particles,area_carbon):
    s_grow_rates = [0]*bucket.n # the growth rate for each bucket [events/s]
    area = [0]*bucket.n # find the area of each bucket assuming all particles have an average radius [m^2]
    n_flux = [0]*bucket.n # the change in the number of particles for each bucket
    
    for i in range(bucket.n):
        area[i] = 2*np.pi*((bucket.r_avg[i])**2)*n_particles[i]
        s_grow_rates[i] = grow_rate*area[i]  # takes the full per area growth rate and spreads it over each bucket based on area
        
    #print(bucket.species)
    #print(s_grow_rates)
    #area_grow_total = sum(area) # the total area of the phase 
        
    
    # only include the rate from the bucket above if it adds to current bucket (aka when it is negative)
    # only include the rate from the bucket below if it adds to current bucket (aka when it is positive)
    if s_grow_rates[1]<0: 
        n_flux[0] = (nuc_rate*area_carbon - s_grow_rates[0] - s_grow_rates[1])
    else: 
        n_flux[0] = (nuc_rate*area_carbon - s_grow_rates[0])
    #n_flux[1:(bucket.n - 1)] = (s_grow_rates[:(bucket.n - 2)] - s_grow_rates[1:(bucket.n - 1)] - s_grow_rates[2:bucket.n] )
    
    for i in range(1,bucket.n-1):
        if (s_grow_rates[i+1]<0):
            if (s_grow_rates[i-1]>0):
                n_flux[i] = (s_grow_rates[i-1] - s_grow_rates[i] - s_grow_rates[i+1] )
        else: 
            if (s_grow_rates[i-1]>0):
                n_flux[i] = (s_grow_rates[i-1] - s_grow_rates[i])
            else:
                n_flux[i] = (- s_grow_rates[i])
            
    if (s_grow_rates[-1]>0):    
        n_flux[-1] = (s_grow_rates[-1] + s_grow_rates[-2])
    else:
        n_flux[-1] = (s_grow_rates[-2])
    #print(n_flux)
    
    return n_flux


def residual(t,SV,SV_dot,resid,user_data):
    # can this file call the cantera directly or will that be in the user data?
    s_k_nuc_S8 = user_data[0]
    s_k_grow_S8 = user_data[1]
    s_k_nuc_Li2S = user_data[2]
    s_k_grow_Li2S = user_data[3]
    SV_index = user_data[4]
    bucket_S8 = user_data[5]
    bucket_Li2S = user_data[6]
    Epsilon_C = user_data[7]
    area_carbon_0 = user_data[8]
    
      
    # read state variable values    
    n_S8 = SV[:SV_index.S8]
    n_Li2S = SV[SV_index.S8:-2]
    Epsilon_S8 = SV[-2]
    Epsilon_Li2S = SV[-1]
    
    a_carbon = area_carbon(bucket_S8,n_S8,bucket_Li2S,n_Li2S,area_carbon_0)
    
    # get the particle growth rates for each bucket [particles/m^2]
    n_flux_S8 = particle_flux(bucket_S8, s_k_nuc_S8, s_k_grow_S8, n_S8, a_carbon)
    n_flux_Li2S = particle_flux(bucket_Li2S, s_k_nuc_Li2S, s_k_grow_Li2S, n_Li2S, a_carbon)
    
    # get the rates of change for each bucket 
    
    ## Set residuals 
    # (starts at the index of the previous species, ends at one less than the index of the current species)
    
    # S8
    resid[:SV_index.S8] = SV_dot[:SV_index.S8] - n_flux_S8

    # Li2S   
    resid[SV_index.S8:SV_index.Li2S] = SV_dot[SV_index.S8:SV_index.Li2S] - n_flux_Li2S
    
    # Volume Fractions
    # right now I am not updating the interfacial areas using the a_m formula (21) so I also cannot use equation (1)
    # to write these residials so they are zero for now
    resid[-2] = SV_dot[-2]
    resid[-1] = SV_dot[-1]
    
