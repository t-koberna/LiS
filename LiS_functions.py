# LiS_functions.py
#
#  This file holds utility functions called by the LiS model.
import numpy as np

class bucket:
    '''
    Holds all of the information for each bucket
    '''
    def __init__(self,number_buckets,bucket_thickness,species_name):
        self.n = number_buckets               # number of buckets
        self.range = bucket_thickness         # the range of radii held in each bucket (aka the thickness) [m]
        self.r_max = [None]*self.n            # the maximum radius of particles in each bucket [m] (need to decide if it is inclusive or exclusive)
        self.r_avg = [None]*self.n            # the average raidus of particles in each bucket [m]
        for indx in range(self.n):
            self.r_max[indx] = self.range*(indx + 1)
            self.r_avg[indx] = self.range*(0.5 + indx)
        self.species = species_name           # holds a string with the species name (maybe don't need this)
        

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

def particle_flux(bucket, nuc_rate, grow_rate, n_particles, epsillon, electrode_thickness):
    s_grow_rates = [0]*bucket.n # the growth rate for each bucket [events/m^3-s]
    area = [0]*bucket.n # find the area of each bucket assuming all particles have an average radius
    n_flux = [0]*bucket.n # the change in the number of particles for each bucket
    for i in range(bucket.n):
        area[i] = 2*np.pi*((bucket.r_avg[i])**2)*n_particles[i]
    area_total = sum(area) # the total area of the phase
    for i in range(bucket.n): # takes the full growth rate and spreads it over each bucket based on percent of the total 
        if area_total == 0:
            s_grow_rates[i] = 0
        else:
            s_grow_rates[i] = grow_rate*area[i]/area_total
   
    n_flux[0] = (nuc_rate - s_grow_rates[0] - s_grow_rates[1])*electrode_thickness*epsillon
    #n_flux[1:(bucket.n - 1)] = (s_grow_rates[:(bucket.n - 2)] - s_grow_rates[1:(bucket.n - 1)] - s_grow_rates[2:bucket.n] )*electrode_thickness*epsillon
    for i in range(1,bucket.n-1):
        n_flux[i] = (s_grow_rates[i-1] - s_grow_rates[i] - s_grow_rates[i+1] )*electrode_thickness*epsillon
    n_flux[-1] = (s_grow_rates[-1] + s_grow_rates[-2])*electrode_thickness*epsillon
    #print(n_flux)
    
    return n_flux


def residual_case1(t,SV,SV_dot,resid,user_data):
    # can this file call the cantera directly or will that be in the user data?
    s_k_nuc_S8 = user_data[0]
    s_k_grow_S8 = user_data[1]
    s_k_nuc_Li2S = user_data[2]
    s_k_grow_Li2S = user_data[3]
    SV_index = user_data[4]
    bucket_S8 = user_data[5]
    bucket_Li2S = user_data[6]
    
    # make this not hard coded (calculated or passed in) in the future
    epsillon_elyte = 0.8
    delta_y = 50*10**-6 # Electrode thickness [m]
    
        
    n_S8 = SV[:SV_index.S8]
    n_Li2S = SV[SV_index.S8:]
    
    # get the particle growth rates for each bucket [particles/m^2]
    n_flux_S8 = particle_flux(bucket_S8, s_k_nuc_S8, s_k_grow_S8, n_S8, epsillon_elyte, delta_y)
    n_flux_Li2S = particle_flux(bucket_Li2S, s_k_nuc_Li2S, s_k_grow_Li2S, n_Li2S, epsillon_elyte, delta_y)
    
    # get the rates of change for each bucket 
    
    ## Set residuals 
    # (starts at the index of the previous species, ends at one less than the index of the current species)
    
    # S8
    resid[:SV_index.S8] = SV_dot[:SV_index.S8] - n_flux_S8

    
    # Li2S
    for ind in range(SV_index.S8,SV_index.Li2S):
        if (SV[ind] <= 0) and (n_flux_Li2S[ind-SV_index.S8]<0):
            resid[ind] = SV_dot[ind]
        else:
            resid[ind] = SV_dot[ind] - n_flux_Li2S[ind-SV_index.S8]
    #resid[SV_index.S8:SV_index.Li2S] = SV_dot[SV_index.S8:SV_index.Li2S] - n_flux_Li2S
    
