# LiS_functions.py
#
#  This file holds utility functions called by the LiS model.
import numpy as np
import matplotlib.pyplot as plt

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
        self.r_nuc = 1e-9
class Index_start:
    '''
    contains all of the index boundaries for the State Variable (SV) vector
    I only track the starts because python indexes in the form {this index}:{one before this index}
    so the starting index of the next section works as the ending index of the previous section
    
    The order of the definitions needs to match the order the species are listed in the SV
    '''
    def __init__(self,n_buckets_Li2O2):
        self.Li2O2 = n_buckets_Li2O2                    

def volume_phase(bucket_phase,n_particles_phase):
    '''
    Find the total volume of a phase deposited on the Cathode
    This is for S_8 and Li_2S
    '''
    Vol_phase = [0]*bucket_phase.n
    for i in range(bucket_phase.n):
        Vol_phase[i] = 2/3*np.pi*((bucket_phase.r_avg[i]+bucket_phase.r_nuc)**3)*n_particles_phase[i]
    Vol_phase = sum(Vol_phase) 
    return Vol_phase

def cs_area_phase(bucket_phase,n_particles_phase):
    '''
    Find the total cross sectional area of a phase deposited on the Cathode
    The cross sectional area is half the surface area
    '''
    CS_area_phase = [0]*bucket_phase.n
    for i in range(bucket_phase.n):
        CS_area_phase[i] = np.pi*((bucket_phase.r_avg[i]+bucket_phase.r_nuc)**2)*n_particles_phase[i]
        #print(bucket_phase.r_avg[i])
        
    CS_area_phase = sum(CS_area_phase)
    #print(CS_area_phase) 
    return CS_area_phase

def area_carbon(bucket_Li2O2,n_particles_Li2O2,area_carbon_0):
    '''
    Finds the cross sectional area occupied by the phases, then uses that to find the area 
    of carbon that is available for nucleation
    '''
    CS_area_Li2O2 = cs_area_phase(bucket_Li2O2,n_particles_Li2O2)
    theta = 1 - np.exp(-CS_area_Li2O2/area_carbon_0)
    area_carbon = (1- theta)*area_carbon_0 #area_carbon_0 - CS_area_Li2O2

    return area_carbon

def particle_flux_1(bucket, nuc_rate_per_area,grow_rate_per_area, n_particles, area_carbon):
    Np_flux = [0]*bucket.n # the change in the number of particles for each bucket
    drdt = np.multiply([1]*bucket.n,grow_rate_per_area) # the average change in radius with time for each bucket [m/s]

    Np_flux[0] = nuc_rate_per_area*area_carbon - drdt[0]/bucket.thickness*n_particles[0]
    Np_flux[1:-1] = drdt[:-2]/bucket.thickness*n_particles[:-2] - drdt[1:-1]/bucket.thickness*n_particles[1:-1]
    Np_flux[-1] = drdt[-2]/bucket.thickness*n_particles[-2]

    return Np_flux

def residual(t,SV,user_data):
    # can this file call the cantera directly or will that be in the user data?
    s_k_nuc_Li2O2_per_area = user_data[0]
    s_k_grow_Li2O2_per_area = user_data[1]
    SV_index = user_data[2]
    bucket_Li2O2 = user_data[3]
    area_carbon_0 = user_data[4]    
    
    resid = np.zeros_like(SV)

    # read state variable values    
    Np_Li2O2 = SV[:SV_index.Li2O2]

    max_cov = 0.999
    start_cov = area_carbon_0*0
    CS_area_Li2O2 = cs_area_phase(bucket_Li2O2,Np_Li2O2) + start_cov
    theta = 1 - np.exp(-CS_area_Li2O2/area_carbon_0)
    
    if theta>max_cov:
        s_k_nuc_Li2O2_per_area = 0
    a_carbon = (1- theta)*area_carbon_0

    if theta>max_cov:
        drdt_Li2O2 = 0
    else:
        drdt_Li2O2 = s_k_grow_Li2O2_per_area*bucket_Li2O2.mv/(1- theta)*3600/(2*(CS_area_Li2O2))*(1.6e-4)#*(a_carbon/10)
    
    # get the particle deposition rates due to nucleation [particles/m^2]
    Np_flux_Li2O2 = particle_flux_1(bucket_Li2O2, s_k_nuc_Li2O2_per_area*3600, drdt_Li2O2, Np_Li2O2, a_carbon)
    
    ## Set residuals 
    resid[:SV_index.Li2O2] = Np_flux_Li2O2
   
    return resid

