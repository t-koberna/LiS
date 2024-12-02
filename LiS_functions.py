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
        #if (bucket.thickness - r_bm_trailing)/bucket.thickness < 0.005:
            #flux_factor[indx_bm_trailing] = bucket.thickness/(bucket.thickness - r_bm_trailing)
            #flux_factor[indx_bm_trailing] = 1
    
    if grow_rate_per_area > 0: 
        Np_flux[0] = nuc_rate_per_area*area_carbon - drdt/bucket.thickness*n_particles[1]*flux_factor[0]
        Np_flux[1:-1] = drdt/bucket.thickness*np.multiply(n_particles[:-2],flux_factor[:-2]) - drdt/bucket.thickness*np.multiply(n_particles[1:-1],flux_factor[1:-1]) 
        Np_flux[-1] = drdt/bucket.thickness*n_particles[-2]*flux_factor[-2]
        #if indx_bm_leading > 0:
            #Np_flux[indx_bm_leading]=Np_flux[indx_bm_leading-1]
        #if r_bm_trailing > 0:
            #Np_flux[indx_bm_trailing]= Np_flux[indx_bm_trailing+2]
        # old code
        #Np_flux[0] = nuc_rate_per_area*area_carbon - drdt/bucket.thickness*n_particles[0]
        #Np_flux[1:-1] =  - drdt/bucket.thickness*n_particles[1:-1] + drdt/bucket.thickness*n_particles[:-2]
        #Np_flux[-1] = drdt/bucket.thickness*n_particles[-2] 
    #print(n_flux)  
    else:
         # Not updated for the bookmarks yet, so far only growth has that 
        Np_flux[0] = nuc_rate_per_area*area_carbon + drdt/bucket.thickness*n_particles[0] - drdt/bucket.thickness*n_particles[1]
        Np_flux[1:-1] =  - drdt/bucket.thickness*n_particles[1:-1] - drdt/bucket.thickness*n_particles[2:]
        Np_flux[-1] = drdt/bucket.thickness*n_particles[-1]
    #print(n_flux)
    #if  indx_bm_leading == 5:
        #print(flux_factor)
        #print(Np_flux)
        #exit()
 
    '''
    if  indx_bm_leading == 5:
        print(flux_factor)
        print(Np_flux)
        print(nuc_rate_per_area*area_carbon)
        exit()
    '''
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
    if t>15:
        s_k_nuc_S8_per_area = 0
    if t>15:
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
    #print(t)
    #if t>100:

def residual_ivp(t,SV, s_k_nuc_S8_per_area,s_k_grow_S8_per_area,s_k_nuc_Li2S_per_area,s_k_grow_Li2S_per_area , SV_index, bucket_S8, bucket_Li2S,area_carbon_0):
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
    dSVdt = np.zeros_like(SV)
    
    # Used to cut off nucleation
    if t>20:
        s_k_nuc_S8_per_area = 0
    if t>20:
        s_k_nuc_Li2S_per_area = 0 
        
    a_carbon = area_carbon(bucket_S8,Np_S8,bucket_Li2S,Np_Li2S,area_carbon_0)
    # get the particle growth rates for each bucket [particles/m^2]
    Np_flux_S8 = particle_flux(bucket_S8, s_k_nuc_S8_per_area, s_k_grow_S8_per_area, Np_S8, bm_S8_front, bm_S8_back, a_carbon)
    Np_flux_Li2S = particle_flux(bucket_Li2S, s_k_nuc_Li2S_per_area, s_k_grow_Li2S_per_area, Np_Li2S, bm_Li2S_front, bm_Li2S_back, a_carbon)
    
    # get the rates of change for each bucket 
    
    ## Set residuals 
    # (starts at the index of the previous species, ends at one less than the index of the current species)
    
    # S8
    dSVdt[:SV_index.S8] =  Np_flux_S8

    # Li2S   
    dSVdt[SV_index.S8:SV_index.Li2S] =  Np_flux_Li2S
    
    # Concentrations In the electroltye
    surface_area_S8 = 2*cs_area_phase(bucket_S8,Np_S8)
    surface_area_Li2S = 2*cs_area_phase(bucket_Li2S,Np_Li2S)
    
    # moles of Li2S and S8
    dSVdt[SV_index.mol_S8_ca] = s_k_nuc_S8_per_area*a_carbon - surface_area_S8*s_k_grow_S8_per_area
    dSVdt[SV_index.mol_Li2S_ca] =  s_k_nuc_Li2S_per_area*a_carbon - surface_area_Li2S*s_k_grow_Li2S_per_area
    
    dSVdt[SV_index.mol_S8_elyte] = - s_k_nuc_S8_per_area*a_carbon + surface_area_S8*s_k_grow_S8_per_area
    dSVdt[SV_index.mol_Li2S_elyte] = -s_k_nuc_Li2S_per_area*a_carbon + surface_area_Li2S*s_k_grow_Li2S_per_area
    
    drdt_Li2S = s_k_grow_Li2S_per_area*bucket_Li2S.mv
    drdt_S8 = s_k_grow_S8_per_area*bucket_S8.mv
    
    # Move the bookmarks
    nuc_cuttoff = 1e-10 # value nucleation needs to be bellow for me to assume the nucleation stage is over 
    # I assume that the process starts with no particles depsosited
    dSVdt[SV_index.bm_S8_front] = drdt_S8
    if sum(Np_S8)> 0 and s_k_nuc_S8_per_area <= nuc_cuttoff:
        dSVdt[SV_index.bm_S8_back] =  drdt_S8
    else:
        dSVdt[SV_index.bm_S8_back] = 0

    dSVdt[SV_index.bm_Li2S_front] =  drdt_Li2S
    if sum(Np_Li2S)> 0 and s_k_nuc_Li2S_per_area <= nuc_cuttoff:
        dSVdt[SV_index.bm_Li2S_back] =   drdt_Li2S
    else:
        dSVdt[SV_index.bm_Li2S_back] = 0
    #print(t)
    #if t>100:
    return dSVdt