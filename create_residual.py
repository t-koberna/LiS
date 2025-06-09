# create_residual.py
#
# This file creates the residual
import cantera as ct
import numpy as np

def residual(t,SV,SV_dot,resid,user_data):
    '''
    
    '''
    input_file, SV_idx, an, sep, i_ext  = user_data

    # So far only the anode has non-zero values, I will work to fill in more as I go
    # Create the cantera objects
    bulk_obj = ct.Solution(input_file, an['bulk-phase'])
    elyte_obj = ct.Solution(input_file, sep['electrolyte-phase'])
    conductor_obj = ct.Solution(input_file, an['conductor-phase'])
    surf_obj = ct.Interface(input_file, an['surf-phase'], [bulk_obj, elyte_obj, conductor_obj])
    
    sdot_electron = surf_obj.get_creation_rates(conductor_obj)  - surf_obj.get_destruction_rates(conductor_obj)
    sdot_Li = surf_obj.get_creation_rates(bulk_obj)  - surf_obj.get_destruction_rates(bulk_obj)

    i_far = -ct.faraday*sdot_electron
    i_dl = i_ext - i_far
    c_dl = an['C_dl']
    resid[SV_idx.ptr['phi_dl_an']] = SV_dot[SV_idx.ptr['phi_dl_an']] - i_dl/c_dl
    resid[SV_idx.ptr['thickness_an']] = SV_dot[SV_idx.ptr['thickness_an']] - sdot_Li*bulk_obj.partial_molar_volumes

    resid[SV_idx.ptr['phi_elyte']] = SV_dot[SV_idx.ptr['phi_elyte']]
    resid[SV_idx.ptr['C_k_elyte']] = SV_dot[SV_idx.ptr['C_k_elyte']]
    resid[SV_idx.ptr['phi_dl_ca']] = SV_dot[SV_idx.ptr['phi_dl_ca']] 
    resid[SV_idx.ptr['phi_ca']] = SV_dot[SV_idx.ptr['phi_ca']]
    resid[SV_idx.ptr['Li2S']] = SV_dot[SV_idx.ptr['Li2S']]
    resid[SV_idx.ptr['S8']] = SV_dot[SV_idx.ptr['S8']]
    resid[SV_idx.ptr['bm_Li2S']] = SV_dot[SV_idx.ptr['bm_Li2S']]
    resid[SV_idx.ptr['bm_S8']] = SV_dot[SV_idx.ptr['bm_S8']]