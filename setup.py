# setup.py
#
# This file sets up the pointers and solution vector for the model 
import cantera as ct
import numpy as np

class SV_pointer:
    '''
    create an object that holds all of the pointers 
    '''
    def __init__(self,input_file, sep, params):
        n_elyte_points = sep['n_points']
        elyte_obj = ct.Solution(input_file, sep['electrolyte-phase'])
        
        self.ptr = {}
        self.ptr['phi_dl_an'] = np.array([0])
        self.ptr['thickness_an'] = np.array([1])
        index_start = self.ptr['thickness_an'][-1] + 1
        self.ptr['phi_elyte'] = np.arange(index_start,index_start + n_elyte_points)
        index_start = self.ptr['phi_elyte'][-1] + 1
        self.ptr['C_k_elyte'] = np.arange(index_start, index_start + elyte_obj.n_species*n_elyte_points)
        index_start = self.ptr['C_k_elyte'][-1] + 1
        self.ptr['phi_dl_ca'] = np.arange(index_start, index_start + 1)
        index_start = self.ptr['phi_dl_ca'][-1] + 1
        self.ptr['phi_ca'] = np.arange(index_start, index_start + 1)
        index_start = self.ptr['phi_ca'][-1] + 1
        self.ptr['Li2S'] = np.arange(index_start, index_start + params['simulations']['number-bins']['Li2S'])
        index_start = self.ptr['Li2S'][-1] + 1
        self.ptr['S8'] = np.arange(index_start, index_start + params['simulations']['number-bins']['S8'])
        index_start = self.ptr['S8'][-1] + 1
        self.ptr['bm_Li2S'] = np.arange(index_start, index_start + 1)
        index_start = self.ptr['bm_Li2S'][-1] + 1
        self.ptr['bm_S8'] = np.arange(index_start, index_start + 1)
        self.num_vars = self.ptr['bm_S8'][-1] + 1

def Solution_Vector_0(SV_idx, sep_inputs, anode_inputs, cathode_inputs, parameters):
    '''
    set the initial values in the solution vector based on the yaml input file 
    '''
    C_k_0_elyte = [species['C_k'] for species in sep_inputs['transport']['diffusion-coefficients']]
    C_k_0 = C_k_0_elyte.copy()
    for i in range(sep_inputs['n_points']-1):
        C_k_0_elyte = np.stack((*C_k_0_elyte,*C_k_0)) 

    SV_0 = np.zeros(SV_idx.num_vars)
    SV_0[SV_idx.ptr['phi_dl_an']] = sep_inputs['phi_0']
    SV_0[SV_idx.ptr['thickness_an']] = anode_inputs['thickness']
    SV_0[SV_idx.ptr['phi_elyte']] = np.array([sep_inputs['phi_0']]*sep_inputs['n_points'])
    SV_0[SV_idx.ptr['C_k_elyte']] = C_k_0_elyte
    SV_0[SV_idx.ptr['phi_dl_ca']] = sep_inputs['phi_0']
    SV_0[SV_idx.ptr['phi_ca']] = cathode_inputs['phi_0']
    SV_0[SV_idx.ptr['Li2S']] = np.zeros(parameters['simulations']['number-bins']['Li2S'])
    SV_0[SV_idx.ptr['S8']] = np.zeros(parameters['simulations']['number-bins']['S8'])
    SV_0[SV_idx.ptr['bm_Li2S']] = 0
    SV_0[SV_idx.ptr['bm_S8']] = 0

    return SV_0