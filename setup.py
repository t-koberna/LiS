# setup.py
#
# This file sets up the pointers and solution vector for the model 
import cantera as ct
import numpy as np

class Parameters:
    '''
    creates a class with the parameters from the yaml file
    '''
    def __init__(self,input_file):
        self.inputs = input_file['parameters']
        T, T_units =  self.inputs['T'].split()
        if T_units == 'C':
            self.T = float(T) + 273.13 # convert from [C] to [K]
        else:
            self.T = float(T)
        P, P_units =  self.inputs['P'].split()
        self.P = float(P)
        i_ext, i_ext_units = self.inputs['simulations']['i_ext'].split()
        self.i_ext = float(i_ext)

class Anode:
    '''
    create a class to hold the properties for the anode object
    '''
    def __init__(self,path, input_file, sep, params):
        self.inputs = input_file['cell-description']['anode']                                          
        self.bulk_obj = ct.Solution(path, self.inputs['bulk-phase'])
        self.bulk_obj.TP = params.T, params.P
        self.conductor_obj = ct.Solution(path, self.inputs['conductor-phase'])
        self.conductor_obj.TP = params.T, params.P
        self.elyte_obj = ct.Solution(path, sep.inputs['electrolyte-phase'])
        self.elyte_obj.TP = params.T, params.P
        C_k_0_elyte = [species['C_k'] for species in sep.inputs['transport']['diffusion-coefficients']]
        self.elyte_obj.X = C_k_0_elyte # it automatically takes in the concetrations and makes them a fraction
        self.surf_obj = ct.Interface(path, self.inputs['surf-phase'], [self.bulk_obj, self.elyte_obj, self.conductor_obj])
        self.surf_obj.TP = params.T, params.P
        
class Seperator:
    '''
    create a class to hold the properties for the seperator object
    '''
    def __init__(self,path, input_file):
        self.inputs = input_file['cell-description']['separator']
        self.elyte_obj = ct.Solution(path, self.inputs['electrolyte-phase'])

class Cathode:
    '''
    create a class to hold the properties for the cathode object
    '''
    def __init__(self, path, input_file, sep, params):
        self.inputs = input_file['cell-description']['cathode']
        self.host_obj = ct.Solution(path, self.inputs['host-phase'])
        self.host_obj.TP = params.T, params.P
        self.elyte_obj = ct.Solution(path, sep.inputs['electrolyte-phase'])
        self.elyte_obj.TP = params.T, params.P
        C_k_0_elyte = [species['C_k'] for species in sep.inputs['transport']['diffusion-coefficients']]
        self.elyte_obj.X = C_k_0_elyte # it automatically takes in the concetrations and makes them a fraction
        self.surf_obj = ct.Interface(path, self.inputs['surf-phase'], [self.host_obj, self.elyte_obj])
        self.surf_obj.TP = params.T, params.P

        # Create the conversion phases and the elyte interfaces
        self.conversion_phases = []
        self.conversion_obj = []
        self.conversion_surf_obj = []

        for i, phase in enumerate(self.inputs["conversion-phases"]):
            self.conversion_phases.append(phase['bulk-name'])
            self.conversion_obj.append(ct.Solution(path, phase["bulk-name"]))
            self.conversion_obj[i].TP = params.T, params.P
            self.conversion_surf_obj.append(ct.Interface(path, phase["surf-name"],
                    [self.elyte_obj, self.host_obj, self.conversion_obj[i]]))


class SV_pointer:
    '''
    create an object that holds all of the pointers 
    '''
    def __init__(self,input_file, sep, params):
        n_elyte_points = sep.inputs['n_points']
        elyte_obj = sep.elyte_obj

        elyte_species = [species['name'] for species in sep.inputs['transport']['diffusion-coefficients']]
        self.elyte_species = elyte_species*n_elyte_points

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
        self.ptr['Li2S'] = np.arange(index_start, index_start + params.inputs['simulations']['number-bins']['Li2S'])
        index_start = self.ptr['Li2S'][-1] + 1
        self.ptr['S8'] = np.arange(index_start, index_start + params.inputs['simulations']['number-bins']['S8'])
        index_start = self.ptr['S8'][-1] + 1
        self.ptr['bm_Li2S'] = np.arange(index_start, index_start + 1)
        index_start = self.ptr['bm_Li2S'][-1] + 1
        self.ptr['bm_S8'] = np.arange(index_start, index_start + 1)
        self.num_vars = self.ptr['bm_S8'][-1] + 1

def Solution_Vector_0(SV_idx, sep, anode, cathode, params):
    '''
    set the initial values in the solution vector based on the yaml input file 
    '''
    C_k_0_elyte = [species['C_k'] for species in sep.inputs['transport']['diffusion-coefficients']]
    C_k_0 = C_k_0_elyte.copy()
    for i in range(sep.inputs['n_points']-1):
        C_k_0_elyte = np.stack((*C_k_0_elyte,*C_k_0)) 

    SV_0 = np.zeros(SV_idx.num_vars)
    SV_0[SV_idx.ptr['phi_dl_an']] = sep.inputs['phi_0']
    SV_0[SV_idx.ptr['thickness_an']] = anode.inputs['thickness']
    SV_0[SV_idx.ptr['phi_elyte']] = np.array([sep.inputs['phi_0']]*sep.inputs['n_points'])
    SV_0[SV_idx.ptr['C_k_elyte']] = C_k_0_elyte
    SV_0[SV_idx.ptr['phi_dl_ca']] = cathode.inputs['phi_0'] - sep.inputs['phi_0']
    SV_0[SV_idx.ptr['phi_ca']] = cathode.inputs['phi_0'] # change this line to cathode.inputs once I make a cathode object
    SV_0[SV_idx.ptr['Li2S']] = np.zeros(params.inputs['simulations']['number-bins']['Li2S'])
    SV_0[SV_idx.ptr['S8']] = np.zeros(params.inputs['simulations']['number-bins']['S8'])
    SV_0[SV_idx.ptr['bm_Li2S']] = 0
    SV_0[SV_idx.ptr['bm_S8']] = 0

    return SV_0