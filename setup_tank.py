# setup_tank.py
#
# This file sets up the Cantera objects, pointers, and solution vector for the model
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


class Tank:
    '''
    create a class to hold the properties and Cantera objects for the electrolyte object
    '''
    def __init__(self,path, input_file, params):
        self.inputs = input_file['cell-description']['tank']
        self.elyte_obj = ct.Solution(path, self.inputs['electrolyte-phase'])
        #C_k_0_elyte = np.ones_like([species['C_k'] for species in self.inputs['transport']['diffusion-coefficients']])
        C_k_0_elyte = [species['C_k'] for species in self.inputs['transport']['diffusion-coefficients']]
        C_total = np.sum(C_k_0_elyte)
        X_k = C_k_0_elyte/C_total
        self.elyte_obj.X = X_k
        self.elyte_obj.TP = params.T, params.P
        

class Solid:
    '''
    create a class to hold the properties and Cantera objects for the cathode object
    '''
    def __init__(self, path, input_file, tank, params):
        self.inputs = input_file['cell-description']['solid']
        self.solid_S8_obj = ct.Solution(path, self.inputs['host-phase_S8'])
        self.solid_Li2S_obj = ct.Solution(path, self.inputs['host-phase_Li2S'])
        self.elyte_obj = tank.elyte_obj # np.copy(
        self.surf_S8_obj = ct.Interface(path, self.inputs['surf-phase_S8'], [self.solid_S8_obj, self.elyte_obj])
        self.surf_Li2S_obj = ct.Interface(path, self.inputs['surf-phase_Li2S'], [self.solid_Li2S_obj, self.elyte_obj])
        self.mv_S8 = self.solid_S8_obj.partial_molar_volumes # m^3/kmol
        self.mv_Li2S = self.solid_Li2S_obj.partial_molar_volumes # m^3/kmol
        self.rho_S8 = self.solid_S8_obj.density # kg/m^3
        self.rho_Li2S = self.solid_Li2S_obj.density # kg/m^3


class SV_pointer:
    '''
    create an object that holds all of the pointers
    '''
    def __init__(self,input_file, tank, params):
        elyte_obj = tank.elyte_obj
        elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]
        self.elyte_species = elyte_species
        self.ptr = {}
        self.ptr['mass_S8'] = np.array([0])
        self.ptr['mass_Li2S'] = np.array([1])
        index_start = self.ptr['mass_Li2S'][-1] + 1
        self.ptr['C_k_elyte'] = np.arange(index_start, index_start + elyte_obj.n_species)
        self.num_vars = self.ptr['C_k_elyte'][-1] + 1

def Solution_Vector_0(SV_idx, tank, solid):
    '''
    set the initial values in the solution vector based on the yaml input file
    '''
    SV_0 = np.zeros(SV_idx.num_vars)
    SV_0[SV_idx.ptr['mass_S8']] = solid.inputs['mass-S8']
    SV_0[SV_idx.ptr['mass_Li2S']] = solid.inputs['mass-Li2S']
    #SV_0[SV_idx.ptr['C_k_elyte']] =  0.1*np.ones_like([species['C_k'] for species in tank.inputs['transport']['diffusion-coefficients']])
    SV_0[SV_idx.ptr['C_k_elyte']] =  [species['C_k'] for species in tank.inputs['transport']['diffusion-coefficients']]


    return SV_0