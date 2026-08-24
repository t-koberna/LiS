# setup_tank.py
#
# This file sets up the Cantera objects, pointers, and solution vector for the model
import cantera as ct
import numpy as np
import pandas as pd
import os

class Parameters:
    '''
    creates a class with the parameters from the yaml file
    The only parameters are th temperature and pressure
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
    Uses the yaml input file to set the initial concentration
    '''
    def __init__(self,path, input_file, params):
        self.inputs = input_file['cell-description']['tank']
        self.input_file = input_file
        self.elyte_obj = ct.Solution(path, self.inputs['electrolyte-phase'])
        C_k_0_elyte = [species['C_k'] for species in self.inputs['transport']['diffusion-coefficients']]
        C_total = np.sum(C_k_0_elyte)
        X_k = C_k_0_elyte/C_total
        self.elyte_obj.X = X_k
        self.elyte_obj.TP = params.T, params.P
        self.write_yaml = 0

class RateTracker:
    def __init__(self,tank):
        self.times = []
        self.rates = []
        self.tank = tank

    def log_step(self, t, y):
        rates = self.tank.elyte_obj.net_rates_of_progress
        self.rates.append([np.copy(rates)])  
 
class Solid:
    '''
    create a class to hold the properties and Cantera objects for the surface objects
    Objects are the solid S8 and Li2S
    '''
    def __init__(self, path, input_file, tank, params):
        self.inputs = input_file['cell-description']['solid']
        self.solid_S8_obj = ct.Solution(path, self.inputs['host-phase_S8'])
        self.solid_Li2S_obj = ct.Solution(path, self.inputs['host-phase_Li2S'])
        self.elyte_obj = tank.elyte_obj 
        self.surf_S8_obj = ct.Interface(path, self.inputs['surf-phase_S8'], [self.solid_S8_obj, self.elyte_obj])
        self.surf_Li2S_obj = ct.Interface(path, self.inputs['surf-phase_Li2S'], [self.solid_Li2S_obj, self.elyte_obj])
        self.mv_S8 = self.solid_S8_obj.partial_molar_volumes                # m^3/kmol
        self.mv_Li2S = self.solid_Li2S_obj.partial_molar_volumes            # m^3/kmol
        self.rho_S8 = self.solid_S8_obj.density                             # kg/m^3
        self.rho_Li2S = self.solid_Li2S_obj.density                         # kg/m^3

class SV_pointer:
    '''
    create an object that holds all of the pointers
    '''
    def __init__(self, tank):
        elyte_obj = tank.elyte_obj
        elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]
        self.elyte_species = elyte_species
        self.ptr = {}
        self.ptr['volume_S8'] = np.array([0])
        self.ptr['volume_Li2S'] = np.array([1])
        self.ptr['mass_S8'] = np.array([2])
        self.ptr['mass_Li2S'] = np.array([3])
        index_start = self.ptr['mass_Li2S'][-1] + 1
        self.ptr['C_k_elyte'] = np.arange(index_start, index_start + elyte_obj.n_species)
        self.num_vars = self.ptr['C_k_elyte'][-1] + 1

def Solution_Vector_0(SV_idx, tank, solid):
    '''
    set the initial values in the solution vector based on the yaml input file
    '''
    SV_0 = np.zeros(SV_idx.num_vars)
    SV_0[SV_idx.ptr['volume_S8']] = solid.inputs['mass-S8']/solid.rho_S8
    SV_0[SV_idx.ptr['volume_Li2S']] = solid.inputs['mass-Li2S']/solid.rho_Li2S
    SV_0[SV_idx.ptr['mass_S8']] = solid.inputs['mass-S8']
    SV_0[SV_idx.ptr['mass_Li2S']] = solid.inputs['mass-Li2S']
    SV_0[SV_idx.ptr['C_k_elyte']] =  [species['C_k'] for species in tank.inputs['transport']['diffusion-coefficients']]

    return SV_0

def SV_0_from_outputs(SV_idx, SV_0_size, solution):
    '''
    Sets the initial values from the outputs from the previous solution
    '''
    SV_0 = np.zeros(SV_0_size)
    sim_outputs_y = np.transpose(solution.y)

    volume_S8, = sim_outputs_y[SV_idx.ptr['volume_S8']]
    volume_Li2S, = sim_outputs_y[SV_idx.ptr['volume_Li2S']]
    mass_S8, = sim_outputs_y[SV_idx.ptr['mass_S8']]
    mass_Li2S, = sim_outputs_y[SV_idx.ptr['mass_Li2S']]
    C_k_elyte = sim_outputs_y[SV_idx.ptr['C_k_elyte']]
    c_k_end = [i[-1] for i in C_k_elyte]
    c_k_end = np.abs(c_k_end)
    SV_0[SV_idx.ptr['volume_S8']] = volume_S8[-1]
    SV_0[SV_idx.ptr['volume_Li2S']] = volume_Li2S[-1]
    SV_0[SV_idx.ptr['mass_S8']] = mass_S8[-1]
    SV_0[SV_idx.ptr['mass_Li2S']] = mass_Li2S[-1]
    SV_0[SV_idx.ptr['C_k_elyte']] =  c_k_end

    return SV_0

def SV_0_from_data(SV_idx, SV_0_size, file_name):
    '''
    Sets the initial values from the end of a data set
    '''
    SV_0 = np.zeros(SV_0_size)
    folder_name = "Data/"+file_name
    os.makedirs(folder_name, exist_ok=True)
    fn = "Outputs.csv"
    fp = f"{folder_name}/{fn}"
    data = pd.read_csv(fp,dtype='float64')
    volume_S8 = data['volume_S8'].to_numpy()
    volume_Li2S = data['volume_Li2S'].to_numpy()
    mass_S8 = data['mass_S8'].to_numpy()
    mass_Li2S = data['mass_Li2S'].to_numpy()
    C_k_elyte = data.loc[:, 'TEGDME(e)':'Li2S(e)'].to_numpy()
    #########################################################################zsxdcfvgbhjnuiko
    # used to go from data whjen no s in the yaml
    #C_k_elyte = np.delete(C_k_elyte, SV_idx.elyte_species.index('S2(e)')+1, axis=1)
    c_k_end = C_k_elyte[-1,:]
    c_k_end = np.abs(c_k_end)
    SV_0[SV_idx.ptr['mass_S8']] = mass_S8[-1]
    SV_0[SV_idx.ptr['mass_Li2S']] = mass_Li2S[-1]
    SV_0[SV_idx.ptr['C_k_elyte']] =  c_k_end
    SV_0[SV_idx.ptr['volume_S8']] = volume_S8[-1]
    SV_0[SV_idx.ptr['volume_Li2S']] = volume_Li2S[-1]

    return SV_0