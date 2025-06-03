"""
    cathode.py
    Class for the cathode
"""
import cantera as ct
import numpy as np

class cathode:
    '''
    create an object for the anode
    '''
    def __init__(self,input_file,inputs,species_name):
        self.thickness = 0
        self.n_variables = 3
        self.pointer = {}
        self.pointer['phi_ed'] = np.array([0])
        self.pointer['phi_dl'] = np.array([1])
        self.pointer['C_k_elyte'] = np.arange(2, 2 + 2)

    def initialize(self, inputs):
        SV = np.zeros([self.n_variables])
        return SV
