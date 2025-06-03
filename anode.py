"""
    anode.py
    Class for the anode
"""
import cantera as ct
import numpy as np

class anode:
    '''
    create an object for the anode
    '''
    def __init__(self,input_file,inputs,species_name):
        self.thickness = 0
        self.n_variables = 5
        self.pointer = {}
        self.pointer['phi_ed'] = np.array([0])
        self.pointer['phi_dl'] = np.array([1])
        self.pointer['thickness'] = np.array([2])
        self.pointer['C_k_elyte'] = np.arange(3, 3 + 2)

    def initialize(self, inputs):
        SV = np.zeros([self.n_variables])
        return SV

