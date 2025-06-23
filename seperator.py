"""
    seperator.py
    Class for the seprator
"""
import cantera as ct
import numpy as np

class seperator:
    '''
    create an object for the anode
    '''
    def __init__(self,input_file,inputs,species_name):
        self.thickness = 0
        self.n_points = 5
        self.n_variables = 5 + 5*2
        self.pointer = {}
        self.pointer['phi_ed'] = np.arange(0, 5)
        self.pointer['C_k_elyte'] = np.arange(5, 5+10)

    def initialize(self, inputs):
        SV = np.zeros([self.n_variables])
        return SV
