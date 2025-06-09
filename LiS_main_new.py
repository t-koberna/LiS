# LiS_main_new.py
#
# This file serves as the main model file.  
#   It is called by the user to run the model

import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup import SV_pointer, Solution_Vector_0
from create_residual import residual
from post_process import create_plots

# Read in the yaml input file
path = Path("Li_Sulfur.yaml")
yaml = YAML(typ='safe')
inputs = yaml.load(path)

anode_inputs = inputs['cell-description']['anode']
sep_inputs = inputs['cell-description']['separator']
cathode_inputs = inputs['cell-description']['cathode']
parameters = inputs['parameters']

T, T_units =  parameters['T'].split()
T = float(T)
P, P_units =  parameters['P'].split()
P = float(P)
i_ext, i_ext_units = parameters['simulations']['i_ext'].split()
i_ext = float(i_ext)

# Set the pointers
SV_idx = SV_pointer(path, sep_inputs, parameters)   

# Set up the solution vector
SV_0 = Solution_Vector_0(SV_idx, sep_inputs, anode_inputs, cathode_inputs, parameters)

# set up and run the solver
time_start = 0 # Initial time [s]
time_end = parameters['simulations']['time_max'] #Final time [s]
tspan = [time_start,time_end]
algvars = []
num_roots = 1 # number of termination checks
def terminate_check(t,SV,SV_dot,return_val,user_data):
    return_val[0] = SV[0] - anode_inputs['thickness'] 
options =  {'userdata':(path, SV_idx, anode_inputs, sep_inputs, i_ext ), 
            'rtol':1e-3,'atol':1e-10, 
            'algebraic_idx':algvars, 'first_step':1e-15,'eventsfn':terminate_check,'num_events':num_roots}

solver = sun.ida.IDA(residual, **options)
SV_dot_0  = np.zeros_like(SV_0)
solution = solver.solve(tspan, SV_0, SV_dot_0)

# process the simulation outputs
sim_outputs =np.stack((*np.transpose(solution.y), solution.t))
create_plots(SV_idx, sim_outputs, sep_inputs)
