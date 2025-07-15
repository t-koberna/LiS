# LiS_main_new.py
#
# This file serves as the main model file.
#   It is called by the user to run the model
import cantera as ct
import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup import SV_pointer, Solution_Vector_0, Anode, Seperator, Parameters, Cathode
from LiS_residual import residual
from post_process import create_plots

# Decide if I am using algebric variables or not (I am workshopping two apraoches for the seperator)
algebraic = 12

# Read in the yaml input file
path = Path("Li_Sulfur.yaml")
yaml = YAML(typ='safe')
inputs = yaml.load(path)

params = Parameters(inputs)
sep = Seperator(path, inputs, params)
anode = Anode(path, inputs, sep, params)
cathode = Cathode(path, inputs, sep, params)

# Set the pointers
SV_idx = SV_pointer(path, sep, params)

# Set up the solution vector
SV_0 = Solution_Vector_0(SV_idx, sep, anode, cathode, params)

# set up and run the solver
time_start = 0 # Initial time [s]
time_end = params.inputs['simulations']['y'] #Final time [s]
tspan = [time_start,time_end]
if algebraic == True:
    algvars = SV_idx.ptr['phi_elyte'].tolist()
    print("algebraics")
else:
    algvars = []
    print("no algebraics")

num_roots = 1 # number of termination checks
def terminate_check(t,SV,SV_dot,return_val,user_data):
    return_val[0] =  SV[ SV_idx.ptr['thickness_an'][0]] # [0] is added to extract a single value from the 1x1 array and I avoid a warning

options =  {'userdata':(SV_idx, anode, sep, cathode, params, algebraic),
            'rtol':1e-4,'atol':1e-12,
            'algebraic_idx':algvars, 'first_step':1e-15,'eventsfn':terminate_check,'num_events':num_roots}

solver = sun.ida.IDA(residual, **options)
SV_dot_0  = np.zeros_like(SV_0)
solution = solver.solve(tspan, SV_0, SV_dot_0)

sim_outputs = np.stack((*np.transpose(solution.y), solution.t))
create_plots(SV_idx, sim_outputs, sep, anode, cathode, params)

params.i_ext = 0.1
options =  {'userdata':(SV_idx, anode, sep, cathode, params, algebraic),
            'rtol':1e-4,'atol':1e-12,
            'algebraic_idx':algvars, 'first_step':1e-15,'eventsfn':terminate_check,'num_events':num_roots}
SV_0 = solution.y[-1,:]
SV_dot_0  = np.zeros_like(SV_0)
solution = solver.solve(tspan, SV_0, SV_dot_0)

# process the simulation outputs
create_plots(SV_idx, sim_outputs, sep, anode, cathode, params)