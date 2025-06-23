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
from create_residual import residual
from post_process import create_plots

# Read in the yaml input file
path = Path("Li_Sulfur.yaml")
yaml = YAML(typ='safe')
inputs = yaml.load(path)

params = Parameters(inputs)
sep = Seperator(path, inputs)
anode = Anode(path, inputs, sep, params)
cathode = Cathode(path, inputs, sep, params)

'''
# Note on equalibrium calculation
print(f"G_Li+(e) = {anode.elyte_obj['Li+(e)'].gibbs_mole/1000/1000} [kJ/mole], I was expecting -278")
print("The value I back calculated that from the final voltage is 278.049 so I am now more confused")
print(f"G_Li(b) = {anode.bulk_obj['Li(b)'].gibbs_mole/1000} [kJ/mole] which is 298.28*0.0291 as expected")
'''

# Set the pointers
SV_idx = SV_pointer(path, sep, params)   

# Set up the solution vector
SV_0 = Solution_Vector_0(SV_idx, sep, anode, cathode, params)

# set up and run the solver
time_start = 0 # Initial time [s]
time_end = params.inputs['simulations']['time_max'] #Final time [s]
tspan = [time_start,time_end]
algvars = SV_idx.ptr['phi_elyte'].tolist()
algvars = []

num_roots = 1 # number of termination checks
def terminate_check(t,SV,SV_dot,return_val,user_data):
    return_val[0] =  SV[ SV_idx.ptr['thickness_an'][0] ]
options =  {'userdata':(SV_idx, anode, sep, cathode, params), 
            'rtol':1e-8,'atol':1e-15, 
            'algebraic_idx':algvars, 'first_step':1e-15,'eventsfn':terminate_check,'num_events':num_roots}

solver = sun.ida.IDA(residual, **options)
SV_dot_0  = np.zeros_like(SV_0)
solution = solver.solve(tspan, SV_0, SV_dot_0)

# process the simulation outputs
sim_outputs =np.stack((*np.transpose(solution.y), solution.t))
create_plots(SV_idx, sim_outputs, sep)


