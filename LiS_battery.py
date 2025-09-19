# LiS_main_new.py
#
# This file serves as the main model file.
#   It is called by the user to run the model
import cantera as ct
import numpy as np
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from setup import SV_pointer, Solution_Vector_0, Anode, Separator, Parameters, Cathode
from residual import residual
from post_process import create_plots

# Decide if I am using algebraic variables or not (I am work shopping two approaches for the separator)
# 1 uses algebraics, any other number does not use algebraics
algebraic = 1

#==================================================================================================
#
#   Set the options and tolerances for the solver   
#
#==================================================================================================

rtol = 1e-4                                                         # Relative tolerance
atol = 1e-12                                                        # Absolute tolerance
first_step = 1e-15                                                  # Size of the initial time step


#==================================================================================================
#
#   Read in the input file, create the separator and electrode objects.
#   Create the pointer and set up the solution vector
#
#==================================================================================================

# Read in the yaml input file
path = Path("Li_Sulfur.yaml")
yaml = YAML(typ='safe')
inputs = yaml.load(path)

params = Parameters(inputs)
# The Cantera objects are initialized during the creation of the following
sep = Separator(path, inputs, params)
anode = Anode(path, inputs, sep, params)
cathode = Cathode(path, inputs, sep, params)

# Set the pointers
SV_idx = SV_pointer(path, sep, params)

# Set up the solution vector
SV_0 = Solution_Vector_0(SV_idx, sep, anode, cathode, params)


#==================================================================================================
#
#   Set termination checks and other simulation options then run the solver.
#   There is an option to run an equilibrating step before charge or discharge.
#
#==================================================================================================

# set up and run the solver
time_start = 0                                                      # Initial time [s]
time_end = params.inputs['simulations']['time_max']                        # Final time [s]

if algebraic == True:
    algvars = SV_idx.ptr['phi_elyte'].tolist()
    print("algebraics")
else:
    algvars = []
    print("no algebraics")

num_roots = 1                                                       # number of termination checks
def terminate_check(t,SV,SV_dot,return_val,user_data):
    return_val[0] =  SV[ SV_idx.ptr['thickness_an'][0]]             # [0] is added to extract a single value from the 1x1 array and I avoid a warning

# Zero current equilibrium step (optional)
if params.inputs['simulations']['equilibrate']['enable']:
    
    time_eq = params.inputs['simulations']['equilibrate']['time']
    print(f"Equilibrating for {time_eq} seconds")
    i_ext = 0.0
    tspan = [time_start,time_eq]
    options =  {'userdata':(SV_idx, i_ext, anode, sep, cathode, params, algebraic),
                'rtol':rtol,'atol':atol,
                'algebraic_idx':algvars, 'first_step':first_step,
                'eventsfn':terminate_check,'num_events':num_roots}

    solver = sun.ida.IDA(residual, **options)
    SV_dot_0  = np.zeros_like(SV_0)
    solution = solver.solve(tspan, SV_0, SV_dot_0)

    sim_outputs = np.stack((*np.transpose(solution.y), solution.t))

    # Initialize the solution vector as the last values from the equilibration
    SV_0 = solution.y[-1,:]
    if params.inputs['simulations']['equilibrate']['plot_flag']:
        create_plots(SV_idx, sim_outputs, sep, anode, cathode, params)

# Run the main simulation
tspan = [time_start,time_end]
i_ext = params.i_ext
options =  {'userdata':(SV_idx, i_ext, anode, sep, cathode, params, algebraic),
            'rtol':rtol,'atol':atol,
            'algebraic_idx':algvars, 'first_step':first_step,
            'eventsfn':terminate_check,'num_events':num_roots}
SV_dot_0  = np.zeros_like(SV_0)
solver = sun.ida.IDA(residual, **options)
solution = solver.solve(tspan, SV_0, SV_dot_0)
sim_outputs = np.stack((*np.transpose(solution.y), solution.t))


#==================================================================================================
#
#   Process the simulation outputs. Generate plots (later on I will include a way to save the data)
#
#==================================================================================================

create_plots(SV_idx, sim_outputs, sep, anode, cathode, params)