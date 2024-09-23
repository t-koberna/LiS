# LiS_main.py
#
# This file serves as the main model file.  
#   It is called by the user to run the model
import numpy as np
import matplotlib.pyplot as plt
from scikits.odes import dae
from LiS_functions import bucket, Index_start, residual_case1

# Anode is on the left at x=0 and Cathode is on the right
# Li -> Li+ + e- (reaction at the anode)
# 1/2S_8 + e- -> 1/2S_8^2- (reaction at the cathode)

'''
Constants
''' 
F = 96485.34 #Faraday's number [C/mol_electron]
R = 8.3145 #Universal gas constant [J/mol-K]

'''
USER INPUTS
'''
## Simulation parameters
V_min = -10 # Minimum cell voltage at which to terminate the integration [V]
V_max = 10 # Maximum cell voltage at which to terminate the integration [V]
X_Li_min = 0.01 # Lithium mole fraction in an electrode at which to terminate the integration [-]
X_Li_max = 0.99 # Lithium mole fraction in an electrode at which to terminate the integration [-]

## Operating Conditions
# If there is more than one external current the simulation will run back to back
i_external = np.array([100]) # external current into the Anode [A/m^2] 
t_sim_max = [1] # the maximum time the battery will be held at each current [s]
T = 298.15 # standard temperature [K]

## Material Properties
rho_C = 2260 # density of carbon [kg/m^3]
rho_S8 = 2070 # density of Sulfur (S8) [kg/m^3]

## Initial Values
Phi_dl_0_an = -0.64 # initial value for Phi_dl for the Anode [V]
Phi_dl_0_ca = 0.34 #0.321 # initial value for Phi_dl for the Cathode [V]
X_S8_0_ca = 0.6 # Initial Mole Fraction of the Cathode for Sulfur (S_8) [-]
C_Li_plus = 1000 # Concentration of Li+ in the Electrolyte [mol/m^3] 
C_std = 1000 # Standard Concentration [mol/m^3] (same as 1 M)
m_S8_0 = 0.1 # Initial mass of Sulfur [kg/m^2]
w_S8_0 = 0.2 # Initial weight percent of the Sulfur (S_8) Phase [kg_C/kg_total]
w_C_0 = 0.2 # Initial weight percent of the Carbon Phase [kg_C/kg_total]

## Material parameters: (Replaced by Cantera?)
sigma_sep = 1.2 # Ionic conductivity for the seperator [1/m-ohm] (this is concentration dependent but it is constant for now)
# Standard Gibbs free energy of formation [J/mol] (need to update)
G_S8 = 40000 
G_S8_min = 40000 
G_Li = -230000
G_Li_plus = -293300
# Standard entropy [J/mol-K]
S_S8 = 4
S_S8_min = 5
S_Li = -11.2
S_Li_plus = 49.7
# Microstructure
t_sep =  1e-5 # thickness of the seperator [m]
n_y_sep = 4 # number of nodes inside the seperator
Delta_y_an = 50*10**-6 # Anode thickness [m]
#r_an = 5*10**-6 # Anode particle radius [m]
#n_r_an = 5 # Number of radial nodes in the anode
Delta_y_ca = 50*10**-6 # Cathode thickness [m]
r_ca = 3*10**-6 # Cathode particle radius [m]
n_r_ca = 5 # Number of radial nodes in the cathode

'''
Parameters
'''
Epsilon_C =  w_C_0*m_S8_0/w_S8_0/rho_C/Delta_y_ca# Volume fraction of the carbon in the cathode [-] (will not change)
Epsilon_S8_0 = m_S8_0/rho_S8/Delta_y_ca # Initial volume fraction of S_8 in the cathode [-]
Epsilon_eltye = 1 - Epsilon_C - Epsilon_S8_0 # Initial volume fraction of electrolyte in the cathode [-]

## Geometry
# Anode


# Cathode
Vol_ca = (4*np.pi/3)*r_ca**3 # Volume of a single anode particle [m^3]
A_surf_ca = 4*np.pi*r_ca**2 # geometric surface area of a single anode particle [m^2]
A_s_ca = 3/r_ca # ratio of surface area to volume for the graphite anode [1/m]
#replace equation
#A_sg_ca = Epsilon_C*Delta_y_ca*A_s_ca # interface surface area per geometric surface area [m^2_interface/m^2_geometric]\

'''
Above are general inputs I will use later on, for now I am starting with a reduced case
'''

h = 1e-4 # height of the tank [m] (not used in case 1)

# Each of the buckets are the same size, with the exception of the final bucket which will extend to infinity
n_bucket_S8 = 4 # number of buckets for S8 [-]
n_bucket_Li2S = 5 # number of buckets for Li2S [-]

t_bucket_S8 = 1e-8 # the radius range (aka thickness) of each bucket for S8 [m]
t_bucket_Li2S = 2e-8 # the radius range (aka thickness) of each bucket for Li2S [m]

bucket_S8 = bucket(n_bucket_S8,t_bucket_S8,"S_8")
bucket_Li2S = bucket(n_bucket_Li2S,t_bucket_Li2S,"Li_2S")

SV_index = Index_start(n_bucket_S8,n_bucket_Li2S)

'''
Initialize the SV vector
'''
sim_inputs = np.zeros(n_bucket_S8 + n_bucket_Li2S)
print(sim_inputs[:SV_index.S8])
# I put S8 on top of Li2S, everything starts at zero for now
sim_inputs[:SV_index.S8] = np.zeros(n_bucket_S8)
sim_inputs[SV_index.S8:SV_index.Li2S] = np.zeros(n_bucket_Li2S)

time_start = 0 # Initial time [s]
time_end = t_sim_max[0] #Final time [s]
times = np.linspace(time_start,time_end,1000)

'''
Integration
'''
S8_limit = 1e-3
Li2S_limit = 1e-3
# Integration Limits 
n_roots = 3 # number of termination checks
def terminate_check(t,SV,SV_dot,return_val,user_data):
    #checks the number of particles in the largest boxes
    return_val[0] = SV[SV_index.S8-1] - S8_limit
    return_val[1] = SV[-1]- Li2S_limit
    neg_check = 1
    for ele in SV:
        if ele < 0:
            neg_check = 1
    return_val[2] = neg_check 
    
    
# I am using a DAE solver, but for now there are no algebraic equations
algvars = []

# I am not sure if these will be params or if the residual can call cantera directly
#[s_k_nuc_S8,s_k_grow_S8,s_k_nuc_Li2S,s_k_grow_Li2S] are the first 4 terms in params [mol/m^3]
params = [0,0,0.008,0.2, SV_index, bucket_S8, bucket_Li2S]
options =  {'user_data':params, 'rtol':1e-8,
        'atol':1e-12, 'algebraic_vars_idx':algvars, 'first_step_size':1e-15,'rootfn':terminate_check,'nr_rootfns':n_roots}
            # , 'compute_initcond':'yp0', 'max_steps':10000}
solver = dae('ida', residual_case1, **options)

SV_0 = sim_inputs
SV_dot_0  = np.zeros_like(SV_0)
solution = solver.solve(times, SV_0, SV_dot_0)
sim_outputs =np.stack((*np.transpose(solution.values.y), solution.values.t))

'''
Post Processing
'''
N_S8  = sim_outputs[:SV_index.S8]
N_Li2S  = sim_outputs[SV_index.S8:SV_index.Li2S] 
time = sim_outputs[-1]

####RGHUIJPOUBVGYCF VBUIONMUVGYCBIOMP< Change this only for error checking
sim_outputs =np.stack((*np.transpose(solution.values.yp), solution.values.t))
N_S8  = sim_outputs[:SV_index.S8]

#print(N_Li2S)
'''
plot the results 
'''

fig1, (ax1, ax2) = plt.subplots(2)
for ind, ele in enumerate(N_S8):
    ax1.plot(time,ele,label=str(ind))
for ind, ele in enumerate(N_Li2S):
    ax2.plot(time,ele,label=str(ind))
ax1.legend(ncol=1, bbox_to_anchor=(1, 0.5),loc = 'center left')
ax1.set_title(r"S$_8$")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("Number of Particles [-]")
ax2.legend(ncol=1, bbox_to_anchor=(1, 0.5),loc = 'center left')
ax2.set_title(r"Li$_2$S")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("Number of Particles [-]")
fig1.tight_layout()

plt.show()