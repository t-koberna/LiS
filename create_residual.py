# create_residual.py
#
# This file creates the residual
import cantera as ct
import numpy as np

def residual(t,SV,SV_dot,resid,user_data):
    '''
    
    '''
    SV_idx, an, sep, ca, params, algebraic  = user_data

    n_elyte_nodes = sep.inputs['n_points']
    n_elyte_species = sep.elyte_obj.n_species

    #### Set the state of the Cantera objects based on SV
    # the potential of the eleyte near the electrode is the same as the potential of the double layer
    an.elyte_obj.electric_potential = SV[SV_idx.ptr['phi_dl_an']] 
    an.elyte_obj.X = SV[SV_idx.ptr['C_k_elyte'][0:n_elyte_species]] # the concetrations in the first elyte node

    # The potential at the cathode is the last node in the elyte relative to the del_phi_dl_ca
    # for now I am using the anode double layer since I do not have the potentials in the seperator working
    ca.elyte_obj.electric_potential = SV[SV_idx.ptr['phi_dl_ca']] #SV[SV_idx.ptr['phi_elyte'][-1]] + SV[SV_idx.ptr['phi_dl_ca']]
    ca.elyte_obj.X = SV[SV_idx.ptr['C_k_elyte'][n_elyte_species*(n_elyte_nodes-1):]] # the concetrations in the final elyte node

    ## Anode
    sdot_electron_an = an.surf_obj.get_net_production_rates(an.conductor_obj) # rate electrons, positive if produce
    sdot_Li_an = an.surf_obj.get_net_production_rates(an.bulk_obj) # rate litium metal, negative if consumed [kmol/m^2-s]
    sdot_Li_ion_an = -sdot_Li_an # the ion is on the oposite side of the chemical equation so it has the opposite rate
    #print(f" Li {sdot_Li_an} e {sdot_electron_an}")

    # I tried to use "interface_current()" to check against this but it wouldn't run
    # Faradaic current density is positive when electrons are consumed
    i_far_an = -ct.faraday*sdot_electron_an # [C/kmol]*[kmol/m^2-s] = [A/m^2]
    i_dl_an = params.i_ext - i_far_an # [A/m^2]
    c_dl_an = an.inputs['C_dl'] # [F/m^2]

    resid[SV_idx.ptr['phi_dl_an']] = SV_dot[SV_idx.ptr['phi_dl_an']] - i_dl_an/c_dl_an # [A/m^2]/[F/m^2]=[C/s-m^2]*[V-m^2/C]=[V/s]
    resid[SV_idx.ptr['thickness_an']] = SV_dot[SV_idx.ptr['thickness_an']] - sdot_Li_an*an.bulk_obj.partial_molar_volumes #[kmol/m^2-s]*[m^3/kmol] = [m/s]

    ## Seperator
    dC_k_elyte_dt, i_io = Dilute_Solution_Theory(SV_idx, an, sep, params, sdot_Li_ion_an, i_dl_an, SV)
    resid[SV_idx.ptr['C_k_elyte']] = SV_dot[SV_idx.ptr['C_k_elyte']]  - dC_k_elyte_dt

    if algebraic == True:
        # I do not have anything to enforce concentraion of charge for this aproach.
        # matches the ionic current (i_io) in one node seperator to the current in the previous nodee
        #   then sets the current in the first node to be equal to the external current. 
        resid[SV_idx.ptr['phi_elyte'][1:]] = i_io[1:] - i_io[:-1] 
        resid[SV_idx.ptr['phi_elyte'][0]] = i_io[0] - params.i_ext #SV_dot[SV_idx.ptr['phi_elyte']] 
    else: 
        # Differentiates Sigma z_k*C_k = 0. This aproach enforces charge nuetrality and also implicitly
        #   solves for the potentials. Kind of like using a DAE, but hopefully more stable. 
        # Issue: I was not getting charge neutrality to be respected unless I put a big multiplyer on the charges
        multiplyer = 1e5
        # I reshape the dC_k_elyte_dt so the dot product with the species charges yeilds the sum of the 
        #   change in charge of the ions in each node
        dC_k_elyte_dt_reshape = np.reshape(dC_k_elyte_dt, (n_elyte_nodes,n_elyte_species))
        resid[SV_idx.ptr['phi_elyte']] = SV_dot[SV_idx.ptr['phi_elyte']] - multiplyer*np.dot(dC_k_elyte_dt_reshape,   sep.elyte_obj.charges)
    
    # If my solution is failing at t=0, I use this so I can trouble shoot the rest
    #resid[SV_idx.ptr['phi_elyte']] = SV_dot[SV_idx.ptr['phi_elyte']]
        
    ## Cathode

    # Problem: the production rate at the cathode is massive and it is breaking the solver. 
    '''
    print(ca.elyte_obj.electric_potential)
    print(SV[SV_idx.ptr['C_k_elyte'][n_elyte_species*(n_elyte_nodes-1):]])
    print(ca.surf_obj.get_net_production_rates(ca.host_obj))
    print(ca.surf_obj.get_net_production_rates(ca.elyte_obj))
    print(an.surf_obj.get_net_production_rates(an.bulk_obj))
    print(an.surf_obj.get_net_production_rates(an.conductor_obj))
    print(an.surf_obj.get_net_production_rates(an.elyte_obj))
    '''
    sdot_electron_ca = ca.surf_obj.get_net_production_rates(ca.host_obj) # rate electrons, positive if produce
    i_far_ca = ct.faraday*sdot_electron_ca # [C/kmol]*[kmol/m^2-s] = [A/m^2]
    i_dl_ca = params.i_ext - i_far_ca # [A/m^2]
    c_dl_ca = ca.inputs['C_dl'] # [F/m^2]
    #print(i_far_ca)
    #print(i_dl_ca/c_dl_ca)
    resid[SV_idx.ptr['phi_dl_ca']] = SV_dot[SV_idx.ptr['phi_dl_ca']]  #- i_dl_ca/c_dl_ca
    # not sure I need a varible for this because it cam be calculated in post processing by adding the delta_ph_dl to the phi_elyte 
    resid[SV_idx.ptr['phi_ca']] = SV_dot[SV_idx.ptr['phi_ca']]

    # Not adressed yet
    resid[SV_idx.ptr['Li2S']] = SV_dot[SV_idx.ptr['Li2S']]
    resid[SV_idx.ptr['S8']] = SV_dot[SV_idx.ptr['S8']]
    resid[SV_idx.ptr['bm_Li2S']] = SV_dot[SV_idx.ptr['bm_Li2S']]
    resid[SV_idx.ptr['bm_S8']] = SV_dot[SV_idx.ptr['bm_S8']]
            
def Dilute_Solution_Theory(SV_idx, an, sep, params, sdot_Li_ion_an, i_dl_an, SV):
    '''
    Find the for the species fluxes in the electroltye
    '''
    n_elyte_nodes = sep.inputs['n_points']
    n_elyte_species = sep.elyte_obj.n_species
    N_k_elyte = np.zeros((n_elyte_nodes+1)*n_elyte_species)

    dy = sep.inputs['thickness']/n_elyte_nodes # the thickness of each node

    # a positive double layer current should take Li+ ions away from from the elyte
    # These two are equal and opposite when the current is zero and that makes sense to me because there is no net current
    # Note: not on that scale, no need to track Li in the double layer? If so, ignore above
    
    # N_k_elyte is calculated at the boundry of each control volume. The first boundry is the anode surface, so there is no
    #   ion flux across there (only sepecies production). The next boundries are halfway between nodes and the species flux 
    #   across those boundries are calculated in the for loop below. 
    R = ct.gas_constant
    F = ct.faraday
    # This loop finds the flux of each species one by one, for the boundries between the middle nodes
    for i in range((n_elyte_nodes-1)*n_elyte_species): # The middle nodes
        current_node = int(i/n_elyte_species)
        species = SV_idx.elyte_species[i] # used to find the charge
        species_int = int(i%n_elyte_species) # used to find the defusion coeff

        D_k = sep.inputs['transport']['diffusion-coefficients'][species_int]['D_k']
        z_k = float(sep.elyte_obj[species].charges)
        # the difference in voltage between the two nodes
        del_phi = (SV[SV_idx.ptr['phi_elyte'][current_node+1]] - SV[SV_idx.ptr['phi_elyte'][current_node]])/dy
        # the average concentration is used for the concentration halfway between nodes
        C_k = (SV[SV_idx.ptr['C_k_elyte'][i]] + SV[SV_idx.ptr['C_k_elyte'][i+n_elyte_species]])/2
        # logic check, If the next node has a larger potential, a positve ion will have a negative flux
        migration = - z_k*D_k*F*C_k*del_phi/(R*params.T)

        del_C = (SV[SV_idx.ptr['C_k_elyte'][i+n_elyte_species]] - SV[SV_idx.ptr['C_k_elyte'][i]])/dy
        # logic check, if the next node has a larger concentration, the species will have a negative flux
        diffusion = -D_k*del_C

        # assumes the bulk velocity of the fluid is zero
        convection = 0

        N_k_elyte[i+n_elyte_species] = (migration + diffusion + convection)

    grad_N_k_node = np.zeros((n_elyte_nodes)*n_elyte_species) #the gradient of the species fluxes for each species, at each node
    i_io = np.zeros(n_elyte_nodes) # the current due to ion transport at each node
    for i in range(n_elyte_nodes):
        # the flux crossing the left boundry, for all species in one node
        entering = N_k_elyte[i*n_elyte_species:i*n_elyte_species+n_elyte_species]
        # the flux crossing the right boundry
        exiting = N_k_elyte[(i+1)*n_elyte_species:(i+1)*n_elyte_species+n_elyte_species]
 
        # for a 1-D model, the gradient is a sigle partial derivative. This partial is delta_N_k divided by dy
        # This sets the gradient for all species in one node
        grad_N_k_node[i*n_elyte_species:i*n_elyte_species+n_elyte_species] = (entering - exiting)/dy
        
        # Ionic current = sum(z_k*N_k*F) for the node
        i_io[i] = F*np.dot((entering - exiting),sep.elyte_obj.charges)

    ## add in the code for the surface and bulk productions here
    dC_k_elyte_dt = grad_N_k_node #+ omega_dot + s_dot 

    dC_k_elyte_dt[0] =  dC_k_elyte_dt[0] + sdot_Li_ion_an # - i_dl_an/ct.faraday # **** Should I include the ions leaving solution to exter the double layer?

    # make sure lithium is being consumed at the other end (the cathode production species are broken right now so I force this to be symetric with the anode)
    dC_k_elyte_dt[n_elyte_species*(n_elyte_nodes-1)] =  dC_k_elyte_dt[n_elyte_species*(n_elyte_nodes-1)] - sdot_Li_ion_an #+ i_dl_an/ct.faraday 

    # add in bulk production and surface production rates
    # first node in the elyte is Lithium
    # Charging the double layer removes Li+ from the electrolyte : - i_dl/ct.faraday [A/m^2]/[C/kmol] = [kmol/m^2-s]
    # The rate Li+ enters the electrolyte has the opposite sign of the rate Li+ in the anode : + sdot_Li [kmol/m^2-s]
    # How do I add in bulk production in the electrolyte if the electrochemical reactions are only happening at the surfaces
    #   do I add in no electrochemical reactions? (I am assuming that electrons cannot be a species in my elyte)
    #   In the past it seems like the elyte only had transport in the middle nodes
    
    #dC_k_elyte_dt[0] = dC_k_elyte_dt[0] + sdot_Li - i_dl/ct.faraday # sould I add this at the end or before the N_k?

    # I have not set the fluxes for the species entering the cathode or interacting with that double layer yet 
    '''
    node = -1 # starts at negative 1 so it gets updated to 0 in the first pass of the loop
    for i in range((n_elyte_nodes)*n_elyte_species):
        if i % n_elyte_species == 0: # sets the state of the cantera object each time I move to the next node
            node = node + 1
            sep.elyte_obj.electric_potential = SV[SV_idx.ptr['phi_elyte'][node]] 
            sep.elyte_obj.X = SV[SV_idx.ptr['phi_dl_an'][node*n_elyte_species:(node+1)*n_elyte_species]]
        omega = sep.elyte_obj.get_net_production_rates()
    ''' 
    return dC_k_elyte_dt, i_io
    