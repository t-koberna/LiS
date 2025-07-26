# create_residual.py
#
# This file creates the residual
import cantera as ct
import numpy as np

def residual(t,SV,SV_dot,resid,user_data):
    '''
    Current convention: A positive current is electrons flowing through the external circuit from the anode to the cathode (discharge).
    The anode is at zero potential. The state variables for the double layer are delta potential differences
        - delta_phi_dl_an = phi_elyte_an - phi_an
        - delta_phi_dl_ca = phi_ca - phi_elyte_ca
    I only use the electrolyte potentials in the migration term and they are deltas between nodes,
        so I chose to calculate them relative to themselves. That means they are all zero to start.
    '''
    SV_idx, i_ext, an, sep, ca, params, algebraic  = user_data

    n_elyte_nodes = sep.inputs['n_points']
    n_elyte_species = sep.elyte_obj.n_species

    ## Set the state of the Cantera objects based on SV
    # The potential of the conductor object is held at zero. I set the potential of the elyte
    #   near the electrode as the same as the potential of the double layer
    an.conductor_obj.electric_potential = 0.
    an.elyte_obj.electric_potential = SV[SV_idx.ptr['phi_dl_an']]
    an.elyte_obj.X = SV[SV_idx.ptr['C_k_elyte'][0:n_elyte_species]] # the concentrations in the first elyte node

    # The potential of the elyte in the interface is the anode double layer potential, plus the total
    #   drop across the separator (last node minus first node)
    # The potential of the host in the interface is the above potential plus del_phi_dl_ca
    ca.elyte_obj.X = SV[SV_idx.ptr['C_k_elyte'][n_elyte_species*(n_elyte_nodes-1):]] # the concentrations in the final elyte node
    ca.elyte_obj.electric_potential = 0
    ca.host_obj.electric_potential = SV[SV_idx.ptr['phi_dl_ca']]


    ## Anode
    # When the double layer is fully charged, faradic current is will be equal to the external current so I am
    #   using the same sign convention for both the external and faradic currents. A positive electron
    #   production rate means electrons are exiting the anode, which should correspond to a positive faradic current.
    # The production rates of Li(s) and e- should be equal and opposite
    sdot_electron_an = an.surf_obj.get_net_production_rates(an.conductor_obj) # rate for electrons, positive if produced
    sdot_Li_an = an.surf_obj.get_net_production_rates(an.bulk_obj) # rate lithium metal, negative if consumed [kmol/m^2-s]
    #print(f" Li {sdot_Li_an} e {sdot_electron_an}")

    # I tried to use "interface_current" to check against this but it wouldn't run
    i_far_an = ct.faraday*sdot_electron_an # [C/kmol]*[kmol/m^2-s] = [A/m^2]
    i_dl_an = i_ext - i_far_an # [A/m^2]
    c_dl_an = an.inputs['C_dl'] # [F/m^2]
    #print(i_far_an)

    # The delta_phi_dl gets larger when there is a positive double layer current
    resid[SV_idx.ptr['phi_dl_an']] = SV_dot[SV_idx.ptr['phi_dl_an']] + i_dl_an/c_dl_an # [A/m^2]/[F/m^2]=[C/s-m^2]*[V-m^2/C]=[V/s]
    # The anode gets thicker when Li(s) is being produced
    resid[SV_idx.ptr['thickness_an']] = \
          SV_dot[SV_idx.ptr['thickness_an']] - sdot_Li_an*an.bulk_obj.partial_molar_volumes #[kmol/m^2-s]*[m^3/kmol] = [m/s]

    ## Cathode
    # I put the code for the cathode before the code for the separator because I need the double layer current for
    #   conservation of charge
    # During discharge (positive case of the sign convention), electrons enter the cathode and are consumed.
    #   So, a negative rate of production of electrons should correspond to a positive faradic current.
    sdot_electron_ca = ca.surf_obj.get_net_production_rates(ca.host_obj) # rate electrons, positive if produced

    # Fraction of geometric surface area available for electrode-elyte reactions
    #  Placeholder. Should eventually be based on S8(s) and Li2S(s) coverage.
    A_frac = 1.

    i_far_ca = -ct.faraday * sdot_electron_ca * A_frac # [C/kmol]*[kmol/m^2-s] = [A/m^2]
    i_dl_ca = i_ext - i_far_ca # [A/m^2]
    c_dl_ca = ca.inputs['C_dl'] # [F/m^2]
    resid[SV_idx.ptr['phi_dl_ca']] = SV_dot[SV_idx.ptr['phi_dl_ca']]  + i_dl_ca/c_dl_ca

    ## Separator
    dC_k_elyte_dt, i_io = elyte_rates(SV_idx, an, ca, sep, params, i_dl_an, i_dl_ca, SV)
    resid[SV_idx.ptr['C_k_elyte']] = SV_dot[SV_idx.ptr['C_k_elyte']]  - dC_k_elyte_dt

    if algebraic == True:
        # I do not have anything to enforce concentration of charge for this approach. But
        #   It should do that implicitly when it solves the algebraic equations.
        #   I match the ionic current (i_io) in one node separator to the current in the previous node
        #   then set the current in the first node to be equal to the external current.
        resid[SV_idx.ptr['phi_elyte'][1:]] = i_io[1:] - i_io[:-1]
        resid[SV_idx.ptr['phi_elyte'][0]] = i_io[0] - i_ext
    else:
        # Differentiates Sigma z_k*C_k = 0. This approach enforces charge neutrality and
        #  also implicitly solves for the potentials. Kind of like using a DAE, but
        #  hopefully more stable.
        # Issue: I was not getting charge neutrality to be respected unless I put a big
        #  multiplier on the charges
        multiplier = 1e10
        # I reshape the dC_k_elyte_dt so the dot product with the species charges
        #   yields the sum of the change in charge of the ions in each node
        dC_k_elyte_dt_reshape = np.reshape(dC_k_elyte_dt, (n_elyte_nodes,n_elyte_species))
        resid[SV_idx.ptr['phi_elyte']] = \
              SV_dot[SV_idx.ptr['phi_elyte']] - multiplier*np.dot(dC_k_elyte_dt_reshape, sep.elyte_obj.charges)

    # Not addressed yet, once dilute solution theory is working, I will add the code for nucleation
    #   and growth. I will switch to concentrated solution theory after that.
    resid[SV_idx.ptr['Li2S']] = SV_dot[SV_idx.ptr['Li2S']]
    resid[SV_idx.ptr['S8']] = SV_dot[SV_idx.ptr['S8']]
    resid[SV_idx.ptr['bm_Li2S']] = SV_dot[SV_idx.ptr['bm_Li2S']]
    resid[SV_idx.ptr['bm_S8']] = SV_dot[SV_idx.ptr['bm_S8']]

def elyte_rates(SV_idx, an, ca, sep, params, i_dl_an, i_dl_ca, SV):
    '''
    Find the for the species fluxes in the electrolyte
    Currently using dilute solution theory
    '''
    n_elyte_nodes = sep.inputs['n_points']
    n_elyte_species = sep.elyte_obj.n_species
    N_k_elyte = np.zeros((n_elyte_nodes+1)*n_elyte_species)

    dy = sep.inputs['thickness']/n_elyte_nodes # the thickness of each node

    # a positive double layer current should take Li+ ions away from from the elyte
    # These two are equal and opposite when the current is zero and that makes sense to me because there is no net current

    # N_k_elyte is calculated at the boundary of each control volume. The first
    #  boundary is the anode surface, so there is no ion flux across there (only
    #  species production). The next boundaries are halfway between nodes and the
    #  species flux across those boundaries are calculated in the for loop below.
    R = ct.gas_constant
    F = ct.faraday
    # This loop finds the flux of each species one by one, for the boundaries between
    #  the nodes:
    #TODO #5
    for i in range((n_elyte_nodes-1)*n_elyte_species):
        current_node = int(i/n_elyte_species)
        species = SV_idx.elyte_species[i] # used to find the charge
        species_int = int(i%n_elyte_species) # used to find the diffusion coeff

        D_k = sep.inputs['transport']['diffusion-coefficients'][species_int]['D_k']
        z_k = float(sep.elyte_obj[species].charges)

        # Electric potential gradient
        del_phi = (SV[SV_idx.ptr['phi_elyte'][current_node+1]]
                   - SV[SV_idx.ptr['phi_elyte'][current_node]])/dy

        # Concentration at the boundary, taken by averaging at node centers:
        C_k = (SV[SV_idx.ptr['C_k_elyte'][i]]
               + SV[SV_idx.ptr['C_k_elyte'][i+n_elyte_species]])/2

        # logic check, If the next node has a larger potential, a positive ion will have
        #  a negative flux
        migration = - z_k*D_k*F*C_k*del_phi/(R*params.T)

        # Concentration gradient:
        del_C = (SV[SV_idx.ptr['C_k_elyte'][i+n_elyte_species]]
                 - SV[SV_idx.ptr['C_k_elyte'][i]])/dy

        # logic check, if the next node has a larger concentration, the species will
        #  have a negative flux
        diffusion = - D_k*del_C

        # assumes the bulk velocity of the fluid is zero
        convection = 0

        #print(diffusion)
        N_k_elyte[i+n_elyte_species] = (migration + diffusion + convection)

    i_io = np.zeros(n_elyte_nodes) # the current due to ion transport at each node
    for i in range(n_elyte_nodes):
        # the flux crossing the left boundary, for all species in one node
        entering = N_k_elyte[i*n_elyte_species:i*n_elyte_species+n_elyte_species]
        # the flux crossing the right boundary
        exiting = N_k_elyte[(i+1)*n_elyte_species:(i+1)*n_elyte_species+n_elyte_species]

        # for a 1-D model, the gradient is a single partial derivative. This partial is delta_N_k divided by dy
        # This sets the gradient for all species in one node
        # grad_N_k_node_o[i*n_elyte_species:i*n_elyte_species+n_elyte_species] = (entering - exiting)/dy

        # Ionic current = sum(z_k*N_k*F) for the node. Only based on transport, not species production
        i_io[i] = F*np.dot((entering - exiting),sep.elyte_obj.charges)

    grad_N_k_node = (N_k_elyte[:-n_elyte_species] - N_k_elyte[n_elyte_species:])/dy

    # print(grad_N_k_node - grad_N_k_node_o)
    #print(i_io)
    # Account for surface and bulk reactions
    s_dot = surface_production(SV, SV_idx, an, ca, sep, n_elyte_nodes, n_elyte_species)
    omega_dot = bulk_production(SV,  SV_idx, sep, n_elyte_nodes, n_elyte_species)
    dC_k_elyte_dt = grad_N_k_node + omega_dot + s_dot

    # Account for ions entering/leaving the double layer
    #TODO #4
    dC_k_elyte_dt[0] =  dC_k_elyte_dt[0] + i_dl_an/ct.faraday
    dC_k_elyte_dt[n_elyte_species*(n_elyte_nodes-1)] =  \
        dC_k_elyte_dt[n_elyte_species*(n_elyte_nodes-1)] - i_dl_ca/ct.faraday

    return dC_k_elyte_dt, i_io

def surface_production(SV, SV_idx, an, ca, sep, n_elyte_nodes, n_elyte_species):
    '''
    Species production on an electrode surface [kmol/m^2-s]
    '''
    s_dot = np.zeros_like(SV[SV_idx.ptr['C_k_elyte']])
    # surface production at the anode
    s_dot[0:n_elyte_species] = an.surf_obj.get_net_production_rates(an.elyte_obj)
    # surface production at the cathode
    s_dot[(n_elyte_nodes-1)*n_elyte_species:n_elyte_nodes*n_elyte_species] = \
        ca.surf_obj.get_net_production_rates(ca.elyte_obj)

    return s_dot

def bulk_production(SV,  SV_idx, sep, n_elyte_nodes, n_elyte_species):
    '''
    Species production in the bulk solution [kmol/m^3-s]
    '''
    omega_dot = np.zeros_like(SV[SV_idx.ptr['C_k_elyte']])
    # Set the concentrations for the nodes in the electrolyte
    for i in range(n_elyte_nodes):
        sep.elyte_obj.X = \
            SV[SV_idx.ptr['C_k_elyte'][i*n_elyte_species:(i+1)*n_elyte_species]]
        omega_dot[i*n_elyte_species:(i+1)*n_elyte_species] = \
            sep.elyte_obj.net_production_rates
    return omega_dot