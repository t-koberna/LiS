import cantera as ct
import numpy as np

def residual(t,SV,resid,user_data):
    SV_idx, tank, solids, params, elyte_volume, max_rate, min_rate = user_data

    # update the concentrations in the elyte objects
    #C_k = np.maximum(SV[SV_idx.ptr['C_k_elyte']],0.0)
    C_k = SV[SV_idx.ptr['C_k_elyte']]
    #print(C_k)
    C_total = np.sum(C_k)
    X_k = C_k/C_total
    tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
    solids.elyte_obj.TPX = solids.elyte_obj.T, solids.elyte_obj.P, X_k

    #tank.elyte_obj.X = X_k
    #solids.elyte_obj.X = X_k

    ## Elyte concentration
    # bulk production
    #switch_factor = 1#0.5 * (1.0 + np.tanh((C_k - 1e-5) / 1e-6))
    #q_dot = tank.elyte_obj.net_production_rates*switch_factor

    # surface production
    min_mass = 1e-17
    mass_S8 = np.maximum(SV[SV_idx.ptr['mass_S8']], min_mass)
    mass_Li2S = np.maximum(SV[SV_idx.ptr['mass_Li2S']], min_mass)

    #mass_S8 = SV[SV_idx.ptr['mass_S8']]
    #mass_Li2S = SV[SV_idx.ptr['mass_Li2S']]

    A_surf_S8 = 2*np.pi*((3*solids.rho_S8/(2*np.pi*(mass_S8+min_mass)))**(2/3))
    s_dot_elyte_S8 = solids.surf_S8_obj.get_net_production_rates(solids.elyte_obj)

    A_surf_Li2S = 2*np.pi*((3*solids.rho_Li2S/(2*np.pi*(mass_Li2S+min_mass)))**(2/3))
    s_dot_elyte_Li2S = solids.surf_Li2S_obj.get_net_production_rates(solids.elyte_obj)

    
    reactants_nu = tank.elyte_obj.reactant_stoich_coeffs()
    products_nu  = tank.elyte_obj.product_stoich_coeffs()
    nu = products_nu - reactants_nu
    reaction_rates = tank.elyte_obj.net_rates_of_progress
    reaction_rates_capped = max_rate * np.tanh(reaction_rates / max_rate)
    reaction_rates_capped_ = np.copy(reaction_rates_capped)
    #print(reaction_rates_capped_)
    low_rate_indx = np.abs(reaction_rates_capped) < min_rate
    reaction_rates_capped[low_rate_indx] = 0.0
    q_dot = np.dot(nu, reaction_rates_capped)
    z = np.array([0,1,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0,0,0,0,0,0])
    #print(q_dot)
    #check = np.dot(q_dot,z)
    #if check !=0:
        #print(check)
        #print(q_dot[1])
        #low_rate_indx = np.abs(reaction_rates_capped_) < min_rate*1e10
        #reaction_rates_capped_[low_rate_indx] = 0.0
        #q_dot = np.dot(nu, reaction_rates_capped_)
        #print('new')
        #print(np.dot(q_dot,z))
        #print(reaction_rates_capped_)
        #print(reaction_rates_capped)
        #q_dot[1] = q_dot[1] - check
    s_dot_elyte_S8 = max_rate * np.tanh(s_dot_elyte_S8 / max_rate)
    s_dot_elyte_S8 = max_rate * np.tanh(s_dot_elyte_S8 / max_rate)
    #print(q_dot)


    '''
    resid[SV_idx.ptr['C_k_elyte']] = SV_dot[SV_idx.ptr['C_k_elyte']] - \
        (q_dot + s_dot_elyte_S8*A_surf_S8/elyte_volume + s_dot_elyte_Li2S*A_surf_Li2S/elyte_volume)
    
    # change in mass
    resid[SV_idx.ptr['mass_S8']] = SV_dot[SV_idx.ptr['mass_S8']] - \
        (-s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')]*solids.rho_S8*solids.mv_S8*A_surf_S8)
    resid[SV_idx.ptr['mass_Li2S']] = SV_dot[SV_idx.ptr['mass_Li2S']] - \
        (-s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')]*solids.rho_Li2S*solids.mv_Li2S*A_surf_Li2S)
    '''
    resid[SV_idx.ptr['C_k_elyte']] = (q_dot + s_dot_elyte_S8*A_surf_S8/elyte_volume + s_dot_elyte_Li2S*A_surf_Li2S/elyte_volume)
    # change in mass
    resid[SV_idx.ptr['mass_S8']] = (-s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')]*solids.rho_S8*solids.mv_S8*A_surf_S8)
    resid[SV_idx.ptr['mass_Li2S']] = (-s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')]*solids.rho_Li2S*solids.mv_Li2S*A_surf_Li2S)
    #print(q_dot)
    #print(s_dot_elyte_S8)
    #print(s_dot_elyte_Li2S)
    #print(t)

def residual_ivp(t,SV, SV_idx, tank, solids, params, elyte_volume, max_rate, min_rate ):
    dSVdt = np.zeros_like(SV)
    # update the concentrations in the elyte objects
    C_k = np.maximum(SV[SV_idx.ptr['C_k_elyte']],0.0)
    #C_k = SV[SV_idx.ptr['C_k_elyte']]

    C_total = np.sum(C_k)
    X_k = C_k/C_total
    tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
    solids.elyte_obj.TPX = solids.elyte_obj.T, solids.elyte_obj.P, X_k

    ## Elyte concentration
    # bulk production
    reactants_nu = tank.elyte_obj.reactant_stoich_coeffs()
    products_nu  = tank.elyte_obj.product_stoich_coeffs()
    nu = products_nu - reactants_nu
    reaction_rates = tank.elyte_obj.net_rates_of_progress
    reaction_rates_capped = max_rate * np.tanh(reaction_rates / max_rate)
    reaction_rates_capped_ = np.copy(reaction_rates_capped)
    #print(reaction_rates_capped_)
    low_rate_indx = np.abs(reaction_rates_capped) < min_rate
    reaction_rates_capped[low_rate_indx] = 0.0
    q_dot = np.dot(nu, reaction_rates_capped)

    # surface production
    mass_S8 = np.maximum(SV[SV_idx.ptr['mass_S8']], 1e-12)
    mass_Li2S = np.maximum(SV[SV_idx.ptr['mass_Li2S']], 1e-12)

    A_surf_S8 = 2*np.pi*((3*solids.rho_S8/(2*np.pi*mass_S8))**(2/3))
    s_dot_elyte_S8 = solids.surf_S8_obj.get_net_production_rates(solids.elyte_obj)

    A_surf_Li2S = 2*np.pi*((3*solids.rho_Li2S/(2*np.pi*mass_Li2S))**(2/3))
    s_dot_elyte_Li2S = solids.surf_Li2S_obj.get_net_production_rates(solids.elyte_obj)

    max_rate = 1e0
    #print(q_dot)
    q_dot = max_rate * np.tanh(q_dot / max_rate)
    s_dot_elyte_S8 = max_rate * np.tanh(s_dot_elyte_S8 / max_rate)
    s_dot_elyte_S8 = max_rate * np.tanh(s_dot_elyte_S8 / max_rate)

    dSVdt[SV_idx.ptr['C_k_elyte']] = (q_dot + s_dot_elyte_S8*A_surf_S8/elyte_volume + s_dot_elyte_Li2S*A_surf_Li2S/elyte_volume)
    # change in mass
    dSVdt[SV_idx.ptr['mass_S8']] = (-s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')]*solids.rho_S8*solids.mv_S8*A_surf_S8)
    dSVdt[SV_idx.ptr['mass_Li2S']] = (-s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')]*solids.rho_Li2S*solids.mv_Li2S*A_surf_Li2S)
    
    return dSVdt
