import numpy as np

def residual(t,SV,resid,user_data):
    np.set_printoptions(suppress=False, precision=1)
    SV_idx, tank, solids, params, total_volume = user_data

    # update the concentrations in the elyte objects
    # Transform back from log concentrations
    C_k = np.exp(SV[SV_idx.ptr['C_k_elyte']].copy())
    lower_C_k_limit = 1e-40
    C_k = np.nan_to_num(C_k, nan=lower_C_k_limit, posinf=lower_C_k_limit, neginf=lower_C_k_limit)
    C_k = np.maximum(C_k,lower_C_k_limit)
    C_total = np.sum(C_k)
    X_k = C_k/C_total

    tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
    solids.elyte_obj.TPX = solids.elyte_obj.T, solids.elyte_obj.P, X_k

    # surface production
    mass_S8 = SV[SV_idx.ptr['mass_S8']]
    mass_Li2S = SV[SV_idx.ptr['mass_Li2S']]
    #this makes sure the solver doesn't fail by guessing a negative mass 
    mass_S8 = np.maximum(mass_S8, 1e-30)
    mass_Li2S =  np.maximum(mass_Li2S, 1e-30)

    # Calculate the surface area and get the solid production rate 
    A_surf_S8 = 4 * np.pi * ((3 * mass_S8) / (4 * np.pi * solids.rho_S8))**(2/3)
    s_dot_elyte_S8 = solids.surf_S8_obj.get_net_production_rates(solids.elyte_obj)

    A_surf_Li2S = 4 * np.pi * ((3 * mass_Li2S) / (4 * np.pi * solids.rho_Li2S))**(2/3)
    s_dot_elyte_Li2S = solids.surf_Li2S_obj.get_net_production_rates(solids.elyte_obj)

    # I need to get individual rates so I recreate the math used to calculate 
    # I like having visibility into the individual rates, that is why I construct q_dot myself
    reactants_nu = tank.elyte_obj.reactant_stoich_coeffs()
    products_nu  = tank.elyte_obj.product_stoich_coeffs()
    nu = products_nu - reactants_nu
    reaction_rates = tank.elyte_obj.net_rates_of_progress
    q_dot = np.dot(nu, reaction_rates) # the species productions from the capped rates 
    
    # The solver was crashing as the volumes approached zero, so I cut off dissolution once they are very small
    if SV[SV_idx.ptr['volume_S8']] < 1e-10:
        # If the production rate in the elyte is positive, that means the species is dissolving
        if s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')] > 0:
           s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')] =0
    if SV[SV_idx.ptr['volume_Li2S']] < 1e-10:
        if s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')] > 0:
            s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')] =0
    ##########################################   

    resid[SV_idx.ptr['mass_S8'][-1]] = (-s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')]*solids.rho_S8*solids.mv_S8*A_surf_S8)
    resid[SV_idx.ptr['mass_Li2S'][-1]] = (-s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')]*solids.rho_Li2S*solids.mv_Li2S*A_surf_Li2S)
    dV_S8_dt = resid[SV_idx.ptr['mass_S8'][-1]]/solids.rho_S8
    dV_Li2S_dt = resid[SV_idx.ptr['mass_Li2S'][-1]]/solids.rho_Li2S
    resid[SV_idx.ptr['volume_S8']] = dV_S8_dt
    resid[SV_idx.ptr['volume_Li2S']] = dV_Li2S_dt
    # I need the change in volume because it impacts the concentration 
    dV_elyte_dt = -(dV_S8_dt + dV_Li2S_dt)
    elyte_volume = total_volume - (SV[SV_idx.ptr['volume_Li2S']] + SV[SV_idx.ptr['volume_S8']])
    resid[SV_idx.ptr['C_k_elyte']] = (q_dot/C_k + s_dot_elyte_S8*A_surf_S8/elyte_volume/C_k 
                        + s_dot_elyte_Li2S*A_surf_Li2S/elyte_volume/C_k - 1/elyte_volume*dV_elyte_dt)

 
    # I print these out so I can track the progress and see if the solver is getting stuck or if the problem is too stiff
    #print(f"{np.max(np.abs(reaction_rates)):.5e}")
    print(f"{t:.5e}") 
    
