import cantera as ct
import numpy as np
import sys 
def residual(t,SV,resid,user_data):
    np.set_printoptions(suppress=False, precision=1)
    SV_idx, tank, solids, params, total_volume, max_rate, min_rate = user_data

    # update the concentrations in the elyte objects
    #C_k = np.maximum(SV[SV_idx.ptr['C_k_elyte']],1e-80)
    C_k = SV[SV_idx.ptr['C_k_elyte']]
    C_total = np.sum(C_k)
    X_k = C_k/C_total
    tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
    solids.elyte_obj.TPX = solids.elyte_obj.T, solids.elyte_obj.P, X_k

    # surface production
    mass_S8 = SV[SV_idx.ptr['mass_S8']]
    mass_Li2S = SV[SV_idx.ptr['mass_Li2S']]
    #this ensures the solver doesn't fail by guessing a negative mass 
    mass_S8 = 0.5 * (mass_S8 + np.abs(mass_S8 + 1e-40))
    mass_Li2S = 0.5 * (mass_Li2S + np.abs(mass_Li2S + 1e-40))

    # Calculate the surface area and get the solid production rate 
    A_surf_S8 = 2 * np.pi * ((3 * mass_S8) / (2 * np.pi * solids.rho_S8))**(2/3)
    s_dot_elyte_S8 = solids.surf_S8_obj.get_net_production_rates(solids.elyte_obj)

    A_surf_Li2S = 2 * np.pi * ((3 * mass_Li2S) / (2 * np.pi * solids.rho_Li2S))**(2/3)
    s_dot_elyte_Li2S = solids.surf_Li2S_obj.get_net_production_rates(solids.elyte_obj)

    ##########################################
    # this section is for capping indivdual rates at a predefined maximum limit
    # I need to get individual rates so I recreate the math used to calculate tank.elyte_obj.net_production_rates
    reactants_nu = tank.elyte_obj.reactant_stoich_coeffs()
    products_nu  = tank.elyte_obj.product_stoich_coeffs()
    nu = products_nu - reactants_nu
    reaction_rates = tank.elyte_obj.net_rates_of_progress
    reaction_rates_capped = max_rate * np.tanh(reaction_rates / max_rate)
    reaction_rates_capped_ = np.copy(reaction_rates_capped)
    reaction_rates_capped_ = reaction_rates#
    #print(reaction_rates_capped_)
    # I do not currently have a miniumum rate, but if i want to set one here is how
    #low_rate_indx = np.abs(reaction_rates_capped) < min_rate
    #reaction_rates_capped[low_rate_indx] = 0.0
    q_dot = np.dot(nu, reaction_rates_capped) # the species productions from the capped rates 
    q_dot_orig = np.dot(nu, reaction_rates) # the species productions from the uncapped rate 

    # The number of moles of each element and the charge of each species, used for conservation checks
    nLi = np.array([tank.elyte_obj.n_atoms(name, 'Li') for name in SV_idx.elyte_species])
    nS = np.array([tank.elyte_obj.n_atoms(name, 'S') for name in SV_idx.elyte_species])
    z = np.array([0,1,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0,0,0,0,0,0])
    
    # check to see what reaction exceeded the limit, I print them out then manually lower the
    #       rate for that reaction in the yaml file    
    check = 0 
    for i in range(tank.elyte_obj.n_reactions):
        rxn = tank.elyte_obj.reaction(i)  
        #print(reaction_rates[i])
        if reaction_rates[i]> max_rate:
            #print(f"{reaction_rates[i]/max_rate:.1e}")
            #print(f"\nReaction {i+1}: {rxn.equation}, {rxn.ID}")
            check = 1
    # check to see if the residuals for the masses exceed the max_rate
    if s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')]*solids.rho_S8*solids.mv_S8*A_surf_S8 > max_rate:
        rate =s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')]*solids.rho_S8*solids.mv_S8*A_surf_S8/max_rate
        #print(f"{rate.item():.1e}") 
        #print('S8s')  
        check = 1      
    if s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')]*solids.rho_Li2S*solids.mv_Li2S*A_surf_Li2S > max_rate:
        #print('Li2Ss') 
        rate = s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')]*solids.rho_Li2S*solids.mv_Li2S*A_surf_Li2S/max_rate
        #print(f"{rate.item():.1e}") 
        check =1
    # break the code if the max rate is exceeded
    #if check ==1 :
        #sys.exit()
  
    # checks for conservation of moles and charge
    #q_dot = tank.elyte_obj.net_production_rates
    #check = np.dot(q_dot,z)
    #print(np.dot(q_dot,nS))
    #if np.dot(q_dot,nS) !=0 or np.dot(q_dot,nLi):
        #q_dot = q_dot*0
    
    '''
    print('line')
    print(np.dot(q_dot_orig,nLi))
    print(np.dot(q_dot,nLi))
    print(np.dot(q_dot_orig,nS))
    print(np.dot(q_dot,nS))
    '''
   

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
    #max_rate_s = max_rate # cap the surface production rates
    #s_dot_elyte_S8 = max_rate_s * np.tanh(s_dot_elyte_S8 / max_rate_s)
    #s_dot_elyte_Li2S = max_rate_s * np.tanh(s_dot_elyte_Li2S / max_rate_s)
    ##########################################

    resid[SV_idx.ptr['mass_S8']] = (-s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')]*solids.rho_S8*solids.mv_S8*A_surf_S8)
    resid[SV_idx.ptr['mass_Li2S']] = (-s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')]*solids.rho_Li2S*solids.mv_Li2S*A_surf_Li2S)
    resid[SV_idx.ptr['volume_S8']] = resid[SV_idx.ptr['mass_S8']]/solids.rho_S8
    resid[SV_idx.ptr['volume_Li2S']] = resid[SV_idx.ptr['mass_Li2S']]/solids.rho_Li2S
    # I need the change in volume because it impacts the concentration 
    dV_elyte_dt = -(resid[SV_idx.ptr['volume_S8']] + resid[SV_idx.ptr['volume_Li2S']])
    elyte_volume = total_volume - (SV[SV_idx.ptr['volume_Li2S']] + SV[SV_idx.ptr['volume_S8']])
    resid[SV_idx.ptr['C_k_elyte']] = (q_dot + s_dot_elyte_S8*A_surf_S8/elyte_volume + s_dot_elyte_Li2S*A_surf_Li2S/elyte_volume - C_k/elyte_volume*dV_elyte_dt)

    #print(q_dot)
    #print(s_dot_elyte_S8)
    #print(s_dot_elyte_Li2S)
    print(t) 
