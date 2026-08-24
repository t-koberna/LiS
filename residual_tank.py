import cantera as ct
import numpy as np
import sys 
from pathlib import Path
import sksundae as sun
from ruamel.yaml import YAML
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarstring import PlainScalarString
import re

def residual(t,SV,resid,user_data):
    np.set_printoptions(suppress=False, precision=1)
    SV_idx, tank, solids, params, total_volume, max_rate, min_rate, path = user_data

    lower_C_k_limit = 1e-60
    # update the concentrations in the elyte objects
    C_k = np.nan_to_num(SV[SV_idx.ptr['C_k_elyte']].copy(), nan=lower_C_k_limit, posinf=lower_C_k_limit, neginf=lower_C_k_limit)
    C_k = np.maximum(C_k,lower_C_k_limit)
    #C_k = SV[SV_idx.ptr['C_k_elyte']]
    C_total = np.sum(C_k)
    X_k = C_k/C_total

    tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
    solids.elyte_obj.TPX = solids.elyte_obj.T, solids.elyte_obj.P, X_k

    
    # surface production
    mass_S8 = SV[SV_idx.ptr['mass_S8']]
    mass_Li2S = SV[SV_idx.ptr['mass_Li2S'][-1]]
    #this makes sure the solver doesn't fail by guessing a negative mass 
    mass_S8 = 0.5 * (mass_S8 + np.abs(mass_S8 + 1e-30))
    mass_Li2S = 0.5 * (mass_Li2S + np.abs(mass_Li2S + 1e-30))

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
    #if t>1e-10:
    #    print(reaction_rates)
    #    gbhjn
    #print(tank.elyte_obj.forward_rate_constants)
    #print(tank.elyte_obj.reverse_rate_constants)
    #wret
    
    yaml = YAML(typ='rt')
    inputs = yaml.load(path)
     
    net_rates = tank.elyte_obj.net_rates_of_progress
    write_yaml = 0
    
    for i, rxn in enumerate(inputs['bulk-elyte-reactions']):
        reaction_rate = np.abs(net_rates[i])
        pre_exp_factor = tank.elyte_obj.reaction(i).rate.pre_exponential_factor
        
        if reaction_rate > max_rate :#and reaction_rate<max_rate*1e5:
            write_yaml = 1
            #new_A = pre_exp_factor * (max_rate / reaction_rate)
            new_A = pre_exp_factor * np.max([(max_rate / reaction_rate),5e-1])


            formatted_str = f"{new_A:.2e}"
            
            rxn['rate-constant']['A'] = formatted_str
    
    write_yaml = 0###############$%^&*()OIJHBGVFTYUIKJNHBGTY&U*IOKJHBGTYU&*IOKJHGTY& 
    if write_yaml == 1:
        print("reducing rate(s)")
        # Step 1: Write file out using ruamel 
        # (This completely preserves all your inline comments and spacing)
        with open(path, 'w') as file:
            yaml.dump(inputs, file)

        # Step 2: Open file as a plain text string and strip the quotes from 'A' fields
        with open(path, 'r', newline="", encoding='utf-8') as file:
            text_content = file.read()

        # This regular expression scans the file for patterns like: A: '2.98e+30' or A: "2.98e+30"
        # and safely converts them to bare scalar numbers: A: 2.98e+30
        cleaned_content = re.sub(r"(A:\s*)['\"]([+-]?\d+\.?\d*e[+-]?\d+)['\"]", r"\1\2", text_content)

        # Overwrite the file with the cleaned plain text string
        with open(path, 'w', newline="", encoding='utf-8') as file:
            file.write(cleaned_content)

        # 3. Reload your Cantera solution matrix cleanly
        tank.elyte_obj = ct.Solution(path, tank.inputs['electrolyte-phase'])

    tank.elyte_obj.TPX = tank.elyte_obj.T, tank.elyte_obj.P, X_k
    reaction_rates_capped = tank.elyte_obj.net_rates_of_progress
  
    #reaction_rates_capped = max_rate * np.tanh(reaction_rates_capped / max_rate)
    #reaction_rates_capped_ = np.copy(reaction_rates_capped)
    reaction_rates_capped_ = reaction_rates
    #print(reaction_rates_capped_)
    # I do not currently have a miniumum rate, but if i want to set one here is how
    if min_rate > 0:
        low_rate_indx = np.abs(reaction_rates_capped) < np.max(np.abs(reaction_rates_capped))/min_rate
        low_rate_indx = np.abs(reaction_rates_capped) < min_rate
        reaction_rates_capped[low_rate_indx] = 0.0
    q_dot = np.dot(nu, reaction_rates_capped) # the species productions from the capped rates 
    q_dot_orig = np.dot(nu, reaction_rates) # the species productions from the uncapped rate 

    # The number of moles of each element and the charge of each species, used for conservation checks
    nLi = np.array([tank.elyte_obj.n_atoms(name, 'Li') for name in SV_idx.elyte_species])
    nS = np.array([tank.elyte_obj.n_atoms(name, 'S') for name in SV_idx.elyte_species])
    z = np.array([0,1,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,-2,0,0,0,0,0,0,0,0])
    
    # check to see what reaction exceeded the limit, I print them out then manually lower the
    #       rate for that reaction in the yaml file    
    '''
    if s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')]*solids.rho_S8*solids.mv_S8*A_surf_S8 > max_rate*1e4:
        rate =s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')]*solids.rho_S8*solids.mv_S8*A_surf_S8/max_rate
        #print(f"{rate.item():.1e}") 
        #print('S8s')  
        check = 1    
        #sys.exit()  
    if s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')]*solids.rho_Li2S*solids.mv_Li2S*A_surf_Li2S > max_rate*1e4:
        rate = s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')]*solids.rho_Li2S*solids.mv_Li2S*A_surf_Li2S/max_rate
        print(f"{rate.item():.1e}") 
        print('Li2Ss') 
        check =1
        #sys.exit()  
    '''
  
    #max_rate_s = max_rate*1e100 # cap the surface production rates
    if SV[SV_idx.ptr['volume_S8']] < 1e-13:
        s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')] =0
    if SV[SV_idx.ptr['volume_Li2S']] < 1e-13:
            s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')] =0
    #s_dot_elyte_S8 = np.sign(s_dot_elyte_S8)*max_rate_s * np.tanh(np.abs(s_dot_elyte_S8) / max_rate_s)
    #s_dot_elyte_Li2S = max_rate_s * np.tanh(s_dot_elyte_Li2S / max_rate_s)
    ##########################################

    resid[SV_idx.ptr['mass_S8'][-1]] = (-s_dot_elyte_S8[SV_idx.elyte_species.index('S8(e)')]*solids.rho_S8*solids.mv_S8*A_surf_S8)
    resid[SV_idx.ptr['mass_Li2S'][-1]] = (-s_dot_elyte_Li2S[SV_idx.elyte_species.index('Li2S(e)')]*solids.rho_Li2S*solids.mv_Li2S*A_surf_Li2S)
    resid[SV_idx.ptr['volume_S8']] = resid[SV_idx.ptr['mass_S8'][-1]]/solids.rho_S8
    resid[SV_idx.ptr['volume_Li2S']] = resid[SV_idx.ptr['mass_Li2S'][-1]]/solids.rho_Li2S
    # I need the change in volume because it impacts the concentration 
    dV_elyte_dt = -(resid[SV_idx.ptr['volume_S8']] + resid[SV_idx.ptr['volume_Li2S']])
    elyte_volume = total_volume - (SV[SV_idx.ptr['volume_Li2S']] + SV[SV_idx.ptr['volume_S8']])
    resid[SV_idx.ptr['C_k_elyte']] = (q_dot + s_dot_elyte_S8*A_surf_S8/elyte_volume + s_dot_elyte_Li2S*A_surf_Li2S/elyte_volume 
                                      - C_k/elyte_volume*dV_elyte_dt)
    #If want to use logspaced time
    #resid[SV_idx.ptr['mass_S8'][-1]] = resid[SV_idx.ptr['mass_S8'][-1]]*np.exp(t)
    #resid[SV_idx.ptr['mass_Li2S'][-1]] = resid[SV_idx.ptr['mass_Li2S'][-1]]*np.exp(t)
    #resid[SV_idx.ptr['volume_Li2S']] = resid[SV_idx.ptr['volume_Li2S']]*np.exp(t)
    #resid[SV_idx.ptr['C_k_elyte']] =  resid[SV_idx.ptr['C_k_elyte']]*np.exp(t)
    #print(q_dot)
    #print(s_dot_elyte_S8)
    #print(s_dot_elyte_Li2S)
    print(f"{np.max(np.abs(reaction_rates_capped)):.5e}")
    #print(s_dot_elyte_S8*A_surf_S8/elyte_volume)
    #print(s_dot_elyte_Li2S*A_surf_Li2S/elyte_volume)
    print(f"{t:.5e}") 
    
