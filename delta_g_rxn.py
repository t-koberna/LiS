import pandas as pd
import numpy as np
from math import floor
from matplotlib import pyplot as plt
import matplotlib as mpl
import cantera as ct
from pathlib import Path
from ruamel.yaml import YAML
from setup_tank import SV_pointer, Solution_Vector_0, SV_0_from_outputs, SV_0_from_data, Tank, Solid, Parameters

# Read in the yaml input file
path = Path("Prune\Li_Sulfur_tank.yaml") # does not include the ions in solution
path = Path("Li_Sulfur_tank_no_ions_Rafa.yaml") # does not include the ions in solution
path = Path("Li_Sulfur_tank_test_no_ions.yaml") # does not include the ions in solution


yaml = YAML(typ='safe')
inputs = yaml.load(path)

params = Parameters(inputs)
tank = Tank(path, inputs, params)
solids = Solid(path, inputs, tank, params)

name_elyte_species = [species['name'] for species in tank.inputs['transport']['diffusion-coefficients']]
C_k_0_elyte = [species['C_k'] for species in tank.inputs['transport']['diffusion-coefficients']]
rxn_eqs = [rxn.equation for rxn in tank.elyte_obj.reactions()]
rxn_rates = [rxn.rate.pre_exponential_factor for rxn in tank.elyte_obj.reactions()]
rxn_rates_str = np.char.mod('%.1e', rxn_rates)

surf = tank.elyte_obj
for i, spec in enumerate(tank.elyte_obj.species()):
    print(f"{i}: {spec.name}")
for i in range(surf.n_reactions):
    rxn = surf.reaction(i)  
    #print(f"\nReaction {i+1}: {rxn.equation}")
    #print(f"  Reactants: {rxn.reactants}")
    #print(f"  Products:  {rxn.products}")
R = surf.reactant_stoich_coeffs()
P = surf.product_stoich_coeffs()
nu = P - R
print(nu)
g_f = tank.elyte_obj.standard_gibbs_RT * ct.gas_constant * tank.elyte_obj.T
g_f = g_f*1e-6 # convert J/kmol to kJ/mol
g_f_rxn =np.dot(nu.T,g_f)
g_f_rxn_str = np.char.mod('%.2f', g_f_rxn)

headers = ['Reaction Equation', r'$\Delta G_f$ [kJ/mol]', 'Pre Exp Factor']
data = np.column_stack((rxn_eqs,g_f_rxn_str,rxn_rates_str))
row_height_inch = 0.25
fig_height = (len(data) * row_height_inch) + 1.0

fig, ax = plt.subplots(figsize=(6, fig_height)) 
ax.axis('off') # Hide graph lines and ticks

# 3. Generate the table
table = ax.table(
    cellText=data, 
    colLabels=headers, 
    loc='center', 
    cellLoc='center'
)

# 4. Style the table (Optional)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5) # Scale row heights for readability
table.auto_set_column_width(col = range(np.size(headers)))
plt.tight_layout()

plt.show()

