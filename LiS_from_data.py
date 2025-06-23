# Creates plots from saved data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LiS_functions import bucket, plot_results

folder_name = "2025-02-11_12-57-09"
#folder_name = "graph3"

fn_bookmarks = "bookmarks.csv"
fn_S8 = "Num_Particles_S8.csv"
fn_Li2S = "Num_Particles_Li2S.csv"
fn_inputs = "inputs.csv"
fn_moles = "moles.csv"
fp_bookmarks = f"{folder_name}/{fn_bookmarks}"
fp_S8 = f"{folder_name}/{fn_S8}"
fp_Li2S = f"{folder_name}/{fn_Li2S}"
fp_inputs = f"{folder_name}/{fn_inputs}"
fp_moles = f"{folder_name}/{fn_moles}"
    
bookmarks_data = pd.read_csv(fp_bookmarks,dtype='float64')
time = bookmarks_data['time'].to_numpy()
time_end = time[-1]
bm_S8_front = bookmarks_data['Bookmark front S8'].to_numpy()
bm_Li2S_front = bookmarks_data['Bookmark front Li2S'].to_numpy()

df_S8 = pd.read_csv(fp_S8,dtype='float64')
num_columns_S8 = df_S8.shape[1]
N_S8 = df_S8.to_numpy()
N_S8 = N_S8.T

df_Li2S = pd.read_csv(fp_Li2S,dtype='float64')
num_columns_Li2S = df_Li2S.shape[1]
N_Li2S = df_Li2S.to_numpy()
N_Li2S = N_Li2S.T

inputs= pd.read_csv(fp_inputs)
t_bucket_S8 = inputs['t_bucket_S8'][0]
t_bucket_Li2S = inputs['t_bucket_Li2S'][0]
mv_S8 = inputs['mv_S8'][0]
mv_Li2S = inputs['mv_Li2S'][0]
h = inputs['h'][0]
area_carbon_0 = inputs['area_carbon_0'][0]
V_elyte_0 = inputs['V_elyte_0'][0]
bucket_S8 = bucket(num_columns_S8,t_bucket_S8,mv_S8,"S_8")
bucket_Li2S = bucket(num_columns_Li2S,t_bucket_Li2S,mv_Li2S,"Li_2S")

# Theses are not inputs, but I didn't want to make a whole new excel file for them
# if I end up needed more output values stored, maybe I will
#r_min_S8 = inputs['r_min_S8'][0]
#r_min_Li2S = inputs['r_min_Li2S'][0]

moles= pd.read_csv(fp_moles,dtype='float64')
mol_S8_elyt = moles['mol_S8_elyt'].to_numpy()
mol_Li2S_elyt = moles['mol_Li2S_elyt'].to_numpy()
mol_S8_ca = moles['mol_S8_ca'].to_numpy()
mol_Li2S_ca = moles['mol_Li2S_ca'].to_numpy()

# pick what plots to display (1 yes, anything else no)
num_particles_bin = 0
cs_area = 0
total_particles = 0
conc_and_moles = 0
vol_frac = 0
time_stamps_bins = 1
bookmark_movement = 0

plot_flags = [num_particles_bin, cs_area, total_particles, conc_and_moles, vol_frac, time_stamps_bins, bookmark_movement]

plot_results(plot_flags, time, N_S8, N_Li2S, bucket_S8, bucket_Li2S, 
        mol_S8_elyt, mol_Li2S_elyt, mol_S8_ca, mol_Li2S_ca,
        bm_S8_front, bm_Li2S_front,
        h, area_carbon_0, V_elyte_0, time_end, folder_name)


plt.show()
