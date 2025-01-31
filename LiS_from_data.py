# Creates plots from saved data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LiS_functions import bucket, plot_results

folder_name = "2025-01-31_10-26-08"

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
bm_S8_back = bookmarks_data['Bookmark back S8'].to_numpy()
bm_Li2S_front = bookmarks_data['Bookmark front Li2S'].to_numpy()
bm_Li2S_back = bookmarks_data['Bookmark back Li2S'].to_numpy()

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

moles= pd.read_csv(fp_moles,dtype='float64')
mol_S8_elyt = moles['mol_S8_elyt'].to_numpy()
mol_Li2S_elyt = moles['mol_Li2S_elyt'].to_numpy()
mol_S8_ca = moles['mol_S8_ca'].to_numpy()
mol_Li2S_ca = moles['mol_Li2S_ca'].to_numpy()

# pick what plots to display (1 yes, anything else no)
num_particles_bin = 1
cs_area = 1
total_particles = 1
conc_and_moles = 1
vol_frac = 1
time_stamps_bins = 1
bookmark_movement = 1

plot_flags = [num_particles_bin, cs_area, total_particles, conc_and_moles, vol_frac, time_stamps_bins, bookmark_movement]

plot_results(plot_flags, time, N_S8, N_Li2S, bucket_S8, bucket_Li2S, 
        mol_S8_elyt, mol_Li2S_elyt, mol_S8_ca, mol_Li2S_ca,
        bm_S8_front, bm_S8_back, bm_Li2S_front, bm_Li2S_back,
        h, area_carbon_0, V_elyte_0, time_end, folder_name)

figg = plt.figure()
plt.plot(time[260:],bm_S8_front[260:]-bm_S8_back[260:])

plt.show()