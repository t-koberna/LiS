import pandas as pd
import numpy as np

df_1 = pd.read_csv("Supplementary/Num_Particles_1.csv",dtype='float64')
nLi2O2_1 = df_1.to_numpy()
nLi2O2_1 = nLi2O2_1.T
nLi2O2_1 = nLi2O2_1[0]

df_2 = pd.read_csv("Supplementary/Num_Particles_2.csv",dtype='float64')
nLi2O2_2 = df_2.to_numpy()
nLi2O2_2 = nLi2O2_2.T
nLi2O2_2 = nLi2O2_2[0]

df_3 = pd.read_csv("Supplementary/Num_Particles_3.csv",dtype='float64')
nLi2O2_3 = df_3.to_numpy()
nLi2O2_3 = nLi2O2_3.T
nLi2O2_3 = nLi2O2_3[0]

n_bins = np.size(nLi2O2_1)

#Error check
e_1_numerator = [0]*n_bins
e_2_numerator = [0]*n_bins

#err =1/n * sum((n_i,j - n_i,3)^2/n_i,ref^2)*w_i,j
#w_i,j = 0.5*(n_i,j/sum(n_i,j)+n_i,3/sum(n_i,3))

for ind, ele in enumerate(nLi2O2_3):
    # if the bin is empty and the other is not, we set that as 100% error
    if ele ==0: 
        e_1_numerator[ind] = 1
        e_2_numerator[ind] = 1
    else:
        e_1_numerator[ind] = np.sqrt(((nLi2O2_1[ind]-ele))**2/((ele)**2))*((nLi2O2_1[ind]/sum(nLi2O2_1) + nLi2O2_3[ind]/sum(nLi2O2_3))/2)
        e_2_numerator[ind] = np.sqrt(((nLi2O2_2[ind]-ele))**2/((ele)**2))*((nLi2O2_2[ind]/sum(nLi2O2_2) + nLi2O2_3[ind]/sum(nLi2O2_3))/2)



e_1= sum(e_1_numerator)/n_bins
e_2= sum(e_2_numerator)/n_bins

print(e_1*100)
print(e_2*100)