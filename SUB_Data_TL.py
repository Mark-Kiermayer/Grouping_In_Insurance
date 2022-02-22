#!/usr/bin/env python
# coding: utf-8

########################## Section 0 - Import Tools  ################################

# ## 0.1. Import Packages
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import gamma, truncnorm, describe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import time, json, pickle, os, random, sobol_seq # random: for setting seed


### 0.2. Import customized functions
from actuarial_functions import get_termlife_reserve_profile
from data_part1_simulation_feature_space import data_simulation_features
from data_part2_calculation_targets import data_simulation_targets
from data_prep_General import data_prep_feautures_scale

# Dataframe representation
pd.set_option('precision', 2)


########################## Section 1 - Global Parameters  #################################

path_data = os.getcwd()+'\\TermLife\\Data\\NEW_'
# General assumptions - will be exported and passed on to connected scripts -> saved at the end of the script
params = {'A': 0.00022, 'B': 2.7*10**(-6), 'c': 1.124, 'int_rate': 0.01}
explan_vars_range = {'age': (25, 67), 'sum_ins': (1000, 1000000), 'duration': (2, 40), 'age_of_contract': (0, 39), 'interest_rate': (-0.01, 0.04)}
# Portofolio assumptions
N_contracts = 100000
n_in = len(explan_vars_range.keys())
# Matrix Version of previous upper/ lower bounds on features
Max_min = np.zeros([n_in,2])
Max_min = np.array([explan_vars_range['age'][0],explan_vars_range['age'][1]+explan_vars_range['duration'][1],
                    explan_vars_range['sum_ins'][0], explan_vars_range['sum_ins'][1], 
                    explan_vars_range['duration'][0], explan_vars_range['duration'][1], 
                    explan_vars_range['age_of_contract'][0], explan_vars_range['age_of_contract'][1], 
                    explan_vars_range['interest_rate'][0], explan_vars_range['interest_rate'][1]]).reshape(-1,2) #,dtype = 'int')



##################################   Section 2 - Data   #########################################
# ## 2.1. Simulation of feature variables
random.seed(42)
# data for full dimension
explan_vars = data_simulation_features(params['A'], params['B'], params['c'], N_contracts, Max_min, N_features= n_in)
# ## 2.2. Calculation of Target Values
# Create Targets
targets = data_simulation_targets(data=explan_vars[:,0:4].astype('int'), dur_max= explan_vars_range['duration'][1], A=params['A'], B=params['B'],c= params['c'],
                                     int_rate= explan_vars[:,4] )


# ## 2.3. Data Visualization and Analysis
fig, ax = plt.subplots(3,2, figsize = (10,8))
ax = ax.flatten()
ax[0].hist(explan_vars[:,0], bins = 150 )#, density = True)
ax[0].set_xlabel(r'Age, $X_1(\omega)$', fontsize = 'large')
ax[0].set_ylabel('Absolute Frequency', fontsize = 'large')
ax[1].hist(explan_vars[:,1], bins = 300)#, density = True)
ax[1].set_xlabel(r'Sum Insured, $X_2(\omega)$', fontsize = 'large')
ax[2].hist(explan_vars[:,2], bins = 150)#, density = True)
ax[2].set_xlabel(r'Duration, $X_3(\omega)$', fontsize = 'large')
ax[2].set_ylabel('Absolute Frequency', fontsize = 'large')
ax[3].hist(explan_vars[:,3], bins = 100)#, density = True)
ax[3].set_xlabel(r'Age of Contract, $X_4(\omega)$', fontsize = 'large')
ax[4].hist(explan_vars[:,4], bins = 100)#, density = True)
ax[4].set_xlabel(r'Interest Rate, $X_5(\omega)$', fontsize = 'large')

plt.tight_layout()
plt.show()


# ### 2.4.4. Exemplary Profile of Policy Values -> uncomment if required
#plt.plot(get_termlife_reserve_profile(age_curr = 49, Sum_ins = 100000, duration = 25, interest = 0.05, 
#                                      age_of_contract= 4, option_past = False), '*')
#         #color = 'black', linewidth = 0.7)
#plt.ylabel('Policy Value', fontsize = 'x-large')
#plt.xlabel(r'Time, $t$', fontsize = 'x-large')
#plt.show()


# ### 2.4.5. Visualization - Range of Targets
# Tabular version of range of targets
df = pd.DataFrame(data= None, index = None, columns = ['25% percentile', 'median', '75% percentile', 'max.'])
df.loc[r'$\max_{t} Y(\omega)_t$']= [np.quantile(a = targets.max(axis=1), q=0.25),np.quantile(a = targets.max(axis=1), q=0.5),
                             np.quantile(a = targets.max(axis=1), q=0.75),targets.max()]
print(df)


# Check Scaling feature of multivariate data
## Parameters for scaling procedure of targets
V_max  = get_termlife_reserve_profile(age_curr=explan_vars_range['age'][1], Sum_ins = explan_vars_range['sum_ins'][1], 
                                      duration=explan_vars_range['duration'][1], interest = explan_vars_range['interest_rate'][0]).max() 
V_min = 0

fig, [ax1, ax2] = plt.subplots(1,2, figsize = (12,4))
ax1.hist(targets.max(axis=1), bins = 500, density = False)
ax1.axvline(targets.max(axis=1).max(),color = 'red', linestyle = '-.', label = 'Maximum Policy Value')
ax1.legend(loc=(0.5,0.8))
ax1.set_ylabel('Absolute Frequency', fontsize = 'x-large')
ax1.set_xlabel('Policy Values', fontsize = 'x-large')
#ax1.set_title('Histogram of maximal reserves per contract')

ax2.hist(2*np.log(1+targets.max(axis=1))/np.log(1+V_max)-1, bins = 40, density=False)
ax2.axvline((np.log(1+targets.max(axis=1))/np.log(1+V_max)).max(),color = 'red', linestyle = '-.')
ax2.set_xlabel('Log-Scaled Policy Values', fontsize = 'x-large')

#ax2.set_title('Histogram of maximal reserves per contract on a log-scale')
plt.tight_layout()
plt.show()


# ## Export Portfolio
# Create Portfolio
Portfolio_x = pd.DataFrame(explan_vars, columns= list(explan_vars_range.keys()))
Portfolio_y = pd.DataFrame(targets, columns = ['t{}'.format(i) for i in range(targets.shape[1])])

# export data
Portfolio_x.to_csv(path_data+'TL_Portfolio.csv')
Portfolio_y.to_csv(path_data+'TL_Targets.csv')


# ## 2.3. Data Preparation
# ### 2.3.1. Scaling


## Parameters for scaling procedure of targets
V_max  = get_termlife_reserve_profile(age_curr=explan_vars_range['age'][1], Sum_ins = explan_vars_range['sum_ins'][1], 
                                      duration=explan_vars_range['duration'][1], interest = explan_vars_range['interest_rate'][0]).max() 
V_min = 0

print('V_max, V_min: ' + str(V_max) + ',  '+ str(V_min))

## Scale feature components to [-1,+1]
data_sc = data_prep_feautures_scale(explan_vars, Max_min, option = 'conditional')
Portfolio_x_scaled = pd.DataFrame(data_sc, columns= list(explan_vars_range.keys()))

# ### 2.3.2. Split (raw and scaled) Data in Training and Test Set

# Split ration training and test data
ratio_tr_tst = 0.7
N_train = int(ratio_tr_tst*N_contracts)

# Complete Data
#X_train, X_test, y_train, y_test = Portfolio_x_scaled[:N_train], Portfolio_x_scaled[N_train:], Portfolio_y[:N_train], Portfolio_y[N_train:]
#X_train_raw, X_test_raw = Portfolio_x[:N_train], Portfolio_x[N_train:]

#random split; originally not performed, as generation of data is random already
X_train, X_test, y_train, y_test = train_test_split(Portfolio_x_scaled, Portfolio_y, train_size = ratio_tr_tst)
id_split_train, id_split_test = X_train.index, X_test.index
X_train_raw, X_test_raw = Portfolio_x.loc[id_split_train], Portfolio_x.loc[id_split_test]

# save data
# Export Data
X_train.to_csv(path_data+'X_train.csv')
X_test.to_csv(path_data+'X_test.csv')
X_train_raw.to_csv(path_data+'X_train_raw.csv')
X_test_raw.to_csv(path_data+'X_test_raw.csv')
y_train.to_csv(path_data+'y_train.csv')
y_test.to_csv(path_data+'y_test.csv')


params['V_max'] = V_max
params['V_min'] = V_min



with open(path_data+'TL_params.pkl', 'wb') as f:
    pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
with open(path_data+'TL_explan_vars_range.pkl', 'wb') as f:
    pickle.dump(explan_vars_range, f, pickle.HIGHEST_PROTOCOL)
print('Data and params saved to: ' + path_data)

import seaborn as sns

sns.pairplot(X_train)
plt.show()
sns.pairplot(X_test)
plt.show()


sns.kdeplot(y_train.values.flatten(), label = 'targets_train')
sns.kdeplot(y_test.values.flatten(), label = 'targets_test')
plt.show()