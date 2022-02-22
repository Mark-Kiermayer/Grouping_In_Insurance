#!/usr/bin/env python
# coding: utf-8

# # Pensions
# ## 0.1.Import Packages
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import gamma, truncnorm, describe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import time, json, pickle, os, random, sobol_seq # random: for setting seed

# ## 0.2. Load Functions
from actuarial_functions import get_pension_reserve, pension_reserve
from data_part1_simulation_feature_space import data_simulation_features
from data_part2_calculation_targets import data_simulation_targets
from data_prep_General import data_prep_feautures_scale, data_prep_split

############ 1. Assumptions  ############

# Portfolio Details
N_contracts = 100000 
input_used = ['Fund','Age', 'Salary', 'Salary_scale', 'Contribution']#, 'interest_rate']
n_in = len(input_used)

# current directory for saving issues
path_data = os.getcwd()+'\\Pensions\\Data\\NEW_'

# Dataframe representation
pd.set_option('precision', 2)
pd.set_option('display.max_colwidth', 40)


# Makeham mortality model
A= 0.00022
B=2.7*10**(-6)
c=1.124
interest_rate = 0.03 # fixed interest rate
early_pension_structure = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1]
pension_age_max = 67

params = {'A': A, 'B': B, 'c': c, 'early_pension_structure': early_pension_structure, 
          'pension_age_max': pension_age_max, 'interest_rate': interest_rate}

# Split ration training and test data
ratio_tr_tst = 0.7
N_train = int(ratio_tr_tst*N_contracts)

# share of validation data
val_share = 0.25

# Range of Variables
age_min, age_max = 25,60
fund_min, fund_max = 0, 500000
salary_min, salary_max = 20000, 200000
salary_scale_min, salary_scale_max = 0.01, 0.05
contribution_min, contribution_max = 0.01, 0.1
#int_min, int_max = -0.01, 0.10 # fixed interest rate interest_rate

explan_vars_range = {'age': [age_min, age_max], 'fund': [fund_min, fund_max], 'salary': [salary_min, salary_max], 
                     'salary_scale': [salary_scale_min, salary_scale_max], 'contribution': [contribution_min, contribution_max]}

# Matrix Version of previous upper/ lower bounds on features
Min_Max = np.array([explan_vars_range['fund'][0],explan_vars_range['fund'][1],
                    explan_vars_range['age'][0], explan_vars_range['age'][1], 
                    explan_vars_range['salary'][0], explan_vars_range['salary'][1],
                    explan_vars_range['salary_scale'][0], explan_vars_range['salary_scale'][1],
                    explan_vars_range['contribution'][0], explan_vars_range['contribution'][1]]).reshape(n_in,2)


################ 2. Simulation of Data ##################
# ## 2.1. Raw Data
random.seed(42)
data_sc = 2*sobol_seq.i4_sobol_generate(dim_num = n_in, n = N_contracts)-1   # in [-1,+1]

data = (data_sc+1)/2*(Min_Max[:,1]-Min_Max[:,0])+Min_Max[:,0]


targets = pension_reserve(data, pension_age_max=pension_age_max, age_min=age_min, interest_std = interest_rate, 
                          ep_structure= early_pension_structure)



for i in range(10):
    plt.plot(targets[i,:])
plt.xlabel('Time')
plt.title('Examplary expected pensions')
plt.show()

# ## 2.2. Train-/ Test-Split
# Complete Data
data_train,data_test = data_prep_split(data,ratio_tr_tst)
data_train_sc, data_test_sc = data_prep_split(data_sc, ratio_tr_tst)
targets_train, targets_test= data_prep_split(targets,ratio_tr_tst)


Portfolio_x_scaled = pd.DataFrame(data_sc, columns= list(explan_vars_range.keys()))
Portfolio_x = pd.DataFrame(data, columns= list(explan_vars_range.keys()))
Portfolio_y = pd.DataFrame(targets, columns = ['t{}'.format(i) for i in range(targets.shape[1])])

# split data
N_train = int(ratio_tr_tst*N_contracts)
X_train, X_test, y_train, y_test = Portfolio_x_scaled[:N_train], Portfolio_x_scaled[N_train:], Portfolio_y[:N_train], Portfolio_y[N_train:]
X_train_raw, X_test_raw = Portfolio_x[:N_train], Portfolio_x[N_train:]


V_max = get_pension_reserve(fund_accum = fund_max, age = age_min, salary = salary_max, 
                            salary_scale = salary_scale_max, contribution = contribution_max, 
                             A = 0.00022, B = 2.7*10**(-6), c = 1.124, interest = interest_rate,
                            pension_age_max = 67, early_pension = early_pension_structure).max()
print('V_max: ' + str(V_max))
# Tabular version of range of targets

df = pd.DataFrame(data= None, index = None, columns = ['25th percentile', 'median', '75th percentile', 'max.'])
df.loc[r'$\max_{t} Y(\omega)_t$']= [np.quantile(a = targets.max(axis=1), q=0.25),np.quantile(a = targets.max(axis=1), q=0.5),
                             np.quantile(a = targets.max(axis=1), q=0.75),targets.max()]
#df.loc['$Y(\omega)$']= [np.quantile(a = targets, q=0.25),np.quantile(a = targets, q=0.5),
#                             np.quantile(a = targets, q=0.75),targets.max()]
print(df)


# save data
X_train.to_csv(path_data+'X_train.csv')
X_test.to_csv(path_data+'X_test.csv')
X_train_raw.to_csv(path_data+'X_train_raw.csv')
X_test_raw.to_csv(path_data+'X_test_raw.csv')
y_train.to_csv(path_data+'y_train.csv')
y_test.to_csv(path_data+'y_test.csv')

Portfolio_x.to_csv(path_data + 'Pension_Portfolio.csv')
Portfolio_y.to_csv(path_data + 'Pension_Targets.csv')
# save params
params['V_max'] = V_max

with open(path_data+'Pension_params.pkl', 'wb') as f:
    pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
with open(path_data+'Pension_explan_vars_range.pkl', 'wb') as f:
    pickle.dump(explan_vars_range, f, pickle.HIGHEST_PROTOCOL)

print('n')
print('Data and params saved in: ' + path_data)
print('n')

#################   3. Analysis of Data/ Targets   ###########################

# FYI - share of times with target-value == 0
#(targets==0).sum()/(N_contracts*(pension_age_max-age_min+1))

# Check Scaling feature of multivariate data
fig, [ax1, ax2] = plt.subplots(1,2, figsize = (12,4))
ax1.hist(targets[targets!=0], bins = 1000, density = False)
ax1.axvline(targets.max(axis=1).max(),color = 'orange', linestyle = '-.', label = 'Maximum Reserve (in Data)')
ax1.axvline(V_max,color = 'red', linestyle = '-.', label = 'Maximum Reserve')

ax1.legend(loc=(0.5,0.8))
ax1.set_ylabel('Absolute Frequency', fontsize = 'large')
ax1.set_xlabel('Policy Values', fontsize = 'large')
#ax1.set_title('Histogram of maximal reserves per contract')

ax2.hist(2*np.log(1+targets[targets!=0])/np.log(1+V_max)-1, bins = 1000, density=False)
ax2.axvline((np.log(1+targets[targets!=0])/np.log(1+V_max)).max(),color = 'red', linestyle = '-.')
ax2.set_xlabel('Log-Scaled Policy Values', fontsize = 'large')

#ax2.set_title('Histogram of maximal reserves per contract on a log-scale')
plt.tight_layout()
plt.show()


sns.kdeplot(y_train.values.flatten(), label='training data')
sns.kdeplot(y_test.values.flatten(), label = 'test data')
plt.title('KDE of targets')
plt.show()


# Compare training and test data
sns.pairplot(X_train)
sns.pairplot(X_test)