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
import time, json, pickle, os, random #, sobol_seq # random: for setting seed

# ## 0.2. Load Functions
from actuarial_functions import get_pension_reserve, pension_reserve
from data_part1_simulation_feature_space import data_simulation_features
from data_part2_calculation_targets import data_simulation_targets
from data_prep_General import data_prep_feautures_scale, data_prep_split

########################## Section 1 - Global Parameters  #################################
cd = os.getcwd()
path_data = cd+'/Pensions/Data/'
# Load general assumptions
with open(path_data+'Pension_params.pkl', 'rb') as f:
    params = pickle.load(f)
with open(path_data+'Pension_explan_vars_range.pkl', 'rb') as f:
    explan_vars_range = pickle.load(f)

print('Parameters imported: ', params)
print('Explanatory variables imported: ', explan_vars_range)
V_max = params['V_max']

############ 1. Assumptions  ############

# Portfolio Details
N_contracts = 100000 
n_in = len(explan_vars_range.keys())
# Split ration training and test data
ratio_tr_tst = 0.7
N_train = int(ratio_tr_tst*N_contracts)

# Dataframe representation
pd.set_option('precision', 2)
pd.set_option('display.max_colwidth', 40)


# Info - Range of Variables
#age_min, age_max = 25,60
#fund_min, fund_max = 0, 500000
#salary_min, salary_max = 20000, 200000
#salary_scale_min, salary_scale_max = 0.01, 0.05
#contribution_min, contribution_max = 0.01, 0.1
#int_min, int_max = -0.01, 0.10 # fixed interest rate interest_rate

#explan_vars_range = {'age': [age_min, age_max], 'fund': [fund_min, fund_max], 'salary': [salary_min, salary_max], 
#                     'salary_scale': [salary_scale_min, salary_scale_max], 'contribution': [contribution_min, contribution_max]}

# Matrix Version of previous upper/ lower bounds on features
Min_Max = np.array([explan_vars_range['fund'][0],explan_vars_range['fund'][1],
                    explan_vars_range['age'][0], explan_vars_range['age'][1], 
                    explan_vars_range['salary'][0], explan_vars_range['salary'][1],
                    explan_vars_range['salary_scale'][0], explan_vars_range['salary_scale'][1],
                    explan_vars_range['contribution'][0], explan_vars_range['contribution'][1]]).reshape(n_in,2)


################ 2. Simulation of Data ##################
# ## 2.1. Raw Data
random.seed(42)

age = np.random.normal(loc=40, scale=7,size=N_contracts).astype('int')
age[age<explan_vars_range['age'][0]] = explan_vars_range['age'][0]
age[age>explan_vars_range['age'][1]] = explan_vars_range['age'][1]
salary = 20000 + np.random.gamma(shape=4, scale =10000, size=N_contracts).astype('int')
salary_scale = np.random.uniform(low=0.01, high=0.05, size=N_contracts)
contribution = np.random.uniform(low=0.01, high=0.05, size=N_contracts)

# retrace fund volume asuming constant salary_scale, constant contribution and entry into workforce at age 25
# fund = \sum_{k=1}^{age-25} salary*(1-salary_scale)^k*contribution = salary_scale*contribution*((1-salary_scale)^{age-25+1}-1)/(1-scale-1)
fund = salary*contribution*((1-salary_scale)**(age-24)-1)/(-salary_scale)


# Visualize explanatory variables of portfolio
fig, ax = plt.subplots(1,5, figsize=(13,3))
ax=ax.flatten()
sns.distplot(age, norm_hist= True, kde=False, color = 'gray', ax=ax[0], bins = np.unique(age))
sns.distplot(fund,norm_hist= True, kde=False, color = 'gray',ax=ax[1])
sns.distplot(salary, norm_hist= True,kde=False, color = 'gray',ax=ax[2])
sns.distplot(salary_scale,norm_hist= True,kde=False, color = 'gray', ax=ax[3])
sns.distplot(contribution,norm_hist= True,kde=False,  color = 'gray',ax=ax[4])

for i in range(5):
    ax[i].set_xlabel(r'$X_{}$'.format(i+1), fontsize = 16)
    ax[i].tick_params(axis='x', labelsize= 14 )
    ax[i].tick_params(axis='y', labelsize= 14 )
    ax[i].xaxis.offsetText.set_fontsize(14)
    ax[i].yaxis.offsetText.set_fontsize(14)

ax[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax[2].ticklabel_format(axis="x", style="sci", scilimits=(0,0))


ax[0].set_xticks([30,40,50,60])
ax[1].set_xticks([0, 50000,100000])
ax[2].set_xticks([0, 60000, 120000])
ax[3].set_xticks([0.01, 0.05])
ax[4].set_xticks([0.01, 0.05])
plt.tight_layout()
sns.despine()
plt.savefig(cd+r'/Matplotlib_figures/Data_Pensions.png', dpi = 400)
plt.savefig(cd+r'/Matplotlib_figures/Data_Pensions.eps', dpi = 400)
# plt.show()
plt.close()
exit()

data = np.transpose(np.array([fund,age,salary,salary_scale, contribution]))
data_sc = data_prep_feautures_scale(data, Min_Max, option = 'standard')
targets = pension_reserve(data, pension_age_max= params['pension_age_max'], age_min= explan_vars_range['age'][0], interest_std = params['interest_rate'], 
                          ep_structure= params['early_pension_structure'])

# visual back-test for computation of targets
for i in range(10):
    plt.plot(targets[i,:])
plt.xlabel('Time')
plt.title('Examplary expected pensions')
plt.show()

# ## 2.2. Train-/ Test-Split -> not relevant for Grouping of NEW_Portfolio !!!
# Complete Data
#data_train,data_test = data_prep_split(data, ratio_tr_tst)
#data_train_sc, data_test_sc = data_prep_split(data_sc, ratio_tr_tst)
#targets_train, targets_test= data_prep_split(targets,ratio_tr_tst)


X = pd.DataFrame(data_sc, columns= list(explan_vars_range.keys()))
X_raw = pd.DataFrame(data, columns= list(explan_vars_range.keys()))
y = pd.DataFrame(targets, columns = ['t{}'.format(i) for i in range(targets.shape[1])])

# export data
X.to_csv(path_data +'NEW_X.csv')
X_raw.to_csv(path_data+'NEW_X_raw.csv')
y.to_csv(path_data +'NEW_y.csv')

V_max = get_pension_reserve(fund_accum = explan_vars_range['fund'][1], age = explan_vars_range['age'][0], salary = explan_vars_range['salary'][1], 
                            salary_scale = explan_vars_range['salary_scale'][1], contribution = explan_vars_range['contribution'][1], 
                             A = 0.00022, B = 2.7*10**(-6), c = 1.124, interest = params['interest_rate'],
                            pension_age_max = 67, early_pension = params['early_pension_structure']).max()
print('V_max: ' + str(V_max))
# Tabular version of range of targets

df = pd.DataFrame(data= None, index = None, columns = ['25th percentile', 'median', '75th percentile', 'max.'])
df.loc[r'$\max_{t} Y(\omega)_t$']= [np.quantile(a = targets.max(axis=1), q=0.25),np.quantile(a = targets.max(axis=1), q=0.5),
                             np.quantile(a = targets.max(axis=1), q=0.75),targets.max()]
print(df)

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
plt.tight_layout()
plt.show()

# Compare training and test data -> backtest scaling procedure: pairplots should visually match (ignoring scale)
sns.pairplot(X)
sns.pairplot(X_raw)
plt.show()