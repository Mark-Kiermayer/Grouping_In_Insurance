#!/usr/bin/env python
# coding: utf-8

########################## Section 0 - Import Tools  ################################

# ## 0.1. Import Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle, os, random 


### 0.2. Import customized functions
from functions.actuarial_functions import get_termlife_reserve_profile, get_historic_interest
from functions.data_part2_calculation_targets import data_simulation_targets
from functions.data_prep_General import data_prep_feautures_scale

# Dataframe representation
pd.set_option('precision', 4)


########################## Section 1 - Global Parameters  #################################
cd = os.getcwd()
path_data = cd+'/TermLife/Data/'
# Load general assumptions
with open(path_data+'TL_params.pkl', 'rb') as f:
    params = pickle.load(f)
with open(path_data+'TL_explan_vars_range.pkl', 'rb') as f:
    explan_vars_range = pickle.load(f)

print('Parameters imported: ', params)
print('Explanatory variables imported: ', explan_vars_range)
print('V_max: ', get_termlife_reserve_profile(age_curr=explan_vars_range['age'][1], Sum_ins = explan_vars_range['sum_ins'][1], 
                                      duration=explan_vars_range['duration'][1], interest = explan_vars_range['interest_rate'][0]).max())
V_max = params['V_max']

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
random.seed(42)
age_init = (explan_vars_range['age'][0]+np.random.gamma(shape = 10, scale = 2.5, size = N_contracts)).astype('int')
sum_ins = (10000 + np.random.gamma(shape = 2, scale = 70000, size = N_contracts)).astype('int')
duration = 5 + np.random.gamma(shape=10, scale =1, size=N_contracts).astype('int')
age_of_contract = (np.random.uniform(size=N_contracts)*duration).astype('int')
duration_rem = duration - age_of_contract
interest_rate = get_historic_interest(age_of_contract)

# adjust age to current age
age = age_init+age_of_contract

# Visualize explanatory variables of portfolio
fig, ax = plt.subplots(1,5, figsize=(13,3))#(9,2))
ax=ax.flatten()
sns.distplot(age, norm_hist= True, kde=False, color = 'gray', ax=ax[0], bins = np.unique(age))
sns.distplot(sum_ins,norm_hist= True, kde=False, color = 'gray',ax=ax[1])
sns.distplot(duration, norm_hist= True,kde=False, color = 'gray',ax=ax[2], bins = np.unique(duration))
sns.distplot(age_of_contract,norm_hist= True,kde=False, color = 'gray', ax=ax[3], bins = np.unique(age_of_contract))

el_rate, count_rate = np.unique(interest_rate, return_counts=True)
ax[4].bar(el_rate, count_rate/len(interest_rate), width = 0.001, color='grey')
# sns.distplot(interest_rate,norm_hist= True,kde=False,  color = 'gray',ax=ax[4])#, bins = np.unique(interest_rate))
for i in range(5):
    ax[i].set_xlabel(r'$X_{}$'.format(i+1), fontsize = 16) #'x-large')
    ax[i].tick_params(axis='x', labelsize= 14 )
    ax[i].tick_params(axis='y', labelsize= 14 )
    ax[i].xaxis.offsetText.set_fontsize(14)
    ax[i].yaxis.offsetText.set_fontsize(14)

plt.tight_layout()
sns.despine()
ax[0].set_xticks([50,75,100])
ax[1].set_xticks([0, 0.5e6,1e6])
ax[2].set_xticks([10,20,30])
ax[3].set_xticks([0,10,20,30])
ax[4].set_xticks([0.01, 0.02,0.03,0.04])
#fig.suptitle('Explanatory Variables of realistic Portfolio.')
plt.savefig(cd+r'/Matplotlib_figures/Data_TL.png', dpi=400)
plt.savefig(cd+r'/Matplotlib_figures/Data_TL.eps', dpi=400)
# plt.show()
plt.close()

# write data in DataFrame
explan_vars = pd.DataFrame(data = None, columns= ['age','sum_ins', 'duration', 'age_of_contract', 'interest_rate'] )
explan_vars['age'], explan_vars['sum_ins'], explan_vars['duration'] = age, sum_ins, duration
explan_vars['age_of_contract'], explan_vars['interest_rate'] = age_of_contract, interest_rate

print(explan_vars.head())
explan_vars=explan_vars.values

# targets
targets = data_simulation_targets(data=explan_vars[:,0:4].astype('int'), dur_max= explan_vars_range['duration'][1], A=params['A'], B=params['B'],c= params['c'],
                                     int_rate= explan_vars[:,4] )

index_no_zero_pad = (targets!=0)
index_no_zero_pad[:,0] = True # include initial reserve, even if 0.

sns.distplot(targets[index_no_zero_pad].flatten())
plt.title('Distribution of Reserve Values (exluding padded zeros).')
plt.show()

# ### 2.4.5. Visualization - Range of Targets
# Tabular version of range of targets
df = pd.DataFrame(data= None, index = None, columns = ['25% percentile', 'median', '75% percentile', 'max.'])
df.loc[r'$\max_{t} Y(\omega)_t$']= [np.quantile(a = targets.max(axis=1), q=0.25),np.quantile(a = targets.max(axis=1), q=0.5),
                             np.quantile(a = targets.max(axis=1), q=0.75),targets.max()]
print(df)

fig, [ax1, ax2] = plt.subplots(1,2, figsize = (12,4))
ax1.hist(targets.max(axis=1), bins = 500, density = False)
ax1.axvline(targets.max(axis=1).max(),color = 'red', linestyle = '-.', label = 'Maximum Policy Value')
ax1.legend(loc=(0.5,0.8))
ax1.set_ylabel('Absolute Frequency', fontsize = 'x-large')
ax1.set_xlabel('Policy Values', fontsize = 'x-large')

ax2.hist(2*np.log(1+targets.max(axis=1))/np.log(1+V_max)-1, bins = 40, density=False)
ax2.axvline((np.log(1+targets.max(axis=1))/np.log(1+V_max)).max(),color = 'red', linestyle = '-.')
ax2.set_xlabel('Log-Scaled Policy Values', fontsize = 'x-large')

#ax2.set_title('Histogram of maximal reserves per contract on a log-scale')
plt.tight_layout()
plt.show()


# Create Portfolio
X_raw = pd.DataFrame(explan_vars, columns= list(explan_vars_range.keys()))
y = pd.DataFrame(targets, columns = ['t{}'.format(i) for i in range(targets.shape[1])])
X = pd.DataFrame(data_prep_feautures_scale(explan_vars, Max_min, option = 'conditional'), 
                columns= list(explan_vars_range.keys()))# export data

# export data
X.to_csv(path_data +'NEW_X.csv')
X_raw.to_csv(path_data+'NEW_X_raw.csv')
y.to_csv(path_data +'NEW_y.csv')
