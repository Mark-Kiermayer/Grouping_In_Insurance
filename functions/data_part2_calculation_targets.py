#!/usr/bin/env python
# coding: utf-8

import numpy as np 
from actuarial_functions import get_termlife_reserve_profile

# ## Calculate Target Values

def data_simulation_targets(data, dur_max, int_rate, A=None, B=None, c= None, shock_mort = 0, shock_int = 0):
    
    N_contracts = data.shape[0]
    # Initilize arrays
    targets = np.zeros([N_contracts,dur_max +1] )
    
    for i in range(N_contracts):
        # Targets of full dimensional feature space with stochastic interest rate
        targets[i,0:max(data[i,2]-data[i,3]+1,0)] = get_termlife_reserve_profile(age_curr = data[i,0], 
                                                                    Sum_ins = data[i,1],duration= data[i,2],  
                                                                    interest = int_rate[i]+shock_int, age_of_contract = data[i,3], 
                                                                    option_past=False, A=A, B=B,c=c, shock_mort=shock_mort)
    return targets        




# Apply classical actuarial computation to obtain target values for given data
# Optionally: Computation also for 1-dimensional projections w.r.t age, sum, duration and age of contract
# dpreciated version -> included early 1-dimensional pre-analysis
def data_simulation_targets_old(data, dur_max,data_age=None, data_sum=None, data_dur=None, data_aoc=None, 
                            age_std=None, sum_std=None, dur_std=None, aoc_std=None, 
                            int_rate = 0.05, option_1dim= False, A=None, B=None, c= None):
    N_contracts = data.shape[0]
    
    # Initilize arrays
    targets = np.zeros([N_contracts,dur_max +1] )
    
    if option_1dim == False:
        if type(int_rate) != type(1):
            for i in range(N_contracts):
                # Targets of full dimensional feature space with stochastic interest rate
                targets[i,0:max(data[i,2]-data[i,3]+1,0)] = get_termlife_reserve_profile(age_curr = data[i,0], 
                                                                         Sum_ins = data[i,1],duration= data[i,2],  
                                                                         interest = int_rate[i], age_of_contract = data[i,3], 
                                                                     option_past=False, A=A, B=B,c=c)
        else:
            for i in range(N_contracts):
                # Targets of full dimensional feature space with fixed interest rate
                targets[i,0:max(data[i,2]-data[i,3]+1,0)] = get_termlife_reserve_profile(age_curr = data[i,0], 
                                                                         Sum_ins = data[i,1],duration= data[i,2],  
                                                                         interest = int_rate, age_of_contract = data[i,3], 
                                                                     option_past=False, A=A, B=B,c=c)
        return targets        
    
    else:    
        # init. 1-dim values
        targets_age, targets_sum, targets_dur, targets_aoc = np.zeros([N_contracts,dur_std+1]), np.zeros([N_contracts,dur_std+1]),np.zeros([N_contracts,dur_max+1]),np.zeros([N_contracts,dur_std+1])
        for i in range(N_contracts):
            # Targets of full dimensional feature space
            targets[i,0:max(data[i,2]-data[i,3]+1,0)] = get_termlife_reserve_profile(age_curr = data[i,0], 
                                                                     Sum_ins = data[i,1],duration= data[i,2],  
                                                                     interest = int_rate, age_of_contract = data[i,3], 
                                                                 option_past=False)
        
            # Targets of projected feature space
            targets_age[i,:] = get_termlife_reserve_profile(age_curr= data_age[i], Sum_ins = sum_std, duration = dur_std, 
                                                            age_of_contract = aoc_std,interest= int_rate,option_past=False)
            targets_sum[i,:] = get_termlife_reserve_profile(age_curr= age_std, Sum_ins = data_sum[i], duration = dur_std, 
                                                            age_of_contract = aoc_std,interest= int_rate,option_past=False)
            targets_dur[i,0:(data_dur[i]+1)] = get_termlife_reserve_profile(age_curr= age_std, Sum_ins = sum_std, 
                                duration = data_dur[i], age_of_contract = aoc_std, interest= int_rate,option_past=False)
            targets_aoc[i,0:max(dur_std-data_aoc[i]+1,0)] = get_termlife_reserve_profile(age_curr= age_std, 
                                                     Sum_ins = sum_std, duration = dur_std, age_of_contract = data_aoc[i], 
                                                                             interest= int_rate,option_past=False)
        return [targets, [targets_age, targets_sum, targets_dur, targets_aoc]]





