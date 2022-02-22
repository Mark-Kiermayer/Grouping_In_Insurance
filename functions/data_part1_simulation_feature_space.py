#!/usr/bin/env python
# coding: utf-8

# # Data Simulation - Part I: Feature Space
import  numpy as np 
import sobol_seq


##### Simulate Data for Reserve Calculation #####

# Option practical: Age of contract is simulated as a random share of the respective duration
# Option all-over: All contracts are simulated as random values on their Min-Max Range

def data_simulation_features(A, B, c, N_contracts, Max_min, N_features=4, option_1dim_data = False):
   
    data = np.zeros([N_contracts,N_features])


    # Use Sobol Sequence to optimize uniform coverage of multidimensional feature space
    data = sobol_seq.i4_sobol_generate(N_features,N_contracts)

    
    # Simulate initial ages of policyholders upon signing the contract
    # Note: We first simulate the initial age at start of contract (-> use age_up = Max_min[0,1]-Max_min[2,0])
    data[:,0] =(Max_min[0,0] + (Max_min[0,1]-Max_min[2,1]-Max_min[0,0])*data[:,0])

    # simulate sums insured
    data[:,1] = (Max_min[1,0]+(Max_min[1,1]-Max_min[1,0])*data[:,1])
    # simulate duration
    data[:,2] = (Max_min[2,0]+(Max_min[2,1]-Max_min[2,0])*data[:,2])
    # random percentage of duration has passed; at least one remaining year of contract (hence dur - 1)
    data[:,3] = data[:,3]*(data[:,2]-1)         
    # obtain current age of policyholder (initial age + age of contract)
    data[:,0] = data[:,0] + data[:,3]
    # ceil data as integers
    data[:,0:4] = data[:,0:4].astype('int')
    
    # optional: interest rate
    if N_features == 5:
        data[:,4] = (Max_min[4,0]+(Max_min[4,1]-Max_min[4,0])*data[:,4])
    
    
    
    if option_1dim_data == True:
        ### Lower-dimensional Datasets
        data_age = data[:,0]
        data_sum = data[:,1]
        data_dur = data[:,2]
        data_aoc = data[:,3]
        

        return [data, [data_age, data_sum, data_dur, data_aoc]]
    else:
        return data

