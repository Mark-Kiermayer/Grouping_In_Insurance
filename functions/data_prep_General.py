import numpy as np
import copy


def data_prep_feautures_scale(data, Max_min, option = 'standard'):
    
    '''
    Perform feature scaling w.r.t. values in Max_min matrix
    Optionally: scale the elapsed duration conditionally to the duration of the contract.
    '''

    #### Scale feature components to [-1,+1]
    # Max_min: Row 0 -> Minima of features, Row 1 -> Maxima of features
    data_sc = np.zeros(shape = data.shape)
    
    data_sc = 2*(data- Max_min[:,0])/(Max_min[:,1]-Max_min[:,0]) -1
    if option == 'conditional':
        # conditional scalin of elapsed duration (alias age of contract) relative to max duration of contract
        data_sc[:,3] = 2*data[:,3]/data[:,2] -1
    
    return data_sc



def data_re_transform_features(data_scaled, Max_min, option = 'conditional'):
    
    '''
    Reverse feature scaling performed by data_prep_feautures_scale().
    '''
    
    #### Transform feature components from [-1,+1] to their previous range
    # Max_min: Row 0 -> Minima of features, Row 1 -> Maxima of features
    data_previous = ((data_scaled+1)/2*(Max_min[:,1]-Max_min[:,0])+Max_min[:,0])
    if option == 'conditional':
        # Re_transform 'Age of contract' separately
        data_previous[:,3] = (data_scaled[:,3]+1)/2*data_previous[:,2]
    
    return np.array(data_previous)#.reshape((-1,len(Max_min)))

def data_prep_change_scale(data, Max_min):

    '''
    Change scaling from min-max to conditional
    Note: This only affects the component 'elapsed duration' (component no. 3) of term life contracts
    '''

    data_new = copy.deepcopy(data)
    # obtain raw elapsed duration
    data_new[:,3] = (data_new[:,3]+1)/2*(data_new[:,2]+1)/2*(Max_min[2,1]-Max_min[2,0])
    # min-max scale elapsed duration
    data_new[:,3] = 2*data_new[:,3]/(Max_min[3,1]-Max_min[3,0])-1
    return data_new



def data_prep_targets_scale(value, scale_up, scale_low =0, logarithmic = False):

    '''
    We check target scaling as preprocessing (in contrast to including it in the neural network architecture) 
    However, eventually we drop it due to inefficency and apply an internal scaling layer. 
    '''
    if logarithmic == False:
        return 2*(value-scale_low)/(scale_up-scale_low)-1
    else:
        return 2*(np.log(1+value)-np.log(1+scale_low))/(np.log(1+scale_up)-np.log(1+scale_low))-1



# Split data w.r.t. a given share/ ratio.
def data_prep_split(data, split_ratio):
    '''Split (raw and scaled) Data in Training and Test Set'''
    N = len(data)
    N_train = int(split_ratio*N)       
    
    return data[0:N_train,],data[N_train:,]
