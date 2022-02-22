import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, sobol_seq, os
from sklearn import cluster
import tensorflow as tf
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Input, Lambda, Average, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


from functions.rnn_functions import combine_models, create_rnn_model
from functions.actuarial_functions import get_termlife_reserve_profile, get_pension_reserve
from functions.data_prep_General import data_re_transform_features, data_prep_feautures_scale, data_prep_change_scale
from functions.visualization_functions import set_size



# Get the number of members for each cluster
def kmeans_counts(data_labels, clusters):
    
    counts = np.zeros(shape = (clusters,1))
    for i in range(clusters):
        counts[i] = (data_labels == i).sum()
    
    return counts

def get_number_of_clusters(data, K_rel):

    '''
    Obtain number of clusters, given data and a relative number K_rel, e.g. 1/100 represents 1 cluster per 100 contracts.
    At the core lies a pre-processing of data w.r.t. the actuarial interest rates (last column)
    '''

    _,count = np.unique(data[:,-1], return_counts=True)
    count = sum((np.ceil(count*K_rel)).astype('int'))
    return count



def dict_to_array(dictionary):
    '''
    transform dictionary format of NN-grouping model points (1 per cluster) to array
    '''

    N = len(dictionary.keys())
    return np.array([dictionary[i][0] for i in range(N)]).reshape((N,-1))

def termlife_km_centroids_prediction(data_kmeans):

    '''
    Function for compact notation of predicting km model points to fund volume at time 0

    Inputs:
    -------
        data_kmeans: numpy array with Kxn data (no. of clusters x features per contract)

    Output:
    -------
        numpy array Kx1, representing policy values of model points at time 0
    '''
    km_low = np.zeros((len(data_kmeans),2))
    km_up = np.zeros((len(data_kmeans),2))

    for i in range(len(data_kmeans)):
        km_low[i,:] = get_termlife_reserve_profile(age_curr = np.floor(data_kmeans[i,0]).astype('int'), 
                                                Sum_ins= data_kmeans[i,1],
                                                duration = np.floor(data_kmeans[i,2]).astype('int'), 
                                                interest = data_kmeans[i,4],
                                                #interest = interest_rate,
                                                age_of_contract = np.floor(data_kmeans[i,3]).astype('int'), 
                                                option_past = False)[0:2]


        km_up[i,:] = get_termlife_reserve_profile( age_curr = np.ceil(data_kmeans[i,0]).astype('int'), 
                                                Sum_ins= data_kmeans[i,1],
                                                duration = np.ceil(data_kmeans[i,2]).astype('int'), 
                                                interest = data_kmeans[i,4], 
                                                #interest = interest_rate,
                                                age_of_contract = np.ceil(data_kmeans[i,3]).astype('int'), 
                                                option_past = False)[0:2]
    pred = (km_low+km_up)/2
    return (pred[:,0], pred[:,1])



def get_representatives_model(N_input=None, input_option_cluster_scale = False, N_features = 5, 
                              scale=1, N_output = 41, N_ensembles = 1, log_scaling = True,  **args):
    

    '''
    Create Model for the determination of a cluster's representative
    Procedure will have to be repeated for each cluster
    Input: N number of members of cluster, scale: scaling factor in Ensemble model
    Output: Model
    '''


    count = 0
    INPUT = Input(shape = (N_input,))
    
    if input_option_cluster_scale:
        INPUT_scale = Input(shape = (1,))
    count +=1
    x = Dense(units = N_features, activation = 'tanh', use_bias = False)(INPUT)
    count +=1
        
    if N_ensembles >1:
        # include previous choice of Ensemble Model        
        OUTPUT = combine_models(input_layer=x, n_ensembles= N_ensembles, 
                                load_weights= False, weights_ensembles = None, 
                                scale = scale, LSTM_nodes= [N_output],  FFN_nodes = [N_output],
                                dense_act_fct= 'tanh',
                                return_option = 'output')
        
    else: # For a 1-Model-Ensemble the average-Layer cannot be evaluated
        OUTPUT = create_rnn_model(model_input=x,widths_rnn= [N_output], widths_ffn=[N_output], 
                                optimizer_type='adam',loss_type='mse', 
                                metric_type='mae', dense_act_fct= 'tanh',
                                dropout_rnn=0, option_recurrent_dropout= False,
                                lambda_layer = True, lambda_scale =scale, 
                                log_scale=log_scaling, return_option = 'output')
        
        
    if input_option_cluster_scale:
        OUTPUT = Lambda( lambda xvar: xvar*INPUT_scale)(OUTPUT)
        model = Model(inputs = [INPUT, INPUT_scale], outputs = OUTPUT)
    else:    
        model = Model(inputs = INPUT, outputs = OUTPUT)

    for i in range(count,len(model.layers)):
        model.layers[i].trainable = False
    
    return model


# version with optimized, functional model setup;
def cluster_ann(y_lst, model_prediction, N_centroids = 1,
                cluster_object = None, context = 'pension', Max_min = None,
                N_epochs = 4000, **args):
    
    '''
    Create a model with a nested prediction model for policy values. Here, we automize the procedure to obtain optimal model points.
    mse-loss, mae-metric, EarlyStopping and Adam(lr, epsilon=10**(-5)) with lr-decay are all hard encoded.
    The lr-schedule is combined with EarlyStopping (incl. patience) and a patience (5 iterations) on the lr-decay which is triggered if the mse does not decrease by at least factor 0.01.

    Inputs:
    --------
        y_lst: List of target values for each clusters.
        model_prediction: pretrained prediction model.
        N_centroids: Number of model points per cluster.
        init_centroids: Centroids of K-Means which serve as initial values.
        context: string, either 'pension' or 'termlife'. Relevant for adding random noise to inital centroids when N_centroids > 1

    Output: list of representatives and list of models
    '''
    # Parameters
    N_clusters = len(y_lst)
    N_features = model_prediction.input_shape[1]
    bool_gpu = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    lr_start = 0.1 # initial lr for Adam
    lr_decay = 0.5 #every
    adam_eps = 10**(-5) # epsilon factor for Adam
    epoch_iteration = 2000 # epochs until learning rate gets decreased
    epoch_toleranz = 5 # patience, if loss doesn't increase despite lowering the learning rate and local fine-tuning
    es = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

    print('----------------------------------------------')
    print('Learning setting: ')
    print('\t Optimizer: Adam(lr={}, epsilon={})'.format(lr_start, adam_eps))
    print('\t Lr-decay: ', lr_decay)
    print('\t Epochs per iter.:', epoch_iteration)
    print('\t Patience (lr | es): ', epoch_toleranz, ' | ', 50)
    print('----------------------------------------------')

    # central quantities from ANN grouping
    init_centroids = cluster_object.cluster_centers_
    if context == 'termlife':
        # transform to KMeans centroids conditional scaling of elapsed duration to improve initial weights
        init_centroids = data_prep_change_scale(init_centroids, Max_min)

    member_counts = kmeans_counts(cluster_object.labels_, N_clusters)
    
    # placeholders
    representatives = {}
    representatives_pv = {}
    history = {}
    times = {}
    t_start = time.time()


    if bool_gpu: 
        with tf.device('/gpu:0'):
            INPUT = Input(shape=(1,))
            model_prediction_copy = clone_model(model_prediction)
            model_prediction_copy.set_weights(model_prediction.get_weights())
            model_prediction_copy.trainable = False
            mps = [Dense(units=N_features, use_bias = False, activation='tanh', name = 'Centroid_{}'.format(i))(INPUT) for i in range(N_centroids)]
            pred = [model_prediction_copy(val) for val in mps]
            # Note: for simplicity we assume equal weights, i.e. average
            if N_centroids > 1:
                OUTPUT = Average()(pred)
            elif N_centroids == 1:
                OUTPUT = pred
            else:
                print('ValueError: Expected integer value for "N_centroids", got: ', N_centroids)

            model_clustering = Model(INPUT, OUTPUT)
            model_clustering.compile(loss='mse', metrics= ['mae'], optimizer = Adam(lr=lr_start, epsilon=adam_eps))
    else: # exact same model setup, without distribution strategy
        INPUT = Input(shape=(1,))
        model_prediction.trainable = False
        mps = [Dense(units=N_features, use_bias = False, activation='tanh', name = 'Centroid_{}'.format(i))(INPUT) for i in range(N_centroids)]
        pred = [model_prediction(val) for val in mps]
        # Note: for simplicity we assume equal weights, i.e. average
        if N_centroids > 1:
            OUTPUT = Average()(pred)
        elif N_centroids == 1:
            OUTPUT = pred
        else:
            print('ValueError: Expected integer value for "N_centroids", got: ', N_centroids)

        model_clustering = Model(INPUT, OUTPUT)
        model_clustering.compile(loss='mse', metrics= ['mae'], optimizer = Adam(lr=lr_start, epsilon=adam_eps))
    
    print('Model set up. Time required: ' + str(np.round_(time.time()-t_start, decimals = 2))+ ' sec.')

    
    for i in range(N_clusters):   
        # Note: empirically centroids can have component \in {-1,+1} 
        # -> not admissible for arctanh() -> add constant for numeric stability to avoid divide by zero error!
        index = (init_centroids == -1) | (init_centroids == 1)
        if  index.sum() > 0:
            init_centroids[index]-= init_centroids[index]*10**(-7)
            print('\t Numeric stability correction applied.' )


        if N_centroids>1:
            random = sobol_seq.i4_sobol_generate(dim_num = 1, n = int(N_centroids/2))
            if context == 'pension': 
                # add noise to age to break symmetrie of NN for multiple model points-> time until maturity/ pension of mps varies
                w_low = [np.array([np.arctanh(init_centroids[i,0])]+
                                    list(init_centroids[i,1]+random[k]*(1-init_centroids[i,1]))+
                                    list(np.arctanh(init_centroids[i,2:]))
                                ).reshape((1,N_features)) for k in range(int(N_centroids/2))]
                w_up = [np.array([np.arctanh(init_centroids[i,0])]+
                                    list(init_centroids[i,1]-random[k]*(1+init_centroids[i,1]))+
                                    list(np.arctanh(init_centroids[i,2:]))
                                ).reshape((1,N_features)) for k in range(int(N_centroids/2))]
            elif context == 'termlife': 
                # add noise to duration to break symmetrie of NN for multiple model points -> time until maturity/ pension of mps varies
                w_low = [np.array(list(np.arctanh(init_centroids[i,0:2]))+
                                    list(np.arctanh(init_centroids[i,2]+random[k]*(1-init_centroids[i,2])))+
                                    list(np.arctanh(init_centroids[i,3:]))
                                ).reshape((1,N_features)) for k in range(int(N_centroids/2))]
                w_up = [np.array(list(np.arctanh(init_centroids[i,0:2]))+
                                    list(np.arctanh(init_centroids[i,2]-random[k]*(1+init_centroids[i,2])))+
                                    list(np.arctanh(init_centroids[i,3:]))
                                ).reshape((1,N_features)) for k in range(int(N_centroids/2))]
            else:
                print('Error: context unknown!')
                exit()
        
        
        if N_centroids ==1:
            w_clustering_init = [np.arctanh(init_centroids[i,:]).reshape((1,N_features))]
        elif N_centroids%2 == 0: 
            # Equal share of ANN centroids initialized left and right of K-means centroid
            w_clustering_init = w_low+w_up
        else:
            # N_centroids uneven -> Initialize 1x K-means centroid and distribute rest equally right and left
            w_clustering_init = [np.arctanh(init_centroids[i,:]).reshape((1,N_features))]+w_low+w_up  
        
        # set weights
        for k in range(N_centroids):
            model_clustering.get_layer(name='Centroid_{}'.format(k)).set_weights([w_clustering_init[k]])

        t_start = time.time()
        print('Model for Cluster {} of {}'.format(i+1,len(y_lst)))        
        print('\t Training in progress')
        
        model_clustering.fit(x=np.array([1],ndmin=2), y=y_lst[i], batch_size=1, epochs = epoch_iteration,#N_epochs, 
                            callbacks= [es], verbose = 0) #callbacks= [es, lr_schedule]
        history[i] = model_clustering.history.history['loss']

        cache_loss = min(history[i])
        cache_toleranz = 0
        for ep in range(N_epochs//epoch_iteration):

            if ((min(history[i])- cache_loss)/cache_loss>0.01) | (ep==0) | (cache_toleranz<epoch_toleranz):    
                if ((min(history[i])- cache_loss)/cache_loss<0.01) & (ep>0):
                    cache_toleranz +=1   
                cache_loss = min(history[i]) # update
                # decrease learning rate (with restored, optimal weights(!!!) by the use of EarlyStopping(..., restore_best_weights=True))
                if bool_gpu: 
                    with tf.device('/gpu:0'):
                        model_clustering.compile(loss='mse', metrics= ['mae'], optimizer = Adam(lr=lr_start*lr_decay**(ep+1), epsilon=adam_eps))
                else:
                    model_clustering.compile(loss='mse', metrics= ['mae'], optimizer = Adam(lr=lr_start*lr_decay**(ep+1), epsilon=adam_eps))
                # Fine tune model
                print(' \t Fine tuning ', ep, '...', ' (mse: ', np.round_(cache_loss,4), ')')


                model_clustering.fit(x=np.array([1],ndmin=2), y=y_lst[i], batch_size=1, epochs = epoch_iteration,#N_epochs, 
                                callbacks= [es], verbose = 0) #callbacks= [es, lr_schedule]
                # update training history
                history[i] += model_clustering.history.history['loss']
            else:
                break # no significant training progress AND patience/ toleranz has been surpassed
            
        times[i] = np.round_(time.time()-t_start, decimals = 2)

        print(' \t Cluster {} completed. Epochs: {}, time passed '.format(i+1, len(history[i]))+ str(times[i])+ ' sec.')

              
        # Save representative (i.e. hidden output)
        # Note: get_weights returns list with weights, no array-type
        representatives[i] = [np.tanh(np.array(model_clustering.get_layer(
            name = 'Centroid_{}'.format(k)).get_weights()[0])) for k in range(N_centroids)]
        # Save respective policy values
        representatives_pv[i] = [ model_prediction.predict(x = [representatives[i][k]])[0] for k in range(N_centroids)]
    
        # run next iteration of i (up to number of clusters)
    # Note: history returned as values, not dict -> dict cannot be dumped using pickle;
    return [representatives, representatives_pv, member_counts, times, history]

def cluster_ann_test(y_lst, model_prediction, N_centroids = 1,
                cluster_object = None, context = 'pension', Max_min = None,
                N_epochs = 4000, es_patience = 100, wd_cluster= None):
    
    '''
    Create a model with a nested prediction model for policy values. Here, we automize the procedure to obtain optimal model points.
    IMPORTANT: The interest rate is not adjusted, but the K-means value is adopted! It is common practice to group only contract of similar actuarial interest rates!
    mse-loss, mae-metric, EarlyStopping and Adam(lr, epsilon=10**(-5)) with lr-decay are all hard encoded.
    The lr-schedule is combined with EarlyStopping (incl. patience) and a patience (5 iterations) on the lr-decay which is triggered if the mse does not decrease by at least factor 0.01.

    Inputs:
    --------
        y_lst: List of target values for each clusters.
        model_prediction: pretrained prediction model.
        N_centroids: Number of model points per cluster.
        init_centroids: Centroids of K-Means which serve as initial values.
        context: string, either 'pension' or 'termlife'. Relevant for adding random noise to inital centroids when N_centroids > 1

    Output: list of representatives and list of models
    '''
    # Parameters
    N_clusters = len(y_lst)
    N_features = model_prediction.input_shape[1]
    bool_gpu = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    lr_start = 0.1 # initial lr for Adam
    lr_decay = 0.5 #every
    adam_eps = 10**(-5) # epsilon factor for Adam
    epoch_iteration = min(2000,N_epochs) # epochs until learning rate gets decreased
    epoch_toleranz = 5 # patience, if loss doesn't increase despite lowering the learning rate and local fine-tuning
    es = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

    print('----------------------------------------------')
    print('Learning setting: ')
    print('\t Optimizer: Adam(lr={}, epsilon={})'.format(lr_start, adam_eps))
    print('\t Lr-decay: ', lr_decay)
    print('\t Epochs per iter.:', epoch_iteration)
    print('\t Patience (lr | es): ', epoch_toleranz, ' | ', 50)
    print('----------------------------------------------')

    # central quantities from ANN grouping
    init_centroids = cluster_object.cluster_centers_
    if context == 'termlife':
        # transform to KMeans centroids conditional scaling of elapsed duration to improve initial weights
        init_centroids = data_prep_change_scale(init_centroids, Max_min)

    member_counts = kmeans_counts(cluster_object.labels_, N_clusters)
    
    # placeholders
    representatives = {}
    representatives_pv = {}
    history = {}
    times = {}
    t_start = time.time()


    if bool_gpu: 
        with tf.device('/gpu:0'):
            INPUT = Input(shape=(1,))
            model_prediction_copy = clone_model(model_prediction)
            model_prediction_copy.set_weights(model_prediction.get_weights())
            model_prediction_copy.trainable = False
            mps_train = [Dense(units=N_features-1, use_bias = False, activation='tanh', name = 'Centroid_train_{}'.format(i))(INPUT) for i in range(N_centroids)]
            mps_fixed = [Dense(units=1, trainable = False, use_bias = False, activation='tanh', name = 'Centroid_fixed_{}'.format(i))(INPUT) for i in range(N_centroids)]
            mps = [Concatenate(name = 'Centroid_{}'.format(i))([mps_train[i],mps_fixed[i]]) for i in range(N_centroids)]
            pred = [model_prediction_copy(val) for val in mps]
            # Note: for simplicity we assume equal weights, i.e. average
            if N_centroids > 1:
                OUTPUT = Average()(pred)
            elif N_centroids == 1:
                OUTPUT = pred
            else:
                print('ValueError: Expected integer value for "N_centroids", got: ', N_centroids)

            model_clustering = Model(INPUT, OUTPUT)
            model_clustering.compile(loss='mse', metrics= ['mae'], optimizer = Adam(lr=lr_start, epsilon=adam_eps))
    else: # exact same model setup, without distribution strategy
        INPUT = Input(shape=(1,))
        model_prediction_copy = clone_model(model_prediction)
        model_prediction_copy.set_weights(model_prediction.get_weights())
        model_prediction_copy.trainable = False
        mps_train = [Dense(units=N_features-1, use_bias = False, activation='tanh', name = 'Centroid_train_{}'.format(i))(INPUT) for i in range(N_centroids)]
        mps_fixed = [Dense(units=1, trainable = False, use_bias = False, activation='tanh', name = 'Centroid_fixed_{}'.format(i))(INPUT) for i in range(N_centroids)]
        mps = [Concatenate(name = 'Centroid_{}'.format(i))([mps_train[i],mps_fixed[i]]) for i in range(N_centroids)]
        pred = [model_prediction_copy(val) for val in mps]
        # Note: for simplicity we assume equal weights, i.e. average
        if N_centroids > 1:
            OUTPUT = Average()(pred)
        elif N_centroids == 1:
            OUTPUT = pred
        else:
            print('ValueError: Expected integer value for "N_centroids", got: ', N_centroids)

        model_clustering = Model(INPUT, OUTPUT)
        model_clustering.compile(loss='mse', metrics= ['mae'], optimizer = Adam(lr=lr_start, epsilon=adam_eps))
    
    print('Model set up. Time required: ' + str(np.round_(time.time()-t_start, decimals = 2))+ ' sec.')

    
    for i in range(N_clusters):     
        # Note: empirically centroids can have component \in {-1,+1} 
        # -> not admissible for arctanh() -> add constant for numeric stability to avoid divide by zero error!
        index = (init_centroids == -1) | (init_centroids == 1)
        if  index.sum() > 0:
            init_centroids[index]-= init_centroids[index]*10**(-7)
            print('\t Numeric stability correction applied.' )


        if N_centroids>1:
            random = sobol_seq.i4_sobol_generate(dim_num = 1, n = int(N_centroids/2))
            if context == 'pension': 
                # add noise to age to break symmetrie of NN for multiple model points-> time until maturity/ pension of mps varies
                w_low = [np.array([np.arctanh(init_centroids[i,0])]+
                                    list(init_centroids[i,1]+random[k]*(1-init_centroids[i,1]))+
                                    list(np.arctanh(init_centroids[i,2:]))
                                ).reshape((1,N_features)) for k in range(int(N_centroids/2))]
                w_up = [np.array([np.arctanh(init_centroids[i,0])]+
                                    list(init_centroids[i,1]-random[k]*(1+init_centroids[i,1]))+
                                    list(np.arctanh(init_centroids[i,2:]))
                                ).reshape((1,N_features)) for k in range(int(N_centroids/2))]
            elif context == 'termlife': 
                # add noise to duration to break symmetrie of NN for multiple model points -> time until maturity/ pension of mps varies
                w_low = [np.array(list(np.arctanh(init_centroids[i,0:2]))+
                                    list(np.arctanh(init_centroids[i,2]+random[k]*(1-init_centroids[i,2])))+
                                    list(np.arctanh(init_centroids[i,3:]))
                                ).reshape((1,N_features)) for k in range(int(N_centroids/2))]
                w_up = [np.array(list(np.arctanh(init_centroids[i,0:2]))+
                                    list(np.arctanh(init_centroids[i,2]-random[k]*(1+init_centroids[i,2])))+
                                    list(np.arctanh(init_centroids[i,3:]))
                                ).reshape((1,N_features)) for k in range(int(N_centroids/2))]
            else:
                print('Error: context unknown!')
                exit()
        
        
        if N_centroids ==1:
            w_clustering_init = [np.arctanh(init_centroids[i,:]).reshape((1,N_features))]
        elif N_centroids%2 == 0: 
            # Equal share of ANN centroids initialized left and right of K-means centroid
            w_clustering_init = w_low+w_up
        else:
            # N_centroids uneven -> Initialize 1x K-means centroid and distribute rest equally right and left
            w_clustering_init = [np.arctanh(init_centroids[i,:]).reshape((1,N_features))]+w_low+w_up  
        
        # Set Weights 
        for k in range(N_centroids):
            model_clustering.get_layer(name='Centroid_train_{}'.format(k)).set_weights([w_clustering_init[k][:,0:-1]])
            model_clustering.get_layer(name='Centroid_fixed_{}'.format(k)).set_weights([w_clustering_init[k][:,-1:]])

        t_start = time.time()
        print('Model for Cluster {} of {}'.format(i+1,len(y_lst)))        
        print('\t Training in progress')
        
        model_clustering.fit(x=np.array([1],ndmin=2), y=y_lst[i], batch_size=1, epochs = epoch_iteration,#N_epochs, 
                            callbacks= [es], verbose = 0) #callbacks= [es, lr_schedule]
        history[i] = model_clustering.history.history['loss']

        cache_loss = min(history[i])
        cache_toleranz = 0
        for ep in range(N_epochs//epoch_iteration):
            if ((min(history[i])- cache_loss)/cache_loss>0.01) | (ep==0) | (cache_toleranz<epoch_toleranz):    
                if ((min(history[i])- cache_loss)/cache_loss<0.01) & (ep>0):
                    cache_toleranz +=1   
                cache_loss = min(history[i]) # update
                # decrease learning rate (with restored, optimal weights(!!!) by the use of EarlyStopping(..., restore_best_weights=True))
                if bool_gpu: 
                    with tf.device('/gpu:0'):
                        model_clustering.compile(loss='mse', metrics= ['mae'], optimizer = Adam(lr=lr_start*lr_decay**(ep+1), epsilon=adam_eps))
                else:
                    model_clustering.compile(loss='mse', metrics= ['mae'], optimizer = Adam(lr=lr_start*lr_decay**(ep+1), epsilon=adam_eps))
                # Fine tune model
                print(' \t Fine tuning ', ep, '...', ' (mse: ', np.round_(cache_loss,4), ')')


                model_clustering.fit(x=np.array([1],ndmin=2), y=y_lst[i], batch_size=1, epochs = epoch_iteration,#N_epochs, 
                                callbacks= [es], verbose = 0) #callbacks= [es, lr_schedule]
                # update training history
                history[i] += model_clustering.history.history['loss']
            else:
                break # no significant training progress AND patience/ toleranz has been surpassed
            

        times[i] = np.round_(time.time()-t_start, decimals = 2)

        print(' \t Cluster {} completed. Epochs: {}, time passed '.format(i+1, len(history[i]))+ str(times[i])+ ' sec.')

              
        # Save representative (i.e. hidden output of trainable & fixed components of calibrated model points)
        # Note: get_weights returns list with weights, no array-type
        representatives[i] = [np.tanh(np.array(
            list(model_clustering.get_layer( name = 'Centroid_train_{}'.format(k)).get_weights()[0][0])+
            list(model_clustering.get_layer( name = 'Centroid_fixed_{}'.format(k)).get_weights()[0][0])
            ).reshape((1,N_features))
            ) for k in range(N_centroids)]
        # Save respective policy values
        representatives_pv[i] = [ model_prediction.predict(x = [representatives[i][k]])[0] for k in range(N_centroids)]
    
        # run next iteration of i (up to number of clusters)
    # Note: history returned as values, not dict -> dict cannot be dumped using pickle;
    return [representatives, representatives_pv, member_counts, times, history]

class pseudo_KMeans_object:

    '''
    Pre-process data set w.r.t. its actuarial interest rate. Then perform KMeans-clustering on the respective subsets.
    Object combines all subset into one collective object.
    '''

    def __init__(self, data, K_rel):

        '''
        Cluster dataset data after subdividing it into clusters based on the indicated interest rate (last component).
        All subsets are then subject to KMeans clustering.

        Inputs:
        -------
            data: 2d numpy array
            K_rel: float. number of model points per contracts, e.g. 1/100.

        Output
        ------
            pseudo_KMeans_object
        '''
        

        # determine presort by interest rate
        interest_unique, count_unique = np.unique(data[:,-1], return_counts=True)
        N_unique = len(interest_unique)
        print('No. of unique interest rate: ', N_unique)

        # key attributes to mimic a KMeans object
        self.K_rel = K_rel # model points pro data, i.e. 1/100 -> 1 model point per 100 contracts in cluster
        self.K_abs = (np.ceil(count_unique*self.K_rel)).astype('int')
        self.labels_ = np.zeros((len(data),))
        self.cluster_centers_ = np.zeros((self.K_abs.sum(), data.shape[1]))        
        
        # object to save kmeans objects
        kmeans_lst = [None]*N_unique*self.K_abs.sum()

        # perform clustering on each subset/ for each unique interest rate
        for i in range(N_unique):
            index = (data[:,-1]==interest_unique[i])
            position = sum([self.K_abs[j] for j in range(i) ])
            # indicate subset by label
            self.labels_[index] = position
            kmeans_lst[i] = cluster.KMeans(init='k-means++', n_clusters=self.K_abs[i], n_init=10).fit(data[index,:])
            # transfer labels of K-means clustering {0,1,..,K} to label of subset
            self.labels_[index] += kmeans_lst[i].labels_

            # transfer centroids of K-means clustering {0,1,..,K} to label of subset
            self.cluster_centers_[position:position+self.K_abs[i],:] = kmeans_lst[i].cluster_centers_

        

# ## Evaluate Results
def analyze_agglomeration_test(baseline, ann_object, y, Max_min, insurance_type = 'pensions', 
                                #ann_representatives = None, ann_prediction = None, 
                                ann_cluster_presort = None,                              
                                option = 'plot', plot_tag = '', individual_clusters = False,
                                n_columns = 5, figsize = (20,30), option_plot_selection = None, 
                                pension_age_max = 67, interest_rate = 0.03, ep_rate = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1], include_ann = True):


    '''
    Compares (visually or statistically) an ann-agglomeration to a kmeans-baseline, based aggregated reserve of the model points.
    Note that KMeans centroid are scaled differently ('standard') than NN model points ('conditional'). This is hard encoded.
    Also, we extract the number of cluster members (for scaling to portfolio view) from the ann_cluster presort, which defaults with the baseline. 
    Hence, we assume the ann_cluster_presort (a kMeans object) to match with the cluster assignment the NN was trained on!
    Note: Statistic evaluation based on absolute differences (updated)!

    Inputs:
    -------
        baseline: kmeans object, i.e. with attributes labels_ or cluster_centers_
        ann_object: list with 0: representatives, 1: predicted pvs, 2: member_count (, 3: training times, 4: training losses)
        y:      target values, i.e. list with mean reserve values per cluster
        insurance_type: string values, either 'termlife' or 'pensions'
        Max_min: Matrix with ranges of explanatory variables; used for re-transforming scaled model points
        ann_representatives: ann model points
        ann_prediction: ann target values, i.e. reserves for each model point
        ann_cluster_presort: kmeans object which the ann grouping was initialized with; can be different to 'baseline' if multiple model points per cluster are used
        option: selection of options, i.e. 'plot' -> visualization of reserves, 'statistics' -> table with respective metrics on reserves
        plot_tag: optional string, which affects the filename of saved plots.
        individual_clusters:    boolean, whether to visualize the comparison of reserves on a per-cluster-basis
        n_columns: number of columns in visualization; rows will be selected accordingly
        figsize:    select size of figure for visualization
        option_plot_selection: Base, None; otherwise, list of integer values representing the selected clusters to be visualized
        *args: pension parameters, as pension_age_max, interest_rate, ep_rate, include_ann (boolean for displaying ann reserves)

    Outputs:
    --------
        visualization of aggregated reserves for model points and/or corresponding statistics

    '''
    # rename references (to be compatible with prev. notation)
    if include_ann:
        ann_representatives = ann_object[0]
        ann_prediction = ann_object[1]
        n_cl_ann = len(ann_representatives)
    #Number of clusters in Kmeans reference
    n_cl = baseline.cluster_centers_.shape[0]
    n_features = baseline.cluster_centers_.shape[1]
    #Number of members per cluster
    count = kmeans_counts(baseline.labels_,n_cl)
    if (ann_cluster_presort==None)&(include_ann==True):
        if n_cl==n_cl_ann:
            ann_cluster_presort=baseline
            print('Note: baseline infered from data!')
        else:
            print('Parameter ann_cluster_presort has not been assigned and cannot be infered from other input!')
            exit()

    if include_ann: # important: kMeans baseline in general != kMeans presort (e.g. if multiple model points per cluster are employed)
        count_ann = kmeans_counts(ann_cluster_presort.labels_, n_cl_ann)
        N_centroids = len(ann_representatives[0]) #N_modelpoints #

        if np.any(count_ann.flatten() != ann_object[2].flatten()):
            print('Baseline counts: ', count_ann.flatten())
            print('Cached counts: ', ann_object[2].flatten())
            raise ValueError('ValError: Count mismatch!')
    

    # Retransformed Data
    if insurance_type == 'termlife':
        data_kmeans = data_re_transform_features(baseline.cluster_centers_, Max_min, option= 'standard')
        if include_ann == True:
            data_ann = {}
            for i in range(n_cl_ann):
                # transform centroids for each cluster individually
                # dtype ann_rep: dictionary
                # dtype centroids: nparrays
                data_ann[i] = [data_re_transform_features(ann_representatives[i][k], 
                                                          Max_min, option= 'conditional').reshape((n_features,))
                               for k in range(N_centroids)]
        
    elif insurance_type == 'pensions':
        data_kmeans = data_re_transform_features(baseline.cluster_centers_, Max_min, option= 'standard')

        if include_ann == True:
            data_ann = {}
            for i in range(n_cl_ann):
                # transform centroids for each cluster individually
                # dtype ann_rep: dictionary
                # dtype centroids: nparrays
                data_ann[i] = [data_re_transform_features(ann_representatives[i][k], 
                                                          Max_min, option= 'standard').reshape((n_features,))
                               for k in range(N_centroids)]
    else: # invalid type
        print('Error: insurance_type not known/ implemented!')
        return

        
    # Calculate targets for (floored and ceiled) centroids for kmeans baseline and ANN model points
    n_out = y.shape[1]
    PV_km_low = np.zeros(shape = (n_cl, n_out))
    PV_km_up = np.zeros(shape = (n_cl, n_out))
    PV_km_pred = np.zeros(shape = (n_cl, n_out))
    PV_ann_up = {} # usage of dictionary, as potentially multiple model points per cluster are used
    PV_ann_low = {} # usage of dictionary, as potentially multiple model points per cluster are used
    
    
    # classical computation of policy values (for termlife or pension)
    for i in range(n_cl):
        if insurance_type == 'termlife':
            #print('Raw kmeans values: ', data_kmeans)    
            PV_km_low[i,0:max(np.floor(data_kmeans[i,2]).astype('int')-
                                  np.floor(data_kmeans[i,3]).astype('int'),0)+1] = get_termlife_reserve_profile(
                                                         age_curr = np.floor(data_kmeans[i,0]).astype('int'), 
                                                         Sum_ins= data_kmeans[i,1],
                                                         duration = np.floor(data_kmeans[i,2]).astype('int'), 
                                                         interest = data_kmeans[i,4],
                                                         #interest = interest_rate,
                                                         age_of_contract = np.floor(data_kmeans[i,3]).astype('int'), 
                                                         option_past = False)


            PV_km_up[i,0:max(np.ceil(data_kmeans[i,2]).astype('int')-
                  np.ceil(data_kmeans[i,3]).astype('int'),0)+1] = get_termlife_reserve_profile(
                                                        age_curr = np.ceil(data_kmeans[i,0]).astype('int'), 
                                                        Sum_ins= data_kmeans[i,1],
                                                        duration = np.ceil(data_kmeans[i,2]).astype('int'), 
                                                        interest = data_kmeans[i,4], 
                                                        #interest = interest_rate,
                                                        age_of_contract = np.ceil(data_kmeans[i,3]).astype('int'), 
                                                        option_past = False)

            # Optional: Also for representatives of ANN
            # Account for that ANN might be based on less clusters but have more centroids
            if (include_ann == True):
                if (i<n_cl_ann):
                    PV_ann_up[i] = np.zeros(shape = (N_centroids, n_out))
                    PV_ann_low[i] = np.zeros(shape = (N_centroids, n_out))

                    for k in range(N_centroids):
                        #print('cluster, centroid: ', i, k)
                        PV_ann_low[i][k][0:max(np.floor(data_ann[i][k][2]).astype('int')-
                                    np.floor(data_ann[i][k][3]).astype('int'),0)+1] = get_termlife_reserve_profile(
                                                                    age_curr = np.floor(data_ann[i][k][0]).astype('int'), 
                                                                    Sum_ins= data_ann[i][k][1],
                                                                    duration = np.floor(data_ann[i][k][2]).astype('int'), 
                                                                    interest = data_ann[i][k][4],
                                                                    #interest = interest_rate,
                                                                    age_of_contract = np.floor(data_ann[i][k][3]).astype('int'), 
                                                                    option_past = False)

                        PV_ann_up[i][k][0:max(np.ceil(data_ann[i][k][2]).astype('int')-
                                    np.ceil(data_ann[i][k][3]).astype('int'),0)+1] = get_termlife_reserve_profile(
                                                                    age_curr = np.ceil(data_ann[i][k][0]).astype('int'), 
                                                                    Sum_ins= data_ann[i][k][1],
                                                                    duration = np.ceil(data_ann[i][k][2]).astype('int'), 
                                                                    interest = data_ann[i][k][4], 
                                                                    #interest = interest_rate,
                                                                    age_of_contract = np.ceil(data_ann[i][k][3]).astype('int'), 
                                                                    option_past = False)

        else: # insurance_type == 'pensions':
            # input_used = ['Fund','Age', 'Salary', 'Salary_scale', 'Contribution', 'interest_rate']
            # Account for that ANN might be based on less clusters but have more centroids
            
            PV_km_low[i,0:max(pension_age_max- np.floor(data_kmeans[i,1]).astype('int'),0)+1] =  get_pension_reserve(fund_accum = data_kmeans[i,0], 
                                     age = np.floor(data_kmeans[i,1]).astype('int'), 
                                     salary = data_kmeans[i,2], salary_scale = data_kmeans[i,3], 
                                     contribution = data_kmeans[i,4], 
                                     A = 0.00022, B = 2.7*10**(-6), c = 1.124, 
                                     interest = interest_rate, # <-- for pension plans: interest fixed, external factor
                                     pension_age_max = pension_age_max, 
                                     early_pension = ep_rate)

            PV_km_up[i,0:max(pension_age_max- np.ceil(data_kmeans[i,1]).astype('int'),0)+1] = get_pension_reserve(fund_accum =data_kmeans[i,0], 
                                     age = np.ceil(data_kmeans[i,1]).astype('int'), 
                                     salary = data_kmeans[i,2], salary_scale = data_kmeans[i,3], 
                                     contribution = data_kmeans[i,4], 
                                     A = 0.00022, B = 2.7*10**(-6), c = 1.124, 
                                      interest = interest_rate, # <-- for pension plans: interest fixed, external factor
                                     pension_age_max = pension_age_max, 
                                     early_pension = ep_rate)

                # Optional: Also for representatives of ANN
            if (include_ann == True):
                if (i<n_cl_ann):
                    PV_ann_up[i] = np.zeros(shape = (N_centroids, n_out))
                    PV_ann_low[i] = np.zeros(shape = (N_centroids, n_out))

                    for k in range(N_centroids):
                        #print('cluster, centroid: ', i, k)
                        PV_ann_low[i][k][0:max(pension_age_max- np.ceil(data_ann[i][k][1]).astype('int'),0)+1] = get_pension_reserve(
                                                fund_accum =data_ann[i][k][0], 
                                                age = np.ceil(data_ann[i][k][1]).astype('int'), 
                                                salary = data_ann[i][k][2], salary_scale = data_ann[i][k][3], 
                                                contribution = data_ann[i][k][4], 
                                                A = 0.00022, B = 2.7*10**(-6), c = 1.124, 
                                                #interest = data_ann[i,5],
                                                interest = interest_rate,
                                                pension_age_max = pension_age_max, 
                                                early_pension = ep_rate)

                        PV_ann_up[i][k][0:max(pension_age_max- np.floor(data_ann[i][k][1]).astype('int'),0)+1] = get_pension_reserve(
                                                fund_accum =data_ann[i][k][0], 
                                                age = np.floor(data_ann[i][k][1]).astype('int'), 
                                                salary = data_ann[i][k][2], salary_scale = data_ann[i][k][3], 
                                                contribution = data_ann[i][k][4], 
                                                A = 0.00022, B = 2.7*10**(-6), c = 1.124, 
                                                #interest = data_ann[i,5],
                                                interest = interest_rate,
                                                pension_age_max = pension_age_max, 
                                                early_pension = ep_rate)

    # Calculate actual targets per cluster 
    targets_cl = np.zeros(shape = (n_cl, n_out))
    for i in range(n_cl):
        index = (baseline.labels_ == i)
        targets_cl[i,:] = y[index,:].sum(axis=0)/count[i]

    PV_km_pred = (PV_km_up[:,:]+PV_km_low[:,:])/2

    if option == 'plot':
        if (n_cl!=n_cl_ann):
                print('Warning: Number of individual clusters differes between ann and KMeans, i.e. n_cl != n_cl_ann.')
        if (individual_clusters == True) & include_ann &(n_cl==n_cl_ann):
            
            if option_plot_selection == None: # Plot all clusters C_1,..,C_K
                fig, ax = plt.subplots(nrows = np.ceil(n_cl/n_columns).astype('int'), 
                                       ncols = min(n_columns, n_cl), figsize = figsize)
                if n_cl > 1:
                    ax = ax.flatten()
                else:
                    ax = [ax]


                for i in range(n_cl):
                    # Actual Targets
                    ax[i].plot(targets_cl[i,:], 'r*', label = 'target')
                    # Reserve based on K-Means clustering
                    
                    ax[i].plot(PV_km_pred[i,:], linestyle = '-', color = 'orange', label = 'KM prediction')
                    ax[i].plot(PV_km_up[i,:], linestyle = ':', color = 'orange', #linewidth = 5, # marker = 'o', 
                               label = 'KM bound')
                    ax[i].plot(PV_km_low[i,:], linestyle = ':', color = 'orange')#, linewidth = 5)#,marker = 'o')

                    if i%n_columns==0: # first column
                        ax[i].set_ylabel('policy value', fontsize = 'large')
                    if i>= (n_columns*(np.ceil(n_cl/n_columns).astype('int')-1)): # last row
                        ax[i].set_xlabel('time, t', fontsize = 'large')

                    if (include_ann == True) & (i<n_cl_ann):
                        
                        # Predicted Reserve by ANN (overall)
                        ax[i].plot(sum(ann_prediction[i][k] for k in range(N_centroids))/N_centroids,
                                       linestyle = '-', color = 'black', 
                                       label = 'NN prediction')
                        ax[i].plot(sum(PV_ann_up[i][k] for k in range(N_centroids))/N_centroids,
                                       linestyle = '--', color = 'black', 
                                       label = 'AN bounds')
                        ax[i].plot(sum(PV_ann_low[i][k] for k in range(N_centroids))/N_centroids,
                                       linestyle = '--', color = 'black')
                        if N_centroids >1:
                            for k in range(N_centroids):

                                # Predicted Reserve by ANN (per centroid)
                                if k == 0:
                                    ax[i].plot(ann_prediction[i][k], linestyle = '-', color = 'grey', 
                                               label = 'NN prediction (MP)')
                                else:
                                    ax[i].plot(ann_prediction[i][k], linestyle = '-', color = 'grey')


                    if i == 0:
                        ax[i].legend()

                fig.tight_layout()
                plt.savefig(os.getcwd()+r'/Matplotlib_figures/Grouping{}_indiv_{}_K{}_C{}.eps'.format(plot_tag, insurance_type, n_cl_ann, N_centroids), format = 'eps')
                plt.show()
            else: ### Plot only selection of clusters C_1,...,C_[option_plot_selection]
                
                
                fig, ax = plt.subplots(nrows = 1, ncols = len(option_plot_selection), figsize=(10,3))
                ax = ax.flatten()

                for i in range(len(option_plot_selection)):
                    # Actual Targets
                    ax[i].plot(targets_cl[option_plot_selection[i],:], 'r*', label = 'target')
                    # Reserve based on K-Means clustering
                    ax[i].plot(PV_km_pred[option_plot_selection[i],:], linestyle = '-', color = 'orange', 
                               label = 'KM prediction')
                    ax[i].plot(PV_km_up[option_plot_selection[i],:], linestyle = ':', color = 'orange', 
                               label = 'KM bound')
                    ax[i].plot(PV_km_low[option_plot_selection[i],:], linestyle = ':', color = 'orange')

                    if i%n_columns==0: # first column
                        ax[i].set_ylabel('policy value', fontsize = 'large')
                    if i>= (len(option_plot_selection)-n_columns): # last row
                        ax[i].set_xlabel('Time, t', fontsize = 'large')

                    if (include_ann == True) & (i<n_cl_ann):
                        
                        # Predicted Reserve by ANN (overall)
                        ax[i].plot(sum(ann_prediction[option_plot_selection[i]][k] for k in range(N_centroids))/N_centroids,
                                       linestyle = '-', color = 'black', label = 'NN prediction')
                        ax[i].plot(sum(PV_ann_up[option_plot_selection[i]][k] for k in range(N_centroids))/N_centroids,
                                       linestyle = ':', color = 'black',label = 'NN bound')
                        ax[i].plot(sum(PV_ann_low[option_plot_selection[i]][k] for k in range(N_centroids))/N_centroids,
                                       linestyle = ':', color = 'black')
                        
                        if N_centroids>1:
                            for k in range(N_centroids):
                            # Predicted Reserve by ANN (per centroid)
                                if k == 0:
                                    ax[i].plot(ann_prediction[option_plot_selection[i]][k], linestyle = '-', color = 'grey', 
                                               label = 'NN prediction (MP)')
                                else:
                                    ax[i].plot(ann_prediction[option_plot_selection[i]][k], linestyle = '-', color = 'grey')

                    if i == 0:
                        ax[i].legend()
                plt.tight_layout()
                plt.savefig(os.getcwd()+r'/Matplotlib_figures/Grouping{}_select_{}_K{}_C{}.eps'.format(plot_tag,insurance_type, n_cl_ann, N_centroids), format = 'eps')
                plt.show()
            
        ### Plot aggregated fit (of policy values)
        if insurance_type == 'termlife':
            _, ax_cum = plt.subplots(1,1,figsize=(4,3))
        elif insurance_type == 'pensions':
            _, ax_cum = plt.subplots(1,1,figsize=set_size(fraction=.5))
        else:
            print('Insurance type unknown!')
            exit()

        ax_cum.plot(y.sum(axis=0), 'r*', label = 'target') # not scaled by numbers
        ax_cum.plot((PV_km_pred*count).sum(axis=0), color = 'orange', linestyle = '-', 
                 label ='KM prediction')
        ax_cum.plot((PV_km_up*count).sum(axis=0), color = 'orange', linestyle = ':', 
                 label ='KM bound')
        ax_cum.plot((PV_km_low*count).sum(axis=0), color = 'orange', linestyle = ':')
        if include_ann == True:
            ax_cum.plot(sum(ann_prediction[i][k]*count_ann[i]/N_centroids for i in range(n_cl_ann) for k in range(N_centroids)), 
                        color = 'black', label = 'NN prediction')
            
            ax_cum.plot(sum(PV_ann_up[i][k]*count_ann[i]/N_centroids for i in range(n_cl_ann) for k in range(N_centroids)), 
                        color = 'black', linestyle = ':', label = 'NN bound')
            ax_cum.plot(sum(PV_ann_low[i][k]*count_ann[i]/N_centroids for i in range(n_cl_ann) for k in range(N_centroids)), 
                        color = 'black', linestyle = ':')
        ax_cum.set_xlabel('time, t', fontsize = 'large')
        ax_cum.set_ylabel('policy value', fontsize = 'large')
        if (n_cl_ann == 100) & (insurance_type == 'termlife') :
            ax_cum.legend() 
        if (insurance_type=='pensions'): # visualization detail for paper    
            if n_cl_ann == 10:
                ax_cum.legend(loc='lower center', bbox_to_anchor=(0.41, 0.0))
            elif N_centroids == 5:
                ax_cum.legend(loc='lower center', bbox_to_anchor=(0.41, 0.0))
        plt.tight_layout()
        if include_ann:
            plt.savefig(os.getcwd()+r'/Matplotlib_figures/Grouping{}_cum_{}_K{}_C{}.eps'.format(plot_tag, insurance_type, n_cl_ann, N_centroids), format = 'eps')
        else:
            ax_cum.set_title(r'K-Means: $K= {}$'.format(n_cl), fontsize= 'large')       
        plt.close()     
    elif option == 'statistic':
        
        # Compare target policy values with policy values of ANN representatives
        # For the PV of ANN representatives we take the mean of the floored (ann_PV_low)
        # and ceiled value (ann_PV_up) as our proxy
        
        df = pd.DataFrame(data=None, index = None, columns = [r'$\overline{re}$', #r'$CL_{0.99,|re{}_t|}$',
                                                              r'$\overline{e}$'])#, ' $CL_{0.99,|e{}_t|}$'])  
        
        ## Cumulative
        index_cum = (y.sum(axis=0)>0)
        
        # Portfolio: K-Means
        diff_km = np.abs((PV_km_pred*count).sum(axis=0) - y.sum(axis=0))
        diff_km_rel = diff_km[index_cum]/y.sum(axis=0)[index_cum]
        df.loc['KM prediction'] = (diff_km_rel.mean(), #np.percentile(np.abs(diff_km_rel),99), 
                               diff_km.mean())#, np.percentile(np.abs(diff_km),99))
        
        # Portfolio: ANN via classical calculation
        diff_up = np.abs(sum((PV_ann_up[i][k])*count_ann[i]/N_centroids for i in range(n_cl_ann) for k in range(N_centroids)) -y.sum(axis=0))
        diff_rel_up = diff_up[index_cum]/y.sum(axis=0)[index_cum]
        diff_low = np.abs(sum((PV_ann_low[i][k])*count_ann[i]/N_centroids for i in range(n_cl_ann) for k in range(N_centroids)) -y.sum(axis=0))
        diff_rel_low = diff_low[index_cum]/(targets_cl*count).sum(axis=0)[index_cum]
        

        # Portfolio: ANN prediction
        diff_pred = np.abs(sum(ann_prediction[i][k]*count_ann[i]/N_centroids for i in range(n_cl_ann) for k in range(N_centroids))-y.sum(axis=0))
        diff_pred_rel = diff_pred[index_cum]/y.sum(axis=0)[index_cum]

        df.loc['NN prediction'] = (diff_pred_rel.mean(),# np.percentile(np.abs(diff_pred_rel),99), 
                                    diff_pred.mean())#, np.percentile(np.abs(diff_pred),99)) 

        df.loc['NN bound (up)'] = (diff_rel_up.mean(),
                                    diff_up.mean())
        
        df.loc['NN bound (low)'] = (diff_rel_low.mean(),# np.percentile(np.abs(diff_rel),99), 
                                    diff_low.mean())#, np.percentile(np.abs(diff),99))  
        

        ann_pred = sum(ann_prediction[i][k]*count_ann[i]/N_centroids for i in range(n_cl_ann) for k in range(N_centroids))
        ann_pred_up = sum(PV_ann_up[i][k]*count_ann[i]/N_centroids for i in range(n_cl_ann) for k in range(N_centroids))
        ann_pred_low = sum(PV_ann_low[i][k]*count_ann[i]/N_centroids for i in range(n_cl_ann) for k in range(N_centroids))
        return df, [diff_km, diff_km_rel], [diff_pred, diff_pred_rel], [diff_up, diff_rel_up], [diff_low, diff_rel_low],  [ann_pred_up, ann_pred_low, ann_pred], [PV_ann_up, PV_ann_low]
    else:
        raise ValueError('Unknown input-option!')
