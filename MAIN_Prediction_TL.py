#!/usr/bin/env python
# coding: utf-8

# ## 0.1. Import Packages
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import gamma, truncnorm, describe
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
import time, json, pickle, os

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model, clone_model
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.layers import Dense, Dropout, Activation, RepeatVector, Average, LSTM, Lambda, Input, Multiply
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from data_prep_General import data_re_transform_features
from rnn_functions import create_multiple_rnn_models, train_individual_ensembles
from statistical_analysis_functions import create_df_model_comparison, model_examine_indivual_fit
from clustering import analyze_agglomeration_test, kmeans_counts
from visualization_functions import visualize_representatives_km_ann, training_progress_visual, ensemble_plot, plot_accuracy_cum
from boosting import ANN_boost

# import data
cd = cd = os.getcwd() + r'/TermLife'
path_data = cd + r'/Data/'
wd_rnn = cd +r'/ipynb_Checkpoints/Prediction'
# dummy if saved models should be loaded (TRUE) or the all models should be recalculated (False)
dummy_load_saved_models = True
bool_latex = True
bool_fine_tune = True
# Dataframe representation
pd.set_option('precision', 2)
# share of validation data
val_share = 0.25
BATCH_replica = 64
SHUFFLE_SIZE = 1000

def is_active(y):
    '''
    Check targets y to see at which times the contract is still active.
    '''
    active = y>0
    active[:,0] = True # at time 0 always active
    return active.astype('int')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # supress all info about loading tf devices, e.g. [..] -> physical GPU (...)
tf_strategy = tf.distribute.MirroredStrategy()#['/gpu:0']) # select specific GPUs
BATCH = BATCH_replica*tf_strategy.num_replicas_in_sync
#print('---------------------------------------------------------')
#print("Num GPUs Available: ", tf_strategy.num_replicas_in_sync)
#print('---------------------------------------------------------')


# data
X_train = pd.read_csv(path_data+'X_train.csv', index_col= 0).values 
X_test = pd.read_csv(path_data+'X_test.csv', index_col= 0).values 
y_train = pd.read_csv(path_data+'y_train.csv', index_col= 0).values
y_test = pd.read_csv(path_data+'y_test.csv', index_col= 0).values
X_train_raw = pd.read_csv(path_data+'X_train_raw.csv', index_col= 0).values
X_test_raw = pd.read_csv(path_data+'X_test_raw.csv', index_col= 0).values 

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
exit()

# For quantitative (regression) model
data_train = tf.data.Dataset.from_tensor_slices((X_train[0:int(len(X_train)*(1-val_share))], y_train[0:int(len(X_train)*(1-val_share))])
                                                ).shuffle(SHUFFLE_SIZE).batch(BATCH_replica).prefetch(tf.data.experimental.AUTOTUNE)
data_val = tf.data.Dataset.from_tensor_slices((X_train[int(len(X_train)*(1-val_share)):], y_train[int(len(X_train)*(1-val_share)):])
                                                ).shuffle(SHUFFLE_SIZE).batch(BATCH_replica).prefetch(tf.data.experimental.AUTOTUNE)
data_test = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(SHUFFLE_SIZE).batch(BATCH_replica).prefetch(tf.data.experimental.AUTOTUNE)
# For qualitative (classification) model
data_qual_train = tf.data.Dataset.from_tensor_slices((X_train[0:int(len(X_train)*(1-val_share))], is_active(y_train[0:int(len(X_train)*(1-val_share))]))
                                                ).shuffle(SHUFFLE_SIZE).batch(BATCH_replica).prefetch(tf.data.experimental.AUTOTUNE)
data_qual_val = tf.data.Dataset.from_tensor_slices((X_train[int(len(X_train)*(1-val_share)):], is_active(y_train[int(len(X_train)*(1-val_share)):]))
                                                ).shuffle(SHUFFLE_SIZE).batch(BATCH_replica).prefetch(tf.data.experimental.AUTOTUNE)
data_qual_test = tf.data.Dataset.from_tensor_slices((X_test, is_active(y_test))).shuffle(SHUFFLE_SIZE).batch(BATCH_replica).prefetch(tf.data.experimental.AUTOTUNE)

# Load general assumptions
with open(path_data+'TL_params.pkl', 'rb') as f:
    params = pickle.load(f)
with open(path_data+'TL_explan_vars_range.pkl', 'rb') as f:
    explan_vars_range = pickle.load(f)

print('Parameters imported: ', params)
print('Explanatory variables imported: ', explan_vars_range)


#################################### Section 1 - Global Parameters  ##################################################
# Portfolio Details
N_contracts = len(X_train)+len(X_test) 
int_rate = params['int_rate']
n_in = len(explan_vars_range.keys())

# Matrix Version of previous upper/ lower bounds on features
Max_min = np.array([explan_vars_range['age'][0],explan_vars_range['age'][1]+explan_vars_range['duration'][1],
                    explan_vars_range['sum_ins'][0], explan_vars_range['sum_ins'][1], 
                    explan_vars_range['duration'][0], explan_vars_range['duration'][1], 
                    explan_vars_range['age_of_contract'][0], explan_vars_range['age_of_contract'][1], 
                    explan_vars_range['interest_rate'][0], explan_vars_range['interest_rate'][1]]).reshape(-1,2)


###################################### Create - Prediction Models  ####################################################

# Parameters
n_timesteps, n_features, n_output = explan_vars_range['duration'][1]+1,n_in, explan_vars_range['duration'][1]+1
#INPUT = Input(shape=(n_features,), name = 'Input')
es = EarlyStopping(monitor= 'val_loss', patience= 50, restore_best_weights=True)

# ## Generate all single-models for ensemble methods
N_epochs = 1500
es_patience = 50

################################ Train MSE Models

if True: # see old baseline
    N_ensembles = 10
    tf_strategy = tf.distribute.MirroredStrategy()
    BATCH = BATCH_replica*tf_strategy.num_replicas_in_sync
    # Create Multiple RNNs with identical configuration
    weak_learners_hist = {}
    with tf_strategy.scope():
        INPUT = Input(shape=(n_features,), name = 'Input')
        models_mse = create_multiple_rnn_models(number=N_ensembles, model_input=INPUT,widths_rnn =[n_output], 
                                        widths_ffn = [n_output], 
                                        dense_act_fct = 'tanh', optimizer_type='adam', loss_type='mse', 
                                        metric_type='mae', dropout_share=0, 
                                        lambda_layer = True, lambda_scale =params['V_max'], log_scale=True, 
                                        model_compile = True, return_option = 'model', branch_name = '')
        models_mae = create_multiple_rnn_models(number=N_ensembles, model_input=INPUT,widths_rnn =[n_output], 
                                widths_ffn = [n_output], 
                                dense_act_fct = 'tanh', optimizer_type='adam', loss_type='mae', 
                                metric_type='mae', dropout_share=0, 
                                lambda_layer = True, lambda_scale = params['V_max'], log_scale=True, 
                                model_compile = True, return_option = 'model', branch_name = '')

    if os.path.isfile(wd_rnn+r'/mse/model_0.h5') & dummy_load_saved_models:
        # load model weights
        for i in range(N_ensembles):
            models_mse[i].load_weights(wd_rnn+r'/mse/model_{}.h5'.format(i))
            #with open(wd_rnn+r'/{}/model_{}_hist.json'.format(train_type,i), 'rb') as f:
            #    weak_learners_hist[i] = pickle.load(f)
    else:
        # Train multiple RNNs with identical configuration
        models_mse, models_mse_hist = train_individual_ensembles(models_mse, X_train, y_train, 
                                                        n_epochs= N_epochs, 
                                                        n_batch= BATCH, es_patience= es_patience,
                                                        path = wd_rnn+r'/mse')
        # Save Model (and History) is integrated in function 'train_individual_ensembles'

    if os.path.isfile(wd_rnn+r'/mae/model_0.h5') & dummy_load_saved_models:
        # load model weights
        for i in range(N_ensembles):
            models_mae[i].load_weights(wd_rnn+r'/mae/model_{}.h5'.format(i))
            #with open(wd_rnn+r'/{}/model_{}_hist.json'.format(train_type,i), 'rb') as f:
            #    weak_learners_hist[i] = pickle.load(f)
    else:
        # Train multiple RNNs with identical configuration
        models_mae, models_mae_hist = train_individual_ensembles(models_mse, X_train, y_train, 
                                                        n_epochs= N_epochs, 
                                                        n_batch= BATCH, es_patience= es_patience,
                                                        path = wd_rnn+r'/mae')
        # Save Model (and History) is integrated in function 'train_individual_ensembles'    

    # Create Ensembles, using pre-trained weak learners
    with tf_strategy.scope():
        #----------------------------------------------------
        N_ensembles = 5
        # Note: cloning of models in order to perform fine-tuning independent of weak learners
        ensemble_mse_5 = clone_model(Model(INPUT, Average()([models_mse[i](INPUT) for i in range(N_ensembles)])))
        ensemble_mse_5.set_weights(Model(INPUT, Average()([models_mse[i](INPUT) for i in range(N_ensembles)])).get_weights())
        ensemble_mse_5.compile(loss = 'mse', metrics=['mae'], optimizer = Adam(0.0001))

        ensemble_mae_5 = clone_model(Model(INPUT, Average()([models_mae[i](INPUT) for i in range(N_ensembles)])))
        ensemble_mae_5.set_weights(Model(INPUT, Average()([models_mae[i](INPUT) for i in range(N_ensembles)])).get_weights())
        ensemble_mae_5.compile(loss = 'mae', optimizer = Adam(0.0001))
        #----------------------------------------------------
        N_ensembles = 10
        ensemble_mse_10 = clone_model(Model(INPUT, Average()([models_mse[i](INPUT) for i in range(N_ensembles)])))
        ensemble_mse_10.set_weights(Model(INPUT, Average()([models_mse[i](INPUT) for i in range(N_ensembles)])).get_weights())
        ensemble_mse_10.compile(loss = 'mse', metrics=['mae'], optimizer = Adam(0.0001))

        ensemble_mae_10 = clone_model(Model(INPUT, Average()([models_mae[i](INPUT) for i in range(N_ensembles)])))
        ensemble_mae_10.set_weights(Model(INPUT, Average()([models_mae[i](INPUT) for i in range(N_ensembles)])).get_weights())  
        ensemble_mae_10.compile(loss = 'mae', optimizer = Adam(0.0001))
        #-----------------------------------------------------


    # fine tuning
    if bool_fine_tune:
        if os.path.isfile(wd_rnn+r'/ensemble_mse_5.h5'):
            ensemble_mse_5.load_weights(wd_rnn+r'/ensemble_mse_5.h5')
            print('Fine-tuned mse-5 ensemble loaded.')
        else:
            print('Fine tuning mse-5 ensemble ...')
            ensemble_mse_5.fit(x=X_train, y=y_train, validation_split = val_share, batch_size= BATCH, epochs=N_epochs, callbacks=[es], verbose = 2)
            print('Ensemble mse-5 fine-tuned for {} epochs.'.format(len(ensemble_mse_5.history.history['loss'])))
            ensemble_mse_5.save_weights(wd_rnn+r'/ensemble_mse_5.h5')

        if os.path.isfile(wd_rnn+r'/ensemble_mse_10.h5'):
            ensemble_mse_10.load_weights(wd_rnn+r'/ensemble_mse_10.h5')
            print('Fine-tuned mse-10 ensemble loaded.')
        else:
            print('Fine tuning mse-10 ensemble ...')
            ensemble_mse_10.fit(x=X_train, y=y_train, validation_split = val_share, batch_size= BATCH, epochs=N_epochs, callbacks=[es], verbose = 2)
            print('Ensemble mse-10 fine-tuned for {} epochs.'.format(len(ensemble_mse_10.history.history['loss'])))
            ensemble_mse_10.save_weights(wd_rnn+r'/ensemble_mse_10.h5')
        
        if os.path.isfile(wd_rnn+r'/ensemble_mae_5.h5'):
            ensemble_mae_5.load_weights(wd_rnn+r'/ensemble_mae_5.h5')
            print('Fine-tuned mae-5 ensemble loaded.')
        else:
            print('Fine tuning mae-5 ensemble ...')
            ensemble_mae_5.fit(x=X_train, y=y_train, validation_split = val_share, batch_size= BATCH, epochs=N_epochs, callbacks=[es], verbose = 2)
            print('Ensemble mae-5 fine-tuned for {} epochs.'.format(len(ensemble_mae_5.history.history['loss'])))
            ensemble_mae_5.save_weights(wd_rnn+r'/ensemble_mae_5.h5')

        if os.path.isfile(wd_rnn+r'/ensemble_mae_10.h5'):
            ensemble_mae_10.load_weights(wd_rnn+r'/ensemble_mae_10.h5')
            print('Fine-tuned mae-10 ensemble loaded.')
        else:
            print('Fine tuning mae-10 ensemble ...')
            ensemble_mae_10.fit(x=X_train, y=y_train, validation_split = val_share, batch_size= BATCH, epochs=N_epochs, callbacks=[es], verbose = 2)
            print('Ensemble mae-10 fine-tuned for {} epochs.'.format(len(ensemble_mae_10.history.history['loss'])))
            ensemble_mae_10.save_weights(wd_rnn+r'/ensemble_mae_10.h5')


        results_statistic = create_df_model_comparison(model_single_lst=models_mse[0:]+models_mae[0:], 
                                x_test = X_test, y_test= y_test,  
                                model_ens_lst = [ensemble_mse_5, ensemble_mse_10,
                                                ensemble_mae_5, ensemble_mae_10], #, ensemble_mse_mae_5, ensemble_mse_mae_10],
                                names_number= ['5', '10','5', '10'], #,'5','10'], 
                                names_loss= ['mse', 'mse','mae','mae'], # 'Mixed','Mixed'],
                                names_loss_single = ['mse']*10+['mae']*10)
        print('Statistics after fine-tuning:')
        print(results_statistic[0])
        if bool_latex:
            with open('TeX_tables/Prediction_TL_Model_Comparison.tex','w') as tf:
                tf.write(results_statistic[0].to_latex())

    # Relate following relative values to absolute Policy Values
    interval_lst = [0,0.001, 0.005, 0.01,0.2,0.4,0.6,0.8,1]


    stat_ENS_0 = model_examine_indivual_fit(model = ensemble_mse_5, data = X_test, 
                            targets = y_test, output_option = 'statistic', PV_max= params['V_max'],
                                            interval_lst= interval_lst)
    print('Statistics for 5-MSE ensemble')
    print(stat_ENS_0, r'\n')
    if bool_latex:
        with open('TeX_tables/Prediction_TL_Model_MSE_5.tex','w') as tf:
            tf.write(stat_ENS_0.to_latex())

    stat_ENS_1 = model_examine_indivual_fit(model = ensemble_mse_10, data = X_test, 
                            targets = y_test, output_option = 'statistic', PV_max= params['V_max'],
                                            interval_lst= interval_lst)
    print('Statistics for 10-MSE ensemble')
    print(stat_ENS_1, r'\n')
    if bool_latex:
        with open('TeX_tables/Prediction_TL_Model_MSE_10.tex','w') as tf:
            tf.write(stat_ENS_1.to_latex())

    stat_ENS_2 = model_examine_indivual_fit(model = ensemble_mae_5, data = X_test, PV_max= params['V_max'],
                            targets = y_test, output_option = 'statistic', interval_lst= interval_lst)
    print('Statistics for 5-MAE ensemble')
    print(stat_ENS_2, r'\n')
    if bool_latex:
        with open('TeX_tables/Prediction_TL_Model_MAE_5.tex','w') as tf:
            tf.write(stat_ENS_2.to_latex())

    stat_ENS_3 = model_examine_indivual_fit(model = ensemble_mae_10, data = X_test, PV_max= params['V_max'],
                            targets = y_test, output_option = 'statistic', interval_lst= interval_lst)
    print('Statistics for 10-MAE ensemble')
    print(stat_ENS_3, r'\n')
    if bool_latex:
        with open('TeX_tables/Prediction_TL_Model_MAE_10.tex','w') as tf:
            tf.write(stat_ENS_3.to_latex())

print('Analysis of prediction models completed!')

