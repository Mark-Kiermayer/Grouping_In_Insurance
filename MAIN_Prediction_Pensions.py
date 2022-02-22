#!/usr/bin/env python
# coding: utf-8

# # Pensions

#################################### Imports Packages ###############################################
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
from tensorflow.keras.layers import Dense, Dropout, Activation, RepeatVector, Average, LSTM, Lambda, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from data_prep_General import data_re_transform_features
from rnn_functions import create_multiple_rnn_models, train_individual_ensembles
from statistical_analysis_functions import create_df_model_comparison, model_examine_indivual_fit
from clustering import analyze_agglomeration_test, kmeans_counts
from visualization_functions import visualize_representatives_km_ann

### Paths
cd = os.getcwd() + '/Pensions' #r"C:\Users\mark.kiermayer\Documents\Python Scripts\NEW Paper (Grouping) - Code - V1\Pensions"
path_data = cd + '/Data/' # import data
wd_rnn = cd + '/ipynb_Checkpoints/Prediction' # save/ load rnns
# dummy if saved models should be loaded (TRUE) or the all models should be recalculated (False)
dummy_load_saved_models = True
bool_fine_tune = True
bool_latex = True
BATCH_replica = 64
val_share = 0.25
tf_strategy = tf.distribute.MirroredStrategy()#['/gpu:0']) # select specific GPUs
BATCH = BATCH_replica*tf_strategy.num_replicas_in_sync

print('---------------------------------------------------------')
print("Num GPUs Available: ", tf_strategy.num_replicas_in_sync)
print('---------------------------------------------------------')

### load data
X_train = pd.read_csv(path_data+'X_train.csv', index_col= 0).values 
X_test = pd.read_csv(path_data+'X_test.csv', index_col= 0).values 
y_train = pd.read_csv(path_data+'y_train.csv', index_col= 0).values
y_test = pd.read_csv(path_data+'y_test.csv', index_col= 0).values
X_train_raw = pd.read_csv(path_data+'X_train_raw.csv', index_col= 0).values
X_test_raw = pd.read_csv(path_data+'X_test_raw.csv', index_col= 0).values 

# Load general assumptions
with open(path_data+'Pension_params.pkl', 'rb') as f:
    params = pickle.load(f)
with open(path_data+'Pension_explan_vars_range.pkl', 'rb') as f:
    explan_vars_range = pickle.load(f)


#################################### Assumptions ###############################################
print('Parameters:' )
print(params)
print('Explanatory variables: ')
print(explan_vars_range)

# Dataframe representation
pd.set_option('precision', 2)
pd.set_option('display.max_colwidth', 40)

# Range of Variables
# Matrix Version of previous upper/ lower bounds on features
Min_Max = np.array([explan_vars_range['fund'][0],explan_vars_range['fund'][1],
                    explan_vars_range['age'][0], explan_vars_range['age'][1], 
                    explan_vars_range['salary'][0], explan_vars_range['salary'][1],
                    explan_vars_range['salary_scale'][0], explan_vars_range['salary_scale'][1],
                    explan_vars_range['contribution'][0], explan_vars_range['contribution'][1]]).reshape(-1,2)


#################################### Build Prediction Models ###############################################

# General settings
n_output = params['pension_age_max']-explan_vars_range['age'][0]+1
n_in = X_train.shape[1]
es = EarlyStopping(monitor= 'val_loss', patience= 100 )
es_patience = 100
N_epochs = 1500

### Single Model Configurations
### MSE Training
N_ensembles = 10

# Create Multiple RNNs with identical configuration
models_mse_hist = {}
with tf_strategy.scope():
    INPUT = Input(shape = (n_in,))
    models_mse = create_multiple_rnn_models(number=N_ensembles, model_input=INPUT,widths_rnn =[n_output],  
                                            widths_ffn = [n_output], 
                                            dense_act_fct = 'tanh', optimizer_type='adam', loss_type='mse', 
                                            metric_type='mae', dropout_share=0, 
                                            lambda_layer = True, lambda_scale =params['V_max'], log_scale=True, 
                                            model_compile = True, return_option = 'model', branch_name = '')

# Either load model(s) or train them
if os.path.isfile(wd_rnn+r'/mse/model_0.h5') & dummy_load_saved_models:
    # load model weights
    for i in range(N_ensembles):
        models_mse[i].load_weights(wd_rnn+r'/mse/model_{}.h5'.format(i))
        with open(wd_rnn+r'/mse/model_{}_hist.json'.format(i), 'rb') as f:
            models_mse_hist[i] = pickle.load(f)
    print('MSE prediction models loaded!')
else:
    # Train multiple RNNs with identical configuration
    for i in range(len(models_mse)):
        models_mse[i].fit(x=X_train, y=y_train, validation_split = val_share, batch_size= BATCH, epochs=N_epochs, callbacks=[es], verbose = 2)
        models_mse[i].save_weights(wd_rnn+r'/mse/model_{}.h5'.format(i))
        models_mse_hist[i] = models_mse[i].history
        with open(wd_rnn+r'/mse/model_{}_hist.json'.format(i), 'wb') as f: # alternative: option #'w'
            pickle.dump(models_mse_hist[i], f, pickle.HIGHEST_PROTOCOL)
    
    #models_mse, models_mse_hist = train_individual_ensembles(models_mse, X_train, y_train, 
    #                                                n_epochs= N_epochs, 
    #                                                n_batch= BATCH, es_patience= es_patience,
    #                                                path = wd_rnn+r'/mse')
    # Note: Save Model (and History) is integrated in function 'train_individual_ensembles'


#### MAE Training
N_ensembles = 10
# Create Multiple RNNs with identical configuration
models_mae_hist = {}
with tf_strategy.scope():
    INPUT = Input(shape = (n_in,))
    models_mae = create_multiple_rnn_models(number=N_ensembles, model_input=INPUT,widths_rnn =[n_output], 
                                    widths_ffn = [n_output], 
                                    dense_act_fct = 'tanh', optimizer_type='adam', loss_type='mae', 
                                    metric_type='mae', dropout_share=0, 
                                    lambda_layer = True, lambda_scale =params['V_max'], log_scale=True, 
                                        model_compile = True, return_option = 'model', branch_name = '')

# Either load model(s) or train them
if os.path.isfile(wd_rnn+r'/mae/model_0.h5') & dummy_load_saved_models:
    # load model weights
    for i in range(N_ensembles):
        models_mae[i].load_weights(wd_rnn+r'/mae/model_{}.h5'.format(i))
        with open(wd_rnn+r'/mae/model_{}_hist.json'.format(i), 'rb') as f:
            models_mae_hist[i] = pickle.load(f)
    print('MAE prediction models loaded!')   
else:
    # Train multiple RNNs with identical configuration
    for i in range(len(models_mse)):
        models_mae[i].fit(x=X_train, y=y_train, validation_split = val_share, batch_size= BATCH, epochs=N_epochs, callbacks=[es], verbose = 2)
        models_mae[i].save_weights(wd_rnn+r'/mae/model_{}.h5'.format(i))
        models_mae_hist[i] = models_mae[i].history
        with open(wd_rnn+r'/mae/model_{}_hist.json'.format(i), 'wb') as f: # alternative: option #'w'
            pickle.dump(models_mae_hist[i], f, pickle.HIGHEST_PROTOCOL)

    #models_mae, models_mae_hist = train_individual_ensembles(models_mae, X_train, y_train, 
    #                                                n_epochs= N_epochs, 
    #                                                n_batch= BATCH, es_patience= es_patience,
    #                                                path = wd_rnn+r'/mae')
    # Save Model (and History) is integrated in function 'train_individual_ensembles'



# Insight in early stopping behaviour
print('Observe early stopping times of mse/mae-trained models:')
for i in range(10):
    print('\t'+str(len(models_mse_hist[i]['val_loss']))+ " / " + str(len(models_mae_hist[i]['val_loss'])))


################ 4.2. Ensemble(s)  ###############################

# Fix Number of Ensembles used
N_ensembles = 5
N_epochs = 4000
dummy_load_saved_models_ensembles = True
batchsize = 64
es_patience = 50


# Create Ensembles, using pre-trained weak learners
with tf_strategy.scope():
    #----------------------------------------------------
    N_ensembles = 5
    # Note: cloning of models in order to perform fine-tuning independent of weak learners
    ensemble_mse_5 = clone_model(Model(INPUT, Average()([models_mse[i](INPUT) for i in range(N_ensembles)])))
    ensemble_mse_5.set_weights(Model(INPUT, Average()([models_mse[i](INPUT) for i in range(N_ensembles)])).get_weights())
    ensemble_mse_5.compile(loss = 'mse', metrics=['mae'], optimizer = Adam(0.001))

    ensemble_mae_5 = clone_model(Model(INPUT, Average()([models_mae[i](INPUT) for i in range(N_ensembles)])))
    ensemble_mae_5.set_weights(Model(INPUT, Average()([models_mae[i](INPUT) for i in range(N_ensembles)])).get_weights())
    ensemble_mae_5.compile(loss = 'mae', optimizer = Adam(0.001))
    #----------------------------------------------------
    N_ensembles = 10
    ensemble_mse_10 = clone_model(Model(INPUT, Average()([models_mse[i](INPUT) for i in range(N_ensembles)])))
    ensemble_mse_10.set_weights(Model(INPUT, Average()([models_mse[i](INPUT) for i in range(N_ensembles)])).get_weights())
    ensemble_mse_10.compile(loss = 'mse', metrics=['mae'], optimizer = Adam(0.001))

    ensemble_mae_10 = clone_model(Model(INPUT, Average()([models_mae[i](INPUT) for i in range(N_ensembles)])))
    ensemble_mae_10.set_weights(Model(INPUT, Average()([models_mae[i](INPUT) for i in range(N_ensembles)])).get_weights())  
    ensemble_mae_10.compile(loss = 'mae', optimizer = Adam(0.001))
    #----------------------------------------------------

    #results_statistic = create_df_model_comparison(model_single_lst=models_mse[0:2]+models_mae[0:2], 
    #                        x_test = X_test, y_test= y_test,  
    #                        model_ens_lst = [ensemble_mse_5, ensemble_mse_10,
    #                                        ensemble_mae_5, ensemble_mae_10], #, ensemble_mse_mae_5, ensemble_mse_mae_10],
    #                        names_number= ['5', '10','5', '10'], #,'5','10'], 
    #                        names_loss= ['MSE', 'MSE','MAE','MAE'], # 'Mixed','Mixed'],
    #                        names_loss_single = ['MSE']*2+['MAE']*2)
    #print('Statistics before fine-tuning:')
    #print(results_statistic[0])
    #print('\n')


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

    if False:
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
            with open('TeX_tables/Prediction_DC_Model_Comparison.tex','w') as tf:
                tf.write(results_statistic[0].to_latex())


# Relate following relative values to absolute Policy Values
interval_lst = [0,0.005, 0.01,0.2,0.4,0.6,0.8,1]


stat_ENS_0 = model_examine_indivual_fit(model = ensemble_mse_5, data = X_test, PV_max= params['V_max'], 
                        targets = y_test, output_option = 'statistic', interval_lst= interval_lst)
print('Statistics for 5-MSE ensemble')
print(stat_ENS_0, r'\n')

if bool_latex:
    with open('TeX_tables/Prediction_DC_Model_MSE_5.tex','w') as tf:
        tf.write(stat_ENS_0.to_latex())


stat_ENS_1 = model_examine_indivual_fit(model = ensemble_mse_10, data = X_test, PV_max= params['V_max'],
                        targets = y_test, output_option = 'statistic', interval_lst= interval_lst)
print('Statistics for 10-MSE ensemble')
print(stat_ENS_1, r'\n')

with open('Prediction_TL_Model_MSE_10.tex','w') as tf:
    tf.write(stat_ENS_1.to_latex())


stat_ENS_2 = model_examine_indivual_fit(model = ensemble_mae_5, data = X_test, PV_max= params['V_max'],
                        targets = y_test, output_option = 'statistic', interval_lst= interval_lst)
print('Statistics for 5-MAE ensemble')
print(stat_ENS_2, r'\n')

if bool_latex:
    with open('TeX_tables/Prediction_DC_Model_MAE_5.tex','w') as tf:
        tf.write(stat_ENS_2.to_latex())


stat_ENS_3 = model_examine_indivual_fit(model = ensemble_mae_10, data = X_test, PV_max= params['V_max'],
                        targets = y_test, output_option = 'statistic', interval_lst= interval_lst)
print('Statistics for 10-MAE ensemble')
print(stat_ENS_3, r'\n')

print(r'\t Analysis of prediction models completed!')