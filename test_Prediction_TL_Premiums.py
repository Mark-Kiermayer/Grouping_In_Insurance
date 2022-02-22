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

from actuarial_functions import get_termlife_premium
from data_prep_General import data_re_transform_features
from rnn_functions import create_multiple_rnn_models, train_individual_ensembles, create_rnn_model
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



# Load general assumptions
with open(path_data+'TL_params.pkl', 'rb') as f:
    params = pickle.load(f)
with open(path_data+'TL_explan_vars_range.pkl', 'rb') as f:
    explan_vars_range = pickle.load(f)

print('Parameters imported: ', params)
print('Explanatory variables imported: ', explan_vars_range)


y_premium_train = np.array([get_termlife_premium(age_init=X_train_raw[i,0]-X_train_raw[i,3], Sum_ins=X_train_raw[i,1], 
                            duration=X_train_raw[i,2].astype('int'),  interest=X_train_raw[i,4], A= params['A'], B=params['B'], c=params['c']) 
                            for i in range(len(X_train_raw))])
y_premium_test = np.array([get_termlife_premium(age_init=X_test_raw[i,0]-X_test_raw[i,3], Sum_ins=X_test_raw[i,1], 
                            duration=X_test_raw[i,2].astype('int'),  interest=X_test_raw[i,4], A= params['A'], B=params['B'], c=params['c']) 
                            for i in range(len(X_test_raw))])

print(describe(y_premium_train))
print(describe(y_premium_test))

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
es_patience = 25

################################ Train MSE Models
tf_strategy = tf.distribute.MirroredStrategy()
BATCH = BATCH_replica*tf_strategy.num_replicas_in_sync
with tf_strategy.scope():
    INPUT = Input(shape=(n_features,), name = 'Input')
    model = create_rnn_model(model_input=INPUT,widths_rnn =[10], widths_ffn = [1],
                        dense_act_fct = 'relu', act_fct_special = False, 
                        option_recurrent_dropout = False, 
                        n_repeat = 41, option_dyn_scaling = False,
                        optimizer_type= Adam(0.001), loss_type='mae', metric_type='mae',
                        dropout_rnn=0, lambda_layer = True, lambda_scale =50000, log_scale=True, 
                        model_compile = True, return_option = 'model', branch_name = '')
                        
print(model.summary())
if os.path.isfile(wd_rnn+r'/model_premium_prediction_mae.h5') & dummy_load_saved_models:
    model.load_weights(wd_rnn+r'/model_premium_prediction_mae.h5')
else:
    model.fit(x=X_train, y=y_premium_train, validation_split=val_share,batch_size=BATCH, epochs=N_epochs, callbacks=[es], verbose = 2)
    model.save_weights(wd_rnn+r'/model_premium_prediction_mae.h5')

model.evaluate(x=X_test, y=y_premium_test)
plt.scatter(y_premium_test, model.predict(X_test))
plt.show()