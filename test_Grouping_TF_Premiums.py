import pandas as pd
import numpy as np
import logging, pickle, os

from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import Input, Average 

from functions.actuarial_functions import get_termlife_premium
from functions.data_prep_General import data_re_transform_features, data_prep_feautures_scale
from functions.rnn_functions import create_multiple_rnn_models, create_rnn_model
from functions.statistical_analysis_functions import model_examine_indivual_fit 
from functions.clustering import analyze_agglomeration_test, dict_to_array, cluster_ann_test 
from functions.visualization_functions import visualize_representatives_km_ann


# import data
cd = os.getcwd() + "/TermLife" #r"C:\Users\mark.kiermayer\Documents\Python Scripts\NEW Paper (Grouping) - Code - V1\Termlife"
path_data = cd + '/Data/'
wd_rnn = cd + r'/ipynb_Checkpoints/Prediction' # path to load prediction model
wd_cluster = cd+r'/ipynb_Checkpoints/Grouping'# path to save grouping 
load_agg_model = True
load_kmeans = True
bool_latex = True
bool_plot = True
logging.basicConfig(filename='logging_grouping_TL.txt',level=logging.WARNING)

# type of prediction model: 'mae' or 'mse' trained
pred_model_type = 'mae'
N_ensembles = 5
# Dataframe representation
pd.set_option('precision', 4)

# data
X = pd.read_csv(path_data+'NEW_X.csv', index_col= 0).values 
X_raw = pd.read_csv(path_data+'NEW_X_raw.csv', index_col= 0).values
y = pd.read_csv(path_data+'NEW_y.csv', index_col= 0).values 

# Load general assumptions
with open(path_data+'TL_params.pkl', 'rb') as f:
    params = pickle.load(f)
with open(path_data+'TL_explan_vars_range.pkl', 'rb') as f:
    explan_vars_range = pickle.load(f)

print('Parameters imported: ', params)
print('Explanatory variables imported: ', explan_vars_range)

# current age, sum ins, duration, elapsed duration, interest rate
y_premium = np.array([get_termlife_premium(age_init = X_raw[i,0]-X_raw[i,3], Sum_ins = X_raw[i,2], 
                        duration = X_raw[i,2].astype('int'),  interest = X_raw[i,4],  A= params['A'], B=params['B'], c=params['c']) 
                        for i in range(len(y))]).reshape((-1,1))

#################################### Section 1 - Global Parameters  ##################################################
# Portfolio Details
N_contracts = len(X) 
int_rate = params['int_rate']
n_in = len(explan_vars_range.keys())

# Matrix Version of previous upper/ lower bounds on features
Max_min = np.array([explan_vars_range['age'][0],explan_vars_range['age'][1]+explan_vars_range['duration'][1],
                    explan_vars_range['sum_ins'][0], explan_vars_range['sum_ins'][1], 
                    explan_vars_range['duration'][0], explan_vars_range['duration'][1], 
                    explan_vars_range['age_of_contract'][0], explan_vars_range['age_of_contract'][1], 
                    explan_vars_range['interest_rate'][0], explan_vars_range['interest_rate'][1]]).reshape(-1,2)

X_backtest = data_prep_feautures_scale(X_raw, Max_min) # max-min-scaled data used for kMeans baseline


###################################### Section 2 - Prediction Model  ####################################################
# Parameters
n_timesteps, n_features, n_output = explan_vars_range['duration'][1]+1,n_in, explan_vars_range['duration'][1]+1
INPUT = Input(shape=(n_features,), name = 'Input')


# Create Multiple RNNs with identical configuration
weak_learners_hist = {}
weak_learners = create_multiple_rnn_models(number=N_ensembles, model_input=INPUT,widths_rnn =[n_output], 
                                  widths_ffn = [n_output], 
                                   dense_act_fct = 'tanh', optimizer_type='adam', loss_type='mse', 
                                   metric_type='mae', dropout_share=0, 
                                   lambda_layer = True, lambda_scale =params['V_max'], log_scale=True, 
                                    model_compile = True, return_option = 'model', branch_name = '')


### Ensembles of weak_learners, Loss: MSE
model_pv = Model(INPUT, Average()([weak_learners[i](INPUT) for i in range(N_ensembles)]))
model_pv.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
if os.path.isfile(wd_rnn+r'/ensemble_{}_{}.h5'.format(pred_model_type,N_ensembles)):
    model_pv.load_weights(wd_rnn+r'/ensemble_{}_{}.h5'.format(pred_model_type,N_ensembles))
    print('-----------------------------------------------------------')
    print('Loaded prediction model with {} ensembles and {} loss.'.format(N_ensembles, pred_model_type))
    print('-----------------------------------------------------------')
else:
    print('No Prediction model available!')
    exit()

if False: # Backtest: Display fit of prediction model on so far unseen data
    stat_ENS = model_examine_indivual_fit(model = model_pv, data = X, PV_max= params['V_max'],
                            targets = y, output_option = 'statistic')
    print('Statistics for ensemble with {} weak learners, trained w.r.t. {}.'.format(N_ensembles, pred_model_type))
    print(stat_ENS, r'\n')

model_prem = create_rnn_model(model_input=INPUT,widths_rnn =[10], widths_ffn = [1],
                        dense_act_fct = 'relu', act_fct_special = False, 
                        option_recurrent_dropout = False, 
                        n_repeat = 41, option_dyn_scaling = False,
                        optimizer_type= Adam(0.001), loss_type='mae', metric_type='mae',
                        dropout_rnn=0, lambda_layer = True, lambda_scale =50000, log_scale=True, 
                        model_compile = True, return_option = 'model', branch_name = '')
                        
if os.path.isfile(wd_rnn+r'/model_premium_prediction_mae.h5'):
    model_prem.load_weights(wd_rnn+r'/model_premium_prediction_mae.h5')
else:
    print('Model for premiums not available!')
    exit()

###########################################################################################################################################
################################################## Agglomeration of Contracts #############################################################
###########################################################################################################################################

## Ensemble to integrate in Clustering procedure, Choose EP with 5 Sub-Models
N_epochs_clustering = 40000
es_patience_clustering = 100

if True:
    N_clusters = 10

    # load or perform kmeans cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_10 = pickle.load(input)
        print('10-means Model loaded!')
    else:
        print('Error: K-means not available!')
        exit()

    # Devide Data in k clusters
    #data_lst_cluster_10 = []
    targets_lst_pv_10 = []
    targets_lst_prem_10 = []
    for i in range(N_clusters):
        index = kMeans_10.labels_ == i
        #data_lst_cluster_10.append(X[index,].reshape((1,index.sum(),n_in)))
        targets_lst_pv_10.append((y[index,].sum(axis=0)/index.sum()).reshape(1,n_output))
        targets_lst_prem_10.append((y_premium[index,].sum(axis=0)/index.sum()).reshape(1,1))


    # load or create NN grouping
    if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/PREMIUM_cluster_object.pkl')&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/PREMIUM_cluster_object.pkl', 'rb') as input:
            cluster_analysis_10 = pickle.load(input)
        print('NN-grouping (MSE) loaded for K=10!')
    else:
        print('Starting NN-grouping with K=10:')
        cluster_analysis_10 = cluster_ann_test(y_lst_pv = targets_lst_pv_10, y_lst_prem=targets_lst_prem_10,
                                        model_pred_pv = model_pv, model_pred_prem= model_prem,
                                        N_centroids= 1, cluster_object = kMeans_10, Max_min = Max_min,
                                        N_epochs = N_epochs_clustering, 
                                        es_patience= es_patience_clustering,
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/PREMIUM_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_10, output, pickle.HIGHEST_PROTOCOL)

    # per cluster view evaluation
    if bool_plot:
        #analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Max_min, insurance_type='termlife',
        #                    ann_object = cluster_analysis_10,
        #                    individual_clusters=True, option= 'plot', n_columns=5, figsize= (20,4))
        analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Max_min, insurance_type='termlife',
                            ann_object = cluster_analysis_10, option_plot_selection= [0,9],
                            individual_clusters=True, option= 'plot', n_columns=5, figsize= (4,1))

    # statistics
    stat_10 = analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Max_min, insurance_type='termlife',
                        ann_object = cluster_analysis_10,
                        #ann_prediction= cluster_analysis_10[1], ann_representatives= cluster_analysis_10[0],
                        individual_clusters=True, option= 'statistic', n_columns=5)
    print('Statistics for grouping with K=10 (MSE):')
    print(stat_10[0])#.style.set_properties(subset=[r'$CL_{0.99,|re{}_t|}$'], **{'width': '60px'})
    if bool_latex:
        with open('TeX_tables/Grouping_TL_K10.tex','w') as f:
            f.write(stat_10[0].to_latex())

    # visualize tradeoff for kmeans and NN model points
    visualize_representatives_km_ann(km_rep= kMeans_10.cluster_centers_, 
                                ann_rep= data_prep_feautures_scale(data_re_transform_features(dict_to_array(cluster_analysis_10[0]), 
                                            option= 'conditional', Max_min=Max_min), Max_min, option = 'standard'), 
                            features=['age', 'sum', 'duration', 'duration (el.)', 'interest'])
