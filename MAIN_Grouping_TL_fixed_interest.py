import pandas as pd
import numpy as np
import pickle, os, logging
logging.basicConfig(filename='logging_grouping_TL.txt',level=logging.WARNING)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Average, Input

from functions.data_prep_General import data_re_transform_features, data_prep_feautures_scale
from functions.rnn_functions import create_multiple_rnn_models
from functions.statistical_analysis_functions import model_examine_indivual_fit
from functions.clustering import analyze_agglomeration_test,  dict_to_array, cluster_ann_test, pseudo_KMeans_object, get_number_of_clusters
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
model_prediction = Model(INPUT, Average()([weak_learners[i](INPUT) for i in range(N_ensembles)]))
model_prediction.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
if os.path.isfile(wd_rnn+r'/ensemble_{}_{}.h5'.format(pred_model_type,N_ensembles)):
    model_prediction.load_weights(wd_rnn+r'/ensemble_{}_{}.h5'.format(pred_model_type,N_ensembles))
    print('-----------------------------------------------------------')
    print('Loaded prediction model with {} ensembles and {} loss.'.format(N_ensembles, pred_model_type))
    print('-----------------------------------------------------------')
else:
    raise ValueError('No Prediction model available!')


###########################################################################################################################################
################################################## Agglomeration of Contracts #############################################################
###########################################################################################################################################

## Ensemble to integrate in Clustering procedure, Choose EP with 5 Sub-Models
model_supervision_clustering =  model_prediction  #### IMPORTANT CHOICE ####
N_epochs_clustering = 40000
es_patience_clustering = 100


#################################################  N = 14 #########################################################
if True:
    K_rel = 1/10000 # approx K=10 as before
    N_clusters = get_number_of_clusters(X_backtest, K_rel=K_rel)
    print('Pre-processing interest rate leads to {} clusters!'.format(N_clusters))

    # Create or load k_means Cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/PRESORT_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/PRESORT_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_presort_14 = pickle.load(input)
        print('14-means (presorted) loaded!')
    else:
        # perform clustering
        if (load_agg_model==True):
            print('Error by user. Trying to change underlying cluster assignment while retaining NN-grouping.')
            exit()
        kMeans_presort_14 = pseudo_KMeans_object(X_backtest, K_rel=K_rel)
        print('14-Means (presorted) created!')
        # save result
        with open(wd_cluster+r'/kMeans_Baseline' + r'/PRESORT_kMeans_{}.pkl'.format(N_clusters), 'wb') as output:
            pickle.dump(kMeans_presort_14, output, pickle.HIGHEST_PROTOCOL)

    # Devide Data in k clusters
    data_lst_cluster_14 = []
    targets_lst_cluster_14 = []
    for i in range(N_clusters):
        index = kMeans_presort_14.labels_ == i
        data_lst_cluster_14.append(X[index,].reshape((1,index.sum(),n_in)))
        targets_lst_cluster_14.append((y[index,].sum(axis=0)/index.sum()).reshape(1,n_output))

    # load or create NN grouping
    if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/PRESORT_cluster_object.pkl')&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/PRESORT_cluster_object.pkl', 'rb') as input:
            cluster_analysis_presort_14 = pickle.load(input)
        print('NN-grouping (MSE) loaded for K=10!')
    else:
        print('Starting NN-grouping with K=14:')
        cluster_analysis_presort_14 = cluster_ann_test(y_lst = targets_lst_cluster_14, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= 1, context = 'termlife',
                                        cluster_object = kMeans_presort_14, Max_min = Max_min,
                                        N_epochs = N_epochs_clustering, 
                                        es_patience= es_patience_clustering,
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/PRESORT_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_presort_14, output, pickle.HIGHEST_PROTOCOL)

    # per cluster view evaluation
    if bool_plot:
        analyze_agglomeration_test(baseline = kMeans_presort_14, y = y, Max_min=Max_min, insurance_type='termlife',
                            ann_object = cluster_analysis_presort_14, option_plot_selection= [0,9],
                            individual_clusters=True, option= 'plot', plot_tag = 'PRESORT', n_columns=5, figsize= (4,1))

    # statistics
    stat_14 = analyze_agglomeration_test(baseline = kMeans_presort_14, y = y, Max_min=Max_min, insurance_type='termlife',
                        ann_object = cluster_analysis_presort_14,
                        individual_clusters=True, option= 'statistic', n_columns=5)
    print('Statistics for grouping with K={} (MSE):'.format(N_clusters))
    print(stat_14[0])
    if bool_latex:
        with open('TeX_tables/Grouping_TL_K{}_PRESORT.tex'.format(N_clusters),'w') as f:
            f.write(stat_14[0].to_latex())

    # visualize tradeoff for kmeans and NN model points
    visualize_representatives_km_ann(km_rep= kMeans_presort_14.cluster_centers_, 
                                ann_rep= data_prep_feautures_scale(data_re_transform_features(dict_to_array(cluster_analysis_presort_14[0]), 
                                option= 'conditional', Max_min=Max_min), Max_min, option = 'standard'), 
                                features=['age', 'sum', 'duration', 'duration (el.)', 'interest'])

#################################################  N = 29 #########################################################

if True:
    K_rel = 1/4000 # approx K=25 as before
    N_clusters = get_number_of_clusters(X_backtest, K_rel=K_rel)
    print('Pre-processing interest rate leads to {} clusters!'.format(N_clusters))

    # Create or load k_means Cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/PRESORT_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/PRESORT_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_presort_29 = pickle.load(input)
        print('29-means (presorted) loaded!')
    else:
        # perform clustering
        if (load_agg_model==True):
            print('Error by user. Trying to change underlying cluster assignment while retaining NN-grouping.')
            exit()
        kMeans_presort_29 = pseudo_KMeans_object(X_backtest, K_rel=K_rel)
        print('29-Means (presorted) created!')
        # save result
        with open(wd_cluster+r'/kMeans_Baseline' + r'/PRESORT_kMeans_{}.pkl'.format(N_clusters), 'wb') as output:
            pickle.dump(kMeans_presort_29, output, pickle.HIGHEST_PROTOCOL)

    # Devide Data in k clusters
    data_lst_cluster_29 = []
    targets_lst_cluster_29 = []
    for i in range(N_clusters):
        index = kMeans_presort_29.labels_ == i
        data_lst_cluster_29.append(X[index,].reshape((1,index.sum(),n_in)))
        targets_lst_cluster_29.append((y[index,].sum(axis=0)/index.sum()).reshape(1,n_output))

    # load or create NN grouping
    if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/PRESORT_cluster_object.pkl')&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/PRESORT_cluster_object.pkl', 'rb') as input:
            cluster_analysis_presort_29 = pickle.load(input)
        print('NN-grouping (MSE) loaded for K=10!')
    else:
        print('Starting NN-grouping with K=29:')
        cluster_analysis_presort_29 = cluster_ann_test(y_lst = targets_lst_cluster_29, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= 1, context = 'termlife',
                                        cluster_object = kMeans_presort_29, Max_min = Max_min,
                                        N_epochs = N_epochs_clustering, 
                                        es_patience= es_patience_clustering,
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/PRESORT_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_presort_29, output, pickle.HIGHEST_PROTOCOL)

    # per cluster view evaluation
    if bool_plot:
        analyze_agglomeration_test(baseline = kMeans_presort_29, y = y, Max_min=Max_min, insurance_type='termlife',
                            ann_object = cluster_analysis_presort_29, option_plot_selection= [0,9],
                            individual_clusters=True, option= 'plot', plot_tag = 'PRESORT', n_columns=5, figsize= (4,1))

    # statistics
    stat_29 = analyze_agglomeration_test(baseline = kMeans_presort_29, y = y, Max_min=Max_min, insurance_type='termlife',
                        ann_object = cluster_analysis_presort_29,
                        individual_clusters=True, option= 'statistic', n_columns=5)
    print('Statistics for grouping with K={} (MSE):'.format(N_clusters))
    print(stat_29[0])
    if bool_latex:
        with open('TeX_tables/Grouping_TL_K{}_PRESORT.tex'.format(N_clusters),'w') as f:
            f.write(stat_29[0].to_latex())

    # visualize tradeoff for kmeans and NN model points
    visualize_representatives_km_ann(km_rep= kMeans_presort_29.cluster_centers_, 
                                ann_rep= data_prep_feautures_scale(data_re_transform_features(dict_to_array(cluster_analysis_presort_29[0]), 
                                option= 'conditional', Max_min=Max_min), Max_min, option = 'standard'), 
                                features=['age', 'sum', 'duration', 'duration (el.)', 'interest'])