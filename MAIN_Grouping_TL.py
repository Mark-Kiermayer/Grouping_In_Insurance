import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import cluster
import matplotlib.pyplot as plt
import  pickle, os

from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Average,  Input

from functions.data_prep_General import data_re_transform_features, data_prep_feautures_scale
from functions.rnn_functions import create_multiple_rnn_models
from functions.statistical_analysis_functions import model_examine_indivual_fit
from functions.clustering import analyze_agglomeration_test, cluster_ann, dict_to_array
from functions.visualization_functions import visualizeMargDistributionTermLife, visualize_representatives_km_ann


# import data
cd = os.getcwd() + r'/TermLife' #r"C:\Users\mark.kiermayer\Documents\Python Scripts\NEW Paper (Grouping) - Code - V1\Termlife"
path_data = cd + r'/Data/'
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
exit()

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



# Visualize realistic portfolio (-> save resulting plot)
visualizeMargDistributionTermLife(X_raw, path = os.path.join(os.getcwd(), r'Matplotlib_figures/Data_TL.eps'))


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

#################################################  N = 10 #########################################################
if True:
    N_clusters = 10

    # load or perform kmeans cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_10 = pickle.load(input)
        print('10-means Model loaded!')
    else:
        # perform clustering
        if load_agg_model==True:
            print('Error by user. Trying to change underlying cluster assignment while retaining NN-grouping.')
            exit()
        kMeans_10 = cluster.KMeans(init='k-means++', n_clusters=N_clusters, n_init=10).fit(X_backtest)
        print('KMeans 10 created!')
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'wb') as output:
            pickle.dump(kMeans_10, output, pickle.HIGHEST_PROTOCOL)

    # Devide Data in k clusters
    data_lst_cluster_10 = []
    targets_lst_cluster_10 = []
    for i in range(N_clusters):
        index = kMeans_10.labels_ == i
        data_lst_cluster_10.append(X[index,].reshape((1,index.sum(),n_in)))
        targets_lst_cluster_10.append((y[index,].sum(axis=0)/index.sum()).reshape(1,n_output))


    # load or create NN grouping
    if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl')&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'rb') as input:
            cluster_analysis_10 = pickle.load(input)
        print('NN-grouping (MSE) loaded for K=10!')
    else:
        print('Starting NN-grouping with K=10:')
        cluster_analysis_10 = cluster_ann(y_lst = targets_lst_cluster_10, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= 1, context = 'termlife',
                                        cluster_object = kMeans_10, Max_min = Max_min,
                                        N_epochs = N_epochs_clustering, 
                                        es_patience= es_patience_clustering,
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_10, output, pickle.HIGHEST_PROTOCOL)

    # per cluster view evaluation
    if bool_plot:
        analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Max_min, insurance_type='termlife',
                            ann_object = cluster_analysis_10, option_plot_selection= [0,3],
                            individual_clusters=True, option= 'plot', n_columns=5, figsize= (4,1))

    # statistics
    stat_10 = analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Max_min, insurance_type='termlife',
                        ann_object = cluster_analysis_10,
                        individual_clusters=True, option= 'statistic', n_columns=5)
    print('Statistics for grouping with K=10 (MSE):')
    print(stat_10[0])
    if bool_latex:
        with open('TeX_tables/Grouping_TL_K10.tex','w') as f:
            f.write(stat_10[0].to_latex())

    # visualize tradeoff for kmeans and NN model points
    visualize_representatives_km_ann(km_rep= kMeans_10.cluster_centers_, 
                                ann_rep= data_prep_feautures_scale(data_re_transform_features(dict_to_array(cluster_analysis_10[0]), 
                                            option= 'conditional', Max_min=Max_min), Max_min, option = 'standard'), 
                            features=['age', 'sum', 'duration', 'duration (el.)', 'interest'])

#################################################  N = 25 #########################################################
if True:
    N_clusters = 25
    # Load or perform kmeans cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_25 = pickle.load(input)
            print('25-means loaded!')
    else:
        # perform clustering
        if load_agg_model==True:
            print('Error by user. Trying to change underlying cluster assignment while retaining NN-grouping.')
            exit()
        kMeans_25 = cluster.KMeans(init='k-means++', n_clusters=N_clusters, n_init=10).fit(X_backtest)
        print('KMeans 25 created!')
        # save result
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'wb') as output:
            pickle.dump(kMeans_25, output, pickle.HIGHEST_PROTOCOL)


    # Devide Data in k clusters
    data_lst_cluster_25 = []
    targets_lst_cluster_25 = []
    for i in range(N_clusters):
        index = kMeans_25.labels_ == i
        data_lst_cluster_25.append(X[index,].reshape((1,index.sum(),n_in)))
        targets_lst_cluster_25.append((y[index,].sum(axis=0)/index.sum()).reshape(1,n_output))

    # load or create NN grouping
    if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl')&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'rb') as input:
            cluster_analysis_25 = pickle.load(input)
        print('NN-grouping loaded for K=25!')
    else:
        print('Starting NN-grouping with K=25:')
        cluster_analysis_25 = cluster_ann(y_lst = targets_lst_cluster_25, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= 1, context = 'termlife',
                                        cluster_object = kMeans_25, Max_min = Max_min,
                                        N_epochs = N_epochs_clustering, 
                                        es_patience= es_patience_clustering,
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_25, output, pickle.HIGHEST_PROTOCOL)

    # per cluster view evaluation
    if bool_plot:
        analyze_agglomeration_test(baseline = kMeans_25, y = y, Max_min=Max_min, insurance_type='termlife',
                            ann_object= cluster_analysis_25,
                            #ann_prediction= cluster_analysis_25[1], ann_representatives= cluster_analysis_25[0],
                            individual_clusters=True, option= 'plot',n_columns=5, figsize= (15,8))

    # statistics
    stat_25 = analyze_agglomeration_test(baseline = kMeans_25, y = y, Max_min=Max_min, insurance_type='termlife',
                        ann_object = cluster_analysis_25,
                        option= 'statistic', n_columns=5)
    print('Statistics for grouping with K=25:')
    print(stat_25[0])
    if bool_latex:
        with open('TeX_tables/Grouping_TL_K25.tex','w') as f:
            f.write(stat_25[0].to_latex())

#################################################  N = 50 #########################################################
if True:
    N_clusters = 50

    # Load or perform kmean cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_50 = pickle.load(input)
        print('Model loaded!')
    else:
        # perform clustering
        if load_agg_model==True:
            print('Error by user. Trying to change underlying cluster assignment while retaining NN-grouping.')
            exit()
        kMeans_50 = cluster.KMeans(init='k-means++', n_clusters=N_clusters, n_init=10).fit(X_backtest)
        print('KMeans 50 created!')
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'wb') as output:
            pickle.dump(kMeans_50, output, pickle.HIGHEST_PROTOCOL)

    # Devide Data in k clusters
    data_lst_cluster_50 = []
    targets_lst_cluster_50 = []
    for i in range(N_clusters):
        index = kMeans_50.labels_ == i
        data_lst_cluster_50.append(X[index,].reshape((1,index.sum(),n_in)))
        targets_lst_cluster_50.append((y[index,].sum(axis=0)/index.sum()).reshape(1,n_output))


    # load or perform NN grouping
    if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl')&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'rb') as input:
            cluster_analysis_50 = pickle.load(input)
        print('NN-grouping loaded for K=50!')
    else:
        print('Starting NN-grouping with K=50:')
        cluster_analysis_50 = cluster_ann(y_lst = targets_lst_cluster_50, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= 1, context = 'termlife',
                                        cluster_object = kMeans_50, Max_min = Max_min,
                                        N_epochs = N_epochs_clustering, 
                                        es_patience= es_patience_clustering,
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_50, output, pickle.HIGHEST_PROTOCOL)


    # per cluster view evaluation
    if bool_plot:
        analyze_agglomeration_test(baseline = kMeans_50, y = y, Max_min=Max_min, insurance_type='termlife',
                            ann_object = cluster_analysis_50,
                            individual_clusters=True, option= 'plot',n_columns=5, figsize= (15,20))

    # statistics
    stat_50 = analyze_agglomeration_test(baseline = kMeans_50, y = y, Max_min=Max_min, insurance_type='termlife',
                        ann_object = cluster_analysis_50,
                        individual_clusters=True, option= 'statistic', n_columns=5)

    print('Statistics for grouping with K=50:')
    print(stat_50[0])
    if bool_latex:
        with open('TeX_tables/Grouping_TL_K50.tex','w') as f:
            f.write(stat_50[0].to_latex())


################################################  N = 100 #########################################################
if True:
    N_clusters = 100

    # Create or load k_means Cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_100 = pickle.load(input)
        print('100-means loaded!')
    else:
        # perform clustering
        if load_agg_model==True:
            print('Error by user. Trying to change underlying cluster assignment while retaining NN-grouping.')
            exit()
        kMeans_100 = cluster.KMeans(init='k-means++', n_clusters=N_clusters, n_init=10).fit(X_backtest)
        print('KMeans 100 created!')
        # save result
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'wb') as output:
            pickle.dump(kMeans_100, output, pickle.HIGHEST_PROTOCOL)

    # Devide Data in k clusters
    data_lst_cluster_100 = []
    targets_lst_cluster_100 = []
    for i in range(N_clusters):
        index = kMeans_100.labels_ == i
        data_lst_cluster_100.append(X[index,].reshape((1,index.sum(),n_in)))
        targets_lst_cluster_100.append((y[index,].sum(axis=0)/index.sum()).reshape(1,n_output))

    # either load or create NN grouping
    if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl')&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'rb') as input:
            cluster_analysis_100 = pickle.load(input)
        print('NN-grouping loaded for K=100!')
    else:
        print('Starting NN-grouping with K=100:')
        cluster_analysis_100 = cluster_ann(y_lst = targets_lst_cluster_100, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= 1, context = 'termlife',
                                        cluster_object = kMeans_100, Max_min = Max_min,
                                        N_epochs = N_epochs_clustering, 
                                        es_patience= es_patience_clustering,
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save grouping object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_100, output, pickle.HIGHEST_PROTOCOL)


    # cluster based visual evaluation
    if bool_plot:
        analyze_agglomeration_test(baseline = kMeans_100, y = y, Max_min=Max_min, insurance_type='termlife',
                            ann_object = cluster_analysis_100,
                            individual_clusters=True, option= 'plot', n_columns=5, figsize= (15,15))


    # statistics
    stat_100 = analyze_agglomeration_test(baseline = kMeans_100, y = y, Max_min=Max_min, insurance_type='termlife',
                        ann_object = cluster_analysis_100,
                        individual_clusters=True, option= 'statistic', n_columns=5)
    print('Statistics for grouping with K=100:')
    print(stat_100[0])
    if bool_latex:
        with open('TeX_tables/Grouping_TL_K100.tex','w') as f:
            f.write(stat_100[0].to_latex())

    #visualize_representatives_km_ann(km_rep= kMeans_100.cluster_centers_, 
    #                            ann_rep= data_prep_feautures_scale(data_re_transform_features(dict_to_array(cluster_analysis_100[0]), 
    #                                        option= 'conditional', Max_min=Max_min), Max_min, option = 'standard'), 
    #                        features=['age', 'sum', 'duration', 'duration (el.)', 'interest'])

# Table for runtimes of clustering models
runtimes = [sum(cluster_analysis_100[3].values())/60,sum(cluster_analysis_50[3].values())/60,
            sum(cluster_analysis_25[3].values())/60, sum(cluster_analysis_10[3].values())/60]
df_agglom_runtimes = pd.DataFrame(data = None, index = None, columns = [r'$K$','','$100$','$50$', '$25$', '$10$'])#, '10_2', '1_10', '5_5'])
df_agglom_runtimes.loc[''] = [r'$\text{Runtime [min]}$',r'$\tilde{P}_\mathcal{N}$']+runtimes
print('Runtime of algorithms:')
print(df_agglom_runtimes)