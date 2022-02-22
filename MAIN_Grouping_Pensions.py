import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import cluster
import matplotlib.pyplot as plt
import time, json, pickle, os

import tensorflow as tf
print('Num. of avail. GPUs: ', len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Average, Input 

from functions.rnn_functions import create_multiple_rnn_models
from functions.clustering import cluster_ann, analyze_agglomeration_test
from functions.visualization_functions import visualizeMargDistributionPension

# import data
cd = os.getcwd() + "/Pensions" #r"C:\Users\mark.kiermayer\Documents\Python Scripts\NEW Paper (Grouping) - Code - V1\Termlife"
path_data = cd + '/Data/'
wd_rnn = cd + r'/ipynb_Checkpoints/Prediction' # path to load prediction model
wd_cluster = cd+r'/ipynb_Checkpoints/Grouping'# path to save grouping 
load_agg_model = True
load_kmeans = True
bool_latex = True

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
with open(path_data+'Pension_params.pkl', 'rb') as f:
    params = pickle.load(f)
with open(path_data+'Pension_explan_vars_range.pkl', 'rb') as f:
    explan_vars_range = pickle.load(f)

print('Parameters imported: ', params)
print('Explanatory variables imported: ', explan_vars_range)


# Range of Variables
# Matrix Version of previous upper/ lower bounds on features
Min_Max = np.array([explan_vars_range['fund'][0],explan_vars_range['fund'][1],
                    explan_vars_range['age'][0], explan_vars_range['age'][1], 
                    explan_vars_range['salary'][0], explan_vars_range['salary'][1],
                    explan_vars_range['salary_scale'][0], explan_vars_range['salary_scale'][1],
                    explan_vars_range['contribution'][0], explan_vars_range['contribution'][1]]).reshape(-1,2)



# Visualize realistic portfolio (-> save resulting plot)
visualizeMargDistributionPension(X_raw, path = os.path.join(os.getcwd(), r'Matplotlib_figures/Data_Pensions.eps'))


#################################### Build Prediction Models ###############################################

# General settings
n_output = params['pension_age_max']-explan_vars_range['age'][0]+1
n_in = X.shape[1]
INPUT = Input(shape = (n_in,))

### Single Model Configurations
### MSE Training

# Create Multiple RNNs with identical configuration
weak_learner_hist = {}
weak_learner = create_multiple_rnn_models(number=N_ensembles, model_input=INPUT,widths_rnn =[n_output],  
                                  widths_ffn = [n_output], 
                                   dense_act_fct = 'tanh', optimizer_type='adam', loss_type='mse', 
                                   metric_type='mae', dropout_share=0, 
                                   lambda_layer = True, lambda_scale =params['V_max'], log_scale=True, 
                                    model_compile = True, return_option = 'model', branch_name = '')


prediction_model = Model(INPUT, Average()([weak_learner[i](INPUT) for i in range(N_ensembles)]))    
if os.path.isfile(wd_rnn+r'/ensemble_{}_{}.h5'.format(pred_model_type, N_ensembles)):
    prediction_model.load_weights(wd_rnn+r'/ensemble_{}_{}.h5'.format(pred_model_type, N_ensembles))
    print('Prediction {}-model for supervision of grouping successfully loaded!'.format(pred_model_type), '\n')
else:
    print('Prediction model not available! Aborting grouping.')
    exit()


###################################################################################################################
#####################################  Section - Grouping #########################################################


## Ensemble to integrate in Clustering procedure, Choose EP with 5 Sub-Models
model_supervision_clustering =  prediction_model  #### IMPORTANT CHOICE ####
N_epochs_clustering = 50000
es_patience_clustering = 50
BATCH = 32

################################################  N = 100 #########################################################
if False:    
    N_clusters = 100

    # Create or load k_means Cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_100 = pickle.load(input)
        print('100-means loaded!')
    else:
        # perform clustering
        #kMeans_100 = cluster.MiniBatchKMeans(n_clusters=N_clusters, batch_size=BATCH, verbose =0).fit(X)
        if load_agg_model==True:
            print('Error by user. Trying to change underlying cluster assignment while retaining NN-grouping.')
            exit()
        kMeans_100 = cluster.KMeans(init='k-means++', n_clusters=N_clusters, n_init=10).fit(X)
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
        # perform grouping by ANN
        cluster_analysis_100 = cluster_ann(y_lst = targets_lst_cluster_100, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= 1, context = 'pension',
                                        cluster_object= kMeans_100, 
                                        N_epochs = N_epochs_clustering, 
                                        es_patience= es_patience_clustering,
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save grouping object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_100, output, pickle.HIGHEST_PROTOCOL)
    print('NN grouping, time: ', cluster_analysis_100[3].sum())


    # cluster based visual evaluation
    #analyze_agglomeration_test(baseline = kMeans_100, y = y, Max_min=Min_Max,
    #                      include_ann= True, ann_prediction= cluster_analysis_100[1], 
    #                      ann_cluster_presort= kMeans_100, ep_rate= params['early_pension_structure'],
    #                           interest_rate=params['interest_rate'],
    #                      ann_representatives= cluster_analysis_100[0], individual_clusters=True,
    #                      figsize = (20,32), insurance_type= 'pensions', pension_age_max= params['pension_age_max'])

    # check statistics
    stat_100 = analyze_agglomeration_test(baseline = kMeans_100, y = y, Max_min=Min_Max,
                        include_ann= True, ann_object= cluster_analysis_100, individual_clusters=True,
                        ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                        option= 'statistic', insurance_type= 'pensions', pension_age_max= params['pension_age_max'])
    print('Statistics for grouping with K=100:')
    print(stat_100[0])


#################################################  N = 50 #########################################################
if False:    
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
        kMeans_50 = cluster.KMeans(init='k-means++', n_clusters=N_clusters, n_init=10).fit(X)
        # save result
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
        # perform agglomeration by ANN # Check if N_ensemble matches supervision model
        cluster_analysis_50 = cluster_ann(y_lst = targets_lst_cluster_50, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= 1, context = 'pension',
                                        cluster_object= kMeans_50,
                                        N_epochs = N_epochs_clustering, 
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_50, output, pickle.HIGHEST_PROTOCOL)
    print('NN grouping, time: ', cluster_analysis_50[3].sum())


    # per cluster view evaluation
    # analyze_agglomeration_test(baseline = kMeans_50, y = y, Max_min=Min_Max,
    #                      include_ann= True, ann_prediction= cluster_analysis_50[1],
    #                      ann_cluster_presort= kMeans_50,
    #                      ann_representatives= cluster_analysis_50[0], individual_clusters=True,
    #                      ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
    #                      figsize = (20,12), insurance_type= 'pensions', pension_age_max= params['pension_age_max'])


    # statistics
    stat_50 = analyze_agglomeration_test(baseline = kMeans_50, y = y, Max_min=Min_Max,
                        include_ann= True, ann_object= cluster_analysis_50, individual_clusters=True,
                        ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                        option= 'statistic', insurance_type= 'pensions', pension_age_max= params['pension_age_max'])

    print('Statistics for grouping with K=50:')
    print(stat_50[0])


#################################################  N = 25 #########################################################
if False:    
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
        kMeans_25 = cluster.KMeans(init='k-means++', n_clusters=N_clusters, n_init=10).fit(X)
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
        # perform agglomeration by ANN # Check if N_ensemble matches supervision model
        cluster_analysis_25 = cluster_ann(y_lst = targets_lst_cluster_25, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= 1, context = 'pension',
                                        cluster_object= kMeans_25,
                                        N_epochs = N_epochs_clustering, 
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_25, output, pickle.HIGHEST_PROTOCOL)
    print('NN grouping, time: ', cluster_analysis_25[3].sum())

    # cluster based visual evaluation
    #analyze_agglomeration_test(baseline = kMeans_25, y = y, Max_min=Min_Max,
    #                      include_ann= True, ann_prediction= cluster_analysis_25[1],
    #                      ann_cluster_presort= kMeans_25,
    #                      ann_representatives= cluster_analysis_25[0], individual_clusters=True,
    #                      ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
    #                      figsize = (20,12), insurance_type= 'pensions', pension_age_max= params['pension_age_max'])

    # statistics
    stat_25 = analyze_agglomeration_test(baseline = kMeans_25, y = y, Max_min=Min_Max,
                        include_ann= True, ann_object= cluster_analysis_25, individual_clusters=True,
                        ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                        option= 'statistic', insurance_type= 'pensions', pension_age_max= params['pension_age_max'])
    print('Statistics for grouping with K=25:')
    print(stat_25[0])


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
        kMeans_10 = cluster.KMeans(init='k-means++', n_clusters=N_clusters, n_init=10).fit(X)
        # save result
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
        #print(os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl'))
        #print(load_agg_model)
        #exit()
        cluster_analysis_10 = cluster_ann(y_lst = targets_lst_cluster_10, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= 1, context = 'pension',
                                        cluster_object= kMeans_10,
                                        N_epochs = N_epochs_clustering, 
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_10, output, pickle.HIGHEST_PROTOCOL)
    print('NN grouping, time: ', sum(cluster_analysis_10[3].values()))

    # cluster based visual evaluation
    analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Min_Max,
                        include_ann= True, ann_object= cluster_analysis_10, individual_clusters=True,
                        ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                        figsize = (20,4), insurance_type= 'pensions', pension_age_max= params['pension_age_max'])

    # statistics
    stat_10 = analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Min_Max,
                        include_ann= True, ann_object= cluster_analysis_10, individual_clusters=True,
                        ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                        option= 'statistic', insurance_type= 'pensions', pension_age_max= params['pension_age_max'])
    print('Statistics for grouping with K=10 (MSE):')
    print(stat_10[0])#.style.set_properties(subset=[r'$CL_{0.99,|re{}_t|}$'], **{'width': '60px'})

    if bool_latex:
            with open('TeX_tables/Grouping_DC_K10.tex','w') as tw:
                tw.write(stat_10[0].to_latex())

    # visualize tradeoff for kmeans and NN model points
    #visualize_representatives_km_ann(km_rep = kMeans_10.cluster_centers_, 
    #                                 ann_rep = cluster_analysis_10_mse[0], features= input_used)

########################################  20=10x2 model points #########################################################
if True:
    N_clusters = 10
    N_centroids = 2


    # load 10-means cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_10 = pickle.load(input)
        print('10-means Model loaded!')
    else:
        print('10-means model not available!')
        exit()

    # Devide Data in k clusters
    data_lst_cluster_10 = []
    targets_lst_cluster_10 = []
    for i in range(N_clusters):
        index = kMeans_10.labels_ == i
        data_lst_cluster_10.append(X[index,].reshape((1,index.sum(),n_in)))
        targets_lst_cluster_10.append((y[index,].sum(axis=0)/index.sum()).reshape(1,n_output))   


    # load or create NN grouping (with 2 model points and 5 clusters)
    if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object_10_2.pkl')&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object_10_2.pkl', 'rb') as input:
            cluster_analysis_10_2 = pickle.load(input)
        print('NN grouping with K=10 and C=2 loaded!')
    else:
        # perform agglomeration by ANN
        # Check whether N_ensembles and model_supervision_clustering match
        cluster_analysis_10_2 = cluster_ann(y_lst = targets_lst_cluster_10, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= N_centroids, context = 'pension',
                                        cluster_object= kMeans_10,
                                        N_epochs = N_epochs_clustering,
                                        wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object_10_2.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_10_2, output, pickle.HIGHEST_PROTOCOL)
    print('NN grouping, time: ', sum(cluster_analysis_10_2[3].values()))
    # cluster based visual evaluation
    #analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Min_Max,
    #                      include_ann= True, ann_prediction= cluster_analysis_10_2[1],
    #                           ann_cluster_presort= kMeans_10,
    #                      ann_representatives= cluster_analysis_10_2[0], individual_clusters=True,
    #                      ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
    #                      figsize = (20,4), insurance_type= 'pensions', pension_age_max= params['pension_age_max'])


    # uncomment for info on training times per cluster
    #for i in range(10):
    #   print(len(cluster_analysis_10_2[-1][i]['loss']))

    # cluster based visual evaluation (selected cluster)
    analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Min_Max,
                        include_ann= True, ann_object= cluster_analysis_10_2, individual_clusters=True,
                        ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                        option_plot_selection= [3,9], n_columns= 2,
                        figsize = (13,3), insurance_type= 'pensions', pension_age_max= params['pension_age_max'])

    # statistics and plot
    stat_10_2 = analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Min_Max,
                        include_ann= True, ann_object= cluster_analysis_10_2, individual_clusters=True,
                        ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                        option='statistic', insurance_type= 'pensions', pension_age_max= params['pension_age_max'])
    print('Statistics for grouping with 20=10x2 model points:')
    print(stat_10_2[0], '\n')


#########################  10 model points (no pre-clustering) ####################################################
if True:
        N_clusters = 1
        N_centroids = 10

        # load or create kmeans cluster assignment
        if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
            # load model weights
            with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
                kMeans_1 = pickle.load(input)
                print('Model loaded!')
        else:
            # perform clustering
            if load_agg_model==True:
                print('Error by user. Trying to change underlying cluster assignment while retaining NN-grouping.')
                exit()
            kMeans_1 = cluster.KMeans(init='k-means++', n_clusters=N_clusters, n_init=10).fit(X)
            # save result
            with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'wb') as output:
                pickle.dump(kMeans_1, output, pickle.HIGHEST_PROTOCOL)


        # Devide Data in k clusters
        data_lst_cluster_1 = []
        targets_lst_cluster_1 = []
        for i in range(N_clusters):
            index = kMeans_1.labels_ == i
            data_lst_cluster_1.append(X[index,].reshape((1,index.sum(),n_in)))
            targets_lst_cluster_1.append((y[index,].sum(axis=0)/index.sum()).reshape(1,n_output))

        # load or create NN grouping
        if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object_1_10.pkl')&load_agg_model:
            # load model weights
            with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object_1_10.pkl', 'rb') as input:
                cluster_analysis_1_10 = pickle.load(input)
            print('NN grouping with K=1 and C=10 loaded!')
        else:
            # perform agglomeration by ANN
            # Check whether N_ensembles and model_supervision_clustering match
            cluster_analysis_1_10 = cluster_ann(y_lst = targets_lst_cluster_1, 
                                            model_prediction = model_supervision_clustering, 
                                            N_centroids= N_centroids, cluster_object= kMeans_1,
                                            N_epochs = N_epochs_clustering)#, wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
            # save agglomeration object
            with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object_1_10.pkl', 'wb') as output:
                pickle.dump(cluster_analysis_1_10, output, pickle.HIGHEST_PROTOCOL)

        print('NN grouping, time: ', sum(cluster_analysis_1_10[3].values()))

        # load 10-means reference
        if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(10))&load_agg_model:
            # load model weights
            with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(10), 'rb') as input:
                kMeans_10 = pickle.load(input)
            print('10-means Model loaded!')
        else:
            print('10-means Model not available!')
            exit()

        # cluster based visual evaluation
        analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Min_Max, ann_cluster_presort= kMeans_1,
                        include_ann= True, ann_object= cluster_analysis_1_10, individual_clusters=False,
                        ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                        figsize = (20,4), insurance_type= 'pensions', pension_age_max= params['pension_age_max'])


        # statistics and plot
        stat_1_10 = analyze_agglomeration_test(baseline = kMeans_10, y = y, Max_min=Min_Max,
                            include_ann= True, ann_cluster_presort= kMeans_1,
                            ann_object= cluster_analysis_1_10, individual_clusters=True,
                            ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                            option= 'statistic', insurance_type= 'pensions', pension_age_max= params['pension_age_max'])
        print('Statistics for grouping with 10=10x1 model points:')
        print(stat_1_10[0], '\n')

        if bool_latex:
            with open('TeX_tables/Grouping_DC_K1_C10.tex','w') as tw:
                tw.write(stat_1_10[0].to_latex())


########################################  25=5x5 model points #########################################################
if True:
    N_clusters = 5
    N_centroids = 5

    # load or create cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_5 = pickle.load(input)
        print('Model loaded!')
    else:
        # perform clustering
        if load_agg_model==True:
            print('Error by user. Trying to change underlying cluster assignment while retaining NN-grouping.')
            exit()
        kMeans_5 = cluster.KMeans(init='k-means++', n_clusters=N_clusters, n_init=10).fit(X)
        # save result
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(N_clusters), 'wb') as output:
            pickle.dump(kMeans_5, output, pickle.HIGHEST_PROTOCOL)

    # Devide Data in k clusters
    data_lst_cluster_5 = []
    targets_lst_cluster_5 = []
    for i in range(N_clusters):
        index = kMeans_5.labels_ == i
        data_lst_cluster_5.append(X[index,].reshape((1,index.sum(),n_in)))
        targets_lst_cluster_5.append((y[index,].sum(axis=0)/index.sum()).reshape(1,n_output))

    # load or create NN grouping
    if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object_5_5.pkl')&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object_5_5.pkl', 'rb') as input:
            cluster_analysis_5_5 = pickle.load(input)
        print('NN grouping with K=5 and C=5 loaded!')
    else:
        # perform agglomeration by ANN
        # Check whether N_ensembles and model_supervision_clustering match
        cluster_analysis_5_5 = cluster_ann(y_lst = targets_lst_cluster_5, 
                                        model_prediction = model_supervision_clustering, 
                                        N_centroids= N_centroids, cluster_object= kMeans_5,
                                        N_epochs = N_epochs_clustering)#, wd_cluster = wd_cluster +r'/K_{}'.format(N_clusters))
        # save agglomeration object
        with open(wd_cluster+r'/K_{}'.format(N_clusters) + r'/NEW_cluster_object_5_5.pkl', 'wb') as output:
            pickle.dump(cluster_analysis_5_5, output, pickle.HIGHEST_PROTOCOL)

    print('NN grouping, time: ', sum(cluster_analysis_5_5[3].values()))


    # Load 25-means reference
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(25))&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/NEW_kMeans_{}.pkl'.format(25), 'rb') as input:
            kMeans_25 = pickle.load(input)
        print('25-means loaded!')
    else:
        print('25-means model not available!')
        exit()

    # cluster based visual evaluation
    analyze_agglomeration_test(baseline = kMeans_25, y = y, Max_min=Min_Max, ann_cluster_presort= kMeans_5,
                        include_ann= True, ann_object= cluster_analysis_5_5, individual_clusters=True,
                        ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                        figsize = (20,4), insurance_type= 'pensions', pension_age_max= params['pension_age_max'])


    # statistics and plot
    stat_5_5 = analyze_agglomeration_test(baseline = kMeans_25, y = y, Max_min=Min_Max,
                        include_ann= True, ann_cluster_presort= kMeans_5,
                        ann_object= cluster_analysis_5_5, individual_clusters=True,
                        ep_rate= params['early_pension_structure'], interest_rate= params['interest_rate'],
                        option= 'statistic',insurance_type= 'pensions', pension_age_max= params['pension_age_max'])
    print('Grouping with 25=5x5 model points')
    print(stat_5_5[0])

    if bool_latex:
        with open('TeX_tables/Grouping_DC_K5_C5.tex','w') as tw:
            tw.write(stat_5_5[0].to_latex())


# Table for runtimes of clustering models
runtimes = [#sum(cluster_analysis_100[3].values())/60,sum(cluster_analysis_50[3].values())/60,
            #sum(cluster_analysis_25[3].values())/60, 
            sum(cluster_analysis_10[3].values())/60,
            sum(cluster_analysis_10_2[3].values())/60, sum(cluster_analysis_1_10[3].values())/60,
            sum(cluster_analysis_5_5[3].values())/60]
df_agglom_runtimes = pd.DataFrame(data = None, index = None, columns = [r'$K$','','$10$', '10_2', '1_10', '5_5'])
df_agglom_runtimes.loc[''] = [r'$\text{Runtime [min]}$',r'$\tilde{P}_\mathcal{N}$']+runtimes
print('Runtime of algorithms:')
print(df_agglom_runtimes)