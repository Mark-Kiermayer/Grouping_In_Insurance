import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter # trim redundant zeros in xticks, e.g. 0(.000)
import pickle, os, logging
logging.basicConfig(filename='logging_grouping_TL.txt',level=logging.WARNING)

from functions.data_prep_General import data_re_transform_features, data_prep_feautures_scale
from functions.actuarial_functions import SCR_interest_analysis, target_investment_return, SCR_analysis
from functions.clustering import kmeans_counts, dict_to_array, termlife_km_centroids_prediction, kmeans_counts
from functions.visualization_functions import visualize_representatives_km_ann, set_size


# import data
cd = os.getcwd() + "/TermLife" #r"C:\Users\mark.kiermayer\Documents\Python Scripts\NEW Paper (Grouping) - Code - V1\Termlife"
path_data = cd + '/Data/'
wd_rnn = cd + r'/ipynb_Checkpoints/Prediction' # path to load prediction model
wd_cluster = cd+r'/ipynb_Checkpoints/Grouping'# path to save grouping 


#########################################################################################################
str_cluster_prefix = r'TEST' ## <------- choice of clustering model type (NEW vs TEST)
clustering_object = r'/'+str_cluster_prefix + r'_cluster_object.pkl'  
#########################################################################################################

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
n_in = len(explan_vars_range.keys())

# Matrix Version of previous upper/ lower bounds on features
Max_min = np.array([explan_vars_range['age'][0],explan_vars_range['age'][1]+explan_vars_range['duration'][1],
                    explan_vars_range['sum_ins'][0], explan_vars_range['sum_ins'][1], 
                    explan_vars_range['duration'][0], explan_vars_range['duration'][1], 
                    explan_vars_range['age_of_contract'][0], explan_vars_range['age_of_contract'][1], 
                    explan_vars_range['interest_rate'][0], explan_vars_range['interest_rate'][1]]).reshape(-1,2)

# max-min-scaled data used for kMeans baseline -> eliminate cond. scaling of elapsed duration for kMeans baseline
X_backtest = data_prep_feautures_scale(X_raw, Max_min) 

########################################## Section 1 b) - stressed target values ######################################

# load K-Means baseline and NN grouping (original or new incl. premium)
if True:
    N_clusters = 14

    # load or perform kmeans cluster assignment
    if os.path.isfile(wd_cluster+r'/kMeans_Baseline' + r'/PRESORT_kMeans_{}.pkl'.format(N_clusters))&load_kmeans:
        # load model weights
        with open(wd_cluster+r'/kMeans_Baseline' + r'/PRESORT_kMeans_{}.pkl'.format(N_clusters), 'rb') as input:
            kMeans_presort = pickle.load(input)
        print('{}-Kmeans (presorted) loaded!'.format(N_clusters))
    else:
        print('{}-Kmeans (presorted) not available!'.format(N_clusters))
        exit()

    # load or create NN grouping
    if os.path.isfile(wd_cluster+r'/K_{}'.format(N_clusters) + r'/PRESORT_cluster_object.pkl')&load_agg_model:
        # load model weights
        with open(wd_cluster+r'/K_{}'.format(N_clusters) +  r'/PRESORT_cluster_object.pkl', 'rb') as input:
            cluster_analysis_presorted = pickle.load(input)
        print('NN-grouping (MSE) loaded for K={}!'.format(N_clusters))
    else:
        print('NN-grouping not available!')
        exit()


nn_model_points = data_re_transform_features(dict_to_array(cluster_analysis_presorted[0]), option= 'conditional', Max_min=Max_min)
nn_fund = (dict_to_array(cluster_analysis_presorted[1])[:,0],dict_to_array(cluster_analysis_presorted[1])[:,1])
nn_counts = cluster_analysis_presorted[2].flatten()

km_model_points = data_re_transform_features(kMeans_presort.cluster_centers_, Max_min, option = 'standard')
km_fund = termlife_km_centroids_prediction(km_model_points)
km_counts = kmeans_counts(kMeans_presort.labels_, N_clusters).flatten()

if True:
    visualize_representatives_km_ann(km_rep= kMeans_presort.cluster_centers_, ann_rep= data_prep_feautures_scale(nn_model_points, Max_min, option = 'standard'), 
                                     features = [r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$'],#features=['age', 'sum', 'duration', 'duration (el.)', 'interest'], 
                                     plot_tag=str_cluster_prefix)

    SCR_analysis(val_true=(X_raw, (y[:,0], y[:,1])), val_nn= (nn_model_points, nn_fund, nn_counts), val_km= (km_model_points, km_fund, km_counts),
                        A= params['A'], B= params['B'], c= params['c'])


# interest rate observed on portfolio level
df_scr = SCR_interest_analysis(val_true=(X_raw,(y[:,0], y[:,1])), val_nn= (nn_model_points, nn_fund, nn_counts), val_km= (km_model_points, km_fund, km_counts),
                    A= params['A'], B= params['B'], c= params['c'])

# determine 
interest_avg = target_investment_return( contracts=X_raw, assets=y[:,0], counts = None, premiums = None, A= params['A'], B= params['B'], c= params['c'])  

# Figure 1 - Lineplot
plt.figure(figsize=set_size(fraction=.5))
plt.plot(df_scr.loc[r'$P$'], 'r*', label = r'$P$')#, markersize=4)
plt.plot(df_scr.loc[r'$P^{(NN)}$'], color = 'black', label = r'$P^{(NN)}$')
plt.plot(df_scr.loc[r'$P^{(KM)}$'], color = 'orange', label = r'$P^{(KM)}$')
plt.ylabel(r'loss $L_{\omega}(\cdot)$')
plt.xlabel(r'interest rate $i_{\omega}$')
plt.xticks([-0.05, -0.025, 0, np.round_(interest_avg,4), 0.05])
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.legend(fontsize=9, labelspacing=.2)
plt.tight_layout()
plt.savefig(os.getcwd()+r'/Matplotlib_figures/SCR_{}_{}_line_plot.eps'.format(N_clusters,'MODIFIED'), format = 'eps')
plt.show()

# Figure 2 - relative differences
if True:
    df_scr_ae = pd.DataFrame(data = None, index=df_scr.columns)
    df_scr_ae[r'$P^{(NN)}$'] = np.abs(df_scr.loc[r'$P^{(NN)}$']-df_scr.loc[r'$P$'])
    df_scr_ae[r'$P^{(KM)}$'] = np.abs(df_scr.loc[r'$P^{(KM)}$']-df_scr.loc[r'$P$'])
    n = len(df_scr_ae)-1

    nn_delta = (df_scr_ae.iloc[1,0]-df_scr_ae.iloc[0,0])/(df_scr_ae.index[1]-df_scr_ae.index[0])
    nn_shift = df_scr_ae.iloc[1,0]-nn_delta*df_scr_ae.index[1]
    nn_zero = -nn_shift/nn_delta

    km_delta = (df_scr_ae.iloc[1,1]-df_scr_ae.iloc[0,1])/(df_scr_ae.index[1]-df_scr_ae.index[0])
    km_shift = df_scr_ae.iloc[1,1]-km_delta*df_scr_ae.index[1]
    km_zero = -km_shift/km_delta

    plt.figure(figsize=set_size(fraction=.5))
    plt.plot([df_scr_ae.index[0],km_zero,df_scr_ae.index[n]], 
            [km_delta*df_scr_ae.index[0]+km_shift, km_delta*km_zero+km_shift, km_delta*km_zero+km_shift-km_delta*(df_scr_ae.index[n]-km_zero)], 
            color='orange', label=r'$P^{(KM)}$')    
    plt.plot([df_scr_ae.index[0],nn_zero, df_scr_ae.index[n]], 
                [nn_delta*df_scr_ae.index[0]+nn_shift, nn_delta*nn_zero+nn_shift,nn_delta*nn_zero+nn_shift-nn_delta*(df_scr_ae.index[n]-nn_zero)], 
                color='black', label=r'$P^{(NN)}$')


    plt.ylabel(r'abs. error')
    plt.xlabel(r'interest rate $i_{\omega}$')
    plt.xticks([-0.05, -0.025, 0, np.round_(interest_avg,4), 0.05])
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.tight_layout()
    plt.savefig(os.getcwd()+r'/Matplotlib_figures/SCR_{}_{}_ae_plot.eps'.format(N_clusters,'MODIFIED'), format = 'eps')
    plt.show()

    print(df_scr_ae)