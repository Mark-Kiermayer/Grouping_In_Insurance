import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

from statistical_analysis_functions import relate_loss


def set_size(width=6.5, fraction=1):

    """
    Set figure dimensions to avoid scaling in LaTeX.
    Use a presumably visually appealing heigth/width ration.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in in
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    width =width/inches_per_pt
    fig_width_pt = width * fraction
    # Golden ratio to set aesthetic figure height: https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim 


def visualizeMargDistributionTermLife(X_raw, path):
    '''
    Visualize explanatory variables of TL portfolio
    '''

    fig, ax = plt.subplots(1,5, figsize=(12,2.5))
    ax=ax.flatten()
    sns.distplot(X_raw[:,0], norm_hist= True, kde=False, color = '#808080', ax=ax[0], bins = np.unique(X_raw[:,0]))
    sns.distplot(X_raw[:,1],norm_hist= True, kde=False, color = '#808080',ax=ax[1])
    sns.distplot(X_raw[:,2], norm_hist= True,kde=False, color = '#808080',ax=ax[2], bins = np.unique(X_raw[:,2]))
    sns.distplot(X_raw[:,3],norm_hist= True,kde=False, color = '#808080', ax=ax[3], bins = np.unique(X_raw[:,3]))

    interest_hist = np.unique(X_raw[:,4], return_counts=True)
    ax[4].bar(x=interest_hist[0], height= interest_hist[1]/sum(interest_hist[1]), width= 0.001,color = '#808080', alpha = .5)
    #sns.distplot(X_raw[:,4],norm_hist= True,kde=False,  color = 'black',ax=ax[4])#, bins = np.unique(X_raw[:,4]))
    for i in range(5):
        ax[i].set_xlabel(r'$X_{}$'.format(i+1))
    plt.tight_layout()
    sns.despine()
    plt.savefig(path, format='eps')
    plt.close()

def visualizeMargDistributionPension(X_raw, path):
    '''
    Visualize explanatory variables of DB portfolio
    '''

    fig, ax = plt.subplots(1,5, figsize=(12,2.5))
    ax=ax.flatten()
    sns.distplot(X_raw[:,1], norm_hist= True, kde=False, color = 'grey', ax=ax[0], bins = np.unique(X_raw[:,1]))
    sns.distplot(X_raw[:,0],norm_hist= True, kde=False, color = 'grey',ax=ax[1])
    ax[1].set_xticks([0,50000,100000])
    sns.distplot(X_raw[:,2], norm_hist= True,kde=False, color = 'grey',ax=ax[2])
    ax[2].set_xticks([0,60000,120000])
    sns.distplot(X_raw[:,3],norm_hist= True,kde=False, color = 'grey', ax=ax[3])
    ax[3].set_xticks([0.01,0.05])
    sns.distplot(X_raw[:,4],norm_hist= True,kde=False,  color = 'grey',ax=ax[4])
    ax[4].set_xticks([0.01,0.05])
    for i in range(5):
        ax[i].set_xlabel(r'$X_{}$'.format(i+1))
    plt.tight_layout()
    sns.despine()
    plt.savefig(path, format='eps')
    plt.close()

 
def training_progress_visual(history, option_validation=True, option_relate = True, y = None, 
                             fig_size = (10,6), model_name = '', option_simple_fig = False):

    '''
    Given a Training History, visualize MSE and MAE (incl. their log-versions)
    '''

    n_epoch = len(history['loss'])
    
    relate_mse_5 = relate_loss(data=y, discrepancy=0.05, measure='mse')
    relate_mse_1 = relate_loss(data=y, discrepancy=0.01, measure='mse')
    relate_mae_5 = relate_loss(data=y, discrepancy=0.05, measure='mae')
    relate_mae_1 = relate_loss(data=y, discrepancy=0.01, measure='mae')
        
    if option_simple_fig == True:
        _, ax = plt.subplots(1,1,figsize = fig_size)
        ax.plot(np.log(history['loss']), label = 'Training Set')
        ax.plot(np.log(history['val_loss']), label = 'Validation Set')
        ax.axhline(np.log(relate_mse_5),xmax = n_epoch,  color = 'black', linestyle = '-.', label = '$q=0.05$')
        ax.axhline(np.log(relate_mse_1),xmax = n_epoch,  color = 'green', linestyle = '-.', label = '$q=0.01$')
        ax.set_ylabel('log(MSE)', fontsize = 'large')
        ax.set_xlabel('Epoch', fontsize = 'large')
        ax.legend()
        plt.show()

    else:
        fig, ax = plt.subplots(2,2,figsize = fig_size)
        ax[0,0].plot(history['loss'], label = 'Training Set') 
        if option_validation: 
            ax[0,0].plot(history['val_loss'], label = 'Validation Set')
        ax[0,0].axhline((relate_mse_5),xmax = n_epoch,  color = 'black', linestyle = '-.', 
                        label = '$q=0.05$')
        ax[0,0].axhline((relate_mse_1),xmax = n_epoch,  color = 'green', linestyle = '-.', 
                        label = '$q=0.01$')
        ax[0,0].set_ylabel('MSE')

        ax[0,1].plot(np.log(history['loss']))
        ax[0,1].axhline(np.log(relate_mse_5),xmax = n_epoch,  color = 'black', linestyle = '-.')
        ax[0,1].axhline(np.log(relate_mse_1),xmax = n_epoch,  color = 'green', linestyle = '-.')
        ax[0,1].set_ylabel('log(MSE)')

        ax[1,0].plot(history['mean_absolute_error']) 
        ax[1,0].axhline((relate_mae_5),xmax = n_epoch,  color = 'black', linestyle = '-.')
        ax[1,0].axhline((relate_mae_1),xmax = n_epoch,  color = 'green', linestyle = '-.')
        ax[1,0].set_ylabel('MAE')
        ax[1,0].set_xlabel('Epoch')

        ax[1,1].plot(np.log(history['mean_absolute_error']))#mean_absolute_error']))
        ax[1,1].axhline(np.log(relate_mae_5),xmax = n_epoch,  color = 'black', linestyle = '-.')
        ax[1,1].axhline(np.log(relate_mae_1),xmax = n_epoch,  color = 'green', linestyle = '-.')
        ax[1,1].set_ylabel('log(MAE)')
        ax[1,1].set_xlabel('Epoch')

        if option_validation:
            ax[0,1].plot(np.log(history['val_loss']))
            ax[1,0].plot(history['val_mean_absolute_error'])
            ax[1,1].plot(np.log(history['val_mean_absolute_error']))

        # Display Legend for Training (and optional Validation) Data
        ax[0,0].legend()

        fig.suptitle('Training Progress of Model '+ model_name)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    

def ensemble_plot(models, model_ens, data, y, contract_display = 0, fig_size = (10,6)):

    '''
    Visualize the fit of an ensemble for an individual contract, highlighting the variance of the underlying weak learners.

    Inputs:
    -------
        models: list of weak learners, alias base models
        model_ens: ensemble model
        data: input values
        y:  target values
        contract_display: integer value; index for which contract data[i] should be displayed
        fig_size: adapt the size of the resulting plot

    Outputs:
    --------
        None; plot is displayed
    '''
    
    _, ax = plt.subplots(1,1,figsize = fig_size)
    n_models = len(models)
    pred = []

    # plot 1st model seperately, in order to label all models only once
    pred.append(models[0].predict(x=data[contract_display:contract_display+1,:]).flatten())
    ax.plot(pred[0], 'black', linestyle = ':', label = 'Single Model(s)')
    for i in range(1,n_models):
        pred.append(models[i].predict(x=data[contract_display:contract_display+1,:]).flatten())
        ax.plot(pred[i], 'black', linestyle = ':')

    # Ensemble
    ax.plot(model_ens.predict(x=data[contract_display:contract_display+1,:]).flatten(),
            color = 'magenta', linestyle = '-', label = 'Ensemble')

    # Display Targets
    ax.plot(y[contract_display,:], '*r', label = 'Target')
    plt.legend()
    ax.set_xlabel('Time, t', fontsize = 'large')
    ax.set_ylabel('Policy Value', fontsize = 'large')
    plt.show()
    


def rnn_single_dim_config_plots(hist,  scale,ref5, ref1,dictionary_lambda= [False,False,True,True], 
                                measure = 'loss', show_val = True, fig_size = (10,6)):

    '''
    Compare multiple rnns w.r.t. their training history
    '''
    
    color_lst = ['blue', 'green', 'red', 'c', 'purple', 'brown', 'yellow']
    x_axis_len = len(hist[0].history['loss'])
    _, ax = plt.subplots(1,1,figsize= fig_size)
    for i in range(len(hist)):
        cache = 1
        if dictionary_lambda[i]==False:
            cache = scale**2
        ax.plot(range(1,x_axis_len+1),np.log(np.array(hist[i].history['loss'])*cache),'r', 
                label = 'Model {}'.format(i), color = color_lst[i%(len(color_lst)+1)])
        if show_val: ax.plot(range(1,x_axis_len+1),np.log(np.array(hist[i].history['val_loss'])*cache),'--r', 
                              color = color_lst[i%(len(color_lst)+1)]) #label = 'Model {} - Validation'.format(i),)
    
    ax.axhline(np.log(ref5),xmax = len(hist[0].history['loss']),  color = 'black', linestyle = '-.',label = '$q=0.05$')
    ax.axhline(np.log(ref1),xmax = len(hist[0].history['loss']),  color = 'grey', linestyle = '-.',label = '$q=0.01$')

    ax.legend()
    ax.set_ylabel('log(MSE)', fontsize = 'large')
    ax.set_xlabel('Epoch', fontsize = 'large')




def visualize_prediction_quality(model, x, y, position = 0, model_name = '', fig_size = (8,4), 
                                 additional_plot = False, add_y = None, normalize_add_y = True,
                                plot_on_ax = False, ax = None, fig = None):

    '''Vizualize the model's prediction in comparison to target values for a selected contract.'''
    
    if plot_on_ax == False:
        fig, ax = plt.subplots(1,1, figsize = fig_size)
    
    # Case I: Single Contract Prediction
    if type(position) == int:
        # Case 0: 2-dimensional Data
        if len(x.shape) == 2:
            pred = model.predict(x[position:position+1,:]).flatten()
        # Case 1: 3-dimensional Data
        elif len(x.shape) == 3:
            pred = model.predict(x[position:position+1,:,:]).flatten()
        else:
            print('Unknown Data Input')
            exit()
        # Plot Prediction
        ax.plot(pred, '-.', label = 'Prediction')
        # Plot Target
        ax.plot(y[position,:], '*r',label = 'Target')
    
    # Case II: Multiple Contract Prediction
    elif type(position) == list or type(position).__module__ ==np.__name__:
        #case = 'II'
        pred = []
        for i in position:
            # Case 0: 2-dimensional Data
            if len(x.shape) == 2:
                pred=model.predict(x[i:i+1,:]).flatten()
            # Case 1: 3-dimensional Data
            elif len(x.shape) == 3:
                pred = model.predict(x[i:i+1,:,:]).flatten()
            else:
                print('Unknown Data Input')
                exit()
            # Plot Predictions:
            if i==position[0]:
                ax.plot(pred, '-.', label = 'Prediction')
                ax.plot(y[i,:], '*',label = 'Target')
            else:
                ax.plot(pred, '-.')
                ax.plot(y[i,:], '*')
        
    else:
        print('Unknown Input Type position.')
    
    ax.set_xlabel('Time, t', fontsize = 'large')
    ax.set_ylabel('Value', fontsize = 'large')
    
    if additional_plot:
        if normalize_add_y:
            add_y = add_y[position,:]/add_y[position,:].max()
        ax.plot(add_y, ':g', label = 'Policy Value \n (scaled)')
        
    ax.legend(loc = 1)
        
    if fig != None:
        fig.suptitle('Visualization of Model '+ model_name + ' for selected, single contract.')



def plot_accuracy_cum(model_lst, x, y, fig_size = (10,6)):
    '''
    For a given model calculate prediction values.
    In the subsequent absolute error and display it relative to the target value.
    Visually, we present the relative error.
    '''
    
    
    pred = model_lst[0].predict(x)
    pred_cum = pred.sum(axis=0)
    y_cum = y.sum(axis=0)
    index_pos = y_cum > 0
    
    # save precisions in dataframe
    stat_columns = list(range(len(index_pos[index_pos==True])))
    df = pd.DataFrame(data = None, index = stat_columns, columns = None)
    
    _, ax = plt.subplots(1,1, figsize = fig_size)

    # Include 2nd x-axis for absolute policy value
    # plot first for better visibility
    ax2 = ax.twinx()
    ax2.set_ylabel('Cumulative Policy Value', color = 'grey', fontsize = 'large')
    ax2.bar(range(len(y_cum)),(y_cum), color = 'grey', alpha = .2)
    ax2.tick_params(axis='y')


    # plots limits for accuracy
    ax.plot(range(y.shape[1]), np.hstack(np.array([np.repeat(0.05, 15), 
                                                 np.repeat(0.1, 15),np.repeat(0.2, y.shape[1]-30)])), '--r')
    ax.plot(range(y.shape[1]), (-1)*np.hstack(np.array([np.repeat(0.05, 15),np.repeat(0.1, 15),
                                          np.repeat(0.2, y.shape[1]-30)])), '--r')
    #Plot models' accuracy
    acc = (pred_cum[index_pos]-y_cum[index_pos])/y_cum[index_pos]
    ax.plot(acc, label = 'Ensemble 0' ) #, color = 'green')
    df.loc[:,'Ensemble'] = list(acc)
    
    # optional: Plot other models
    for i in range(1, len(model_lst)):
        pred = model_lst[i].predict(x)
        pred_cum = pred.sum(axis=0)
        y_cum = y.sum(axis=0)
        index_pos = y_cum > 0
        acc = (pred_cum[index_pos]-y_cum[index_pos])/y_cum[index_pos]
        ax.plot(acc, label = 'Ensemble '+str(i)) #, color = 'green')
        df.loc[:,'Ensemble '+str(i)] = list(acc)
            
    
    ax.tick_params(axis='y')
    ax.set_ylabel('rce${}_t$', fontsize = 'large')
    ax.set_xlabel('Time, $t$', fontsize = 'large')
    ax.legend()

    plt.tight_layout()
    plt.show()
    
    
    return df



#### Comparison of Representative Contracts of K-Means and ANN Approaches
## Input: K-Means and ANN-Representatives

def visualize_representatives_km_ann(km_rep, ann_rep, features = ['age', 'Sum_ins', 'duration','age_of_contract'], plot_tag=''):
                                    #,option = 'standard'):
    
    n_features = len(features)
    n_cl = len(ann_rep)
    
    if type(ann_rep)== type({}):
        # transform dictionary to np.array
        ann_rep = np.array(list(ann_rep.values())).reshape(-1,n_features)
    

    #colors = ['green', 'blue', 'orange', 'black','red', 'grey', 'pink'] # Adapt if for features used
    markers = ['o','+','>', '<', 'x']
    # create a figure and axis
    _, ax = plt.subplots(figsize=set_size(fraction=.5))
    ax.plot([-1,1],[-1,1], linestyle = '--', linewidth=.5, color = 'grey')
    # plot each data-point
    for i in range(n_features):
        ax.scatter(ann_rep[:,i],km_rep[:,i], s = 20, marker=markers[i], label = features[i], color= 'black') #colors[i])

    # set a title and labels
    #ax.set_title('Comparison of Representatives', fontsize = 'large')
    ax.set_xlabel('NN model point', fontsize = 'large')
    ax.set_ylabel('KM centroid', fontsize = 'large')
    ax.set_xlim([-1.1,1.4])
    ax.set_xticks(np.arange(-1,1.5,0.5))
    ax.set_yticks(np.arange(-1,1.5,0.5))

    ax.legend(loc='center right')#, title='contract features')
    plt.savefig(os.getcwd()+r'/Matplotlib_figures/Grouping{}_termlife_mps_K{}_C1.eps'.format(plot_tag, n_cl), format = 'eps')
    plt.show()


# def plot_feature_structure(x, y,feature_name = 'Age', pos = 10):
#     #! depreciated
#     plt.plot(x, y[:,pos], 'o')
#     plt.ylabel('Policy Value (time fixed)', fontsize = 'large')
#     plt.xlabel(feature_name, fontsize = 'large')
#     plt.show()


## Aim here is the same as for plot_feature_structure
## This function presents the target value (at some fixed point in time) w.r.t. the feature component's value
## for all feature components

# def plot_all_features_structure(x_lst, y_lst, names_lst = ['Age', 'Sum Insured','Duration', 'Age of Contract'],
#                                 pos_lst = [10,10,1,0], fig_size = (12,8)):
    
#     #! depreciated
#     n_features = len(x_lst)
#     _, ax = plt.subplots(2,2,figsize= fig_size)
#     ax = ax.flatten()
#     for i in range(n_features):
#         ax[i].plot(x_lst[i], y_lst[i][:,pos_lst[i]], '.')
#         ax[i].set_xlabel(names_lst[i], fontsize = 'large')
#         if i in [0,2]:
#             ax[i].set_ylabel('Policy Value')
#         i+=1