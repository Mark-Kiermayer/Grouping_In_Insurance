import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import describe


def evaluate_split_congruence(x_train, x_test):

    '''Evaluate if the split (train-test) has common data, respectively determine share'''
    
    n_test = x_test.shape[0]
    n_match = 0
    train_lst = x_train.tolist()
    test_lst = x_test.tolist()
    for i in range(n_test):
        if test_lst[i] in train_lst:
            n_match+=1
    
    return n_match/n_test


def relate_loss(data, discrepancy, measure = 'mse'):

    '''
    Get references for MSE or AE Values
    Determine baseline MSE or MAE for a constant relative discrepancy (given %) per time point.
    '''
    val = 0
    if measure == 'mse':
        val = (np.square(discrepancy*data.flatten())).mean()
    elif measure ==  'mae':
         val = (np.abs(discrepancy*data.flatten())).mean()  
    return val



def calc_row_df(prediction, target, measure_type = 'absolute_error',lambda_layer = True, dropout_layer = True, 
                option_relative = False,row_name = None):

    '''
    Given a prediction, a target and information about the underlying model's configuration, 
    i.e. Usage of Dropout or type of Lambda-Scaling Layer
    
    Output:
    -------
    The function returns a list of properties for SE or AE.
    This list will eventually be added to a table, to compare models with different configurations
    '''
    
    metric = 0
    if measure_type == 'absolute_error':
        metric = (np.abs(prediction-target)).flatten()
    elif measure_type == 'squared_error':
        metric = (np.square(prediction-target)).flatten()
    else:
        print('Measure_type unknown')
        return
    
    statistic = describe(metric)
    if lambda_layer ==True:
        lambda_layer = 'yes'
    else:
        lambda_layer = 'no'
    if dropout_layer ==True:
        dropout_layer = 'yes'
    else:
        dropout_layer = 'no'
    
    if row_name == None:
        return dropout_layer, lambda_layer, statistic[1][0], statistic[1][1], statistic[2], statistic[3]
    else:
        return dropout_layer, lambda_layer,[statistic[1][0], statistic[1][1], statistic[2], statistic[3]], row_name





def create_df_model_comparison(model_single_lst,x_test, y_test, model_ens_lst = [None], 
                               threshold = 0, wre_measure_option = True,
                               discount_option = False, names_number = None, names_loss = None,
                               names_loss_single = None,
                               discount_val = 1, version = 'new'):


    '''
    For Several Single-, Ensemble- and Ensemble incl. Qual. - Models perform descriptive analysis
    For Non-Zero Target Values (which are optionally above a threshold): Look at Absolute Error per timepoint relative to the target value.
    For Zero-Target Values (or optionally target values below threshold): Look at Absolute Error per time point
    
    Optionally: Also include a Weighted Relative (Absolute) Error (WRAE) where WRAE = target/sum(targets at given time) * error for contract at given time
    '''
    
    #Error catching, if no names of loss functions or number of models in ensemble provided
    if names_number == None:
        names_number = ['']*len(model_ens_lst)
        names_loss = ['']*len(model_ens_lst)
        
    if names_loss_single == None:
        names_loss_single = ['']*len(model_single_lst)
    
    # initialize variables to use for later storage purposes
    n_lst = len(model_single_lst)
    pred = []
    pred_ens = []
    diff = []
    diff_ens = []
    wre = []
    wre_ens = []
    

    # Determine times where at least one contract is still active, 
    # i.e. time where we can calculate the weighted error w.r.t. the sum of reserves at that point in time
    index_pv_cum = (y_test.sum(axis=0)>0)
    y_test_cum = y_test.sum(axis=0)[index_pv_cum]
    
    df = pd.DataFrame(data=None, index = None, columns = [r'\ell(\cdot,\cdot)',r'$N_{bag}$',
                                                          r'$\overline{e}$',
                                                          r'$pc_{0.99, |e_{t,x}|}$',
                                                          r'$\overline{wre}$',
                                                          r'$pc_{0.99, |wre_{t,x}|}$'] )
        
        
    # For all standard (individual) Models do as follows:
    for i in range(n_lst):
        # Calculate Predictions
        pred.append(model_single_lst[i].predict(x_test))
        # Calculate Errors for Zero-Target Time Points and Non-Zero Target Time Points seperately
        diff.append(np.abs(pred[i]-y_test))#.flatten()
        wre.append(diff[i][:,index_pv_cum]/y_test_cum)
        if discount_option:
            # discount each year j by (discount factor)^j # No discounting included, discount_val = 1
            wre[i] = wre[i]/discount_val**np.linspace(0,index_pv_cum.sum()-1, index_pv_cum)

        # add statistics to table
        df.loc['Model {}'.format(i)] = (names_loss_single[i], '1',
                                       (diff[i].flatten()).mean(), 
                                        np.percentile(np.abs((diff[i].flatten())), 99),
                                        wre[i].flatten().mean(), 
                                        np.percentile(np.abs((wre[i].flatten())), 99))

    # Option Model Ensemble: 
    if model_ens_lst[0] != None:
        for i in range(len(model_ens_lst)):
            pred_ens.append(model_ens_lst[i].predict(x_test))
            diff_ens.append(np.abs(pred_ens[i]-y_test))#.flatten()
            wre_ens.append(diff_ens[i][:,index_pv_cum]/y_test_cum)
            if discount_option:
                # discount each year j by (discount factor)^j
                wre_ens[i] = wre_ens[i]/discount_val**np.linspace(0,index_pv_cum.sum()-1, index_pv_cum)

            # Write statistics in table
            df.loc['Ensemble {}'.format(i)] = (names_loss[i], names_number[i],
                                               diff_ens[i].flatten().mean(),
                                               np.percentile(np.abs((diff_ens[i].flatten())), 99),
                                               wre_ens[i].flatten().mean(), 
                                               np.percentile(np.abs((wre_ens[i].flatten())), 99))


    return df, diff_ens, wre_ens

def model_examine_indivual_fit(data, targets, model=None, output_option = 'plot', PV_max = 1,
                               interval_lst = [0,0.005, 0.01,0.2,0.4,0.6,0.8,1]):

    '''
    Compute relative errors of a given model for different percentiles of the target values.
    We split contracts w.r.t. their maximum (true) target value and assign them into baskets. 
    Then, we compute avg. relative (absolute) errors and the 99th-percentile for each basket, but only considering times when the target is positive, i.e. not zero.

    Inputs:
    -------
        model: model to be evaluated
        data:   explanatory variables
        targets: target values
        output_option: Either 'statistic' (returns table) or 'plot' (return scatter plot of values)

    Outputs:
    --------
        table or scatterplot of values
    '''
    n_contracts = len(targets)
    # Max Target Reserve per contract
    targets_max = targets.max(axis=1)#implicitely assuming every contract has at least one year with target >0     
    if sum(targets_max == 0) >1:
        print('ValError: Data contains matured contract(s)!')
        print('Computation aborted.')
        exit()

    # calculate average precision per contract, for times with target>0
    index = targets>0    
    if model != None:
        precision_avg = np.zeros(shape=(n_contracts,))
        prediction = model.predict(x=data)
        for i in range(n_contracts):
            precision_avg[i] = (np.abs(prediction[i,index[i,:]]-targets[i,index[i,:]])/targets[i,index[i,:]]).mean()
        

    if (output_option =='statistic') | (output_option == 'both'):
        targets_max_overall = PV_max #targets_max.max()
        n_stat = len(interval_lst)
        stat_columns = [None]*(n_stat-1)
        for i in range(1,n_stat):
            stat_columns[i-1] = '{}-{}'.format(interval_lst[i-1], interval_lst[i])
        
        df = pd.DataFrame(data=None, index = None, columns = stat_columns )
        
        # Calculate average of average precisions per intervals (of contracts' max reserves)
        row_avg = [None]*(n_stat-1)
        row_stat = [None]*(n_stat-1)
        count = [None]*(n_stat-1)
        for i in range(n_stat-1):
            index_interval = (targets_max >= targets_max_overall*interval_lst[i])&(targets_max < targets_max_overall*interval_lst[i+1])            
            count[i] = int(sum(index_interval))
            if model != None:
                if count[i]>0:
                    row_avg[i] = precision_avg[index_interval].mean()
                    row_stat[i] = np.percentile(np.abs(precision_avg[index_interval]), 99)
                else:
                    row_avg[i]= 'n.a.'
                    row_stat[i] = 'n.a.'
        
        # Add Average Precision for all intervall to dataframe
        df.loc['max. PV'] = [np.round(interval_lst[i]*PV_max, 2) for i in range(1,n_stat)]
        df.loc['# data'] =  count
        if model != None:
            df.loc[r'$\overline{re}$'] = row_avg
            df.loc[r'$pc_{0.99, |re_{t,x}|}$'] = row_stat # Note: Prediction - Target <0 -> Underestimation
        
        if output_option != 'both':
            return df
        
        
    if (output_option =='plot') | (output_option == 'both'):
        plt.scatter(targets_max,precision_avg )
        plt.xlabel('Max. Reserve of Contract', fontsize = 'large')
        plt.ylabel('Average relative Error of Contract', fontsize = 'large')
        plt.show()
        
        if output_option == 'plot':
            return
        else:
            return df
        
    else:
        raise ValueError('output_option unknown!')


def relate_ens_to_q(x, y , x_plain,EA_lst = [None], EAQ_lst = [None], EP_lst = [None] ):

    '''
    Relate ensemble models to a q-value (average percentage discrepancy related to MSE or MAE)
    '''
    
    stat_columns = ['']*(2+len(EA_lst)+len(EAQ_lst)+len(EP_lst))
    row_mse  = ['']*(2+len(EA_lst)+len(EAQ_lst)+len(EP_lst))
    row_mae = ['']*(2+len(EA_lst)+len(EAQ_lst)+len(EP_lst))
    for i in range(len(stat_columns)):
        if i<2:
            if i==0:
                stat_columns[i] = 'q=0.05'
            elif i==1:
                stat_columns[i] = 'q=0.01'
            else:
                print('Error')
                return
        elif i<len(EA_lst)+2:
            stat_columns[i] = 'EA {}'.format(i-2)
        elif i< len(EA_lst)+len(EAQ_lst)+2:
            stat_columns[i] = 'EAQ {}'.format(i-len(EA_lst)-2)
        else:
            stat_columns[i] = 'EP {}'.format(i-len(EA_lst)-len(EAQ_lst)-2)
        
    df = pd.DataFrame(data=None, index = None, columns = stat_columns )
    
    row_mse[0], row_mse[1] =np.log(relate_loss(y,0.05, 'mse')), np.log(relate_loss(y,0.01, 'mse'))
    row_mae[0], row_mae[1] =np.log(relate_loss(y,0.05, 'mae')), np.log(relate_loss(y,0.01, 'mae'))
    for i in range(len(stat_columns)):
        if i<2:
            print('') #do nothing
        elif i<len(EA_lst)+2:
            row_mse[i] = np.log(((EA_lst[i-2].predict(x)-y)**2).mean())
            row_mae[i] = np.log(np.abs(EA_lst[i-2].predict(x)-y).mean())
        elif i< len(EA_lst)+len(EAQ_lst)+2:
            row_mse[i] = np.log(((EAQ_lst[i-len(EA_lst)-2].predict(x)-y)**2).mean())
            row_mae[i] = np.log(np.abs(EAQ_lst[i-len(EA_lst)-2].predict(x)-y).mean())
        else:
            row_mse[i] = np.log(((EP_lst[i-len(EA_lst)-len(EAQ_lst)-2].predict(x_plain)-y)**2).mean())
            row_mae[i] = np.log(np.abs(EP_lst[i-len(EA_lst)-len(EAQ_lst)-2].predict(x_plain)-y).mean())
            
    df.loc['log(MSE)'] = row_mse
    df.loc['log(MAE)'] = row_mae
    
    return df
