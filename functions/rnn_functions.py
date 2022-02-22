import numpy as np
import time
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM, RepeatVector, Activation, Lambda, Average
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def create_rnn_model(model_input,widths_rnn =[41,41], widths_ffn = [41],
                     dense_act_fct = 'linear', act_fct_special = False, 
                     option_recurrent_dropout = True, 
                     n_repeat = 41, option_dyn_scaling = False,
                     optimizer_type='adam', loss_type='mse', metric_type='mae',
                     dropout_rnn=0.2, lambda_layer = True, lambda_scale =1, log_scale=False, 
                     model_compile = True, return_option = 'model', branch_name = ''):#, input_type = '2D'): 

    
    n_rnn = len(widths_rnn)
    x = RepeatVector(n = n_repeat)(model_input)

    
    for i in range(n_rnn-1):
        x = LSTM(units = widths_rnn[i], activation='tanh', recurrent_activation='sigmoid', 
                    dropout=dropout_rnn, recurrent_dropout=dropout_rnn*option_recurrent_dropout, 
                     return_sequences=True, 
                     return_state=False, name = 'RNN_{}{}'.format(branch_name,i+1))(x)
        
        # Optional: Include dynamic scaling factor
        if option_dyn_scaling & (i==0):
            y = Dense(1, name = 'Dynamic_Scaling')(x)
            
    # Insert Last LSTM Layer manually with return_sequences = False (to match input format for Dense Layer)
    x = LSTM(units=widths_rnn[n_rnn-1], activation='tanh', recurrent_activation='sigmoid', 
                 dropout=dropout_rnn, recurrent_dropout=dropout_rnn*option_recurrent_dropout, 
                 return_sequences=False, 
                 return_state=False, name = 'RNN_{}{}'.format(branch_name,n_rnn))(x)
    
    # Optional: Include dynamic scaling factor
    if option_dyn_scaling & (n_rnn == 1):
            y = Dense(1, name = 'Dynamic_Scaling')(x)
    
    
    # Now address Feed-Forward Part of Network
    for k in range(len(widths_ffn)):        
        # Final Dense Layer
        if act_fct_special:
            x = Dense(widths_ffn[k], name = 'Dense_{}'.format(branch_name)+str(k+1))(x)
            x = dense_act_fct(x)
        else:
            x = Dense(widths_ffn[k], name = 'Dense_{}'.format(branch_name)+str(k+1))(x)
            if dense_act_fct != 'linear':
                x = Activation(activation = dense_act_fct, name = dense_act_fct + branch_name)(x)
                
    # Deterministic scaling Layer
    if lambda_layer:
        if log_scale:
            if option_dyn_scaling:
                x = Lambda(lambda x_var: tf.exp((x_var[0]+1)/2*np.log(1+x_var[1]))-1, 
                           name = 'Log_Scaling_Layer{}'.format(branch_name))([x,y])
            else:
                x = Lambda(lambda x_var: tf.exp((x_var+1)/2*np.log(1+lambda_scale))-1, 
                           name = 'Log_Scaling_Layer{}'.format(branch_name))(x)
        else:
            x = Lambda(lambda x_var: (x_var+1)/2*lambda_scale, name = 'Scaling_Layer{}'.format(branch_name))(x)    
    
    if return_option == 'model':
        # Model Configuration
        model = Model(inputs=model_input, outputs=x)
        # Compile model
        if model_compile: 
            model.compile(loss = loss_type, optimizer = optimizer_type, metrics = [metric_type] )
        return model
    else:
        return x





def create_multiple_rnn_models(number, model_input,widths_rnn =[41,41],  n_output=41, widths_ffn = [41], 
                               dense_act_fct = 'linear', optimizer_type='adam', loss_type='mse', 
                               metric_type='mae', dropout_share=0, 
                               lambda_layer = True, lambda_scale =1, log_scale=False, model_compile = True, 
                               return_option = 'model', **args):
    
    '''
    Create multiple models of the same type
    Relevant for creating an ensemble model
    '''

    # Note: option_CUDA=False; for tf.keras CudNNLSTM not implemented
    models = []
    for i in range(number):
        
        models.append(create_rnn_model(model_input = model_input, widths_rnn = widths_rnn, 
                                       widths_ffn = widths_ffn, dense_act_fct = dense_act_fct, 
                                       option_recurrent_dropout = False, 
                                       option_dyn_scaling = False,
                                       optimizer_type=optimizer_type, loss_type=loss_type, 
                                       metric_type=metric_type,
                                       dropout_rnn =dropout_share, lambda_layer = lambda_layer, 
                                       lambda_scale =lambda_scale, log_scale=log_scale, 
                                       model_compile = model_compile, return_option = return_option, 
                                       branch_name = str(i)))
        
    # depending on the return_option used in create_rnn_model, models can be a Tensor output or a compiled model
    return models





def multiple_models_get_weights(models_lst):

    '''
    Input: list of models
    Output: list of their weight configurations
    Purpose: Use list of weigts to transfer them to a ensemble model
    '''
    
    w = []
    for model in models_lst:
        w.append(model.get_weights())
    return w



# 

def train_individual_ensembles( models_lst, x_train, y_train, n_batch = 100, n_epochs = 20, val_share = 0.25, 
                               es_patience = 15, 
                               path = None):
    
    '''Train all models in a list of models'''

    n_ensembles = len(models_lst)   
    hist = {}
    
    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=es_patience, restore_best_weights=True)
    


    for i in range(n_ensembles):
        print('Training Model {} of '.format(i+1)+str(n_ensembles))
        t = time.time()

        # save only best configuration
        hist[i] = models_lst[i].fit(x_train, y_train,batch_size = n_batch, epochs = n_epochs,
                                      validation_split = val_share, verbose=0, callbacks= [es]).history#, mc] ).history

        # Save model
        models_lst.save_weights(path+r'/model_{}.h5'.format(i))

        # Save history
        with open(path+r'/model_{}_hist.json'.format(i), 'wb') as f: # alternative: option #'w'
            pickle.dump(hist[i], f, pickle.HIGHEST_PROTOCOL)
        
        #model_collection.append(model)
        print('END of Model {}'.format(i+1)+'. Time passed: ' + str(int((time.time()-t)*100)/100) + ' sec.')
    return models_lst, hist





 
# def combine_models(input_layer, n_ensembles =1, load_weights = False, weights_ensembles = None, 
#                    scale = 1, LSTM_nodes = [41,41], FFN_nodes = [41], dense_act_fct = 'linear', 
#                   dropout_share = 0, return_option = 'model',
#                   optimizer_type = 'adam', loss_type = 'mse', metric_type = 'mae', index = ''):

#     '''
#     #! Depreciated: combine_models() suspended by more effiecient implementation using tf.keras.model.Model()
#     Create multiple models, load weights and return the ensemble
#     '''
    
    
#     output_ensemble = []
#     for i in range(n_ensembles):
#         # Note: option_CUDA=False; for tf.keras CudNNLSTM not implemented
#         model_ens = create_rnn_model(model_input = input_layer, widths_rnn = LSTM_nodes, 
#                                    widths_ffn = FFN_nodes, dense_act_fct = dense_act_fct, 
#                                    option_recurrent_dropout = False, 
#                                    option_dyn_scaling = False,
#                                    optimizer_type=optimizer_type, loss_type=loss_type, 
#                                    metric_type=metric_type,
#                                    dropout_rnn =dropout_share, lambda_layer = True, 
#                                    lambda_scale =scale, log_scale=True, 
#                                    model_compile = True, return_option = return_option, 
#                                    branch_name = index+str(i))
#         if return_option == 'model':
#         # Load weights; Not reasonable for non-model returns
#             if load_weights:
#                 model_ens.set_weights(weights_ensembles[i])
#             # Save the single-models' outputs in a list -> will be used as input to Average-Layer
#             output_ensemble.append(model_ens.outputs[0])
#         else:
#             # For return_option 'output' ANN-objective .outputs does not exist
#             output_ensemble.append(model_ens)
    
#     # Combine Ensembles by Averaging them
#     output_avg = Average(name = 'Ensembles_Combine'+index)(output_ensemble)
    
#     if return_option == 'model':
#         model = Model(inputs = [input_layer], outputs = output_avg)
#         model.compile(optimizer='adam',loss='mse',metrics= ['mae'])
        
#         return model
#     else:
#         return output_avg