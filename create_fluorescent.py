# -*- coding: utf-8 -*-
"""
Created on Tue July 1 17:03:57 2025
generate the traces
@author: Jon & Hongpeng
"""
import numpy as np
from scipy.io import savemat
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from burstInfer.get_adjusted import get_adjusted
from burstInfer.ms2_loading_coeff import ms2_loading_coeff
import os

seed_setter = np.random.randint(0,1000000000) # Make sure this is random for actual run
np.random.seed(seed_setter)

cwd = os.getcwd()
parent = os.path.dirname(cwd)
data_folder_head = parent  + '/' + 'data/'


#%% generate the promotor state
def generate_promotor(transition_probabilities, number_of_traces, lengh_of_each_trace, starting_value):
    chain_matrix = np.ones((number_of_traces,lengh_of_each_trace))

    for j in range(number_of_traces):
        # starting_value = 1
        chain_length = lengh_of_each_trace
        chain = np.zeros((chain_length))
        chain[0]=starting_value.squeeze()[j]
        for i in range(1, chain_length):
            this_step_distribution = transition_probabilities[int(chain[i - 1])]
            cumulative_distribution = np.cumsum(this_step_distribution)
            r = np.random.rand()
            chain[i] = np.where(cumulative_distribution>r)[0][0]
        chain_matrix[j,:] = chain
    return chain_matrix

#hyper-parameters
onoff_dynamic_transition = False # this is the methods used in Magnus's paper
# onoff_dynamic_transition = True   #this is to generate the sequence which include different transition parameters, which we call it dynamic transition
number_of_traces = 500
lengh_of_each_trace = 200

#parameters for dynamic positions 
number_of_segement_for_dynamic_traces = 20
segment_length= 10

#define transition matrix
k_off_off = 0.8810
k_on_on = 0.8567



if not onoff_dynamic_transition:
    k_on_array = k_on_on
    transition_probabilities = np.zeros((2,2)) #[OFF-OFF, OFF-ON; ON-OFF, ON-ON]
    transition_probabilities[0,0] = k_off_off
    transition_probabilities[0,1] = 1 - k_off_off
    transition_probabilities[1,1] = k_on_on
    transition_probabilities[1,0] = 1 - k_on_on
    starting_value = np.ones(100)
    chain_matrix = generate_promotor(transition_probabilities, number_of_traces, lengh_of_each_trace, starting_value)
    promotor_csv_filename = data_folder_head + "synthetic_data_promotor" + '.csv'
else:
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    xdata = np.arange(start=-2, stop=2, step = 4/number_of_segement_for_dynamic_traces, dtype='float')
    k_on_array = sigmoid(xdata)
    chain_matrix = np.zeros((number_of_traces, number_of_segement_for_dynamic_traces*segment_length))
    segment_save_index_end = segment_length
    segment_save_index_start = 0
    starting_value = np.ones(number_of_traces) #random, 0 1 should be fine


    for i in range(k_on_array.shape[0]):
        cur_kon = k_on_array[i]
        cur_transition_probabilities = np.zeros((2,2))
        cur_transition_probabilities[0, 0] = k_off_off
        cur_transition_probabilities[0, 1] = 1 - k_off_off
        cur_transition_probabilities[1, 1] = cur_kon
        cur_transition_probabilities[1, 0] = 1 - cur_kon
        cur_chain_matrix = generate_promotor(cur_transition_probabilities, number_of_traces, segment_length + 1, starting_value)
        starting_value = cur_chain_matrix[:,-1]
        if i == 0:
            chain_matrix[:,segment_save_index_start:segment_save_index_end] = cur_chain_matrix[:,0:-1]#.squeeze()#.reshape(chain_matrix[:,segment_save_index_start:segment_save_index_end].size())
        else:
            chain_matrix[:, segment_save_index_start:segment_save_index_end] = cur_chain_matrix[:, 1:]
        segment_save_index_end += segment_length
        segment_save_index_start += segment_length
    promotor_csv_filename = data_folder_head + "synthetic_make_data_promotor_dynamic_transition_segement_" + str(number_of_segement_for_dynamic_traces) + '_length_' + str(segment_length) + '.csv'

#saving promoter states
sampling_dataframe = pd.DataFrame(chain_matrix)
sampling_dataframe.to_csv(promotor_csv_filename)


#%% generate the fluorescence traces

#read parameters from saved synthetic promotor states
ms2_signals = genfromtxt(promotor_csv_filename, delimiter=',', skip_header=1)
signal_holder = ms2_signals[:, 1:]
n_traces = len(signal_holder)
length_of_each_trace = len(signal_holder[0])
synthetic_x = np.arange(0,length_of_each_trace)

# Initialisation key parameters
K = 2 # the number of promotor states: ON/OFF
W = 2 # window size

#define parameters for emission model, which is Gaussian model including mean and variance
mu = np.zeros((K,1))
mu[0,0] = 7096.5295359189 #mean value for the OFF state
mu[1,0] = 48700.3752 #mean value for the ON state
noise = 17364.9315703868

t_MS2 = 30
deltaT = 20
kappa = t_MS2 / deltaT    # the definition of kappa is in Table 1 in https://github.com/GarciaLab/cpHMM/blob/master/cpHMM_documentation.pdf


#%%
# MS2 coefficient calculation
ms2_coeff = ms2_loading_coeff(kappa, W)
# ms2_coeff_flipped = np.flip(ms2_coeff, 1)
count_reduction_manual = np.zeros((1,W-1)) 
for t in np.arange(0,W-1):
    count_reduction_manual[0,t] = np.sum(ms2_coeff[0,t+1:])
count_reduction_manual = np.reshape(count_reduction_manual, (W-1,1))


mask = np.int32((2**W)-1)
fluorescence_holder = np.zeros((n_traces,length_of_each_trace))
get_adjusted_trace_record = []

for i in np.arange(0, len(fluorescence_holder)):
    tmp_get_adjusted_trace_record = []
    single_promoter = np.expand_dims(signal_holder[i,:], axis = 0)
    single_trace = np.zeros((1,length_of_each_trace))
    t = 0
    window_storage = int(single_promoter[0,0])
    single_trace[0,t] = ((get_adjusted(window_storage, K, W, ms2_coeff)[0] * mu[1,0]) + (get_adjusted(window_storage, K, W, ms2_coeff)[1] * mu[0,0])) + np.random.normal(0, noise)  #gaussian noise
    tmp_get_adjusted_trace_record.append(get_adjusted(window_storage, K, W, ms2_coeff))

    window_storage = 0
    t = 1
    present_state_list = []
    present_state_list.append(int(single_promoter[0,0]))
    while t < length_of_each_trace:
        present_state = int(single_promoter[0,t])
        #present_state_list.append(present_state)
        window_storage = np.bitwise_and((present_state_list[t-1] << 1) + present_state, mask)#这里是为了限制窗口的位置。前面的(present_state_list[t-1] << 1) + present_state代表的是不加限制的整个序列；而这里我们规定了只有window内部的算数，因此要和31做
        #逻辑运算
        present_state_list.append(window_storage)
        single_trace[0,t] = ((get_adjusted(window_storage, K, W, ms2_coeff)[0] * mu[1,0]) + (get_adjusted(window_storage, K, W, ms2_coeff)[1] * mu[0,0])) + np.random.normal(0, noise)
        t = t + 1
        tmp_get_adjusted_trace_record.append(get_adjusted(window_storage, K, W, ms2_coeff)) 
        
    fluorescence_holder[i,:] = single_trace
    get_adjusted_trace_record.append(tmp_get_adjusted_trace_record)

sampling_dataframe = pd.DataFrame(fluorescence_holder)
if onoff_dynamic_transition:
    fluorescence_csv_name = data_folder_head + "fluorescent_traces_synthetic_data_W_" + str(W) + "_trace_"+ str(n_traces) + "_noise_" + str(noise) + "_dynamic_segement_" + str(number_of_segement_for_dynamic_traces) + '_length_' + str(segment_length) + '.csv'
    fluorescence_mat_name = data_folder_head + "fluorescent_traces_synthetic_data_W_" + str(W) + "_trace_"+ str(n_traces) + "_noise_" + str(noise) + "_dynamic_segement_" + str(number_of_segement_for_dynamic_traces) + '_length_' + str(segment_length) + '.mat'
else:
    fluorescence_csv_name = data_folder_head + "code_check_fluorescent_traces_synthetic_data_W_" + str(
        W) + "_trace_" + str(
        n_traces) + "_noise_" + str(noise) + '.csv'
    fluorescence_mat_name = data_folder_head + "code_check_fluorescent_traces_synthetic_data_W_" + str(
        W) + "_trace_" + str(
        n_traces) + "_noise_" + str(noise) + '.mat'

sampling_dataframe.to_csv(fluorescence_csv_name)
savemat(fluorescence_mat_name,{
            'Input': fluorescence_holder,
            'Output':signal_holder,
            'deltaT':deltaT,
            'k_on_array':k_on_array,
            'k_off_off':k_off_off,
            'W':W,
            't_MS2':t_MS2,
            'noise':noise,
            'mean_on':mu[1,0],
            'mean_off':mu[0,0],
            'number_of_segement_for_dynamic_traces':2,
            'segment_length' : 40
        }
        )
print("finish code")

#%% plot some traces for visualisation
# for j in np.arange(3,15):
#     plt.figure(j)
#     plt.plot(synthetic_x, fluorescence_holder[j,:].flatten())
    
# =============================================================================
plt.figure(15)
plt.step(synthetic_x, signal_holder[40,:])
plt.figure(16)
plt.plot(synthetic_x, fluorescence_holder[40,:].flatten())

plt.figure(17)
plt.step(synthetic_x, signal_holder[42,:])
plt.figure(18)
plt.plot(synthetic_x, fluorescence_holder[42,:].flatten())

plt.show()
# =============================================================================


