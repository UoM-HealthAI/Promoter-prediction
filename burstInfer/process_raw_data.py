# -*- coding: utf-8 -*-
"""
Created on Wed Jan 8 11:22:25 2020

@author: Jon
"""
import numpy as np

def process_raw_data(signals, cutoff):
    # Build list of fluorescence signals
    number_of_signals = len(signals)
    signal_struct = []
    mean_matrix = np.zeros((number_of_signals, 1))
    max_matrix = np.zeros((number_of_signals, 1))
    length_container = []
    for u in np.arange(0, number_of_signals):
        requested_signal = signals[u, cutoff:]
        requested_signal2 = requested_signal[~np.isnan(requested_signal)]
        mean_matrix[u,] = np.mean(requested_signal2, axis=0)
        max_matrix[u,] = np.max(requested_signal2, axis=0)
        signal_struct.append(np.reshape(requested_signal2, (1, len(requested_signal2))))
        length_container.append(len(requested_signal2))

    matrix_mean = np.mean(mean_matrix)
    matrix_max = np.max(max_matrix)
    unique_lengths = np.unique(length_container)

    output_dict = {}
    output_dict['Processed Signals'] = signal_struct
    print("The shape of the Processed Signals is: ", np.shape(output_dict['Processed Signals']))
    output_dict['Matrix Mean'] = matrix_mean
    print("The shape of the Matrix Mean is: ", np.shape(output_dict['Matrix Mean']))
    output_dict['Matrix Max'] = matrix_max
    print("The shape of the Matrix Max is: ", np.shape(output_dict['Matrix Max']))
    output_dict['Signal Lengths'] = unique_lengths
    print("The shape of the Signal Lengths is: ", np.shape(output_dict['Signal Lengths']))

    return output_dict
