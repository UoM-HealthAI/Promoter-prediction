# -*- coding: utf-8 -*-
"""
Created on Oct 2025

@author: Jingyu
"""
import numpy as np

def log_sum_exp(arr, signs):
    arr_max = np.max(arr[:,:])
    term2_array = np.multiply(signs, np.exp(arr-arr_max))
    term2 = np.sum(term2_array)
    logsum = np.array([arr_max + np.log(np.abs(term2)), np.sign(term2)])
    return logsum

from scipy.special import logsumexp

def log_sum_exp_scipy(arr, signs):
    log_abs, sgn = logsumexp(arr, b=signs, return_sign=True)
    return np.array([log_abs, sgn])
