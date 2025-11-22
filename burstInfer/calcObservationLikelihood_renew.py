# -*- coding: utf-8 -*-
"""
Renewed on Oct 2025

Original: Jon (2020)
Edited by: Jingyu (2025)
"""

from numba import njit
import numpy as np

_TINY = np.finfo(np.float64).tiny

@njit(cache=True)
def calcObservationLikelihood(lambda_logF, noise_tempF, dataF, veef,
                              INPUT_STATE, K, W, ms2):
    """
    高斯观测 log-likelihood（方案 A：仍使用全窗口 W 的核）。
    约定：ms2 为 1D，LSB(p=0) 对应 ms2[0]；K=2（OFF/ON）。
    不再依赖 get_adjusted，避免 1D/2D 索引问题。
    """
    # 噪声下限，避免除零
    sigma = noise_tempF
    if sigma <= 0.0:
        sigma = 1e-8

    # 有效窗口
    W_eff = W if W <= ms2.shape[0] else ms2.shape[0]

    # 按位累计 ones/zeros 的核权重
    s = int(INPUT_STATE)
    ones = 0.0
    zeros = 0.0
    for p in range(W_eff):
        w = ms2[p]
        if (s & 1) == 1:
            ones += w
        else:
            zeros += w
        s >>= 1

    # 线性均值：zeros*veef[0,0] + ones*veef[1,0]
    mu = zeros * veef[0, 0] + ones * veef[1, 0]
    resid = dataF - mu

    # 高斯 log-likelihood
    return 0.5 * (lambda_logF - np.log(2.0 * np.pi)) - 0.5 * (resid * resid) / (sigma * sigma + _TINY)
