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
def compute_dynamic_F(state, length, W, K, ms2, count_reduction_manual):
    """
    返回 (log_f1_terms, log_f0_terms)，每个形状 (1, L)。
    log_f1_terms[0,t] = log(∑_{p<=min(t,W-1)} [bit_p==1]*ms2[p])
    log_f0_terms[0,t] = log(∑_{p<=min(t,W-1)} [bit_p==0]*ms2[p])
    仅支持 K=2（OFF/ON）。
    约定：ms2 为 1D，LSB 对应 ms2[0]。
    """
    L = int(length)
    if K != 2:
        out1 = np.full((1, L), -np.inf)
        out0 = np.full((1, L), -np.inf)
        return (out1, out0)

    W_eff = W if W <= ms2.shape[0] else ms2.shape[0]

    # 展开 W 位
    bits = np.empty(W_eff, dtype=np.int64)
    s = int(state)
    for p in range(W_eff):
        bits[p] = (s >> p) & 1

    # 前缀和
    ps = np.empty(W_eff)
    ones_ps = np.empty(W_eff)
    acc = 0.0
    acc1 = 0.0
    for p in range(W_eff):
        w = ms2[p]
        acc += w
        if bits[p] == 1:
            acc1 += w
        ps[p] = acc
        ones_ps[p] = acc1

    ones_full = ones_ps[W_eff - 1]
    zeros_full = ps[W_eff - 1] - ones_full

    log_f1 = np.empty((1, L))
    log_f0 = np.empty((1, L))
    for t in range(L):
        eff = t if t < W_eff else (W_eff - 1)
        if t < W_eff:
            ones = ones_ps[eff]
            zeros = ps[eff] - ones_ps[eff]
        else:
            ones = ones_full
            zeros = zeros_full
        if ones <= 0.0:
            ones = _TINY
        if zeros <= 0.0:
            zeros = _TINY
        log_f1[0, t] = np.log(ones)
        log_f0[0, t] = np.log(zeros)
    return (log_f1, log_f0)
