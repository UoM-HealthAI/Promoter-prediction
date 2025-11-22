"""
Renewed on Oct 2025

Original: Jon (2020)
Edited by: Jingyu (2025)
"""

from numba import njit

@njit(cache=True)
def get_adjusted(state, K, W, ms2):
    """
    计算给定复合状态 state 在窗口 W 上：
      ones = ∑_{p=0..W-1} [bit_p==1] * ms2[p]
      zeros= ∑_{p=0..W-1} [bit_p==0] * ms2[p]
    约定：ms2 为 1D，LSB(p=0) 对应 ms2[0]。
    返回 (ones, zeros) 的 tuple（nopython 友好）。
    """
    W_eff = W if W <= ms2.shape[0] else ms2.shape[0]
    s = int(state)
    ones = 0.0
    zeros = 0.0
    for p in range(W_eff):
        w = ms2[p]
        if (s & 1) == 1:
            ones += w
        else:
            zeros += w
        s = s >> 1
    return (ones, zeros)
