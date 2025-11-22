# -*- coding: utf-8 -*-
"""
Created on Oct 2025

@author: Jingyu
"""
import numpy as np
import itertools
from scipy.special import logsumexp

def v_log_solve(m_lg, m_sgn, b_lg, b_sgn):
    """
    解 A v = b，其中 A 的元素给成 (log|A_ij|, sign(A_ij))，b 给成 (log|b_i|, sign(b_i))。
    返回 2×n 矩阵：
      第 1 行：log|v_j|
      第 2 行：sign(v_j) ∈ {-1, 0, +1}
    注意：实现基于 Cramer 法则 + 莱布尼茨公式，时间复杂度 O(n! · n)，仅适用于很小的 n（如 n≤6）。
    """

    # ---------- 辅助函数 ----------
    def permutation_parity_zero_based(perm):
        inv = 0
        for i in range(len(perm)):
            for j in range(i+1, len(perm)):
                if perm[i] > perm[j]:
                    inv += 1
        return 1.0 if (inv % 2 == 0) else -1.0

    def log_sum_exp_signed(arr, signs):
        arr = np.asarray(arr, dtype=float).reshape(-1)
        signs = np.asarray(signs, dtype=float).reshape(-1)

        # 同时出现 sign>0 与 sign<0 的 +inf ⇒ ∞ - ∞ 不定
        pos_inf_pos = np.any((arr == np.inf) & (signs > 0))
        pos_inf_neg = np.any((arr == np.inf) & (signs < 0))
        if pos_inf_pos and pos_inf_neg:
            return np.array([np.nan, 0.0])

        # 单侧 +inf 直接短路
        if pos_inf_pos:
            return np.array([np.inf, 1.0])
        if pos_inf_neg:
            return np.array([np.inf, -1.0])

        # 掩掉 NaN；-inf 保留（对应 0 项）
        mask = np.isfinite(arr) & np.isfinite(signs)
        if not np.any(mask):
            return np.array([-np.inf, 0.0])

        log_abs, sgn = logsumexp(arr[mask], b=signs[mask], return_sign=True)
        if sgn == 0.0:
            return np.array([-np.inf, 0.0])
        return np.array([log_abs, sgn])

    def find_perms(n):
        perm_arr = np.array(list(itertools.permutations(np.arange(n))), dtype=int)
        return perm_arr.shape[0], perm_arr

    def log_determinant(M_lg, M_sgn):
        n = M_lg.shape[0]
        assert M_lg.shape == (n, n) and M_sgn.shape == (n, n)

        perm_num, perm_list = find_perms(n)
        logs = np.empty((perm_num,), dtype=float)
        signs = np.empty((perm_num,), dtype=float)

        rows = np.arange(n)
        for i in range(perm_num):
            cols = perm_list[i]
            logs[i]  = np.sum(M_lg[rows, cols])                 # log|∏ a_ii| = ∑ log|a_ii|
            signs[i] = permutation_parity_zero_based(cols) * np.prod(M_sgn[rows, cols])

        log_abs_det, sign_det = log_sum_exp_signed(logs, signs)
        return log_abs_det, sign_det

    # ---------- 输入整理与基本检查 ----------
    m_lg  = np.asarray(m_lg,  dtype=float)
    m_sgn = np.asarray(m_sgn, dtype=float)

    n = m_lg.shape[0]
    assert m_lg.shape == (n, n) and m_sgn.shape == (n, n), "m_lg/m_sgn 必须是 n×n"

    # b 向量拉平成长度 n
    b_lg  = np.asarray(b_lg,  dtype=float).reshape(-1)
    b_sgn = np.asarray(b_sgn, dtype=float).reshape(-1)
    assert b_lg.shape[0] == n and b_sgn.shape[0] == n, "b_lg/b_sgn 的长度必须等于 n"

    v_lgs  = np.full((n, 1), -np.inf)   # 默认 0（log|0|=-inf）
    v_sgns = np.zeros((n, 1))           # 默认 0

    det_log_abs, det_sign = log_determinant(m_lg, m_sgn)

    # 若 det(A)=0（完全抵消或某列/行全 0），系统奇异：全部返回 0
    if (not np.isfinite(det_log_abs)) or (det_sign == 0.0):
        return np.vstack((v_lgs.T, v_sgns.T))

    # ---------- Cramer 法则 ----------
    for j in range(n):
        m_log_j = m_lg.copy()
        m_log_j[:, j] = b_lg

        m_sgn_j = m_sgn.copy()
        m_sgn_j[:, j] = b_sgn

        detj_log_abs, detj_sign = log_determinant(m_log_j, m_sgn_j)

        # 分子=0 ⇒ 该分量为 0
        if (not np.isfinite(detj_log_abs)) or (detj_sign == 0.0):
            v_lgs[j, 0]  = -np.inf
            v_sgns[j, 0] = 0.0
            continue

        # 正常情况：log|x_j| = log|det(A_j)| - log|det(A)|；sign(x_j) = sign(det(A_j))*sign(det(A))
        v_lgs[j, 0]  = detj_log_abs - det_log_abs
        v_sgns[j, 0] = detj_sign * det_sign

    return np.vstack((v_lgs.T, v_sgns.T))
