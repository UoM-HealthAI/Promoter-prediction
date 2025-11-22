# -*- coding: utf-8 -*-
"""
Renewed on Oct 2025

Original: Jon (2020)
Edited by: Jingyu (2025)
"""

import numpy as np
from scipy.special import logsumexp
from burstInfer.calcObservationLikelihood_renew import calcObservationLikelihood

def forward_backward(pi0_log, lambda_log, data, noise_temp, v, K, W,
                     ms2_coeff_flipped, states_container_unused, off_off, off_on,
                     on_off, on_on, PERMITTED_MEMORY, trace_length,
                     log_likelihoods_unused, logL_tot_unused, em_step_idx, i_tr):
    """
    Beam-style 前向后向（受限复合状态 PERMITTED_MEMORY）。
    返回：
      - Gamma: list 长度 L；每个元素是 {state:int -> log gamma_t(state)}
      - off_off_results / off_on_results / on_off_results / on_on_results: list[float]，每个是一条转移的 log ξ
      - logL: float，整条序列 log-likelihood
    约定：
      - 复合状态按 W 位编码，LSB 对应当前位；下一时刻做 ((s<<1)+bit) & mask。
      - 发射 log 似然由 calcObservationLikelihood(...) 提供。
    """
    # ---- 准备 ----
    data = np.asarray(data, dtype=float).ravel()
    L = int(trace_length) if trace_length is not None else int(data.size)
    assert L == data.size, "trace_length 与 data 长度不一致"
    assert K == 2, "当前实现仅支持 K=2（OFF/ON）"

    mask = (1 << W) - 1
    TINY = np.finfo(float).tiny

    # 便捷函数：给定前一状态 LSB 和目标 bit，取转移对数概率
    def trans_log(prev_bit, curr_bit):
        if prev_bit == 0 and curr_bit == 0: return off_off
        if prev_bit == 0 and curr_bit == 1: return off_on
        if prev_bit == 1 and curr_bit == 0: return on_off
        return on_on  # 1->1

    # 发射 log 似然包装（与原工程一致）
    def emit_log(t, state_code):
        # data[t] 是观测；state_code 是复合状态编号
        return float(calcObservationLikelihood(lambda_log, noise_temp, data[t], v,
                                              int(state_code), K, W, ms2_coeff_flipped))

    # ---- Forward（α，beam 合并+截断）----
    beam_states = []  # 每时刻的状态列表（list[int]）
    beam_alphas = []  # 每时刻的对数 alpha（np.ndarray shape (S_t,)）

    # t=0：两个单比特状态 0/1
    alpha0 = pi0_log[0, 0] + emit_log(0, 0)
    alpha1 = pi0_log[1, 0] + emit_log(0, 1)
    s0 = [0, 1]
    a0 = np.array([alpha0, alpha1], dtype=float)
    # 截断
    if len(s0) > PERMITTED_MEMORY:
        idx = np.argsort(a0)[::-1][:PERMITTED_MEMORY]
        s0 = [s0[i] for i in idx]
        a0 = a0[idx]
    beam_states.append(s0)
    beam_alphas.append(a0)

    # 之后各时刻
    for t in range(1, L):
        prev_states = beam_states[-1]
        prev_alpha = beam_alphas[-1]
        # 先收集到“目标状态 -> 候选 logalpha 列表”
        target_map = {}  # state -> list of candidate logalpha
        for k, s_prev in enumerate(prev_states):
            a_prev = prev_alpha[k]
            prev_bit = s_prev & 1
            # 两个可能目标
            for bit in (0, 1):
                s_curr = ((s_prev << 1) + bit) & mask
                a_cand = a_prev + trans_log(prev_bit, bit) + emit_log(t, s_curr)
                if s_curr not in target_map:
                    target_map[s_curr] = [a_cand]
                else:
                    target_map[s_curr].append(a_cand)
        # 合并重复目标（logsumexp）
        states_t = list(target_map.keys())
        alphas_t = np.array([logsumexp(np.array(target_map[s], dtype=float))
                             for s in states_t], dtype=float)
        # beam 截断
        if len(states_t) > PERMITTED_MEMORY:
            idx = np.argsort(alphas_t)[::-1][:PERMITTED_MEMORY]
            states_t = [states_t[i] for i in idx]
            alphas_t = alphas_t[idx]
        beam_states.append(states_t)
        beam_alphas.append(alphas_t)

    # 终端对数似然
    logL = logsumexp(beam_alphas[-1]) if len(beam_alphas[-1]) else -np.inf

    # ---- Backward（β）----
    betas = [None] * L
    # t=L-1: beta = 0
    last_states = beam_states[-1]
    betas[-1] = {s: 0.0 for s in last_states}

    for t in range(L - 2, -1, -1):
        curr = {}  # state -> log beta_t(state)
        states_t = beam_states[t]
        states_t1 = set(beam_states[t + 1])
        for s_prev in states_t:
            prev_bit = s_prev & 1
            terms = []
            for bit in (0, 1):
                s_next = ((s_prev << 1) + bit) & mask
                if s_next in states_t1:
                    term = (betas[t + 1][s_next]
                            + trans_log(prev_bit, bit)
                            + emit_log(t + 1, s_next))
                    terms.append(term)
            curr[s_prev] = logsumexp(terms) if terms else -np.inf
        betas[t] = curr

    # ---- Gamma （每时刻、每状态的后验对数）----
    Gammas = []
    for t in range(L):
        g = {}
        a = beam_alphas[t]
        st = beam_states[t]
        for i, s in enumerate(st):
            val = a[i] + betas[t][s] - logL
            g[int(s)] = float(val)
        Gammas.append(g)

    # ---- Xi（分流到四个容器：off_off / off_on / on_off / on_on）----
    off_off_container, off_on_container, on_off_container, on_on_container = [], [], [], []
    for t in range(1, L):
        st_prev = beam_states[t - 1]
        st_curr_set = set(beam_states[t])
        for s_prev in st_prev:
            prev_bit = s_prev & 1
            a_prev = beam_alphas[t - 1][st_prev.index(s_prev)]
            for bit in (0, 1):
                s_curr = ((s_prev << 1) + bit) & mask
                if s_curr in st_curr_set:
                    xi_log = (a_prev
                              + trans_log(prev_bit, bit)
                              + emit_log(t, s_curr)
                              + betas[t][s_curr]
                              - logL)
                    # 分流
                    if prev_bit == 0 and bit == 0:
                        off_off_container.append(float(xi_log))
                    elif prev_bit == 0 and bit == 1:
                        off_on_container.append(float(xi_log))
                    elif prev_bit == 1 and bit == 0:
                        on_off_container.append(float(xi_log))
                    else:
                        on_on_container.append(float(xi_log))

    return {
        'Gamma': Gammas,
        'off_off_results': np.array(off_off_container, dtype=float),
        'off_on_results':  np.array(off_on_container,  dtype=float),
        'on_off_results':  np.array(on_off_container,  dtype=float),
        'on_on_results':   np.array(on_on_container,   dtype=float),
        'logL': float(logL),
    }
