# -*- coding: utf-8 -*-
"""
Renewed on Oct 2025

Original: Jon (2020)
Edited by: Jingyu (2025)
"""

import numpy as np

try:
    from scipy.special import logsumexp as _lse
except Exception:
    # 兜底实现
    def _lse(a, axis=None, keepdims=False):
        a = np.asarray(a, dtype=float)
        m = np.nanmax(a, axis=axis, keepdims=True)
        res = m + np.log(np.nansum(np.exp(a - m), axis=axis, keepdims=True))
        if not keepdims:
            res = np.squeeze(res, axis=axis)
        return res

from burstInfer.v_log_solve import v_log_solve
from burstInfer.log_sum_exp import log_sum_exp_scipy as log_sum_exp
from burstInfer.ms2_loading_coeff import ms2_loading_coeff

from burstInfer.forward_backward_renew import forward_backward
from burstInfer.get_adjusted_renew import get_adjusted
from burstInfer.compute_dynamic_F_renew import compute_dynamic_F

_TINY = np.finfo(float).tiny

class HMM:
    def __init__(self, K, W, t_MS2, deltaT, kappa, compound_states, processed_signals):
        self.K = int(K)
        self.W = int(W)
        self.t_MS2 = float(t_MS2)
        self.deltaT = float(deltaT)
        self.kappa = float(kappa)
        self.compound_states = int(compound_states)
        self.processed_signals = processed_signals  # dict; 'Processed Signals' 为 list[np.ndarray 1D]

    # -------- helpers (本类内部用) --------
    @staticmethod
    def _clip_prob_vec(p, eps=1e-12):
        p = np.asarray(p, dtype=float)
        p = np.clip(p, eps, 1.0)
        p = p / p.sum()
        return p

    @staticmethod
    def _safe_log(x):
        return np.log(np.clip(x, _TINY, np.inf))

    # ------------------------------------

    def initialise_parameters(self):
        K, W = self.K, self.W
        matrix_max = float(self.processed_signals.get('Matrix Max', 1.0))
        matrix_mean = float(self.processed_signals.get('Matrix Mean', 1.0))

        # 安全的尺度（避免 0/NaN）
        scale_max = matrix_max if np.isfinite(matrix_max) and matrix_max > 0 else 1.0
        scale_mean = matrix_mean if np.isfinite(matrix_mean) and matrix_mean > 0 else 1.0

        # v_init: Kx1，给“ON”更大幅度，“OFF”更小幅度；对 K>2 做平滑差异
        v_init = np.linspace(0.15, 0.65, K).reshape(K, 1) * (2.0 / max(W, 1)) * scale_max
        # 避免 0
        v_init = np.clip(v_init, 1e-6, None)

        # 噪声初值（标准差）：0.2~0.8 * matrix_mean，且至少为很小正数
        noise_init = np.clip(0.5 * scale_mean, 1e-6, None)

        # 初始先验 pi0（列向量）
        pi0_init = self._clip_prob_vec(np.ones(K) / K)
        pi0_log = self._safe_log(pi0_init).reshape(K, 1)

        # 转移矩阵 A（列归一化：A[:, j] = P(s_t = i | s_{t-1} = j)）
        A_init = np.zeros((K, K), dtype=float)
        for j in range(K):
            # 偏向于“状态保持”，但保留跳转概率
            stay = 0.8
            A_init[:, j] = (1.0 - stay) / (K - 1)
            A_init[j, j] = stay
        # clip 并归一化
        for j in range(K):
            A_init[:, j] = self._clip_prob_vec(A_init[:, j])
        A_log = self._safe_log(A_init)

        # lambda / noise 参数化（与你原始关系保持一致）
        lambda_log = -2.0 * self._safe_log(noise_init)
        noise_temp = float(noise_init)

        v_logs = self._safe_log(np.abs(v_init))

        parameter_dict = {
            'pi0_log': pi0_log,         # Kx1
            'A_temp': A_init.copy(),    # 冗余保留以兼容旧代码
            'A_log': A_log,             # KxK（列为条件）
            'v': v_init,                # Kx1
            'lambda_log': lambda_log,   # 标量（log-域）
            'noise_temp': noise_temp,   # 标量（σ）
            'v_logs': v_logs,           # Kx1（对数）
        }
        return parameter_dict

    def EM(self, initialised_parameters, n_steps, n_traces, PERMITTED_MEMORY, eps, seed_setter):
        K, W, kappa = self.K, self.W, self.kappa
        signal_struct = self.processed_signals['Processed Signals']  # list of 1D arrays

        # 解包参数（log 空间）
        A_log = np.array(initialised_parameters['A_log'], dtype=float)
        lambda_log = float(initialised_parameters['lambda_log'])
        noise_temp = float(initialised_parameters['noise_temp'])
        pi0_log = np.array(initialised_parameters['pi0_log'], dtype=float)
        v = np.array(initialised_parameters['v'], dtype=float).reshape(K, 1)
        v_logs = np.array(initialised_parameters['v_logs'], dtype=float).reshape(K, 1)

        # MS2 kernel
        ms2_coeff = ms2_loading_coeff(kappa, W)            # 期望形状(1,W)或(W,)
        ms2_coeff = np.asarray(ms2_coeff, dtype=float)
        if ms2_coeff.ndim == 2 and ms2_coeff.shape[0] == 1:
            ms2_coeff = ms2_coeff.ravel()
        # 若你的前向/后向实现要求“翻转”，在此统一做
        ms2_coeff_flipped = ms2_coeff[::-1].copy()

        # 统计总长度
        fluo_length_total = int(sum(int(np.asarray(x, dtype=float).size) for x in signal_struct))
        if fluo_length_total <= 0:
            raise ValueError("No observations in Processed Signals.")

        # 全局日志似然（每步一个）
        logL_tot = np.full((1, n_steps), -np.inf, dtype=float)

        # EM 迭代
        converged = False
        for bw in range(int(n_steps)):
            print(f'EM step number: {bw}')
            # E 步累加器（log 空间：-inf 表示 log(0)）
            pi0_terms = np.full((1, K), -np.inf)     # 行向量
            A_terms = np.full((K, K), -np.inf)       # KxK
            lambda_terms = -np.inf                   # 标量
            v_M_terms = np.full((K, K), -np.inf)     # KxK
            v_b_terms_log = np.full((1, K), -np.inf) # 1xK
            v_b_terms_sign = np.ones((1, K), dtype=float)

            # 本轮总 log-likelihood（forward_backward 应返回并在此累加）
            ll_round = 0.0

            for i_tr in range(int(n_traces)):
                data = np.asarray(signal_struct[i_tr], dtype=float).ravel()
                L = int(data.size)

                # 观测的 log|x| 与符号（避免 log(0)）
                x_abs = np.abs(data) + _TINY
                x_term_logs = np.log(x_abs)  # 形状 (L,)
                x_term_signs = np.sign(data)
                x_term_signs[x_term_signs == 0.0] = 1.0  # 0 当作 + 号

                # 前向后向（须由你已实现的函数提供）
                fb = forward_backward(
                    pi0_log, lambda_log, data, noise_temp, v, K, W,
                    ms2_coeff_flipped, [],               # states_container 未使用可传空
                    A_log[0, 0], A_log[1, 0],            # off_off, off_on
                    A_log[0, 1], A_log[1, 1],            # on_off, on_on（K=2 情况）
                    PERMITTED_MEMORY, L, None, None, bw, i_tr
                )
                gammas = fb['Gamma']  # 按你原有接口：list[dict]，每个 t 一个字典：{state -> log γ_t(state)}
                # 可选：若 forward_backward 返回 logL，可在此累加
                if 'logL' in fb and np.isfinite(fb['logL']):
                    ll_round += fb['logL']

                # A 的计数（对数域累加）
                def _append_A_term(term_prev, arr):
                    if arr.size == 0:
                        return term_prev
                    # term_prev 和 arr 都是 log 概率/权重
                    return _lse(np.concatenate([term_prev.reshape(1, 1), arr.reshape(-1, 1)], axis=0), axis=0)

                # 这些容器里是单个标量的 list；先做清洗再拼接
                def _sanitize(arr_like):
                    x = np.asarray(arr_like, dtype=float).reshape(-1)
                    x = x[np.isfinite(x)]
                    return x

                off_off_array = _sanitize(fb['off_off_results'])
                off_on_array = _sanitize(fb['off_on_results'])
                on_off_array = _sanitize(fb['on_off_results'])
                on_on_array = _sanitize(fb['on_on_results'])

                A_terms[0, 0] = float(_append_A_term(np.array([[A_terms[0, 0]]]), off_off_array))
                A_terms[1, 0] = float(_append_A_term(np.array([[A_terms[1, 0]]]), off_on_array))
                A_terms[0, 1] = float(_append_A_term(np.array([[A_terms[0, 1]]]), on_off_array))
                A_terms[1, 1] = float(_append_A_term(np.array([[A_terms[1, 1]]]), on_on_array))

                # pi0：用 t=0 的 γ（你的原逻辑是用最后一个，容易引入偏差，这里按常规定义用 t=0）
                g0 = gammas[0]
                for m in range(K):
                    pi0_terms[0, m] = np.logaddexp(pi0_terms[0, m], g0.get(m, -np.inf))

                # —— 以下三块按你原始思路，但做了稳定化与缓存 ——

                # 预计算 dynamic_F：对给定 key 和 (m,n) 及所有 t 的矩阵，仅算一次并缓存
                # cache_F[(key)] -> dict with 'F_m'[K, L], 'F_n'[K, L]
                cache_F = {}

                # λ（噪声）项：∑_t ∑_state γ_t(state) * (x_t - sum_k v_k * coeff_k(state,t))^2
                # 你的 get_adjusted 返回 [one_accumulator, zero_accumulator]；保持一致
                for t in range(L):
                    for key, g_log in gammas[t].items():
                        if not np.isfinite(g_log):
                            continue
                        adj = get_adjusted(int(key), K, W, ms2_coeff)  # [ones, zeros]
                        mu_t = adj[1] * float(v[0, 0]) + adj[0] * float(v[1, 0])  # K=2 情形
                        resid2 = (data[t] - mu_t) ** 2 + _TINY
                        lambda_terms = np.logaddexp(lambda_terms, g_log + np.log(resid2))

                # v_M_terms：双重和的对数累加（尽量避免重复 compute_dynamic_F）
                # compute_dynamic_F(key, L, W, K, ms2_coeff_flipped, count_reduction_manual) -> 返回 K 个通道的 F
                # 为避免重复，我们缓存每个 key 的 F 结果
                for t in range(L):
                    for key, g_log in gammas[t].items():
                        if not np.isfinite(g_log):
                            continue
                        key = int(key)
                        if key not in cache_F:
                            F = compute_dynamic_F(key, L, W, K, ms2_coeff_flipped, None)  # 你的实现里第二个参数是 count_reduction_manual；如需请传入
                            # 期望 F 形状：[K][0, t]；这里统一成 (K, L) 的数组
                            F_arr = np.zeros((K, L), dtype=float)
                            for k_idx in range(K):
                                F_arr[k_idx, :] = np.asarray(F[k_idx][0, :]).ravel()
                            cache_F[key] = F_arr
                        F_arr = cache_F[key]
                        for m in range(K):
                            for n in range(K):
                                v_M_terms[m, n] = np.logaddexp(v_M_terms[m, n], g_log + F_arr[n, t] + F_arr[m, t])

                # v_b_terms（带符号的 log-sum-exp 累加）
                for m in range(K):
                    terms_log = []
                    terms_sgn = []
                    for t in range(L):
                        for key, g_log in gammas[t].items():
                            if not np.isfinite(g_log):
                                continue
                            key = int(key)
                            if key not in cache_F:
                                F = compute_dynamic_F(key, L, W, K, ms2_coeff_flipped, None)
                                F_arr = np.zeros((K, L), dtype=float)
                                for k_idx in range(K):
                                    F_arr[k_idx, :] = np.asarray(F[k_idx][0, :]).ravel()
                                cache_F[key] = F_arr
                            F_arr = cache_F[key]
                            terms_log.append(x_term_logs[t] + g_log + F_arr[m, t])
                            terms_sgn.append(x_term_signs[t])
                    if len(terms_log) > 0:
                        assign1 = np.concatenate([np.array([[v_b_terms_log[0, m]]]), np.array(terms_log, ndmin=2)], axis=1)
                        assign2 = np.concatenate([np.array([[v_b_terms_sign[0, m]]]), np.array(terms_sgn, ndmin=2)], axis=1)
                        tmp_log, tmp_sign = log_sum_exp(assign1, assign2)  # 你的“带符号和”的实现
                        v_b_terms_log[0, m] = tmp_log
                        v_b_terms_sign[0, m] = tmp_sign

            # ============ M 步 ============

            # pi0
            pi0_old = np.exp(pi0_log)
            pi0_log = (pi0_terms - np.log(float(n_traces))).reshape(K, 1)
            pi0_norm_rel_change = np.linalg.norm(np.exp(pi0_log) - pi0_old, 2) / (np.linalg.norm(pi0_old, 2) + 1e-12)

            # A 列归一化（log 空间）
            A_old = np.exp(A_log)
            for j in range(K):
                col = A_terms[:, j]
                col = col - _lse(col)  # 归一化
                A_log[:, j] = col
            A_norm_rel_change = np.linalg.norm(np.exp(A_log) - A_old, 2) / (np.linalg.norm(A_old, 2) + 1e-12)

            # lambda -> noise 更新
            lambda_log_old = lambda_log
            # 注意：使用全体观测数 fluo_length_total
            lambda_log = np.log(float(fluo_length_total)) - float(lambda_terms)
            noise_log_old = -0.5 * lambda_log_old
            noise_log = -0.5 * lambda_log
            noise_temp = float(np.exp(noise_log))
            noise_rel_change = abs(np.exp(noise_log) - np.exp(noise_log_old)) / (np.exp(noise_log_old) + 1e-12)

            # v（解对数线性方程）
            v_logs_old = v_logs.copy()
            v_updated = v_log_solve(v_M_terms, np.ones((K, K)), v_b_terms_log, v_b_terms_sign)
            v_logs = np.asarray(v_updated[0, :], dtype=float).reshape(K, 1)
            v = np.exp(v_logs)
            v_norm_rel_change = np.linalg.norm(np.exp(v_logs) - np.exp(v_logs_old), 2) / (np.linalg.norm(np.exp(v_logs_old), 2) + 1e-12)

            # 记录 ll（若 forward_backward 没回传 logL，可留空或由你内部更新）
            logL_tot[0, bw] = ll_round if np.isfinite(ll_round) else logL_tot[0, bw]

            logL_norm_change = 0.0
            if bw > 0 and np.isfinite(logL_tot[0, bw - 1]) and np.isfinite(logL_tot[0, bw]):
                logL_norm_change = abs(logL_tot[0, bw] - logL_tot[0, bw - 1]) / float(fluo_length_total)

            print(pi0_norm_rel_change)
            print(A_norm_rel_change)
            print(noise_rel_change)
            print(v_norm_rel_change)
            print(logL_norm_change)
            print('A:\n', np.exp(A_log))
            print('pi0:\n', np.exp(pi0_log))
            print('noise:\n', np.exp(noise_log))
            print('v:\n', np.exp(v_logs))
            print('lltot:\n', logL_tot[0, bw])

            # 收敛判据（任一相对变化量的最大值 < eps）
            if max(pi0_norm_rel_change, A_norm_rel_change, noise_rel_change, v_norm_rel_change, logL_norm_change) < float(eps):
                converged = True
                logL_tot = logL_tot[:, :bw + 1]
                break

        # 输出
        output_dict = {
            'A': np.exp(A_log),
            'pi0': np.exp(pi0_log).reshape(K, 1),
            'v': np.exp(v_logs).reshape(K, ),
            'noise': np.exp(noise_log),
            'logL': float(logL_tot[0, -1]) if logL_tot.size else np.nan,
            'EM seed': seed_setter,
            'converged': converged,
            'n_em_steps': int(logL_tot.shape[1]),
        }
        return output_dict

    def EM_fixed(self, initialised_parameters, n_steps, n_traces, PERMITTED_MEMORY,
                 eps, seed_setter):

        K, W, kappa = self.K, self.W, self.kappa
        signal_struct = self.processed_signals['Processed Signals']

        # 解包参数
        A_log = np.array(initialised_parameters['A_log'], dtype=float)  # 固定不更新
        lambda_log = float(initialised_parameters['lambda_log'])
        noise_temp = float(initialised_parameters['noise_temp'])
        pi0_log = np.array(initialised_parameters['pi0_log'], dtype=float).reshape(K, 1)
        v = np.array(initialised_parameters['v'], dtype=float).reshape(K, 1)
        v_logs = np.array(initialised_parameters['v_logs'], dtype=float).reshape(K, 1)

        # MS2 kernel，统一成 1D，并做“时间最近位对应 LSB”的翻转
        ms2_coeff = ms2_loading_coeff(kappa, W)
        ms2_coeff = np.asarray(ms2_coeff, dtype=np.float64).ravel()  # ← 强制 1D
        ms2_coeff_flipped = ms2_coeff[::-1].copy()  # LSB 对齐

        # 统计总观测长度
        fluo_length_total = int(sum(int(np.asarray(x, dtype=float).size) for x in signal_struct))
        if fluo_length_total <= 0:
            raise ValueError("No observations in Processed Signals.")

        # 全局 log-likelihood 记录
        logL_tot = np.full((1, n_steps), -np.inf, dtype=float)
        # 某些 forward_backward 旧实现需要这两个占位参数；保持兼容
        log_likelihoods = np.full((1, n_traces), -np.inf, dtype=float)

        TINY = np.finfo(float).tiny
        converged = False

        for bw in range(int(n_steps)):
            print('EM (fixed) step number:', bw)

            # E 步累加器（对数域）
            pi0_terms = np.full((1, K), -np.inf)
            lambda_terms = -np.inf
            v_M_terms = np.full((K, K), -np.inf)
            v_b_terms_log = np.full((1, K), -np.inf)
            v_b_terms_sign = np.ones((1, K), dtype=float)

            ll_round = 0.0  # 汇总本轮 logL（若 forward_backward 返回）

            for i_tr in range(int(n_traces)):
                # 1D 轨迹
                data = np.asarray(signal_struct[i_tr], dtype=float).ravel()
                L = int(data.size)

                # 观测的 log|x| 与符号（避免 log(0)）
                x_abs = np.abs(data) + TINY
                x_term_logs = np.log(x_abs)  # (L,)
                x_term_signs = np.sign(data)
                x_term_signs[x_term_signs == 0.0] = 1.0

                # 固定 A：便于向旧 forward_backward 传标量 off/on
                off_off, off_on = A_log[0, 0], A_log[1, 0]
                on_off, on_on = A_log[0, 1], A_log[1, 1]

                # 前向后向
                fb = forward_backward(
                    pi0_log, lambda_log, data, noise_temp, v, K, W,
                    ms2_coeff_flipped, [],  # states_container 兼容占位
                    off_off, off_on, on_off, on_on,
                    PERMITTED_MEMORY, L,
                    log_likelihoods, logL_tot, bw, i_tr
                )
                gammas = fb['Gamma']  # list[dict]：每个 t 的 {state -> log γ_t(state)}
                if 'logL' in fb and np.isfinite(fb['logL']):
                    ll_round += fb['logL']

                # ------- pi0：用 t=0 的 γ -------
                g0 = gammas[0]
                updated_pi0 = False
                for m in range(K):
                    if m in g0 and np.isfinite(g0[m]):
                        pi0_terms[0, m] = np.logaddexp(pi0_terms[0, m], g0[m])
                        updated_pi0 = True
                # 若 Gamma 不含 base-state 键，保持 pi0 不变
                if not updated_pi0:
                    for m in range(K):
                        pi0_terms[0, m] = np.logaddexp(pi0_terms[0, m], float(pi0_log[m, 0]))

                # ------- λ（噪声）项 -------
                # ∑_t ∑_{compound state} γ_t(state) * (x_t - μ_t)^2
                for t in range(L):
                    for key, g_log in gammas[t].items():
                        if not np.isfinite(g_log):
                            continue
                        adj = get_adjusted(int(key), K, W, ms2_coeff)  # [ones, zeros]
                        mu_t = adj[1] * float(v[0, 0]) + adj[0] * float(v[1, 0])  # K=2
                        resid2 = (data[t] - mu_t) ** 2 + TINY
                        lambda_terms = np.logaddexp(lambda_terms, g_log + np.log(resid2))

                # ------- 预计算 dynamic_F（缓存） -------
                cache_F = {}

                # v_M_terms：∑ γ * F_n * F_m（对数域累加）
                for t in range(L):
                    for key, g_log in gammas[t].items():
                        if not np.isfinite(g_log):
                            continue
                        key = int(key)
                        if key not in cache_F:
                            F = compute_dynamic_F(key, L, W, K, ms2_coeff_flipped, None)
                            # 适配 (K, L)
                            F_arr = np.zeros((K, L), dtype=float)
                            for k_idx in range(K):
                                F_arr[k_idx, :] = np.asarray(F[k_idx][0, :]).ravel()
                            cache_F[key] = F_arr
                        F_arr = cache_F[key]
                        for m in range(K):
                            for n in range(K):
                                v_M_terms[m, n] = np.logaddexp(v_M_terms[m, n], g_log + F_arr[n, t] + F_arr[m, t])

                # v_b_terms：带符号的 log-sum-exp
                for m in range(K):
                    terms_log = []
                    terms_sgn = []
                    for t in range(L):
                        for key, g_log in gammas[t].items():
                            if not np.isfinite(g_log):
                                continue
                            key = int(key)
                            if key not in cache_F:
                                F = compute_dynamic_F(key, L, W, K, ms2_coeff_flipped, None)
                                F_arr = np.zeros((K, L), dtype=float)
                                for k_idx in range(K):
                                    F_arr[k_idx, :] = np.asarray(F[k_idx][0, :]).ravel()
                                cache_F[key] = F_arr
                            F_arr = cache_F[key]
                            terms_log.append(x_term_logs[t] + g_log + F_arr[m, t])
                            terms_sgn.append(x_term_signs[t])
                    if len(terms_log) > 0:
                        assign1 = np.concatenate([np.array([[v_b_terms_log[0, m]]]),
                                                  np.array(terms_log, ndmin=2)], axis=1)
                        assign2 = np.concatenate([np.array([[v_b_terms_sign[0, m]]]),
                                                  np.array(terms_sgn, ndmin=2)], axis=1)
                        tmp_log, tmp_sign = log_sum_exp(assign1, assign2)
                        v_b_terms_log[0, m] = tmp_log
                        v_b_terms_sign[0, m] = tmp_sign

            # ============ M 步（A 固定） ============
            # pi0
            pi0_old = np.exp(pi0_log)
            pi0_log = (pi0_terms - np.log(float(n_traces))).reshape(K, 1)
            pi0_norm_rel_change = np.linalg.norm(np.exp(pi0_log) - pi0_old, 2) / (np.linalg.norm(pi0_old, 2) + 1e-12)

            # A 固定：仅确保归一化（列向量 log 归一化）
            for j in range(K):
                col = A_log[:, j]
                col = col - (np.log(np.sum(np.exp(col - np.max(col)))) + np.max(col))
                A_log[:, j] = col
            A_old = np.exp(initialised_parameters['A_log'])
            A_norm_rel_change = np.linalg.norm(np.exp(A_log) - A_old, 2) / (np.linalg.norm(A_old, 2) + 1e-12)

            # λ（噪声）
            lambda_log_old = lambda_log
            lambda_log = np.log(float(fluo_length_total)) - float(lambda_terms)
            noise_log_old = -0.5 * lambda_log_old
            noise_log = -0.5 * lambda_log
            noise_temp = float(np.exp(noise_log))
            noise_rel_change = abs(np.exp(noise_log) - np.exp(noise_log_old)) / (np.exp(noise_log_old) + 1e-12)

            # v
            v_logs_old = v_logs.copy()
            v_updated = v_log_solve(v_M_terms, np.ones((K, K)), v_b_terms_log, v_b_terms_sign)
            v_logs = np.asarray(v_updated[0, :], dtype=float).reshape(K, 1)
            v = np.exp(v_logs)
            v_norm_rel_change = np.linalg.norm(np.exp(v_logs) - np.exp(v_logs_old), 2) / (
                        np.linalg.norm(np.exp(v_logs_old), 2) + 1e-12)

            # 记录 ll（如果 forward_backward 有返回）
            logL_tot[0, bw] = ll_round if np.isfinite(ll_round) else logL_tot[0, bw]
            logL_norm_change = 0.0
            if bw > 0 and np.isfinite(logL_tot[0, bw - 1]) and np.isfinite(logL_tot[0, bw]):
                logL_norm_change = abs(logL_tot[0, bw] - logL_tot[0, bw - 1]) / float(fluo_length_total)

            print(pi0_norm_rel_change)
            print(A_norm_rel_change)
            print(noise_rel_change)
            print(v_norm_rel_change)
            print(logL_norm_change)
            print('A:\n', np.exp(A_log))
            print('pi0:\n', np.exp(pi0_log))
            print('noise:\n', np.exp(noise_log))
            print('v:\n', np.exp(v_logs))
            print('lltot:\n', logL_tot[0, bw])

            if max(pi0_norm_rel_change, A_norm_rel_change, noise_rel_change, v_norm_rel_change,
                   logL_norm_change) < float(eps):
                converged = True
                logL_tot = logL_tot[:, :bw + 1]
                break

        # 返回（保持你主程序期望的键）
        output_dict = {
            'A': np.exp(A_log),
            'pi0': np.exp(pi0_log),
            'v': np.exp(v_logs),
            'noise': np.exp(noise_log),
            'logL': float(logL_tot[0, -1]) if logL_tot.size else np.nan,
            'EM seed': seed_setter,

            # 这些是后续 EM_with_priors 需要的“先验/初值”形式
            'lambda_log': lambda_log,
            'v_logs': v_logs,
            'noise_temp': noise_temp,
            'pi0_log': pi0_log,
        }
        return output_dict

    def EM_with_priors(self, initialised_parameters, n_steps, n_traces, PERMITTED_MEMORY,
                       eps, seed_setter):
        import numpy as np
        from scipy.special import logsumexp

        # ---------- 小工具 ----------
        def safe_logaddexp(a, b):
            a = np.asarray(a, float);
            b = np.asarray(b, float)
            a0 = np.where(np.isnan(a), -np.inf, a)
            b0 = np.where(np.isnan(b), -np.inf, b)
            out = np.logaddexp(a0, b0)
            out = np.where(np.isposinf(a) | np.isposinf(b), np.inf, out)
            out = np.where(np.isneginf(a) & np.isneginf(b), -np.inf, out)
            return out

        def safe_log_normalize(logv, axis=None):
            logv = np.asarray(logv, float)
            if axis is None:
                if not np.any(np.isfinite(logv)):
                    return np.full_like(logv, -np.log(logv.size))
                Z = logsumexp(np.where(np.isfinite(logv), logv, -np.inf))
                if np.isneginf(Z):
                    return np.full_like(logv, -np.log(logv.size))
                return logv - Z
            finite = np.isfinite(logv)
            Z = logsumexp(np.where(finite, logv, -np.inf), axis=axis, keepdims=True)
            all_bad = ~np.any(finite, axis=axis, keepdims=True)
            Kx = logv.shape[axis]
            normed = logv - Z
            return np.where(all_bad, -np.log(Kx), normed)

        def col_log_softmax(M):
            M = np.asarray(M, float)
            M_f = np.where(np.isfinite(M), M, -np.inf)
            Z = logsumexp(M_f, axis=0, keepdims=True)
            bad = (~np.isfinite(Z)) | np.isneginf(Z)
            good = ~bad
            Kx = M.shape[0]
            M_norm = np.empty_like(M)
            if np.any(good):
                M_norm[:, good[0]] = M[:, good[0]] - Z[:, good[0]]
            if np.any(bad):
                M_norm[:, bad[0]] = -np.log(Kx)
            return M_norm

        def vec_l2(x):
            x = np.asarray(x, float).ravel()
            x = np.nan_to_num(x, nan=0.0, posinf=np.inf, neginf=-np.inf)
            if not np.all(np.isfinite(x)):
                return np.inf
            return float(np.sqrt(np.dot(x, x)))

        # ---------- 读入数据与先验 ----------
        K, W, kappa = self.K, self.W, self.kappa
        signal_struct = self.processed_signals['Processed Signals']

        # A/π0/v/noise 以 EM_fixed 结果为“先验起点”
        if 'A' in initialised_parameters:
            A_log = np.log(np.asarray(initialised_parameters['A'], float))
        else:
            # 兜底：均匀
            A_log = np.log(np.full((K, K), 1.0 / K))
        if 'pi0_log' in initialised_parameters:
            pi0_log = np.asarray(initialised_parameters['pi0_log'], float).reshape(K, 1)
        elif 'pi0' in initialised_parameters:
            pi0_log = np.log(np.asarray(initialised_parameters['pi0'], float)).reshape(K, 1)
        else:
            pi0_log = np.log(np.full((K, 1), 1.0 / K))
        if 'v_logs' in initialised_parameters:
            v_logs = np.asarray(initialised_parameters['v_logs'], float).reshape(K, 1)
            v = np.exp(v_logs)
        elif 'v' in initialised_parameters:
            v = np.asarray(initialised_parameters['v'], float).reshape(K, 1)
            v_logs = np.log(v + 1e-300)
        else:
            v = np.ones((K, 1), float);
            v_logs = np.log(v)
        if 'lambda_log' in initialised_parameters:
            lambda_log = float(initialised_parameters['lambda_log'])
            noise_temp = float(np.exp(-0.5 * lambda_log))
        elif 'noise' in initialised_parameters:
            noise_temp = float(initialised_parameters['noise'])
            noise_temp = max(noise_temp, 1e-8)
            lambda_log = -2.0 * np.log(noise_temp)
        else:
            noise_temp = 1.0
            lambda_log = -2.0 * np.log(noise_temp)

        # MS2 内核：1D，并翻转与 LSB 对齐
        ms2_coeff = ms2_loading_coeff(kappa, W)
        ms2_coeff = np.asarray(ms2_coeff, dtype=float).ravel()  # 1D
        ms2_coeff_flipped = ms2_coeff[::-1].copy()  # LSB ↔ ms2[0]

        # 全体观测点数（用于 λ 的样本数）
        fluo_length_total = int(sum(int(np.asarray(x, float).ravel().size) for x in signal_struct))
        if fluo_length_total <= 0:
            raise ValueError("No observations in Processed Signals.")

        logL_tot = np.full((1, n_steps), -np.inf, dtype=float)
        log_likelihoods = np.full((1, n_traces), -np.inf, dtype=float)

        TINY = np.finfo(float).tiny

        # ---------- EM 迭代 ----------
        for bw in range(int(n_steps)):
            print('EM (with priors) step number:', bw)

            # E 步累加器（log 域）
            pi0_terms = np.full((1, K), -np.inf)
            A_terms = np.full((K, K), -np.inf)
            lambda_terms = -np.inf
            v_M_terms = np.full((K, K), -np.inf)
            v_b_terms_log = np.full((1, K), -np.inf)
            v_b_terms_sign = np.ones((1, K), dtype=float)

            ll_round = 0.0

            for i_tr in range(int(n_traces)):
                data = np.asarray(signal_struct[i_tr], dtype=float).ravel()
                L = int(data.size)

                # 观测的 log|x| 与符号
                x_abs = np.abs(data) + TINY
                x_term_logs = np.log(x_abs)  # (L,)
                x_term_signs = np.sign(data)
                x_term_signs[x_term_signs == 0.0] = 1.0

                # 固定当前 A 的四个转移 log 概率
                off_off, off_on = A_log[0, 0], A_log[1, 0]
                on_off, on_on = A_log[0, 1], A_log[1, 1]

                # 前向后向（beam）
                fb = forward_backward(
                    pi0_log, lambda_log, data, noise_temp, v, K, W,
                    ms2_coeff_flipped, [],  # states_container 占位
                    off_off, off_on, on_off, on_on,
                    PERMITTED_MEMORY, L,
                    log_likelihoods, logL_tot, bw, i_tr
                )
                Gammas = fb['Gamma']  # list of dict
                off_off_container = fb['off_off_results']
                off_on_container = fb['off_on_results']
                on_off_container = fb['on_off_results']
                on_on_container = fb['on_on_results']
                if 'logL' in fb and np.isfinite(fb['logL']):
                    ll_round += fb['logL']

                # ---- π0：用 t=0 的 γ（缺失则按 -inf 忽略）----
                g0 = Gammas[0]
                for m in range(K):
                    pi0_terms[0, m] = safe_logaddexp(pi0_terms[0, m], g0.get(m, -np.inf))

                # ---- A：四类转移的 log ξ 聚合 ----
                def _agg_A_term(cur_log, cont):
                    if cont.size == 0:
                        return cur_log
                    # 允许把当前值也拼进去（防空）
                    arr = np.concatenate([np.array([cur_log], dtype=float), cont.reshape(-1)], axis=0)
                    return logsumexp(arr)

                A_terms[0, 0] = _agg_A_term(A_terms[0, 0], off_off_container)
                A_terms[1, 0] = _agg_A_term(A_terms[1, 0], off_on_container)
                A_terms[0, 1] = _agg_A_term(A_terms[0, 1], on_off_container)
                A_terms[1, 1] = _agg_A_term(A_terms[1, 1], on_on_container)

                # ---- λ（噪声）：∑ γ * (x_t - μ_t(state))^2 ----
                # 方案 A：μ_t 用全窗口的 ones/zeros（不随 t 变）
                W_eff = min(W, ms2_coeff.size)
                for t in range(L):
                    for key, g_log in Gammas[t].items():
                        if not np.isfinite(g_log):
                            continue
                        s = int(key)
                        ones = 0.0;
                        zeros = 0.0;
                        ss = s
                        for p in range(W_eff):
                            w = ms2_coeff_flipped[p]
                            if (ss & 1) == 1:
                                ones += w
                            else:
                                zeros += w
                            ss >>= 1
                        mu_t = zeros * float(v[0, 0]) + ones * float(v[1, 0])
                        resid2 = (data[t] - mu_t) ** 2 + TINY
                        lambda_terms = np.logaddexp(lambda_terms, g_log + np.log(resid2))

                # ---- v 的二次项/一次项：缓存 dynamic F，避免重复算 ----
                cache_F = {}

                # v_M: ∑ γ * F_n * F_m
                for t in range(L):
                    for key, g_log in Gammas[t].items():
                        if not np.isfinite(g_log):
                            continue
                        s = int(key)
                        if s not in cache_F:
                            F = compute_dynamic_F(s, L, W, K, ms2_coeff_flipped, None)
                            F_arr = np.zeros((K, L), dtype=float)
                            for k_idx in range(K):
                                F_arr[k_idx, :] = np.asarray(F[k_idx][0, :]).ravel()
                            cache_F[s] = F_arr
                        F_arr = cache_F[s]
                        for m in range(K):
                            for n in range(K):
                                v_M_terms[m, n] = np.logaddexp(v_M_terms[m, n], g_log + F_arr[n, t] + F_arr[m, t])

                # v_b: 带符号的 log-sum-exp
                for m in range(K):
                    terms_log = []
                    terms_sgn = []
                    for t in range(L):
                        for key, g_log in Gammas[t].items():
                            if not np.isfinite(g_log):
                                continue
                            s = int(key)
                            if s not in cache_F:
                                F = compute_dynamic_F(s, L, W, K, ms2_coeff_flipped, None)
                                F_arr = np.zeros((K, L), dtype=float)
                                for k_idx in range(K):
                                    F_arr[k_idx, :] = np.asarray(F[k_idx][0, :]).ravel()
                                cache_F[s] = F_arr
                            F_arr = cache_F[s]
                            terms_log.append(x_term_logs[t] + g_log + F_arr[m, t])
                            terms_sgn.append(x_term_signs[t])
                    if len(terms_log) > 0:
                        assign1 = np.concatenate([np.array([[v_b_terms_log[0, m]]]),
                                                  np.array(terms_log, ndmin=2)], axis=1)
                        assign2 = np.concatenate([np.array([[v_b_terms_sign[0, m]]]),
                                                  np.array(terms_sgn, ndmin=2)], axis=1)
                        tmp_log, tmp_sign = log_sum_exp(assign1, assign2)
                        v_b_terms_log[0, m] = tmp_log
                        v_b_terms_sign[0, m] = tmp_sign

            # ---- M 步：π0/A/λ/v ----
            # π0
            pi0_log_old = pi0_log.copy()
            pi0_log = safe_log_normalize(pi0_terms).reshape(K, 1)
            pi0_old_prob = np.exp(safe_log_normalize(pi0_log_old.ravel()))
            pi0_new_prob = np.exp(pi0_log.ravel())
            den_pi0 = max(vec_l2(pi0_old_prob), 1e-12)
            pi0_norm_rel_change = vec_l2(pi0_new_prob - pi0_old_prob) / den_pi0

            # A：列 softmax
            A_old = np.exp(A_log)
            A_log = col_log_softmax(A_terms)
            den_A = max(np.linalg.norm(A_old, 'fro'), 1e-12)
            A_norm_rel_change = np.linalg.norm(np.exp(A_log) - A_old, 'fro') / den_A

            # λ / noise
            lambda_log_old = lambda_log
            lambda_log = np.log(float(fluo_length_total)) - float(lambda_terms)
            noise_log_old = -0.5 * lambda_log_old
            noise_log = -0.5 * lambda_log
            noise_temp = float(np.exp(noise_log))
            noise_rel_change = abs(np.exp(noise_log) - np.exp(noise_log_old)) / (np.exp(noise_log_old) + 1e-12)

            # v
            v_logs_old = v_logs.copy()
            m_sign = np.ones((K, K), dtype=float)
            m_log = v_M_terms
            b_sign = v_b_terms_sign
            b_log = v_b_terms_log
            v_updated = v_log_solve(m_log, m_sign, b_log, b_sign)
            v_logs = np.asarray(v_updated[0, :], dtype=float).reshape(K, 1)
            v = np.exp(v_logs)
            den_v = max(vec_l2(np.exp(v_logs_old)), 1e-12)
            v_norm_rel_change = vec_l2(np.exp(v_logs) - np.exp(v_logs_old)) / den_v

            # logL 变化（forward_backward 已经返回每条轨迹的 logL，我们累加）
            logL_tot[0, bw] = ll_round
            logL_norm_change = 0.0
            if bw > 0 and np.isfinite(logL_tot[0, bw - 1]) and np.isfinite(logL_tot[0, bw]):
                logL_norm_change = abs(logL_tot[0, bw] - logL_tot[0, bw - 1]) / float(fluo_length_total)

            # ---- 日志 / 收敛判据 ----
            print(pi0_norm_rel_change)
            print(A_norm_rel_change)
            print(noise_rel_change)
            print(v_norm_rel_change)
            print(logL_norm_change)
            print('A:\n', np.exp(A_log))
            print('pi0:\n', np.exp(pi0_log))
            print('noise:\n', np.exp(noise_log))
            print('v:\n', np.exp(v_logs))
            print('lltot:\n', logL_tot[0, bw])

            if max(pi0_norm_rel_change, A_norm_rel_change, noise_rel_change,
                   v_norm_rel_change, logL_norm_change) < float(eps):
                logL_tot = logL_tot[:, :bw + 1]
                break

        return {
            'A': np.exp(A_log),
            'pi0': np.exp(pi0_log),
            'v': np.exp(v_logs),
            'noise': np.exp(noise_log),
            'logL': float(logL_tot[0, -1]) if logL_tot.size else np.nan,
            'EM seed': seed_setter,
        }
