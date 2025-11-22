# -*- coding: utf-8 -*-
"""
Renewed on Oct 2025

Original: Jon (2020)
Edited by: Jingyu (2025)
"""

import numpy as np
import pandas as pd

def export_em_parameters(params):
    """
    将 EM 结果导出为一行 DataFrame。
    兼容以下形状/键：
      - A: (2,2)
      - pi0: (2,), (2,1), (1,2) 或 pi0_log（则取 exp）
      - v: (2,), (2,1), (1,2)
      - noise 或 lambda_log（则 noise = exp(-0.5*lambda_log)）
      - EM seed / logL
    列顺序与原脚本保持一致：
      ['Random Seed','p_off_off','p_off_on','p_on_off','p_on_on',
       'pi0_on','pi0_off','mu_off','mu_on','noise','logL']
    """
    # --- 读取并规范形状 ---
    A = np.asarray(params.get('A', np.full((2,2), np.nan)), dtype=float)
    # pi0：优先用概率；没有就从 log 概率恢复
    if 'pi0' in params:
        pi0_vec = np.asarray(params['pi0'], dtype=float).reshape(-1)
    elif 'pi0_log' in params:
        pi0_vec = np.exp(np.asarray(params['pi0_log'], dtype=float)).reshape(-1)
    else:
        pi0_vec = np.array([np.nan, np.nan], dtype=float)
    # v（mu）
    if 'v' in params:
        mu_vec = np.asarray(params['v'], dtype=float).reshape(-1)
    else:
        mu_vec = np.array([np.nan, np.nan], dtype=float)
    # noise
    if 'noise' in params:
        noise_val = float(params['noise'])
    elif 'lambda_log' in params:
        noise_val = float(np.exp(-0.5 * float(params['lambda_log'])))
    else:
        noise_val = np.nan
    # 其他
    seed_val = params.get('EM seed', np.nan)
    logL_val = params.get('logL', np.nan)

    # --- 安全访问（长度不够用 NaN 补）---
    def _get(vec, i):
        return float(vec[i]) if i < vec.size else np.nan

    # 约定：索引 0=OFF，1=ON（与 v 的含义一致）
    p_off_off = A[0,0] if A.shape == (2,2) else np.nan
    p_off_on  = A[1,0] if A.shape == (2,2) else np.nan
    p_on_off  = A[0,1] if A.shape == (2,2) else np.nan
    p_on_on   = A[1,1] if A.shape == (2,2) else np.nan

    pi0_off = _get(pi0_vec, 0)
    pi0_on  = _get(pi0_vec, 1)

    mu_off = _get(mu_vec, 0)
    mu_on  = _get(mu_vec, 1)

    # --- 组装输出（与原列顺序一致）---
    row = [
        seed_val,
        p_off_off, p_off_on, p_on_off, p_on_on,
        pi0_on,    pi0_off,  # 注意：原脚本列名顺序是 on 后 off
        mu_off,    mu_on,
        noise_val,
        logL_val
    ]
    cols = ['Random Seed', 'p_off_off', 'p_off_on', 'p_on_off', 'p_on_on',
            'pi0_on', 'pi0_off', 'mu_off', 'mu_on', 'noise', 'logL']

    return pd.DataFrame([row], columns=cols)
