# -*- coding: utf-8 -*-
"""
Created on Oct 2025

@author: Jingyu
"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import genfromtxt
from burstInfer.process_raw_data_renew import process_raw_data
from burstInfer.HMM_renew import HMM
from burstInfer.export_em_parameters_renew import export_em_parameters

# =======================
# Configuration (edit here)
# =======================
CSV_PATH = 'uwt_e1_no_bd.csv'
HAS_ROW_INDEX_COL = True      # 首列是行索引？如果是，先删掉
META_COLS = 11                # 前 META_COLS 列为元数据，轨迹从此列起
CLUSTER_COL_AFTER_STRIP = 1   # 去掉首列索引后的 cluster 列位置
CLUSTER_KEEP_VALUE = 0        # 仅保留该 cluster 的数据

DROP_LOWEST_N_BY_MEAN = 20    # 丢掉均值最小的前 N 条（在轨迹列上算均值）
# 或者用分位数阈值（两者二选一；如果用分位数，将 DROP_LOWEST_N_BY_MEAN 设为 0）
WEAK_MEAN_QUANTILE = None     # 例如 0.05 表示丢掉均值位于后 5% 的轨迹

SEED = 963456000
PLOT_FIG = True
FIG_OUT = f'example_trace_{SEED}.png'

# HMM / EM
K = 2  # Number of promoter states (always 2 for this demo)
EPS = 1e-3 # Convergence tolerance
N_STEPS_FIXED = 1 # Number of maximum EM steps
N_STEPS_PRIORS = 1 # Number of maximum EM steps with priors
PERMITTED_MEMORY = 256 # Number of allowed compound states
# =======================

# MS2 / kernel related
W = 19 # Window size
t_MS2 = 30    # Time for Pol II to traverse MS2 probe (s)
deltaT = 20   # Time resolution (s)
kappa = t_MS2 / deltaT
# =======================

def main():
    # Reproducibility and warnings
    np.random.seed(SEED)
    np.seterr(divide='warn', invalid='warn', over='warn')  # 更易发现数值问题

    # ---- Read data
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    ms2_signals = genfromtxt(CSV_PATH, delimiter=',', skip_header=1)
    if ms2_signals.ndim != 2:
        raise ValueError(f"CSV content must be 2D, got shape {ms2_signals.shape}")

    # ---- Optional: strip a leading row-index column
    if HAS_ROW_INDEX_COL:
        data = ms2_signals[:, 1:]
    else:
        data = ms2_signals

    n_rows, n_cols = data.shape
    if n_cols <= META_COLS:
        raise ValueError(
            f"n_cols={n_cols} <= META_COLS={META_COLS}; please check META_COLS or the CSV format."
        )

    # ---- Compute per-trace mean on the time-series part ONLY
    traces_only = data[:, META_COLS:]
    per_trace_mean = np.nanmean(traces_only, axis=1)

    # ---- Drop weak traces (robust way)
    order = np.argsort(per_trace_mean)  # ascending
    drop_mask = np.zeros(n_rows, dtype=bool)
    if DROP_LOWEST_N_BY_MEAN and DROP_LOWEST_N_BY_MEAN > 0:
        drop_mask[order[:DROP_LOWEST_N_BY_MEAN]] = True
    if WEAK_MEAN_QUANTILE is not None:
        thr = np.nanquantile(per_trace_mean, WEAK_MEAN_QUANTILE)
        drop_mask |= (per_trace_mean <= thr)

    weak_signals_removed = data[~drop_mask, :]

    # ---- Cluster filter (positioned in metadata columns)
    # 注意：这里假设 cluster 列还在元数据区域内
    if CLUSTER_COL_AFTER_STRIP >= META_COLS:
        raise ValueError(
            f"CLUSTER_COL_AFTER_STRIP={CLUSTER_COL_AFTER_STRIP} must be < META_COLS={META_COLS}"
        )
    filtered_by_cluster = weak_signals_removed[
        weak_signals_removed[:, CLUSTER_COL_AFTER_STRIP] == CLUSTER_KEEP_VALUE
    ]
    if filtered_by_cluster.size == 0:
        raise ValueError("No traces left after cluster filtering; adjust filter settings.")

    # ---- Process signals (hand off meta+trace matrix; function will slice from cutoff=META_COLS)
    processed_signals = process_raw_data(filtered_by_cluster, META_COLS)

    # Basic introspection (兼容字典接口)
    if not isinstance(processed_signals, dict):
        raise TypeError("process_raw_data must return a dict for downstream HMM code.")
    if 'Processed Signals' not in processed_signals:
        raise KeyError("Missing key 'Processed Signals' in processed_signals.")
    n_traces = len(processed_signals['Processed Signals'])
    print(f"[INFO] Traces after preprocessing: {n_traces}")

    # ---- Plot a random trace (non-blocking save)
    if PLOT_FIG and n_traces > 0:
        idx = np.random.randint(0, n_traces)
        tr = np.asarray(processed_signals['Processed Signals'][idx]).ravel()
        y_min, y_max = np.nanmin(tr), np.nanmax(tr)
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            print("[WARN] Selected trace contains non-finite values; skipping plot.")
        else:
            # 防止 singular y-limits
            if y_max - y_min < 1e-9:
                pad = 1.0 if y_max == 0 else 0.05 * abs(y_max)
                y_min, y_max = y_min - pad, y_max + pad

            plt.figure(figsize=(10, 4))
            plt.plot(tr, label=f'Trace {idx}')
            plt.ylim([y_min, y_max])
            plt.xlabel('Time (frames)')
            plt.ylabel('Fluorescence')
            plt.title(f'Example Processed Fluorescence Trace (seed={SEED}, idx={idx})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_OUT, dpi=150)
            plt.close()
            print(f"[INFO] Saved example trace to {FIG_OUT}")

    # ---- HMM sanity checks
    compound_states = K ** W
    if PERMITTED_MEMORY > compound_states:
        print(f"[INFO] PERMITTED_MEMORY={PERMITTED_MEMORY} > 2**W={compound_states}, "
              f"clamping to {compound_states}.")
    permitted_memory_eff = min(PERMITTED_MEMORY, compound_states)

    print(f"[INFO] K={K}, W={W}, K**W={compound_states}, PERMITTED_MEMORY={permitted_memory_eff}, "
          f"kappa={kappa:.3f} (= t_MS2/deltaT)")

    # ---- Create and initialize HMM
    demoHMM = HMM(K, W, t_MS2, deltaT, kappa, compound_states, processed_signals)
    init_params = demoHMM.initialise_parameters()
    print("------------- Initial Parameters -------------")
    print("The initial parameters are:", init_params)
    print("----------------------------------------------")

    # ---- EM: “Method B”
    # 先用固定转移估计发射/噪声，再带先验全面训练
    param_priors = demoHMM.EM_fixed(init_params, N_STEPS_FIXED, n_traces,
                                    permitted_memory_eff, EPS * 10, SEED)
    print("------- Parameters after EM_fixed -------")
    print("The parameters after EM_fixed are:", param_priors)
    print("-----------------------------------------")
    learned_params = demoHMM.EM_with_priors(param_priors, N_STEPS_PRIORS, n_traces,
                                            permitted_memory_eff, EPS, SEED)
    print("------------- Learned Parameters -------------")
    print("The learned parameters are:", learned_params)
    print("----------------------------------------------")

    # ---- Export results
    export_dict = export_em_parameters(learned_params)
    results_df = pd.DataFrame(export_dict)
    out_csv = f'renew_result_{SEED}.csv'
    results_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved parameters to {out_csv}")
    # 可选打印
    with pd.option_context('display.max_columns', None):
        print(results_df.head())

if __name__ == "__main__":
    main()
