# -*- coding: utf-8 -*-
"""
Renewed on Oct 2025

Original: Jon (2020)
Edited by: Jingyu (2025)
"""
import numpy as np

def _trim_and_fill_1d(signal_1d: np.ndarray):
    """裁掉首尾 NaN；对内部 NaN 线性插值；若仅1个有效点则常数填充；若无有效点返回 None。"""
    sig = np.asarray(signal_1d, dtype=float)
    finite = np.isfinite(sig)
    if not finite.any():
        return None

    idx = np.flatnonzero(finite)
    start, end = int(idx[0]), int(idx[-1])
    seg = sig[start:end + 1].astype(float, copy=True)

    miss = ~np.isfinite(seg)
    if miss.any():
        x = np.arange(seg.size)
        good = ~miss
        if good.sum() == 1:
            # 只有一个有效点：用该值填满
            seg[miss] = seg[good][0]
        else:
            seg[miss] = np.interp(x[miss], x[good], seg[good])
    return seg

def process_raw_data(signals, cutoff):
    """
    Parameters
    ----------
    signals : 2D array-like
        行为不同轨迹，列为 meta + trace
    cutoff : int
        从该列起（含）为轨迹数据

    Returns
    -------
    dict with keys:
      - 'Processed Signals' : list of 1D np.ndarray(float64)
      - 'Matrix Mean'       : float（逐轨迹均值再取全体均值）
      - 'Matrix Max'        : float（逐轨迹最大值的全体最大）
      - 'Signal Lengths'    : np.ndarray(unique lengths, int)
      - 'Kept Indices'      : np.ndarray(int) 保留下来的原始行索引
      - 'Dropped Empty Indices' : np.ndarray(int) 被丢弃（全 NaN/空）的行索引
    """
    arr = np.asarray(signals, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"`signals` must be 2D, got shape {arr.shape}")

    cutoff = int(cutoff)
    if not (0 <= cutoff < arr.shape[1]):
        raise ValueError(f"`cutoff` must be in [0, {arr.shape[1]-1}], got {cutoff}")

    n_rows = arr.shape[0]
    processed = []
    kept_idx = []
    dropped_empty = []

    means = []
    maxs = []
    lengths = []

    for u in range(n_rows):
        raw = arr[u, cutoff:]  # 仅时序部分
        seg = _trim_and_fill_1d(raw)
        if seg is None or seg.size == 0 or not np.isfinite(seg).all():
            dropped_empty.append(u)
            continue

        seg = np.ascontiguousarray(seg, dtype=float)  # 统一 1D float64
        processed.append(seg)
        kept_idx.append(u)

        means.append(float(np.mean(seg)))
        maxs.append(float(np.max(seg)))
        lengths.append(int(seg.size))

    output = {
        'Processed Signals': processed,
        'Matrix Mean': float(np.mean(means)) if means else np.nan,
        'Matrix Max': float(np.max(maxs)) if maxs else np.nan,
        'Signal Lengths': np.unique(lengths) if lengths else np.array([], dtype=int),
        'Kept Indices': np.array(kept_idx, dtype=int),
        'Dropped Empty Indices': np.array(dropped_empty, dtype=int),
    }
    return output
