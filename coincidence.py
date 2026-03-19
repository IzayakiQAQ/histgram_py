import numpy as np
from scipy.optimize import curve_fit


def _gaussian(x, baseline, center, sigma, amplitude):
    """四参数高斯模型。"""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + baseline


from typing import Tuple

def coincidence_peak(
        signal: np.ndarray,
        idler: np.ndarray,
        bin_width: int,
        bin_num: int,
        save_dir: str = None,
        index: int = 0,
) -> float:
    """对一个时间片段的时间戳做符合计数并拟合高斯峰。

    Parameters
    ----------
    signal : np.ndarray  dtype=int64，已排序
    idler  : np.ndarray  dtype=int64，已排序，已加偏移修正
    bin_width : int  直方图 bin 宽（ps）
    bin_num   : int  直方图 bin 数

    Returns
    -------
    float
        拟合得到的峰值中心位置偏移量（ps）
    """
    window = bin_width * bin_num  

    # 把符合窗口居中到 signal 周围
    signal_shifted = signal - window // 2

    # 1. 批量搜索：每个 signal 对应 idler 的起止索引
    lo = np.searchsorted(idler, signal_shifted,          side='left')
    hi = np.searchsorted(idler, signal_shifted + window, side='right')

    # 2. 收集时间差
    parts = [idler[l:h] - s
             for s, l, h in zip(signal_shifted, lo, hi)
             if h > l]

    if not parts:
        return 0.0

    diffs = np.concatenate(parts)
    if diffs.size == 0:
        return 0.0

    # 3. 直方图生成 (拟通用宽度，用于寻找大范围高斯峰)
    hist, edges = np.histogram(diffs, bins=bin_num, range=(0, window))

    if hist.max() == 0:
        return 0.0

    # 4. 高斯拟合
    x = edges[:-1].astype(np.float64)   # bin 左边沿
    peak_idx = int(np.argmax(hist))
    
    # 初始参数估算 [baseline, center, sigma, amplitude]
    p0 = [
        float(hist[0]),                   
        float(x[peak_idx]),               
        100.0,                            
        float(hist[peak_idx] - hist[0]),  
    ]

    try:
        popt, _ = curve_fit(
            _gaussian, x, hist.astype(np.float64),
            p0=p0,
            maxfev=50_000,
            ftol=1e-9, xtol=1e-9,   
        )
        center = float(popt[1])
    except RuntimeError:
        # 拟合失败时退化到直方图峰值位置
        center = float(x[peak_idx])

    # ── 5. 生成并保存 1ps 分辨率的裁剪直方图 (65536个点) ────
    if save_dir is not None:
        import os
        # 提取落入 [0, window) 内的有效 diffs 做 bincount
        valid_diffs = diffs[(diffs >= 0) & (diffs < window)]
        hist_1ps = np.bincount(valid_diffs, minlength=window)
        
        target_points = 65536
        half_points = target_points // 2
        
        # 找到 1ps 精度下的最大峰值点
        max_idx_1ps = int(np.argmax(hist_1ps))
        
        left = max_idx_1ps - half_points + 1
        right = max_idx_1ps + half_points + 1
        
        total_len = len(hist_1ps)
        if total_len < target_points:
            left = 0
            right = total_len
        else:
            if left < 0:
                left = 0
                right = target_points
            elif right > total_len:
                right = total_len
                left = total_len - target_points
                
        select_hist = hist_1ps[left:right]
        select_time = np.arange(left, right)
        
        out_path = os.path.join(save_dir, f'hist_raw_{index:05d}.csv')
        data_to_save = np.column_stack((select_time, select_hist))
        np.savetxt(out_path, data_to_save, delimiter=",", fmt="%d")

    return center - window / 2
