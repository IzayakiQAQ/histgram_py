import numpy as np
from scipy.optimize import curve_fit


def _gaussian(x, baseline, center, sigma, amplitude):
    """四参数高斯模型。"""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + baseline


def coincidence_peak(
        signal: np.ndarray,
        idler: np.ndarray,
        bin_width: int,
        bin_num: int,
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

    # 3. 直方图生成
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

    return center - window / 2
