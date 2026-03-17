import numpy as np
from scipy.signal import fftconvolve


def estimate_time_offset(
        sig_head: np.ndarray,
        idl_head: np.ndarray,
        window_ps: int,
        n_frames: int,
) -> int:
    """估算 signal 与 idler 之间的全局时间偏移。

    Parameters
    ----------
    sig_head : np.ndarray  dtype=int64
        signal 文件头部 window_ps 时间范围内的时间戳
    idl_head : np.ndarray  dtype=int64
        idler 文件头部 window_ps 时间范围内的时间戳
    window_ps : int
        互相关使用的时间窗口长度（ps）
    n_frames : int
        直方图 bin 数

    Returns
    -------
    int
        timeDiff（ps）：idler + timeDiff ≈ signal 的时间基准
    """
    start = int(sig_head[0])

    # ── signal 直方图 ───────────────────────────────────────────────────
    sig_hist, sig_edges = np.histogram(
        sig_head,
        bins=n_frames,
        range=(start, start + window_ps),
    )

    # ── idler 直方图（先对齐到 signal 起点）─────────────────────────────
    idl_offset = int(idl_head[0]) - start
    idl_hist, _ = np.histogram(
        idl_head - idl_offset,
        bins=n_frames,
        range=(start, start + window_ps),
    )

    # ── FFT 互相关 ──────────────────────────────────────────────────────
    corr = fftconvolve(sig_hist.astype(np.float64),
                       idl_hist[::-1].astype(np.float64),
                       mode='full')

    # 结果中心在 len-1 处
    center = len(sig_hist) - 1
    idx = int(np.argmax(corr)) - center

    bin_width_ps = window_ps / n_frames
    time_diff = int(round(idx * bin_width_ps)) - idl_offset

    print(f"[correlation] bin_width={bin_width_ps:.2f} ps, "
          f"argmax_offset={idx}, idl_offset={idl_offset}, "
          f"timeDiff={time_diff} ps")
    return time_diff
