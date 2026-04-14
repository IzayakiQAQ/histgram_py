import numpy as np
from scipy.signal import fftconvolve


def estimate_time_offset(
        sig_head: np.ndarray,
        idl_head: np.ndarray,
        window_ps: int,
        n_frames: int,
) -> int:
    """Estimate signal-idler time offset with the original FFT cross-correlation."""
    start = int(sig_head[0])

    sig_hist, _ = np.histogram(
        sig_head,
        bins=n_frames,
        range=(start, start + window_ps),
    )

    idl_offset = int(idl_head[0]) - start
    idl_hist, _ = np.histogram(
        idl_head - idl_offset,
        bins=n_frames,
        range=(start, start + window_ps),
    )

    sh = sig_hist.astype(np.float64)
    ih = idl_hist[::-1].astype(np.float64)
    sh -= sh.mean()
    ih -= ih.mean()

    corr = fftconvolve(sh, ih, mode="full")

    center = len(sig_hist) - 1
    idx = int(np.argmax(corr)) - center

    bin_width_ps = window_ps / n_frames
    time_diff = int(round(idx * bin_width_ps)) - idl_offset

    print(
        f"[correlation] bin_width={bin_width_ps:.2f} ps, "
        f"argmax_offset={idx}, idl_offset={idl_offset}, "
        f"timeDiff={time_diff} ps"
    )
    return time_diff
