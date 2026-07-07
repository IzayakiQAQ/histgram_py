import numpy as np
from scipy.optimize import curve_fit


def _gaussian(x, baseline, center, sigma, amplitude):
    """Four-parameter Gaussian model."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + baseline


def coincidence_peak(
        signal: np.ndarray,
        idler: np.ndarray,
        bin_width: int,
        bin_num: int,
        save_dir: str = None,
        index: int = 0,
        fit_half_window_bins: int = 80,
        save_hist_bin_width_ps: int = 1,
        save_hist_center_ps: int = None,
        save_hist_points: int = 65536,
) -> float:
    """Estimate the coincidence peak center for one time slice."""
    window = int(bin_width) * int(bin_num)

    # Center the coincidence window around each signal timestamp.
    signal_shifted = signal - window // 2

    lo = np.searchsorted(idler, signal_shifted, side='left')
    hi = np.searchsorted(idler, signal_shifted + window, side='right')

    parts = [
        idler[l:h] - s
        for s, l, h in zip(signal_shifted, lo, hi)
        if h > l
    ]

    if not parts:
        return 0.0

    diffs = np.concatenate(parts)
    if diffs.size == 0:
        return 0.0

    hist, edges = np.histogram(diffs, bins=bin_num, range=(0, window))
    if hist.max() == 0:
        return 0.0

    x = edges[:-1].astype(np.float64)
    peak_idx = int(np.argmax(hist))

    # Fit only around the strongest coarse bin to avoid global-noise bias.
    half_bins = max(int(fit_half_window_bins), 1)
    left = max(peak_idx - half_bins, 0)
    right = min(peak_idx + half_bins + 1, hist.size)
    xx = x[left:right]
    yy = hist[left:right].astype(np.float64)

    center0 = float(x[peak_idx])
    baseline0 = float(np.median(yy))
    amplitude0 = float(max(yy.max() - baseline0, 1.0))
    sigma0 = max(float(bin_width) * 2.0, 50.0)

    try:
        popt, _ = curve_fit(
            _gaussian,
            xx,
            yy,
            p0=[baseline0, center0, sigma0, amplitude0],
            bounds=(
                [0.0, center0 - half_bins * float(bin_width), 1.0, 0.0],
                [np.inf, center0 + half_bins * float(bin_width), float(window), np.inf],
            ),
            maxfev=50_000,
            ftol=1e-9,
            xtol=1e-9,
        )
        center = float(popt[1])
    except Exception:
        center = center0

    if save_dir is not None:
        import os

        valid_diffs = diffs[(diffs >= 0) & (diffs < window)]
        hist_1ps = np.bincount(valid_diffs, minlength=window)
        save_hist_bin_width_ps = max(int(save_hist_bin_width_ps), 1)
        # `center` is already the absolute peak position inside [0, window).
        center_abs_1ps = int(round(center))
        center_abs_1ps = max(0, min(center_abs_1ps, window - 1))
        local_half_width_ps = max(int(bin_width) * 10, 2000)
        local_left = max(center_abs_1ps - local_half_width_ps, 0)
        local_right = min(center_abs_1ps + local_half_width_ps + 1, window)
        if local_right > local_left:
            local_peak = int(np.argmax(hist_1ps[local_left:local_right]))
            center_abs_1ps = local_left + local_peak

        if save_hist_bin_width_ps == 1:
            hist_to_save = hist_1ps
            time_to_save = np.arange(hist_1ps.size, dtype=np.int64)
            center_idx = center_abs_1ps
        else:
            n_full = (hist_1ps.size // save_hist_bin_width_ps) * save_hist_bin_width_ps
            if n_full <= 0:
                return center - window / 2
            hist_trimmed = hist_1ps[:n_full]
            hist_to_save = hist_trimmed.reshape(-1, save_hist_bin_width_ps).sum(axis=1)
            time_to_save = np.arange(hist_to_save.size, dtype=np.int64) * save_hist_bin_width_ps
            center_idx = center_abs_1ps // save_hist_bin_width_ps

        if save_hist_center_ps is not None:
            fixed_center = int(round(save_hist_center_ps))
            fixed_center = max(0, min(fixed_center, window - 1))
            center_idx = fixed_center // save_hist_bin_width_ps

        target_points = max(int(save_hist_points), 1)
        half_points = target_points // 2

        left = center_idx - half_points + 1
        right = center_idx + half_points + 1

        total_len = len(hist_to_save)
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

        select_hist = hist_to_save[left:right]
        select_time = time_to_save[left:right]

        out_path = os.path.join(save_dir, f'hist_raw_{index:05d}.csv')
        data_to_save = np.column_stack((select_time, select_hist))
        np.savetxt(out_path, data_to_save, delimiter=",", fmt="%d")

    return center - window / 2
