import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def _gaussian(x, baseline, center, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + baseline


def coincidence_two_peaks(
    signal: np.ndarray,
    idler: np.ndarray,
    bin_width: int,
    bin_num: int,
    *,
    min_separation_ps: int | None = None,
    fit_half_window_bins: int = 80,
    fit_half_window_ps_1ps: int | None = None,
    save_dir: str | None = None,
    index: int = 0,
    prefix: str = "hist",
    target_points: int = 65536,
    debug: bool = False,
) -> tuple[float, float]:
    """
    Like coincidence.coincidence_peak(), but for a clear double-peak structure.

    Returns
    -------
    (peak1_center_ps, peak2_center_ps) as float (ps), each expressed as:
        fitted_center - window/2
    so the meaning matches coincidence_peak()'s return convention.
    """
    window = int(bin_width) * int(bin_num)

    signal_shifted = signal - window // 2

    lo = np.searchsorted(idler, signal_shifted, side="left")
    hi = np.searchsorted(idler, signal_shifted + window, side="right")

    parts = [
        idler[l:h] - s
        for s, l, h in zip(signal_shifted, lo, hi)
        if h > l
    ]

    if not parts:
        return 0.0, 0.0

    diffs = np.concatenate(parts)
    if diffs.size == 0:
        return 0.0, 0.0

    # Build 1ps-resolution histogram (window length is typically 1e6, OK in memory).
    valid_diffs = diffs[(diffs >= 0) & (diffs < window)]
    if valid_diffs.size == 0:
        return 0.0, 0.0
    hist_1ps = np.bincount(valid_diffs.astype(np.int64), minlength=window)
    if hist_1ps.max() == 0:
        return 0.0, 0.0

    # If caller didn't provide a 1ps fitting window, derive from coarse-bin window.
    if fit_half_window_ps_1ps is None:
        fit_half_window_ps_1ps = int(max(1, int(fit_half_window_bins) * int(bin_width)))

    if min_separation_ps is None:
        min_separation_ps = max(5 * int(bin_width), 1)

    # Smooth a bit for robust peak picking (moving average).
    y1 = hist_1ps.astype(np.float64)
    k = 7
    if y1.size >= k:
        kernel = np.ones(k, dtype=np.float64) / float(k)
        y_s = np.convolve(y1, kernel, mode="same")
    else:
        y_s = y1

    min_sep_1ps = max(int(min_separation_ps), 1)

    # Peak picking on 1ps histogram
    peak_idxs, props = find_peaks(y_s, distance=min_sep_1ps)
    if peak_idxs.size == 0:
        c = float(int(np.argmax(y_s)))
        out = c - window / 2.0
        return out, out

    # Rank peaks by height (simple and robust for your use case)
    peak_idxs = np.array(peak_idxs, dtype=int)
    order = np.argsort(y_s[peak_idxs])[::-1]
    peak_idxs = peak_idxs[order]

    # Pick the top two distinct peaks; if only one exists, duplicate it
    top = peak_idxs[:2]
    if top.size == 1:
        top = np.array([top[0], top[0]], dtype=int)

    centers = []
    for pidx in top:
        left = max(int(pidx) - int(fit_half_window_ps_1ps), 0)
        right = min(int(pidx) + int(fit_half_window_ps_1ps) + 1, int(hist_1ps.size))
        xx = np.arange(left, right, dtype=np.float64)  # 1ps grid
        yy = hist_1ps[left:right].astype(np.float64)

        if xx.size < 6 or yy.max() <= 0:
            centers.append(float(pidx))
            continue

        baseline0 = float(np.median(yy))
        amplitude0 = float(max(yy.max() - baseline0, 1.0))
        center0 = float(pidx)
        sigma0 = max(50.0, float(bin_width) * 2.0)  # in ps

        p0 = [baseline0, center0, sigma0, amplitude0]

        try:
            popt, _ = curve_fit(
                _gaussian,
                xx,
                yy,
                p0=p0,
                maxfev=50_000,
                ftol=1e-9,
                xtol=1e-9,
            )
            centers.append(float(popt[1]))
        except Exception:
            centers.append(float(center0))

    centers.sort()

    c1 = centers[0] - window / 2.0
    c2 = centers[1] - window / 2.0

    # Optional: save 1ps-resolution histogram segments around each peak.
    if save_dir is not None:
        import os

        os.makedirs(save_dir, exist_ok=True)

        def _save_one(center_rel_ps: float, peak_name: str):
            center_abs = int(round(center_rel_ps + window / 2.0))
            center_abs = max(0, min(center_abs, window - 1))

            tp = int(target_points)
            if tp <= 0:
                return
            half = tp // 2

            left = center_abs - half + 1
            right = center_abs + half + 1

            total_len = int(hist_1ps.size)
            if total_len <= tp:
                left = 0
                right = total_len
            else:
                if left < 0:
                    left = 0
                    right = tp
                elif right > total_len:
                    right = total_len
                    left = total_len - tp

            select_hist = hist_1ps[left:right]
            select_time = np.arange(left, right, dtype=np.int64)

            out_path = os.path.join(save_dir, f"{prefix}_{peak_name}_{index:05d}.csv")
            data_to_save = np.column_stack((select_time, select_hist))
            np.savetxt(out_path, data_to_save, delimiter=",", fmt="%d")

        _save_one(c1, "p1")
        _save_one(c2, "p2")

        if debug:
            # Save a small debug text file for quick inspection.
            dbg_path = os.path.join(save_dir, f"{prefix}_dbg_{index:05d}.txt")
            with open(dbg_path, "w", encoding="utf-8") as df:
                df.write(f"window_ps={window}\n")
                df.write(f"min_separation_ps={min_separation_ps}\n")
                df.write(f"fit_half_window_ps_1ps={fit_half_window_ps_1ps}\n")
                df.write(f"raw_peak_idxs_1ps={top.tolist()}\n")
                df.write(f"centers_abs_1ps={centers}\n")
                df.write(f"centers_rel_ps={(c1, c2)}\n")

    return (c1, c2)
