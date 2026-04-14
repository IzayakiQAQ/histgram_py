import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

try:
    import config1 as cfg
except ImportError:
    import config as cfg


def _gaussian(x, baseline, center, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + baseline


def _smooth_counts(counts: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or counts.size < kernel_size:
        return counts.astype(np.float64)
    kernel = np.ones(int(kernel_size), dtype=np.float64) / float(kernel_size)
    return np.convolve(counts.astype(np.float64), kernel, mode="same")


def fit_left_peak(
    hist_path: Path,
    *,
    smooth_kernel: int,
    min_distance_ps: int,
    prominence_frac: float,
    min_prominence: float,
    fit_half_window_ps: int,
) -> dict:
    data = np.loadtxt(hist_path, delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected histogram format: {hist_path}")

    x = data[:, 0]
    y = data[:, 1]

    if x.size == 0 or np.max(y) <= 0:
        return {
            "center_ps": 0.0,
            "raw_peak_ps": 0.0,
            "peak_count": 0.0,
            "fit_ok": False,
        }

    y_s = _smooth_counts(y, smooth_kernel)
    prominence = max(float(np.max(y_s)) * float(prominence_frac), float(min_prominence))

    peak_idxs, props = find_peaks(
        y_s,
        distance=max(int(min_distance_ps), 1),
        prominence=prominence,
    )

    if peak_idxs.size == 0:
        peak_idx = int(np.argmax(y_s))
        peak_count = float(y[peak_idx])
    else:
        peak_idx = int(np.min(peak_idxs))
        peak_count = float(y[peak_idx])

    center0 = float(x[peak_idx])
    left = max(peak_idx - int(fit_half_window_ps), 0)
    right = min(peak_idx + int(fit_half_window_ps) + 1, x.size)

    xx = x[left:right]
    yy = y[left:right]
    fit_ok = False
    center = center0

    if xx.size >= 6 and np.max(yy) > 0:
        baseline0 = float(np.median(yy))
        amplitude0 = float(max(np.max(yy) - baseline0, 1.0))
        sigma0 = max(30.0, float(fit_half_window_ps) / 3.0)
        p0 = [baseline0, center0, sigma0, amplitude0]
        bounds = (
            [0.0, center0 - fit_half_window_ps, 1.0, 0.0],
            [np.inf, center0 + fit_half_window_ps, fit_half_window_ps * 2.0, np.inf],
        )
        try:
            popt, _ = curve_fit(
                _gaussian,
                xx,
                yy,
                p0=p0,
                bounds=bounds,
                maxfev=50_000,
            )
            center = float(popt[1])
            fit_ok = True
        except Exception:
            center = center0

    return {
        "center_ps": center,
        "raw_peak_ps": center0,
        "peak_count": peak_count,
        "fit_ok": fit_ok,
    }


def _sorted_hist_files(hist_dir: Path) -> list[Path]:
    return sorted(hist_dir.glob("hist_raw_*.csv"))


def _default_root_dir() -> Path:
    out_root = getattr(cfg, "OUTPUT_DIR", None) or cfg.DIR
    return Path(out_root)


def parse_args():
    root = _default_root_dir()
    parser = argparse.ArgumentParser(
        description="Reprocess saved histogram CSVs by selecting the leftmost significant peak."
    )
    parser.add_argument("--single-pair-dir", default="", help="If set, only reprocess this histogram folder and write one peak column.")
    parser.add_argument("--single-label", default="leftpeak_ps", help="Column label suffix used in single-pair mode.")
    parser.add_argument("--root-dir", default=str(root), help="Root directory containing pair histogram folders.")
    parser.add_argument("--pair0-dir", default="pair0_histograms_raw_ps", help="Pair 0 histogram folder name or absolute path.")
    parser.add_argument("--pair1-dir", default="pair1_histograms_raw_1ps", help="Pair 1 histogram folder name or absolute path.")
    parser.add_argument("--out", default="hcf_leftpeak.csv", help="Output CSV path or filename under root-dir.")
    parser.add_argument("--debug-out", default="hcf_leftpeak_debug.csv", help="Debug CSV path or filename under root-dir.")
    parser.add_argument("--smooth-kernel", type=int, default=7, help="Moving-average kernel size for peak detection.")
    parser.add_argument("--min-distance-ps", type=int, default=200, help="Minimum separation between detected peaks.")
    parser.add_argument("--prominence-frac", type=float, default=0.18, help="Prominence threshold as a fraction of per-slice max.")
    parser.add_argument("--min-prominence", type=float, default=5.0, help="Minimum absolute prominence threshold.")
    parser.add_argument("--fit-half-window-ps", type=int, default=180, help="Half-window width for local Gaussian fitting.")
    parser.add_argument(
        "--split-step-ps",
        type=int,
        default=int(cfg.SPLIT_STEP_PS),
        help="Slice duration in ps; used to generate time_s.",
    )
    parser.add_argument("--max-slices", type=int, default=0, help="Only process the first N slices; 0 means all.")
    return parser.parse_args()


def _resolve_under_root(root_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root_dir / p


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    single_pair_dir = _resolve_under_root(root_dir, args.single_pair_dir) if args.single_pair_dir else None
    pair0_dir = _resolve_under_root(root_dir, args.pair0_dir)
    pair1_dir = _resolve_under_root(root_dir, args.pair1_dir)
    out_path = _resolve_under_root(root_dir, args.out)
    debug_path = _resolve_under_root(root_dir, args.debug_out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    if single_pair_dir is not None:
        files = _sorted_hist_files(single_pair_dir)
        if not files:
            raise FileNotFoundError(f"No histogram files found in {single_pair_dir}")
        n = len(files)
        if args.max_slices > 0:
            n = min(n, int(args.max_slices))

        with open(out_path, "w", newline="", encoding="utf-8") as fout, open(
            debug_path, "w", newline="", encoding="utf-8"
        ) as fdbg:
            writer = csv.writer(fout)
            dbg = csv.writer(fdbg)
            writer.writerow(["time_s", args.single_label])
            dbg.writerow(
                [
                    "time_s",
                    "file",
                    "center_ps",
                    "raw_peak_ps",
                    "peak_count",
                    "fit_ok",
                ]
            )

            for i in range(n):
                res = fit_left_peak(
                    files[i],
                    smooth_kernel=args.smooth_kernel,
                    min_distance_ps=args.min_distance_ps,
                    prominence_frac=args.prominence_frac,
                    min_prominence=args.min_prominence,
                    fit_half_window_ps=args.fit_half_window_ps,
                )
                time_s = (i + 1) * args.split_step_ps / 1e12
                writer.writerow([f"{time_s:.6f}", f"{res['center_ps']:.6f}"])
                dbg.writerow(
                    [
                        f"{time_s:.6f}",
                        files[i].name,
                        f"{res['center_ps']:.6f}",
                        f"{res['raw_peak_ps']:.6f}",
                        f"{res['peak_count']:.6f}",
                        int(res["fit_ok"]),
                    ]
                )
    else:
        files0 = _sorted_hist_files(pair0_dir)
        files1 = _sorted_hist_files(pair1_dir)
        if not files0:
            raise FileNotFoundError(f"No histogram files found in {pair0_dir}")
        if not files1:
            raise FileNotFoundError(f"No histogram files found in {pair1_dir}")

        n = min(len(files0), len(files1))
        if args.max_slices > 0:
            n = min(n, int(args.max_slices))

        with open(out_path, "w", newline="", encoding="utf-8") as fout, open(
            debug_path, "w", newline="", encoding="utf-8"
        ) as fdbg:
            writer = csv.writer(fout)
            dbg = csv.writer(fdbg)

            writer.writerow(["time_s", "ch1-ch4_leftpeak_ps", "ch2-ch3_leftpeak_ps", "clock_correction_ps"])
            dbg.writerow(
                [
                    "time_s",
                    "pair0_file",
                    "pair0_center_ps",
                    "pair0_raw_peak_ps",
                    "pair0_peak_count",
                    "pair0_fit_ok",
                    "pair1_file",
                    "pair1_center_ps",
                    "pair1_raw_peak_ps",
                    "pair1_peak_count",
                    "pair1_fit_ok",
                ]
            )

            for i in range(n):
                r0 = fit_left_peak(
                    files0[i],
                    smooth_kernel=args.smooth_kernel,
                    min_distance_ps=args.min_distance_ps,
                    prominence_frac=args.prominence_frac,
                    min_prominence=args.min_prominence,
                    fit_half_window_ps=args.fit_half_window_ps,
                )
                r1 = fit_left_peak(
                    files1[i],
                    smooth_kernel=args.smooth_kernel,
                    min_distance_ps=args.min_distance_ps,
                    prominence_frac=args.prominence_frac,
                    min_prominence=args.min_prominence,
                    fit_half_window_ps=args.fit_half_window_ps,
                )

                time_s = (i + 1) * args.split_step_ps / 1e12
                clock_correction = (r0["center_ps"] - r1["center_ps"]) / 2.0

                writer.writerow(
                    [
                        f"{time_s:.6f}",
                        f"{r0['center_ps']:.6f}",
                        f"{r1['center_ps']:.6f}",
                        f"{clock_correction:.6f}",
                    ]
                )
                dbg.writerow(
                    [
                        f"{time_s:.6f}",
                        files0[i].name,
                        f"{r0['center_ps']:.6f}",
                        f"{r0['raw_peak_ps']:.6f}",
                        f"{r0['peak_count']:.6f}",
                        int(r0["fit_ok"]),
                        files1[i].name,
                        f"{r1['center_ps']:.6f}",
                        f"{r1['raw_peak_ps']:.6f}",
                        f"{r1['peak_count']:.6f}",
                        int(r1["fit_ok"]),
                    ]
                )

    print(f"[done] wrote {out_path}")
    print(f"[done] wrote {debug_path}")
    print(f"[done] processed {n} slices")


if __name__ == "__main__":
    main()
