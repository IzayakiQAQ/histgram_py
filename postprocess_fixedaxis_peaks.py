import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


def _load_job_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    defaults = data.get("defaults", {})
    jobs = data.get("jobs", [])
    if not jobs:
        raise ValueError(f"No jobs in {path}")
    return defaults, jobs[0]


def _pair_value(pair: dict[str, Any], defaults: dict[str, Any], key: str, fallback: Any) -> Any:
    return pair.get(key, defaults.get(key, fallback))


def _gaussian(x: np.ndarray, baseline: float, amplitude: float, center: float, sigma: float) -> np.ndarray:
    return baseline + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _rebin(t: np.ndarray, y: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    factor = max(int(factor), 1)
    if factor == 1:
        return t, y
    n_full = (y.size // factor) * factor
    if n_full <= 0:
        return t, y
    tt = t[:n_full].reshape(-1, factor).mean(axis=1)
    yy = y[:n_full].reshape(-1, factor).sum(axis=1)
    return tt, yy


def _gaussian_center_from_hist(
    hist_path: Path,
    *,
    box_bins: int,
    fit_bin_factor: int,
    fit_half_width_ps: float,
    center_bound_ps: float,
    sigma_min_ps: float,
    sigma_max_ps: float,
) -> tuple[float, dict[str, Any]]:
    data = np.loadtxt(str(hist_path), delimiter=",", dtype=np.float64)
    t = data[:, 0]
    y = data[:, 1]

    box_bins = max(1, min(int(box_bins), y.size))

    csum = np.cumsum(np.r_[0.0, y])
    rolling = csum[box_bins:] - csum[:-box_bins]
    best_left = int(np.argmax(rolling))
    best_sum = float(rolling[best_left])
    coarse_center_index = best_left + (box_bins - 1) / 2.0
    coarse_center = float(np.interp(coarse_center_index, np.arange(t.size), t))

    fit_t, fit_y = _rebin(t, y, fit_bin_factor)
    mask = (fit_t >= coarse_center - fit_half_width_ps) & (fit_t <= coarse_center + fit_half_width_ps)
    xx = fit_t[mask]
    yy = fit_y[mask]
    if xx.size < 8:
        raise ValueError(f"Not enough fit points for {hist_path}")

    baseline0 = float(np.percentile(yy, 10))
    amplitude0 = float(max(yy.max() - baseline0, 1.0))
    sigma0 = float(min(max(fit_half_width_ps / 3.0, sigma_min_ps), sigma_max_ps))

    fit_success = 1.0
    fit_error = ""
    try:
        popt, _ = curve_fit(
            _gaussian,
            xx,
            yy,
            p0=[baseline0, amplitude0, coarse_center, sigma0],
            bounds=(
                [0.0, 0.0, coarse_center - center_bound_ps, sigma_min_ps],
                [np.inf, np.inf, coarse_center + center_bound_ps, sigma_max_ps],
            ),
            maxfev=50_000,
            ftol=1e-9,
            xtol=1e-9,
        )
        baseline, amplitude, center, sigma = [float(v) for v in popt]
    except Exception as exc:
        fit_success = 0.0
        fit_error = type(exc).__name__
        baseline, amplitude, center, sigma = baseline0, amplitude0, coarse_center, sigma0

    fitted = _gaussian(xx, baseline, amplitude, center, sigma)
    residual_rms = float(np.sqrt(np.mean((yy - fitted) ** 2))) if xx.size else float("nan")

    return center, {
        "center_hist_ps": center,
        "coarse_center_hist_ps": coarse_center,
        "sigma_ps": sigma,
        "amplitude": amplitude,
        "baseline": baseline,
        "fit_success": fit_success,
        "residual_rms": residual_rms,
        "box_sum": best_sum,
        "fit_error": fit_error,
        "max_bin_count": float(y.max()),
        "total_count": float(y.sum()),
    }


def _fit_hist_task(task: tuple[str, int, int, float, float, float, float]) -> tuple[float, dict[str, Any]]:
    hist_path, box_bins, fit_bin_factor, fit_half_width_ps, center_bound_ps, sigma_min_ps, sigma_max_ps = task
    return _gaussian_center_from_hist(
        Path(hist_path),
        box_bins=box_bins,
        fit_bin_factor=fit_bin_factor,
        fit_half_width_ps=fit_half_width_ps,
        center_bound_ps=center_bound_ps,
        sigma_min_ps=sigma_min_ps,
        sigma_max_ps=sigma_max_ps,
    )


def _read_times(old_csv: Path) -> list[float]:
    with open(old_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [float(row["time_s"]) for row in reader]


def postprocess(args: argparse.Namespace) -> None:
    job_dir = Path(args.job_dir)
    defaults, job = _load_job_file(Path(args.job_file))

    bin_width_ps = int(defaults.get("bin_width_ps", 20))
    bin_num = int(defaults.get("bin_num", 10000))
    window_half_ps = float(bin_width_ps * bin_num) / 2.0

    times = _read_times(job_dir / "singlepeak_peaks.csv")
    pair_results: list[tuple[str, list[float], list[dict[str, Any]]]] = []
    workers = max(1, int(args.workers))

    for pair in job["pairs"]:
        label = pair["label"]
        hist_width_ps = int(_pair_value(pair, defaults, "save_hist_bin_width_ps", 1))
        hist_dir = job_dir / f"{label}_histograms_raw_{hist_width_ps}ps"
        time_diff_ps = float(pair["time_diff_ps"])

        box_bins = max(1, int(round(args.box_width_ps / hist_width_ps)))
        fit_bin_factor = max(1, int(round(args.fit_bin_width_ps / hist_width_ps)))
        files = sorted(hist_dir.glob("hist_raw_*.csv"))
        if not files:
            raise FileNotFoundError(f"No hist_raw files in {hist_dir}")

        tasks = [
            (
                str(fp),
                box_bins,
                fit_bin_factor,
                args.fit_half_width_ps,
                args.center_bound_ps,
                args.sigma_min_ps,
                args.sigma_max_ps,
            )
            for fp in files
        ]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                fitted = list(executor.map(_fit_hist_task, tasks, chunksize=16))
        else:
            fitted = [_fit_hist_task(task) for task in tasks]

        peaks: list[float] = []
        qualities: list[dict[str, Any]] = []
        for center_hist_ps, quality in fitted:
            corrected_peak_ps = center_hist_ps - window_half_ps - time_diff_ps
            peaks.append(corrected_peak_ps)
            qualities.append(quality)

        pair_results.append((label, peaks, qualities))

    n = min(len(times), *(len(values) for _, values, _ in pair_results))
    out_csv = job_dir / args.output_name
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["time_s"] + [f"{label}_peak_ps" for label, _, _ in pair_results]
        if len(pair_results) >= 2:
            header.append("clock_correction_ps")
        writer.writerow(header)
        arrays = [np.asarray(values[:n], dtype=np.float64) for _, values, _ in pair_results]
        for i in range(n):
            row = [f"{times[i]:.6f}"] + [f"{float(arr[i]):.6f}" for arr in arrays]
            if len(arrays) >= 2:
                row.append(f"{float((arrays[0][i] - arrays[1][i]) / 2.0):.6f}")
            writer.writerow(row)

    quality_csv = job_dir / args.quality_name
    with open(quality_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "time_s",
            "pair",
            "center_hist_ps",
            "coarse_center_hist_ps",
            "sigma_ps",
            "amplitude",
            "baseline",
            "fit_success",
            "residual_rms",
            "box_sum",
            "fit_error",
            "max_bin_count",
            "total_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, _, qualities in pair_results:
            for i, quality in enumerate(qualities[:n]):
                writer.writerow({"time_s": f"{times[i]:.6f}", "pair": label, **quality})

    print(f"[gaussian] wrote {out_csv}")
    print(f"[gaussian] wrote {quality_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess fixed-axis saved histograms with a broad-window Gaussian peak fit."
    )
    parser.add_argument("--job-file", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--box-width-ps", type=float, default=10000.0)
    parser.add_argument("--fit-bin-width-ps", type=float, default=20.0)
    parser.add_argument("--fit-half-width-ps", type=float, default=25000.0)
    parser.add_argument("--center-bound-ps", type=float, default=15000.0)
    parser.add_argument("--sigma-min-ps", type=float, default=500.0)
    parser.add_argument("--sigma-max-ps", type=float, default=30000.0)
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--output-name", default="singlepeak_peaks_gaussian.csv")
    parser.add_argument("--quality-name", default="singlepeak_peak_quality_gaussian.csv")
    return parser.parse_args()


if __name__ == "__main__":
    postprocess(parse_args())
