import argparse
import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import config as cfg
except ImportError:
    cfg = None

from coincidence import coincidence_peak
from correlation import estimate_time_offset


def _cfg_value(name: str, default: Any) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


@dataclass
class BatchSettings:
    split_step_ps: int
    total_time_ps: int
    correlation_window_ps: int
    correlation_frames: int
    bin_width_ps: int
    bin_num: int
    read_chunk_size: int
    save_hist_bin_width_ps: int
    save_hist_center_ps: int | None
    save_hist_points: int
    fit_half_window_bins: int
    workers: int
    max_slices: int
    save_histograms: bool


def _sanitize_name(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.ASCII)
    return text.strip("._") or fallback


def _resolve_path(value: str | list[str], root: Path) -> str | list[str]:
    def one(v: str) -> str:
        p = Path(v)
        return str(p if p.is_absolute() else root / p)

    if isinstance(value, list):
        return [one(v) for v in value]
    return one(value)


def _seconds_to_ps(value: float) -> int:
    return int(round(float(value) * 1e12))


def _load_jobs(job_file: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(job_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {}, data
    if not isinstance(data, dict):
        raise ValueError("Job file must contain a JSON object or a JSON list.")

    defaults = data.get("defaults", {})
    jobs = data.get("jobs", [])
    if not isinstance(defaults, dict) or not isinstance(jobs, list):
        raise ValueError("Job file must use {'defaults': {...}, 'jobs': [...]} format.")
    return defaults, jobs


def _merge_settings(args, defaults: dict[str, Any]) -> BatchSettings:
    split_step_ps = int(
        defaults.get("split_step_ps", _seconds_to_ps(defaults.get("split_step_s", args.split_step_s)))
    )
    total_time_ps = int(
        defaults.get("total_time_ps", _seconds_to_ps(defaults.get("total_time_s", args.total_time_s)))
    )
    correlation_window_ps = int(
        defaults.get(
            "correlation_window_ps",
            _seconds_to_ps(defaults.get("correlation_window_s", args.correlation_window_s)),
        )
    )

    max_slices = int(defaults.get("max_slices", args.max_slices))
    if max_slices <= 0 and total_time_ps > 0:
        max_slices = int(total_time_ps // split_step_ps)

    save_hist_center_ps = defaults.get("save_hist_center_ps", args.save_hist_center_ps)
    if save_hist_center_ps in ("", None):
        save_hist_center_ps = None
    else:
        save_hist_center_ps = int(save_hist_center_ps)

    return BatchSettings(
        split_step_ps=split_step_ps,
        total_time_ps=total_time_ps,
        correlation_window_ps=correlation_window_ps,
        correlation_frames=int(defaults.get("correlation_frames", args.correlation_frames)),
        bin_width_ps=int(defaults.get("bin_width_ps", args.bin_width_ps)),
        bin_num=int(defaults.get("bin_num", args.bin_num)),
        read_chunk_size=int(defaults.get("read_chunk_size", args.read_chunk_size)),
        save_hist_bin_width_ps=int(defaults.get("save_hist_bin_width_ps", args.save_hist_bin_width_ps)),
        save_hist_center_ps=save_hist_center_ps,
        save_hist_points=int(defaults.get("save_hist_points", args.save_hist_points)),
        fit_half_window_bins=int(defaults.get("fit_half_window_bins", args.fit_half_window_bins)),
        workers=max(int(defaults.get("workers", args.workers)), 1),
        max_slices=max_slices,
        save_histograms=not bool(defaults.get("no_hist", args.no_hist)),
    )


def _estimate_or_use_manual_time_diff(sig_reader, idl_reader, settings: BatchSettings, manual_time_diff_ps):
    if manual_time_diff_ps is not None:
        time_diff = int(manual_time_diff_ps)
        print(f"[Phase 0] use manual timeDiff = {time_diff} ps ({time_diff / 1e12:.6f} s)")
        return time_diff

    print(f"[Phase 0] read head {settings.correlation_window_ps / 1e12:.0f}s, estimate timeDiff ...")
    sig_head = sig_reader.peek_head(settings.correlation_window_ps)
    idl_head = idl_reader.peek_head(settings.correlation_window_ps)

    if sig_head.size == 0:
        raise ValueError("Signal head is empty; cannot estimate timeDiff.")
    if idl_head.size == 0:
        raise ValueError("Idler head is empty; cannot estimate timeDiff.")

    print(f"  signal head: {sig_head.size} events, start={sig_head[0]}, end={sig_head[-1]}")
    print(f"  idler  head: {idl_head.size} events, start={idl_head[0]}, end={idl_head[-1]}")

    time_diff = estimate_time_offset(
        sig_head,
        idl_head,
        settings.correlation_window_ps,
        settings.correlation_frames,
    )
    print(f"  timeDiff = {time_diff} ps ({time_diff / 1e12:.6f} s)")
    return time_diff


def process_singlepeak_pair(
    *,
    label: str,
    signal_paths,
    idler_paths,
    save_dir: Path | None,
    settings: BatchSettings,
    manual_time_diff_ps: int | None,
) -> list[float]:
    from ttbin_reader import StreamingTTBinReader

    print(f"\n{'=' * 60}")
    print(f"[pair  ] {label}")
    print(f"[signal] {signal_paths}")
    print(f"[idler ] {idler_paths}")

    sig_reader = StreamingTTBinReader(signal_paths, settings.read_chunk_size)
    idl_reader = StreamingTTBinReader(idler_paths, settings.read_chunk_size)
    time_diff = _estimate_or_use_manual_time_diff(
        sig_reader,
        idl_reader,
        settings,
        manual_time_diff_ps,
    )

    print(f"[Phase 1] streaming slices ({settings.split_step_ps / 1e12:.0f}s per slice) ...")
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    sig_gen = sig_reader.iter_time_windows(settings.split_step_ps, offset_ps=0)
    idl_gen = idl_reader.iter_time_windows(settings.split_step_ps, offset_ps=time_diff)
    results: list[float] = []
    limit = settings.workers * 2
    total = settings.max_slices if settings.max_slices > 0 else None

    with tqdm(total=total, desc=f"  {label}", unit="slice") as pbar:
        with ProcessPoolExecutor(max_workers=settings.workers) as executor:
            futures = []

            for j, (sc, ic) in enumerate(zip(sig_gen, idl_gen)):
                if settings.max_slices > 0 and j >= settings.max_slices:
                    break

                fut = executor.submit(
                    coincidence_peak,
                    sc,
                    ic,
                    settings.bin_width_ps,
                    settings.bin_num,
                    str(save_dir) if save_dir is not None else None,
                    j + 1,
                    settings.fit_half_window_bins,
                    settings.save_hist_bin_width_ps,
                    settings.save_hist_center_ps,
                    settings.save_hist_points,
                )
                futures.append(fut)

                if len(futures) > limit:
                    peak = futures.pop(0).result()
                    results.append(float(peak) - float(time_diff))
                    pbar.update(1)

            for fut in futures:
                peak = fut.result()
                results.append(float(peak) - float(time_diff))
                pbar.update(1)

    print(f"  slices processed: {len(results)}")
    return results


def _write_job_csv(job_dir: Path, pair_results: list[tuple[str, list[float]]], settings: BatchSettings):
    if not pair_results:
        raise ValueError("No pair results to write.")

    n = min(len(values) for _, values in pair_results)
    out_path = job_dir / "singlepeak_peaks.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["time_s"] + [f"{label}_peak_ps" for label, _ in pair_results]
        if len(pair_results) >= 2:
            header.append("clock_correction_ps")
        writer.writerow(header)

        arrays = [np.asarray(values[:n], dtype=np.float64) for _, values in pair_results]
        for i in range(n):
            time_s = (i + 1) * settings.split_step_ps / 1e12
            row = [f"{time_s:.6f}"] + [f"{float(arr[i]):.6f}" for arr in arrays]
            if len(arrays) >= 2:
                row.append(f"{float((arrays[0][i] - arrays[1][i]) / 2.0):.6f}")
            writer.writerow(row)

    if len(pair_results) >= 2:
        data_py_path = job_dir / "data_py.csv"
        a0 = np.asarray(pair_results[0][1][:n], dtype=np.float64)
        a1 = np.asarray(pair_results[1][1][:n], dtype=np.float64)
        with open(data_py_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for i in range(n):
                time_s = (i + 1) * settings.split_step_ps / 1e12
                writer.writerow([f"{time_s:.6f}", f"{float((a0[i] - a1[i]) / 2.0):.6f}"])

    print(f"[Output] wrote {out_path}")
    return out_path, n


def run_job(job: dict[str, Any], output_root: Path, settings: BatchSettings):
    if not isinstance(job, dict):
        raise ValueError("Each job must be a JSON object.")

    root = Path(job.get("root", "."))
    name = _sanitize_name(job.get("name") or root.name, "job")
    if job.get("output_dir"):
        job_dir = Path(job["output_dir"])
        if not job_dir.is_absolute():
            job_dir = output_root / job_dir
    else:
        job_dir = output_root / name
    job_dir.mkdir(parents=True, exist_ok=True)

    pairs = job.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"Job {name!r} must contain a non-empty 'pairs' list.")

    pair_results: list[tuple[str, list[float]]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"Job {name!r} pair {index} must be a JSON object.")
        label = _sanitize_name(pair.get("label"), f"pair{index}")
        if "signal" not in pair or "idler" not in pair:
            raise ValueError(f"Job {name!r} pair {label!r} must contain signal and idler.")

        signal_paths = _resolve_path(pair["signal"], root)
        idler_paths = _resolve_path(pair["idler"], root)
        hist_width = int(pair.get("save_hist_bin_width_ps", settings.save_hist_bin_width_ps))
        hist_center = pair.get("save_hist_center_ps", settings.save_hist_center_ps)
        if hist_center in ("", None):
            hist_center = None
        else:
            hist_center = int(hist_center)
        hist_points = int(pair.get("save_hist_points", settings.save_hist_points))
        pair_settings = BatchSettings(
            **{
                **settings.__dict__,
                "save_hist_bin_width_ps": hist_width,
                "save_hist_center_ps": hist_center,
                "save_hist_points": hist_points,
            }
        )
        save_dir = None
        if pair_settings.save_histograms:
            save_dir = job_dir / f"{label}_histograms_raw_{hist_width}ps"

        results = process_singlepeak_pair(
            label=label,
            signal_paths=signal_paths,
            idler_paths=idler_paths,
            save_dir=save_dir,
            settings=pair_settings,
            manual_time_diff_ps=pair.get("time_diff_ps"),
        )
        pair_results.append((label, results))

    out_path, n_slices = _write_job_csv(job_dir, pair_results, settings)
    return {
        "job": name,
        "status": "ok",
        "pairs": len(pair_results),
        "slices": n_slices,
        "output": str(out_path),
        "error": "",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process single-peak timestamp data into saved histograms and peak CSVs."
    )
    parser.add_argument("--job-file", required=True, help="JSON file describing one or more data groups.")
    parser.add_argument("--output-root", default="", help="Batch output root. Overrides job-file output_root.")
    parser.add_argument("--split-step-s", type=float, default=float(_cfg_value("SPLIT_STEP_PS", int(10e12))) / 1e12)
    parser.add_argument("--total-time-s", type=float, default=float(_cfg_value("SPLIT_TIME_PS", int(86400e12))) / 1e12)
    parser.add_argument(
        "--correlation-window-s",
        type=float,
        default=float(_cfg_value("CORRELATION_WINDOW_PS", int(4e12))) / 1e12,
    )
    parser.add_argument("--correlation-frames", type=int, default=int(_cfg_value("CORRELATION_FRAMES", int(4e7))))
    parser.add_argument("--bin-width-ps", type=int, default=int(_cfg_value("BIN_WIDTH_PS", 20)))
    parser.add_argument("--bin-num", type=int, default=int(_cfg_value("BIN_NUM", 10_000)))
    parser.add_argument("--read-chunk-size", type=int, default=int(_cfg_value("READ_CHUNK_SIZE", 2_000_000)))
    parser.add_argument("--save-hist-bin-width-ps", type=int, default=1)
    parser.add_argument("--save-hist-center-ps", type=int, default=None)
    parser.add_argument("--save-hist-points", type=int, default=65536)
    parser.add_argument("--fit-half-window-bins", type=int, default=80)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-jobs", type=int, default=0, help="Only process the first N jobs; 0 means all.")
    parser.add_argument("--max-slices", type=int, default=0, help="Only process the first N slices; 0 uses total-time-s.")
    parser.add_argument("--no-hist", action="store_true", help="Do not save per-slice raw histogram CSVs.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with later jobs if one job fails.")
    parser.add_argument("--validate-only", action="store_true", help="Validate job structure and print resolved paths only.")
    return parser.parse_args()


def main():
    args = parse_args()
    job_file = Path(args.job_file)
    defaults, jobs = _load_jobs(job_file)
    settings = _merge_settings(args, defaults)

    output_root = Path(args.output_root or defaults.get("output_root", job_file.with_suffix("").name + "_out"))

    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]
    if not jobs:
        raise ValueError("No jobs found in job file.")

    if args.validate_only:
        _validate_jobs(jobs, output_root)
        return

    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "batch_singlepeak_summary.csv"
    summary_rows = []

    print(f"[Batch] jobs={len(jobs)}, output_root={output_root}")
    print(
        "[Batch] "
        f"split={settings.split_step_ps / 1e12:.6g}s, "
        f"max_slices={settings.max_slices or 'all'}, "
        f"hist_bin={settings.save_hist_bin_width_ps}ps, "
        f"workers={settings.workers}"
    )

    for i, job in enumerate(jobs, start=1):
        job_name = job.get("name", f"job{i}") if isinstance(job, dict) else f"job{i}"
        print(f"\n[Batch] job {i}/{len(jobs)}: {job_name}")
        try:
            summary_rows.append(run_job(job, output_root, settings))
        except Exception as exc:
            row = {
                "job": str(job_name),
                "status": "failed",
                "pairs": 0,
                "slices": 0,
                "output": "",
                "error": str(exc),
            }
            summary_rows.append(row)
            if not args.continue_on_error:
                _write_summary(summary_path, summary_rows)
                raise
            print(f"[Error] {job_name}: {exc}")

    _write_summary(summary_path, summary_rows)
    print(f"\n[Batch] wrote {summary_path}")


def _validate_jobs(jobs: list[dict[str, Any]], output_root: Path):
    print(f"[Validate] jobs={len(jobs)}, output_root={output_root}")
    for i, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise ValueError(f"Job {i} must be a JSON object.")
        root = Path(job.get("root", "."))
        name = _sanitize_name(job.get("name") or root.name, f"job{i}")
        pairs = job.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError(f"Job {name!r} must contain a non-empty 'pairs' list.")

        print(f"\n[{i}] {name}")
        print(f"  root: {root}")
        if job.get("output_dir"):
            job_dir = Path(job["output_dir"])
            if not job_dir.is_absolute():
                job_dir = output_root / job_dir
        else:
            job_dir = output_root / name
        print(f"  out : {job_dir}")

        for j, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                raise ValueError(f"Job {name!r} pair {j} must be a JSON object.")
            if "signal" not in pair or "idler" not in pair:
                raise ValueError(f"Job {name!r} pair {j} must contain signal and idler.")
            label = _sanitize_name(pair.get("label"), f"pair{j}")
            print(f"  pair {j}: {label}")
            print(f"    signal: {_resolve_path(pair['signal'], root)}")
            print(f"    idler : {_resolve_path(pair['idler'], root)}")
            if pair.get("time_diff_ps") is not None:
                print(f"    manual timeDiff: {int(pair['time_diff_ps'])} ps")


def _write_summary(summary_path: Path, rows: list[dict[str, Any]]):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["job", "status", "pairs", "slices", "output", "error"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
