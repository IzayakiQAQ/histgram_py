import argparse
import csv
import os
import sys
import re

from tqdm import tqdm

# Allow running from this folder with repo-root imports.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pipeline_variants.dualpeak_single_idler.config_dualpeak_single_idler as cfg
from coincidence_dualpeak import coincidence_two_peaks
from correlation import estimate_time_offset
from ttbin_reader import StreamingTTBinReader


def _normalize_ttbin_paths(val):
    """
    Allow passing multiple volumes via:
      - Python list in config
      - command line string separated by ';' or ','
    """
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return s
        if (";" in s) or ("," in s):
            parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
            if len(parts) > 1:
                return parts
        return s
    return val


def _parse_args():
    p = argparse.ArgumentParser(
        description="Dual-peak coincidence pipeline (single idler + two signals)."
    )
    p.add_argument("--signal-a", default=cfg.SIGNAL_A, help="Path to signal A .ttbin (or ';' separated volumes).")
    p.add_argument("--signal-b", default=cfg.SIGNAL_B, help="Path to signal B .ttbin (or ';' separated volumes).")
    p.add_argument("--idler", default=cfg.IDLER, help="Path to idler .ttbin (or ';' separated volumes).")
    p.add_argument("--out", default=cfg.SAVE_FILE_PATH, help="Output CSV path.")
    p.add_argument("--hist-save-dir", default=getattr(cfg, "HIST_SAVE_DIR", ""), help="Directory to save 1ps histogram segments (empty disables).")
    p.add_argument("--hist-target-points", type=int, default=getattr(cfg, "HIST_TARGET_POINTS", 65536), help="Points to save per peak segment (default 65536).")
    p.add_argument("--corr-window-ps", type=int, default=cfg.CORRELATION_WINDOW_PS, help="Correlation head window (ps).")
    p.add_argument("--corr-frames", type=int, default=cfg.CORRELATION_FRAMES, help="Correlation histogram frames (bins).")
    p.add_argument("--min-sep-ps", type=int, default=cfg.DUALPEAK_MIN_SEPARATION_PS, help="Min separation between two peaks (ps).")
    p.add_argument("--fit-half-window-bins", type=int, default=cfg.DUALPEAK_FIT_HALF_WINDOW_BINS, help="Fit window half-size (bins).")
    p.add_argument("--debug-hist", action="store_true", help="Write extra debug files alongside saved histograms.")
    return p.parse_args()


def _validate_paths(signal_a, signal_b, idler):
    missing = []
    if not signal_a:
        missing.append("SIGNAL_A")
    if not signal_b:
        missing.append("SIGNAL_B")
    if not idler:
        missing.append("IDLER")
    if missing:
        cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config_dualpeak_single_idler.py"))
        raise ValueError(
            "Config is missing required paths: " + ", ".join(missing) +
            f". Edit {cfg_path} or pass them via command line:\n"
            "  python .\\pipeline_variants\\dualpeak_single_idler\\pipeline_dualpeak_single_idler.py "
            "--signal-a <path> --signal-b <path> --idler <path> --out <csv>"
        )


def main():
    args = _parse_args()
    signal_a = _normalize_ttbin_paths(args.signal_a)
    signal_b = _normalize_ttbin_paths(args.signal_b)
    idler = _normalize_ttbin_paths(args.idler)

    _validate_paths(signal_a, signal_b, idler)

    # Default all outputs to the idler folder unless explicitly overridden.
    idler_first = idler[0] if isinstance(idler, list) and idler else idler
    idler_dir = os.path.dirname(os.path.abspath(idler_first)) if idler_first else os.getcwd()

    print("\n" + "=" * 60, flush=True)
    print("[signal A]", signal_a, flush=True)
    print("[signal B]", signal_b, flush=True)
    print("[idler   ]", idler, flush=True)

    # Readers: idler is used twice with different timeDiff offsets, so use two reader instances.
    chunk_size = getattr(cfg, "READ_CHUNK_SIZE", 2_000_000)
    r_sig_a = StreamingTTBinReader(signal_a, chunk_size=chunk_size)
    r_sig_b = StreamingTTBinReader(signal_b, chunk_size=chunk_size)
    r_idl_1 = StreamingTTBinReader(idler, chunk_size=chunk_size)
    r_idl_2 = StreamingTTBinReader(idler, chunk_size=chunk_size)

    import time
    corr_window_ps = int(args.corr_window_ps)
    corr_frames = int(args.corr_frames)
    print(
        f"\n[Phase 0] peek_head {corr_window_ps/1e12:.0f}s, corr_frames={corr_frames} (this can be slow if very large) ...",
        flush=True,
    )
    t0 = time.time()
    h_sa = r_sig_a.peek_head(corr_window_ps)
    h_sb = r_sig_b.peek_head(corr_window_ps)
    h_i1 = r_idl_1.peek_head(corr_window_ps)
    h_i2 = r_idl_2.peek_head(corr_window_ps)
    print(f"  peek_head done in {time.time() - t0:.2f}s", flush=True)

    t1 = time.time()
    t_a = estimate_time_offset(h_sa, h_i1, corr_window_ps, corr_frames)
    t_b = estimate_time_offset(h_sb, h_i2, corr_window_ps, corr_frames)
    print(f"  estimate_time_offset done in {time.time() - t1:.2f}s", flush=True)
    print(f"  timeDiff A = {t_a} ps ({t_a/1e12:.6f} s)")
    print(f"  timeDiff B = {t_b} ps ({t_b/1e12:.6f} s)")

    del h_sa, h_sb, h_i1, h_i2

    print(f"\n[Phase 1] streaming slices ({cfg.SPLIT_STEP_PS/1e12:.0f}s per slice) ...")
    n_slices_est = int(cfg.SPLIT_TIME_PS // cfg.SPLIT_STEP_PS)

    gen_sa = r_sig_a.iter_time_windows(cfg.SPLIT_STEP_PS, offset_ps=0)
    gen_sb = r_sig_b.iter_time_windows(cfg.SPLIT_STEP_PS, offset_ps=0)
    gen_i1 = r_idl_1.iter_time_windows(cfg.SPLIT_STEP_PS, offset_ps=t_a)
    gen_i2 = r_idl_2.iter_time_windows(cfg.SPLIT_STEP_PS, offset_ps=t_b)

    out_path = args.out
    if out_path == cfg.SAVE_FILE_PATH and not os.path.isabs(out_path):
        out_path = os.path.join(idler_dir, out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    hist_save_dir = (args.hist_save_dir or "").strip()
    if not hist_save_dir:
        # Default folder name; lives next to the idler file.
        hist_save_dir = os.path.join(idler_dir, "dualpeak_single_idler_hists")
    if hist_save_dir:
        os.makedirs(hist_save_dir, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_s",
                "A_peak1_ps",
                "A_peak2_ps",
                "B_peak1_ps",
                "B_peak2_ps",
            ]
        )

        with tqdm(total=n_slices_est, desc="  dualpeak", unit="slice") as pbar:
            for i, (sa, sb, i1, i2) in enumerate(zip(gen_sa, gen_sb, gen_i1, gen_i2)):
                if i >= n_slices_est:
                    break

                a1, a2 = coincidence_two_peaks(
                    sa,
                    i1,
                    cfg.BIN_WIDTH_PS,
                    cfg.BIN_NUM,
                    min_separation_ps=args.min_sep_ps,
                    fit_half_window_bins=args.fit_half_window_bins,
                    save_dir=(os.path.join(hist_save_dir, "A") if hist_save_dir else None),
                    index=i + 1,
                    prefix="hist",
                    target_points=args.hist_target_points,
                    debug=args.debug_hist,
                )
                b1, b2 = coincidence_two_peaks(
                    sb,
                    i2,
                    cfg.BIN_WIDTH_PS,
                    cfg.BIN_NUM,
                    min_separation_ps=args.min_sep_ps,
                    fit_half_window_bins=args.fit_half_window_bins,
                    save_dir=(os.path.join(hist_save_dir, "B") if hist_save_dir else None),
                    index=i + 1,
                    prefix="hist",
                    target_points=args.hist_target_points,
                    debug=args.debug_hist,
                )

                # Express each center back in the "signal time base" like original pipeline did:
                # coincidence returns center relative to window mid; original did (peak - timeDiff).
                a1 -= t_a
                a2 -= t_a
                b1 -= t_b
                b2 -= t_b

                t_s = (i + 1) * cfg.SPLIT_STEP_PS / 1e12
                w.writerow(
                    [
                        f"{t_s:.6f}",
                        f"{float(a1):.6f}",
                        f"{float(a2):.6f}",
                        f"{float(b1):.6f}",
                        f"{float(b2):.6f}",
                    ]
                )
                pbar.update(1)

    print(f"\n[Output] wrote {out_path}")


if __name__ == "__main__":
    main()
