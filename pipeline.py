import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

import config as cfg
from coincidence import coincidence_peak
from correlation import estimate_time_offset
from ttbin_reader import StreamingTTBinReader


def _output_root_dir() -> str:
    out_root = getattr(cfg, "OUTPUT_DIR", None)
    return out_root if out_root else cfg.DIR


def _resolve_input_paths(names):
    if isinstance(names, str):
        return names if os.path.isabs(names) else os.path.join(cfg.DIR, names)
    return [n if os.path.isabs(n) else os.path.join(cfg.DIR, n) for n in names]


def _save_hist_bin_width_for_pair(pair_index: int) -> int:
    widths = getattr(cfg, "SAVE_HIST_BIN_WIDTHS_PS", None)
    if not widths:
        return 1
    if pair_index < len(widths):
        return int(widths[pair_index])
    return int(widths[-1])


def process_pair(
    signal_paths,
    idler_paths,
    save_dir=None,
    save_hist_bin_width_ps: int = 1,
    manual_time_diff_ps: int | None = None,
) -> list:
    print(f"\n{'=' * 60}")
    print(f"[signal] {signal_paths}")
    print(f"[idler ] {idler_paths}")

    sig_reader = StreamingTTBinReader(signal_paths, cfg.READ_CHUNK_SIZE)
    idl_reader = StreamingTTBinReader(idler_paths, cfg.READ_CHUNK_SIZE)

    if manual_time_diff_ps is None:
        print(f"\n[Phase 0] read head {cfg.CORRELATION_WINDOW_PS/1e12:.0f}s, estimate timeDiff ...")
        sig_head = sig_reader.peek_head(cfg.CORRELATION_WINDOW_PS)
        idl_head = idl_reader.peek_head(cfg.CORRELATION_WINDOW_PS)

        print(f"  signal head: {sig_head.size} events, start={sig_head[0]}, end={sig_head[-1]}")
        print(f"  idler  head: {idl_head.size} events, start={idl_head[0]}, end={idl_head[-1]}")

        time_diff = estimate_time_offset(
            sig_head,
            idl_head,
            cfg.CORRELATION_WINDOW_PS,
            cfg.CORRELATION_FRAMES,
        )
        print(f"  timeDiff = {time_diff} ps ({time_diff / 1e12:.6f} s)")
        del sig_head, idl_head
    else:
        time_diff = int(manual_time_diff_ps)
        print(f"\n[Phase 0] use manual timeDiff = {time_diff} ps ({time_diff / 1e12:.6f} s)")

    print(f"\n[Phase 1] streaming slices ({cfg.SPLIT_STEP_PS/1e12:.0f}s per slice) ...")
    n_chunks_est = int(cfg.SPLIT_TIME_PS // cfg.SPLIT_STEP_PS)

    results = []
    sig_gen = sig_reader.iter_time_windows(cfg.SPLIT_STEP_PS, offset_ps=0)
    idl_gen = idl_reader.iter_time_windows(cfg.SPLIT_STEP_PS, offset_ps=time_diff)

    with tqdm(total=n_chunks_est, desc="  coincidence", unit="slice") as pbar:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            limit = os.cpu_count() * 2 if os.cpu_count() else 4

            for j, (sc, ic) in enumerate(zip(sig_gen, idl_gen)):
                if j >= n_chunks_est:
                    break

                fut = executor.submit(
                    coincidence_peak,
                    sc,
                    ic,
                    cfg.BIN_WIDTH_PS,
                    cfg.BIN_NUM,
                    save_dir,
                    j + 1,
                    80,
                    save_hist_bin_width_ps,
                )
                futures.append(fut)

                if len(futures) > limit:
                    peak = futures.pop(0).result()
                    results.append(peak - time_diff)
                    pbar.update(1)

            for fut in futures:
                peak = fut.result()
                results.append(peak - time_diff)
                pbar.update(1)

    print(f"  slices processed: {len(results)}")
    return results


def main():
    all_results = []

    for i, pair in enumerate(cfg.FILE_PAIRS):
        save_hist_bin_width_ps = _save_hist_bin_width_for_pair(i)
        pair_dir = os.path.join(
            _output_root_dir(),
            f"pair{i}_histograms_raw_{save_hist_bin_width_ps}ps",
        )
        os.makedirs(pair_dir, exist_ok=True)

        signal_paths = _resolve_input_paths(pair["signal"])
        idler_paths = _resolve_input_paths(pair["idler"])
        results = process_pair(
            signal_paths,
            idler_paths,
            save_dir=pair_dir,
            save_hist_bin_width_ps=save_hist_bin_width_ps,
            manual_time_diff_ps=pair.get("time_diff_ps"),
        )
        all_results.append(results)

    n = min(len(r) for r in all_results)
    row_data = [np.array(r[:n]) for r in all_results]
    clock_correction = (row_data[0] - row_data[1]) / 2.0

    n_slices = int(cfg.SPLIT_TIME_PS // cfg.SPLIT_STEP_PS)
    out_path = cfg.SAVE_FILE_PATH
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"\n[Output] writing {out_path} ...")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "ch1-ch4_ps", "ch2-ch3_ps", "clock_correction_ps"])
        for i in range(min(n, n_slices)):
            t_s = (i + 1) * cfg.SPLIT_STEP_PS / 1e12
            writer.writerow(
                [
                    f"{t_s:.6f}",
                    f"{float(row_data[0][i]):.6f}",
                    f"{float(row_data[1][i]):.6f}",
                    f"{float(clock_correction[i]):.6f}",
                ]
            )

    data_csv_path = os.path.join(_output_root_dir(), "data_py.csv")
    with open(data_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i in range(min(n, n_slices)):
            t_s = (i + 1) * cfg.SPLIT_STEP_PS / 1e12
            writer.writerow([f"{t_s:.6f}", f"{float(clock_correction[i]):.6f}"])

    print(f"  main CSV: {out_path}")
    print(f"  data    : {data_csv_path}")
    print(f"\nDone. Processed {min(n, n_slices)} slices.")


if __name__ == "__main__":
    main()
