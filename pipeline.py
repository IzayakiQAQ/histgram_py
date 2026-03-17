"""
pipeline.py — 主流程

流程：
  1. 对每对 (signal, idler) 文件：
     a. peek_head → 互相关 → timeDiff
     b. 流式 iter_time_windows → 逐片 coincidence_peak → 立即写 CSV 列
  2. 处理完 2 对后，计算 (pair1 - pair2) / 2 作为时钟修正列
  3. 覆盖写最终 CSV（time, ch1-ch4, ch2-ch3, clock correction）

内存峰值：单片段 × 2 路 ≈ 数十 MB，与文件总大小无关。
"""

import csv
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import config as cfg
from ttbin_reader import StreamingTTBinReader
from correlation import estimate_time_offset
from coincidence import coincidence_peak



# ── 处理单对文件，返回该对的 results 列表 ───────────────────────────────────
def process_pair(signal_paths, idler_paths) -> list:
    """
    对单对 (signal, idler) 文件做流式处理。signal_paths/idler_paths 可以是
    单个文件路径字符串，也可以是分卷文件路径列表。
    """
    print(f"\n{'='*60}")
    print(f"[signal] {signal_paths}")
    print(f"[idler ] {idler_paths}")

    sig_reader = StreamingTTBinReader(signal_paths, cfg.READ_CHUNK_SIZE)
    idl_reader = StreamingTTBinReader(idler_paths,  cfg.READ_CHUNK_SIZE)

    # ── 阶段 0：只读头部，估算时间偏移 ──────────────────────────────────
    print(f"\n[Phase 0] 读取头部 {cfg.CORRELATION_WINDOW_PS/1e12:.0f}s 数据，估算 timeDiff ...")
    sig_head = sig_reader.peek_head(cfg.CORRELATION_WINDOW_PS)
    idl_head = idl_reader.peek_head(cfg.CORRELATION_WINDOW_PS)

    print(f"  signal head: {sig_head.size} events, "
          f"start={sig_head[0]}, end={sig_head[-1]}")
    print(f"  idler  head: {idl_head.size} events, "
          f"start={idl_head[0]}, end={idl_head[-1]}")

    time_diff = estimate_time_offset(
        sig_head, idl_head,
        cfg.CORRELATION_WINDOW_PS,
        cfg.CORRELATION_FRAMES,
    )
    print(f"  timeDiff = {time_diff} ps ({time_diff/1e12:.6f} s)")

    del sig_head, idl_head   # 立即释放，不再需要

    # ── 阶段 1：流式逐片处理 ──────────────────────────────────────────────
    print(f"\n[Phase 1] 流式分片处理（每片 {cfg.SPLIT_STEP_PS/1e12:.0f}s）...")
    n_chunks_est = int(cfg.SPLIT_TIME_PS // cfg.SPLIT_STEP_PS)

    results = []

    sig_gen = sig_reader.iter_time_windows(cfg.SPLIT_STEP_PS, offset_ps=0)
    idl_gen = idl_reader.iter_time_windows(cfg.SPLIT_STEP_PS, offset_ps=time_diff)

    with tqdm(total=n_chunks_est, desc='  coincidence', unit='slice') as pbar:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            
            # 限制在运行的任务数，防止读取太快导致内存积压 (CPU_COUNT * 2)
            limit = os.cpu_count() * 2 if os.cpu_count() else 4
            
            for j, (sc, ic) in enumerate(zip(sig_gen, idl_gen)):
                # 提交任务
                fut = executor.submit(coincidence_peak, sc, ic, cfg.BIN_WIDTH_PS, cfg.BIN_NUM)
                futures.append(fut)
                
                # 如果积压超过限制，等待最前面的一个完成
                if len(futures) > limit:
                    peak = futures.pop(0).result()
                    results.append(peak - time_diff)
                    pbar.update(1)
            
            # 处理剩余的任务
            for fut in futures:
                peak = fut.result()
                results.append(peak - time_diff)
                pbar.update(1)

    print(f"  处理片段数: {len(results)}")
    return results


# ── 主函数 ───────────────────────────────────────────────────────────────────
def main():
    all_results = []   # list of list，每个元素对应一对文件的 results

    for i, pair in enumerate(cfg.FILE_PAIRS):
        def _full_paths(names):
            if isinstance(names, str):
                return os.path.join(cfg.DIR, names)
            return [os.path.join(cfg.DIR, n) for n in names]
        signal_paths = _full_paths(pair['signal'])
        idler_paths  = _full_paths(pair['idler'])
        results = process_pair(signal_paths, idler_paths)
        all_results.append(results)

    # ── 对齐长度（两对文件片段数可能因文件时长略有差异）──────────────────
    n = min(len(r) for r in all_results)
    row_data = [np.array(r[:n]) for r in all_results]

    # ── 计算时钟修正列 ──────────────────────────────────────────────────
    clock_correction = (row_data[0] - row_data[1]) / 2.0

    # ── 写入 CSV ────────────────────────────────────────────────────────
    n_slices = int(cfg.SPLIT_TIME_PS // cfg.SPLIT_STEP_PS)
    out_path = cfg.SAVE_FILE_PATH
    print(f"\n[Output] 写入 {out_path} ...")

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'ch1-ch4', 'ch2-ch3', 'clock correction'])
        for i in range(min(n, n_slices)):
            t_s = (i + 1) * cfg.SPLIT_STEP_PS / 1e12
            writer.writerow([
                f'{t_s:.0f}',
                f'{row_data[0][i]:.4f}',
                f'{row_data[1][i]:.4f}',
                f'{clock_correction[i]:.4f}',
            ])

    # ── 附加写 data.csv ─────────────────────────────────────────────────
    data_csv_path = os.path.join(cfg.DIR, 'data_py.csv')
    with open(data_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for i in range(min(n, n_slices)):
            t_s = (i + 1) * cfg.SPLIT_STEP_PS / 1e12
            writer.writerow([f'{t_s:.0f}', f'{clock_correction[i]:.4f}'])

    print(f"  主 CSV : {out_path}")
    print(f"  data   : {data_csv_path}")
    print(f"\n完成！共处理 {min(n, n_slices)} 个时间片段。")


if __name__ == '__main__':
    main()
