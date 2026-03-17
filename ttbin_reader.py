"""
ttbin_reader.py — 流式读取 .ttbin 时间戳，按时间窗口逐片 yield

核心设计：
  - 永不全量加载：用 TimeTagger.FileReader.hasData() + getData(chunk) 循环
  - 维护 leftover 缓冲：上一次读取中超出时间窗口右边界的部分留给下一片用
  - 用 np.searchsorted 替代递归 binary_search（快且无栈溢出风险）
  - timeDiff 偏移以就地 += 方式施加，不产生额外副本
  - 支持分卷文件：将所有分卷路径列表传给 TimeTagger.FileReader
"""

import os
import re
from typing import List, Union
import numpy as np
import TimeTagger


class StreamingTTBinReader:
    """逐块读取一个或多个分卷 .ttbin 文件，按时间窗口生成时间戳片段。

    Parameters
    ----------
    file_paths : str 或 list of str
        单个 .ttbin 文件路径，或分卷文件路径列表（按顺序排列）。
        TimeTagger.FileReader 会把列表中的文件顺序拼接读取。
    chunk_size : int
        每次调用 FileReader.getData() 请求的最大事件数
        （200万条 × 8 bytes ≈ 16 MB，对内存友好）
    """

    def __init__(self, file_paths: Union[str, List[str]], chunk_size: int = 2_000_000):
        # 自动转换路径
        processed_paths = []
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        for p in file_paths:
            # 如果是单个路径，尝试自动发现全部分卷
            volumes = self.find_ttbin_volumes(p)
            processed_paths.extend(volumes)

        if not processed_paths:
             raise FileNotFoundError(f"找不到任何匹配的 .ttbin 文件: {file_paths}")

        self._paths = processed_paths
        self._chunk_size = chunk_size

    @staticmethod
    def find_ttbin_volumes(path: str) -> List[str]:
        """按顺序寻找所有分卷文件。
        示例: 
          输入 'data.ttbin' -> 寻找 'data.ttbin.1', 'data.ttbin.2' ...
          输入 'data.ttbin.1' -> 寻找 'data.ttbin.1', 'data.ttbin.2' ...
        """
        if not path:
            return []
            
        # 1. 提取基础路径（去掉 .N 后缀）
        # 匹配 xxx.ttbin 或 xxx.ttbin.N
        match = re.search(r'^(.*?\.ttbin)(\.\d+)?$', path)
        if not match:
            # 如果不符合 .ttbin 结尾，直接返回原路径（让 FileReader 处理可能的报错）
            return [path]
            
        base_prefix = match.group(1)
        dirname = os.path.dirname(path)
        basename = os.path.basename(base_prefix)
        
        # 2. 在目录下搜寻
        search_dir = dirname if dirname else '.'
        if not os.path.isdir(search_dir):
            return [path]
            
        found_volumes = []
        
        # 首先检查是否存在 base_prefix 本身（通常是 0 字节主文件，FileReader 建议跳过或它会自动处理，
        # 但有些版本需要主文件。不过根据本代码逻辑，我们更倾向于直接拼接物理分卷 .1, .2...）
        # 按照 Swabian 惯例，.1, .2... 是实际数据文件。
        
        i = 1
        while True:
            vol_path = f"{base_prefix}.{i}"
            if os.path.isfile(vol_path):
                found_volumes.append(vol_path)
                i += 1
            else:
                break
                
        # 如果一个分卷都没找到，但原文件存在，则至少返回原文件
        if not found_volumes and os.path.isfile(path):
            return [path]
            
        return found_volumes if found_volumes else [path]

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def peek_head(self, duration_ps: int) -> np.ndarray:
        """只读文件头部 duration_ps 范围内的时间戳，用于互相关估算。

        每次调用都重新打开文件，不影响后续 iter_time_windows 的状态。

        Parameters
        ----------
        duration_ps : int
            从文件第一条时间戳起的时间窗口长度（单位 ps）

        Returns
        -------
        np.ndarray  dtype=int64
        """
        reader = TimeTagger.FileReader(self._paths)
        result_chunks = []
        start_ts = None

        while reader.hasData():
            buf = reader.getData(n_events=self._chunk_size)
            ts = np.array(buf.getTimestamps(), dtype=np.int64)
            if ts.size == 0:
                continue

            if start_ts is None:
                start_ts = int(ts[0])

            end_ts = start_ts + duration_ps
            cut = int(np.searchsorted(ts, end_ts, side='right'))
            result_chunks.append(ts[:cut])

            if cut < ts.size:
                # 已超出所需范围，停止读取
                break

        return np.concatenate(result_chunks) if result_chunks else np.array([], dtype=np.int64)

    def iter_time_windows(self,
                          step_ps: int,
                          offset_ps: int = 0):
        """生成器：按 step_ps 时间宽度逐片 yield 时间戳数组。

        内存中同时只有：leftover 缓冲 + 当前一个 READ_CHUNK。

        Parameters
        ----------
        step_ps : int
            每个时间窗口的宽度（单位 ps）
        offset_ps : int
            在 yield 前对所有时间戳加上此偏移（用于 idler timeDiff 修正）。
            以就地 += 施加，不产生额外数组副本。

        Yields
        ------
        np.ndarray  dtype=int64
            经 offset_ps 修正后的时间戳片段
        """
        reader = TimeTagger.FileReader(self._paths)
        leftover = np.array([], dtype=np.int64)
        window_start = None   # 当前窗口的起始时间（原始时间戳，不含 offset）

        def _yield_chunk(arr: np.ndarray) -> np.ndarray:
            """拷贝一份数组并施加偏移，然后 yield。"""
            out = arr.copy()
            if offset_ps != 0:
                out += offset_ps
            return out

        while True:
            # ── 1. 读取新的一批数据 ──────────────────────────────────
            if reader.hasData():
                buf = reader.getData(n_events=self._chunk_size)
                new_ts = np.array(buf.getTimestamps(), dtype=np.int64)
            else:
                new_ts = np.array([], dtype=np.int64)

            # ── 2. 拼接到 leftover ───────────────────────────────────
            if leftover.size > 0 and new_ts.size > 0:
                combined = np.concatenate([leftover, new_ts])
                leftover = np.array([], dtype=np.int64)
            elif leftover.size > 0:
                combined = leftover
                leftover = np.array([], dtype=np.int64)
            elif new_ts.size > 0:
                combined = new_ts
            else:
                # 文件已读完且无剩余数据，退出
                break

            # ── 3. 初始化第一个窗口起点（用数据中首个时间戳）──────────
            if window_start is None:
                window_start = int(combined[0])

            # ── 4. 从 combined 中切出尽可能多的完整时间窗口 ─────────
            while combined.size > 0:
                window_end = window_start + step_ps
                cut = int(np.searchsorted(combined, window_end, side='right'))

                if cut < combined.size:
                    # 找到了完整的窗口边界
                    chunk = combined[:cut]
                    combined = combined[cut:]
                    yield _yield_chunk(chunk)
                    window_start = window_end
                else:
                    # combined 的末尾还没到窗口右边界
                    if reader.hasData():
                        # 还有数据可读，把 combined 存为 leftover 继续读
                        leftover = combined
                        combined = np.array([], dtype=np.int64)
                        break   # 跳出内层 while，继续外层循环读新 chunk
                    else:
                        # 文件读完，最后一片（可能不足 step_ps）直接 yield
                        yield _yield_chunk(combined)
                        combined = np.array([], dtype=np.int64)
                        break

            # ── 5. 若文件已读完且无 leftover，退出外层 while ─────────
            if not reader.hasData() and leftover.size == 0:
                break
