# histgram_py

面向 Swabian TimeTagger `.ttbin` 时间戳数据的流式符合直方图处理工具。

这个仓库主要解决两类问题：

1. 原始时间戳数据量很大，不能一次性全部读进内存
2. 需要按时间片做符合峰提取，并保存原始直方图用于后处理和诊断

当前版本已经包含：

- 主双路 pair 处理流程
- 原始符合直方图导出
- 左峰重处理脚本
- 双峰数据处理工具
- 单 idler 双 signal 的变体流程
- 针对互相关错位场景的合成测试

## 仓库结构

- `pipeline.py`
  主流程。按 `config.py` 中的配置处理两对输入数据。
- `ttbin_reader.py`
  流式读取 `.ttbin` 文件，支持自动发现分卷文件。
- `correlation.py`
  用 FFT 互相关估计粗略 `timeDiff`。
- `coincidence.py`
  生成符合直方图、做局部高斯拟合，并导出原始直方图 CSV。
- `reprocess_hist_leftpeak.py`
  对已经导出的直方图重新处理，按“最左显著峰”规则提取峰位。
- `coincidence_dualpeak.py`
  适合明显双峰结构的数据。
- `pipeline_variants/dualpeak_single_idler/`
  单个 idler 对两路 signal 的双峰处理变体。
- `tests_misaligned/test_time_offset_misaligned.py`
  对“时间头部不完全对应”的互相关情况做回归测试。

## 主要功能

- 流式处理大体量 `.ttbin` 数据
- 自动识别 `.ttbin.1`、`.ttbin.2` 等分卷
- 先做粗互相关，再做分片符合峰提取
- 支持每个 pair 单独设置保存的直方图分辨率
- 支持导出 `1 ps` 或 `100 ps` 等不同分辨率的原始直方图
- 支持对已保存直方图做二次后处理
- 支持双峰场景的专门处理逻辑

## 依赖环境

- Python 3.x
- Swabian TimeTagger Python 包
- `numpy`
- `scipy`
- `tqdm`

安装 Python 依赖：

```bash
pip install numpy scipy tqdm
```

## 主流程怎么用

### 1. 修改配置文件

先编辑 `config.py`。

最常用的字段有：

- `FILE_PAIRS`
  输入文件对。每个元素是一对 `signal` / `idler`。
- `OUTPUT_DIR`
  输出根目录。
- `SAVE_HIST_BIN_WIDTHS_PS`
  每个 pair 保存原始直方图时使用的分辨率，单位 `ps`。
- `CORRELATION_WINDOW_PS`
  用于粗互相关的头部时间长度。
- `CORRELATION_FRAMES`
  互相关时使用的 bin 数。
- `SPLIT_STEP_PS`
  每个时间片的长度。
- `BIN_WIDTH_PS`
  符合峰粗直方图 bin 宽度。
- `BIN_NUM`
  符合峰粗直方图 bin 数。

如果某一路你已经知道可信的 `timeDiff`，可以在对应的 `FILE_PAIRS` 项里直接写：

```python
'time_diff_ps': 58500000
```

这样该 pair 会跳过自动互相关，直接使用手动值。

### 2. 运行主流程

```bash
python .\pipeline.py
```

典型输出包括：

- `pair0_histograms_raw_<N>ps/`
- `pair1_histograms_raw_<N>ps/`
- `hcf.csv`
- `data_py.csv`

其中 `<N>` 对应 `SAVE_HIST_BIN_WIDTHS_PS` 里设置的保存分辨率。

## 当前直方图保存逻辑

`coincidence.py` 现在的保存流程是：

1. 先用 `BIN_WIDTH_PS` 和 `BIN_NUM` 生成粗直方图
2. 找到最强粗峰
3. 只在粗峰附近做局部高斯拟合
4. 在内部建立完整 `1 ps` 直方图
5. 在拟合中心附近再次做局部 `1 ps` 重定位
6. 以这个局部峰为中心裁剪出固定长度窗口并保存

这样做的目的，是避免保存窗口被单个噪声尖点带偏。

## 已保存直方图的重处理

如果主流程已经跑完，但数据是特殊情况，比如：

- 三峰，只想取最左峰
- 双峰，只想取第一个峰
- 想在不重跑 `.ttbin` 的前提下重新提取峰位

可以使用：

```bash
python .\reprocess_hist_leftpeak.py --root-dir <输出目录>
```

这个脚本会：

- 读取已保存的 `hist_raw_*.csv`
- 先做轻微平滑
- 找显著峰
- 取最左显著峰
- 在该峰附近做局部高斯拟合
- 输出新的结果 CSV 和调试 CSV

常见输出：

- `hcf_leftpeak.csv`
- `hcf_leftpeak_debug.csv`

## 双峰工具

如果数据本身就是明确双峰结构，可以用：

- `coincidence_dualpeak.py`

它会在一个时间片里找两个峰，并分别返回两个峰中心。

如果你的实验结构是“一个 idler 对两路 signal”，可以看：

- `pipeline_variants/dualpeak_single_idler/`

里面包含：

- `config_dualpeak_single_idler.py`
- `pipeline_dualpeak_single_idler.py`

使用前先在配置文件中填写真实数据路径，再运行：

```bash
python .\pipeline_variants\dualpeak_single_idler\pipeline_dualpeak_single_idler.py
```

## 测试

运行互相关错位场景的合成测试：

```bash
python .\tests_misaligned\test_time_offset_misaligned.py
```

做基础语法检查：

```bash
python -m py_compile coincidence.py pipeline.py correlation.py
```

## 说明

- `config.py` 默认是本地实验配置文件，通常会带本机路径，不建议直接原样分享。
- `.csv` 输出结果默认被 `.gitignore` 忽略，不会直接进 Git。
- 临时调试目录也已经加入忽略规则。

## 适合从哪里开始看

如果你第一次看这个仓库，建议按这个顺序：

1. `config.py`
2. `pipeline.py`
3. `coincidence.py`
4. `correlation.py`
5. `reprocess_hist_leftpeak.py`

如果你是在处理多峰特殊数据，优先看：

1. `coincidence.py`
2. `coincidence_dualpeak.py`
3. `reprocess_hist_leftpeak.py`
