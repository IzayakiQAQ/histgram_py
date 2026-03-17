# Histogram Processing Pipeline (histgram_py)

这是一个专为 TimeTagger 原始时间戳数据（.ttbin）设计的、高内存效率的符合计数处理流水线。

## 技术栈 (Technology Stack)

- **语言**: Python 3.x
- **核心库**:
    - **NumPy**: 用于高性能向量化数值计算。
    - **SciPy**: 提供高性能 FFT 互相关与非线性最小二乘高斯拟合（`curve_fit`）。
    - **TimeTagger (Swabian Instruments)**: 调用官方 Python API 进行二进制文件流式读取。
    - **tqdm**: 提供可视化的处理进度反馈。

## 核心特性

1. **内存友好**: 通过分片（Chunking）流式读取，即使处理 24h+ 的 TB 级原始数据，内存占用也能稳定在数百 MB 以内。
2. **自动分卷识别**: 支持 Swabian 的物理分卷（.1.ttbin, .2.ttbin...），只需填写主文件名即可自动关联。
3. **并行加速**: 采用多进程并行处理符合峰拟合计算，大幅缩短长时间序列的处理耗时。
4. **即时写出**: 每段结果实时写入 CSV，防止程序意外中断导致数据丢失。

## 操作方式 (Getting Started)

### 1. 环境准备
确保已安装 Swabian Instruments 的 TimeTagger 软件及其对应的 Python 库，并安装以下依赖：
```bash
pip install numpy scipy tqdm
```

### 2. 配置参数
在运行前，请修改 `config.py`：
- **`DIR`**: 设置您的数据存放文件夹。
- **`FILE_PAIRS`**: 设置需要比对的文件对。只需写主文件名（如 `test.ttbin`），系统会自动寻找物理分卷。
- **`BIN_WIDTH_PS` / `BIN_NUM`**: 调整符合计数的直方图精度。

### 3. 启动分析
直接运行主程序即可：
```bash
python pipeline.py
```

## 文件结构
- `pipeline.py`: 核心调度逻辑。
- `ttbin_reader.py`: 负责高效的 .ttbin 流式读取与自动分卷。
- `correlation.py`: 负责寻找 signal 与 idler 之间的全局时间对齐。
- `coincidence.py`: 向量化符合计数与高斯拟合。
- `config.py`: 所有可调参数的集中地。
