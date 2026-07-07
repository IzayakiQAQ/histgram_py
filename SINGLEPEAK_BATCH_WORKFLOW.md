# Single-Peak Batch Workflow

This repository contains the current workflow used for TimeTagger `.ttbin`
single-peak processing:

1. Read split `.ttbin` files in streaming mode.
2. Estimate or use a manual `time_diff_ps`.
3. Process data in 10 s slices.
4. Save fixed-axis 1 ps raw histograms.
5. Refit saved histograms with a Gaussian model.
6. Optionally combine the 31 and 42 peak CSV files into clock correction.

## Required Scripts

- `run_singlepeak_batch.py`
  Main batch entry point. Reads a JSON job file, streams `.ttbin` data,
  writes `singlepeak_peaks.csv`, and saves per-slice raw histograms.

- `postprocess_fixedaxis_peaks.py`
  Reads saved fixed-axis raw histograms and performs Gaussian peak fitting.
  Writes `singlepeak_peaks_gaussian.csv` and
  `singlepeak_peak_quality_gaussian.csv`.

- `coincidence.py`
  Computes one-slice coincidence histograms and supports fixed-axis histogram
  export through `save_hist_center_ps` and `save_hist_points`.

- `ttbin_reader.py`
  Streaming `.ttbin` reader. It automatically discovers split files such as
  `data.1.ttbin`, `data.2.ttbin`, etc.

- `correlation.py`
  FFT-based coarse `time_diff_ps` estimation.

- `config.py`
  Legacy/default configuration loaded by scripts when a value is not provided
  by JSON.

## Typical Commands

Run a short check job first:

```powershell
python .\run_singlepeak_batch.py --job-file .\singlepeak_50km_280hz_20260629_31_fixedaxis_check_1ps.json
```

Run a full or partial batch job:

```powershell
python .\run_singlepeak_batch.py --job-file .\singlepeak_50km_280hz_20260629_31_fixedaxis_10000s_1ps.json
```

Run Gaussian postprocessing:

```powershell
python .\postprocess_fixedaxis_peaks.py `
  --job-file .\singlepeak_50km_280hz_20260629_31_fixedaxis_10000s_1ps.json `
  --job-dir "E:\lzy\测试结果\2026.6.29 50km 280Hz\单峰_31_1ps固定横坐标_10000s\20260629_50km_280hz_ch3_ch1_fixedaxis_10000s_1ps" `
  --box-width-ps 10000 `
  --fit-bin-width-ps 20 `
  --fit-half-width-ps 25000 `
  --center-bound-ps 15000 `
  --sigma-min-ps 20 `
  --sigma-max-ps 30000 `
  --workers 8
```

## Notes

- Use `max_slices = 1000` for the first 10000 s when the slice length is 10 s.
- Use fixed-axis histogram saving with `save_hist_center_ps = 100000` and
  `save_hist_points = 65536` for visual comparison across slices.
- If the saved fixed-axis histogram misses the peak, adjust `time_diff_ps` so
  the fitted peak falls near the fixed save center.
- `.ttbin`, `.csv`, and `.log` files are ignored by Git.
