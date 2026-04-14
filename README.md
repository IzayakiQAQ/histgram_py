# histgram_py

Streaming coincidence-histogram processing tools for Swabian TimeTagger `.ttbin` data.

This repository is built around long-run, large-volume timestamp data where loading everything into memory is not practical. The core workflow reads `.ttbin` files in chunks, estimates the signal-idler time offset, slices the data into fixed time windows, extracts coincidence peaks, and writes per-slice results plus optional raw histogram CSVs.

## What Is In This Repo

- `pipeline.py`
  Main two-pair processing pipeline.
- `ttbin_reader.py`
  Streaming `.ttbin` reader with automatic multi-volume discovery.
- `correlation.py`
  FFT-based coarse time-offset estimation.
- `coincidence.py`
  Coincidence histogram generation, local Gaussian fitting, and raw histogram export.
- `reprocess_hist_leftpeak.py`
  Reprocesses saved histogram CSVs by selecting the leftmost significant peak and optionally fitting it.
- `coincidence_dualpeak.py`
  Utility for datasets with a clear two-peak structure.
- `pipeline_variants/dualpeak_single_idler/`
  Variant pipeline for one idler shared by two signal channels.
- `tests_misaligned/test_time_offset_misaligned.py`
  Synthetic regression test for offset estimation under partial overlap and noise.

## Main Features

- Streaming processing for large `.ttbin` datasets
- Automatic discovery of split volumes such as `.ttbin.1`, `.ttbin.2`, ...
- FFT-based coarse offset estimation
- Per-slice coincidence peak extraction with local Gaussian fitting
- Raw histogram export at configurable saved resolution such as `1 ps` or `100 ps`
- Histogram reprocessing tools for special cases like double-peak or triple-peak data

## Requirements

- Python 3.x
- Swabian TimeTagger Python package
- `numpy`
- `scipy`
- `tqdm`

Install the Python dependencies with:

```bash
pip install numpy scipy tqdm
```

## Typical Workflow

### 1. Configure the main run

Edit `config.py` for your local dataset and output paths.

Important fields:

- `FILE_PAIRS`
  Input signal/idler pairs. Each path can be a single `.ttbin` or a list of volumes.
- `OUTPUT_DIR`
  Root directory for generated histograms and CSV outputs.
- `SAVE_HIST_BIN_WIDTHS_PS`
  Saved raw histogram resolution for each pair.
- `CORRELATION_WINDOW_PS`
  Head-window duration used for coarse offset estimation.
- `CORRELATION_FRAMES`
  Histogram bins used during coarse correlation.
- `SPLIT_STEP_PS`
  Per-slice duration.
- `BIN_WIDTH_PS`, `BIN_NUM`
  Coarse coincidence histogram settings used for peak finding.

If a pair already has a trusted offset, you can provide:

```python
'time_diff_ps': <known_offset_in_ps>
```

inside the corresponding `FILE_PAIRS` entry to bypass automatic correlation for that pair.

### 2. Run the main pipeline

```bash
python .\pipeline.py
```

Outputs typically include:

- `pair0_histograms_raw_<N>ps/`
- `pair1_histograms_raw_<N>ps/`
- `hcf.csv`
- `data_py.csv`

where `<N>` is the saved histogram resolution from `SAVE_HIST_BIN_WIDTHS_PS`.

### 3. Reprocess saved histograms when needed

For special datasets where the raw histogram contains multiple peaks and you want a controlled post-processing rule, use:

```bash
python .\reprocess_hist_leftpeak.py --root-dir <run_output_dir>
```

This script can:

- select the leftmost significant peak
- fit that peak locally
- write a cleaned output CSV
- write a debug CSV with raw peak positions and fit status

### 4. Use the dual-peak variant if your dataset needs it

The variant pipeline lives under:

- `pipeline_variants/dualpeak_single_idler/`

Set the paths in:

- `pipeline_variants/dualpeak_single_idler/config_dualpeak_single_idler.py`

Then run:

```bash
python .\pipeline_variants\dualpeak_single_idler\pipeline_dualpeak_single_idler.py
```

## Raw Histogram Saving Logic

The current `coincidence.py` behavior is:

1. Build the coarse coincidence histogram using `BIN_WIDTH_PS`
2. Find the strongest coarse bin
3. Fit locally around that coarse peak
4. Build the full `1 ps` histogram internally
5. Re-center the saved window around the fitted peak neighborhood
6. Save either:
   - full `1 ps` bins, or
   - rebinned output such as `100 ps`

This is meant to avoid saving windows centered on isolated noise spikes.

## Tests

Run the synthetic correlation test with:

```bash
python .\tests_misaligned\test_time_offset_misaligned.py
```

You can also do a quick syntax check on the main scripts with:

```bash
python -m py_compile coincidence.py pipeline.py correlation.py
```

## Notes

- `config.py` is intentionally local and experiment-specific. Before sharing your own branch, review the embedded file paths.
- CSV outputs are ignored by `.gitignore`.
- Temporary debug folders such as `_tmp_pair0_save_check/` are also ignored.

## Repository Status

This repository now contains:

- the main two-pair processing pipeline
- a histogram reprocessing tool for left-peak extraction
- a dual-peak helper and a dual-peak pipeline variant
- a regression test for misaligned coarse-correlation inputs

If you are updating the code for a new experiment, the usual starting points are `config.py`, `pipeline.py`, and `coincidence.py`.
