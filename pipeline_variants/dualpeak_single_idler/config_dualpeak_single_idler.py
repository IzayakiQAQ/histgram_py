"""
Dual-peak pipeline configuration (single idler + two signals).

This config is intentionally separate from the original config.py so you can
run both workflows side-by-side.
"""

# -------- Paths --------
# Set these to your real data locations.
#
# Each can be:
#   - a string path to a .ttbin (volume auto-discovery supported)
#   - a list of strings (explicit volumes)
#
# Example:
#   SIGNAL_A = r"E:\data\sigA.ttbin"
#   SIGNAL_B = r"E:\data\sigB.ttbin"
#   IDLER    = r"E:\data\idler.ttbin"
SIGNAL_A = r""
SIGNAL_B = r""
IDLER = r""

# Output CSV path
SAVE_FILE_PATH = r"dualpeak_single_idler.csv"

# Optional: save 1ps-resolution histogram segments around each peak for each slice.
# Leave empty to disable (faster).
HIST_SAVE_DIR = r""
HIST_TARGET_POINTS = 65536


# -------- Correlation (Phase 0) --------
CORRELATION_WINDOW_PS = int(4e12)  # 4 s
CORRELATION_FRAMES = int(4e7)      # keep same as original unless you choose to tune


# -------- Slicing (Phase 1) --------
SPLIT_STEP_PS = int(10e12)         # 10 s per slice
SPLIT_TIME_PS = int(86400 * 1e12)  # 24 h total


# -------- Coincidence histogram --------
BIN_WIDTH_PS = 100
BIN_NUM = 10_000


# -------- Dual-peak detection/fit --------
# Minimum separation between the two peaks (ps). If you don't know yet, start with:
#   roughly (peak2 - peak1) / 2
DUALPEAK_MIN_SEPARATION_PS = 5_000

# Fitting window (in histogram bins) on each side of the detected peak bin.
DUALPEAK_FIT_HALF_WINDOW_BINS = 80
