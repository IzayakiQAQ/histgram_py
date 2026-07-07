import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(csv_path: Path, out_path: Path):
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected histogram format: {csv_path}")

    x = data[:, 0]
    y = data[:, 1]
    peak_idx = int(np.argmax(y)) if y.size else 0
    peak_x = float(x[peak_idx]) if y.size else 0.0
    peak_y = float(y[peak_idx]) if y.size else 0.0

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    ax.plot(x, y, linewidth=0.8)
    ax.axvline(peak_x, color="tab:red", linewidth=0.9, alpha=0.8)
    ax.set_title(f"{csv_path.parent.name}  peak={peak_x:.0f} ps  count={peak_y:.0f}")
    ax.set_xlabel("time_ps")
    ax.set_ylabel("counts")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot saved histogram CSV files.")
    parser.add_argument("root", help="Root directory containing *_histograms_raw_* folders.")
    parser.add_argument("--pattern", default="hist_raw_00001.csv", help="Histogram CSV filename to plot.")
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.glob(f"**/{args.pattern}"))
    if not files:
        raise FileNotFoundError(f"No {args.pattern} files found under {root}")

    for csv_path in files:
        out_path = csv_path.with_suffix(".png")
        plot_histogram(csv_path, out_path)
        print(out_path)

    print(f"[done] plotted {len(files)} histograms")


if __name__ == "__main__":
    main()
