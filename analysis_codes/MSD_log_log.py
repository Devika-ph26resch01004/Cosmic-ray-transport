#!/usr/bin/env python3
"""
Plot the parallel mean-square displacement (MSD) on log-log axes.

Input CSV columns:
    time_s
    mean_square_parallel_displacement_AU2

A reference alpha=1 line illustrates diffusive-like scaling (MSD proportional
to t). A reference alpha=2 line illustrates ballistic-like scaling (MSD
proportional to t**2). These are visual guides, not fitted regimes.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS: edit these for each run
# ============================================================
INPUT_DIR = Path(".")
INPUT_FILE = "D:/PhD/PhD courses/HEPA/Project/data/1000Mev/ensemble_msd_N500_1000MeV.csv"
OUTPUT_DIR = None  # None saves outputs in INPUT_DIR

# Optional time limits. None means use the full available positive data.
PLOT_START_S = None
PLOT_END_S = None

# Draw reference scaling guides, anchored at this time and MSD.
SHOW_REFERENCE_LINES = True
REFERENCE_TIME_S = 500.0
REFERENCE_MSD_AU2 = 0.03
REFERENCE_TIME_MIN_S = 10.0
REFERENCE_TIME_MAX_S = None  # None means last available time

# ============================================================

def load_msd(path):
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path.resolve()}")

    df = pd.read_csv(path)
    required = ["time_s", "msd_z_AU2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}; "
            f"available columns: {df.columns.tolist()}"
        )

    t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
    msd = pd.to_numeric(
        df["msd_z_AU2"], errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(msd) & (t > 0) & (msd > 0)
    t, msd = t[valid], msd[valid]
    order = np.argsort(t)
    t, msd = t[order], msd[order]

    if len(t) < 3:
        raise ValueError("Not enough positive, finite MSD samples.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time values must be unique and strictly increasing.")
    return t, msd


def main():
    output_dir = INPUT_DIR if OUTPUT_DIR is None else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = INPUT_DIR / INPUT_FILE
    t, msd = load_msd(input_path)

    start = t[0] if PLOT_START_S is None else float(PLOT_START_S)
    end = t[-1] if PLOT_END_S is None else float(PLOT_END_S)
    mask = (t >= start) & (t <= end)
    if np.count_nonzero(mask) < 3:
        raise ValueError(f"Fewer than 3 valid points in plotting interval {start:g}–{end:g} s.")
    tp, mp = t[mask], msd[mask]

    out_csv = output_dir / "loglog_msd_plot_data.csv"
    pd.DataFrame({"time_s": tp, "MSD_AU2": mp}).to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(tp, mp, linewidth=1.8, label="Simulated parallel MSD")

    if SHOW_REFERENCE_LINES:
        ref_start = max(REFERENCE_TIME_MIN_S, tp[0])
        ref_end = tp[-1] if REFERENCE_TIME_MAX_S is None else min(float(REFERENCE_TIME_MAX_S), tp[-1])
        if ref_start < ref_end and REFERENCE_TIME_S > 0 and REFERENCE_MSD_AU2 > 0:
            tr = np.logspace(np.log10(ref_start), np.log10(ref_end), 200)
            # Both guide lines pass through the chosen anchor point.
            msd_alpha1 = REFERENCE_MSD_AU2 * (tr / REFERENCE_TIME_S)
            msd_alpha2 = REFERENCE_MSD_AU2 * (tr / REFERENCE_TIME_S) ** 2
            ax.loglog(tr, msd_alpha1, "--", linewidth=1.4,
                      label=r"Reference slope $\alpha=1$")
            ax.loglog(tr, msd_alpha2, ":", linewidth=1.4,
                      label=r"Reference slope $\alpha=2$")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\langle \Delta z^2 \rangle$ (AU$^2$)")
    ax.set_title("Parallel MSD (log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = output_dir / "loglog_msd.png"
    fig.savefig(out_png, dpi=200)
    plt.show()
    plt.close(fig)
    
    print("Log-log MSD plot complete")
    print(f"Input file : {input_path.resolve()}")
    print(f"Time range : {tp[0]:.6g}–{tp[-1]:.6g} s")
    print(f"CSV saved  : {out_csv.resolve()}")
    print(f"Plot saved : {out_png.resolve()}")
    print("Reference lines are visual guides only; they are not fits to the data.")


if __name__ == "__main__":
    main()

