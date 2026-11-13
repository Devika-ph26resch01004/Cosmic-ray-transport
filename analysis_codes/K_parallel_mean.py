
#!/usr/bin/env python3
"""
Calculate parallel diffusion coefficients from exactly three
independent turbulent-field realisations.

For each realisation:
    MSD(t) = intercept + slope*t
    kappa_parallel = slope/2

The final result is the mean and sample standard deviation
of the three individual diffusion coefficients.

Input CSV columns:
    time_s
    msd_z_AU2
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_DIR = Path(r"D:/PhD/PhD courses/HEPA/Project/data/1000Mev/")

INPUT_FILES = [
    "ensemble_msd_N500_1000MeV.csv"
    #"ensemble_msd_N300_100MeV_R2.csv",
    #"ensemble_msd_N300_Lc_0.01_100MeV.csv",
]
'''
INPUT_FILES = [
    "ensemble_msd_N300_Lc_0.01.csv",
    "ensemble_msd_N300_Lc_0.01_R2.csv",
    "ensemble_msd_N300_Lc_0.01_R3.csv",
]

INPUT_FILES = [
    "ensemble_msd_N300_316MeV_R1.csv",
    "ensemble_msd_N300_316MeV_R2.csv",
    "ensemble_msd_N300_Lc_0.01_316MeV.csv",
]
'''
ENERGY_LABEL = "100 MeV"

# Common fitting interval
FIT_START_S = 200.0
FIT_END_S = 4000.0

# Output directory
OUTPUT_DIR = INPUT_DIR / "diffusion_analysis"

# Astronomical unit in metres
AU_M = 149_597_870_700.0

# ============================================================
# DATA LOADING
# ============================================================

def load_msd(filename):

    path = INPUT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path.resolve()}"
        )

    df = pd.read_csv(path)

    required = ["time_s", "msd_z_AU2"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns {missing} in {filename}. "
            f"Available columns: {df.columns.tolist()}"
        )

    t = pd.to_numeric(
        df["time_s"], errors="coerce"
    ).to_numpy(dtype=float)

    msd = pd.to_numeric(
        df["msd_z_AU2"], errors="coerce"
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(t)
        & np.isfinite(msd)
        & (t > 0)
        & (msd >= 0)
    )

    t = t[valid]
    msd = msd[valid]

    order = np.argsort(t)
    t = t[order]
    msd = msd[order]

    if len(t) < 3:
        raise ValueError(
            f"Insufficient valid data in {filename}"
        )

    if np.any(np.diff(t) <= 0):
        raise ValueError(
            f"Time values must be unique and increasing "
            f"in {filename}"
        )

    return t, msd


# ============================================================
# LINEAR FIT AND DIFFUSION COEFFICIENT
# ============================================================

def fit_msd(t, msd, start_s, end_s):

    mask = (
        (t >= start_s)
        & (t <= end_s)
        & (t > 0)
    )

    tf = t[mask]
    msdf = msd[mask]

    if len(tf) < 3:
        raise ValueError(
            f"Only {len(tf)} points in fitting interval "
            f"{start_s}–{end_s} s."
        )

    # Convert MSD from AU^2 to m^2
    msdf_m2 = msdf * AU_M**2

    # Linear least-squares fit
    slope, intercept = np.polyfit(
        tf, msdf_m2, 1
    )

    fitted = slope * tf + intercept

    # R-squared
    residuals = msdf_m2 - fitted

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum(
        (msdf_m2 - np.mean(msdf_m2))**2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0 else np.nan
    )

    # Parallel diffusion coefficient
    kappa = 0.5 * slope

    return {
        "start_s": tf[0],
        "end_s": tf[-1],
        "n_samples": len(tf),
        "slope_m2_s": slope,
        "intercept_m2": intercept,
        "kappa_parallel_m2_s": kappa,
        "r_squared": r2,
        "t_fit": tf,
        "msd_fit_AU2": msdf,
        "fitted_msd_AU2": fitted / AU_M**2,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    if len(INPUT_FILES) != 3:
        raise ValueError(
            "Exactly three turbulent-field realisations "
            "must be supplied."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []
    plot_data = []

    # --------------------------------------------------------
    # Process exactly three realisations
    # --------------------------------------------------------

    for i, filename in enumerate(INPUT_FILES):

        print(
            f"\nProcessing realisation {i + 1}: {filename}"
        )

        t, msd = load_msd(filename)

        result = fit_msd(
            t,
            msd,
            FIT_START_S,
            FIT_END_S
        )

        results.append(result)

        print(
            f"  Fitting interval : "
            f"{result['start_s']:.2f}–"
            f"{result['end_s']:.2f} s"
        )

        print(
            f"  Samples          : "
            f"{result['n_samples']}"
        )

        print(
            f"  Slope            : "
            f"{result['slope_m2_s']:.6e} m^2/s"
        )

        print(
            f"  Kappa_parallel   : "
            f"{result['kappa_parallel_m2_s']:.6e} m^2/s"
        )

        print(
            f"  R^2              : "
            f"{result['r_squared']:.6f}"
        )

        plot_data.append({
            "realisation": i + 1,
            "filename": filename,
            "fit_start_s": result["start_s"],
            "fit_end_s": result["end_s"],
            "n_samples": result["n_samples"],
            "slope_m2_s": result["slope_m2_s"],
            "intercept_m2": result["intercept_m2"],
            "kappa_parallel_m2_s":
                result["kappa_parallel_m2_s"],
            "r_squared": result["r_squared"],
        })

    # ========================================================
    # MEAN AND SAMPLE STANDARD DEVIATION
    # ========================================================

    kappas = np.array([
        r["kappa_parallel_m2_s"]
        for r in results
    ])

    mean_kappa = np.mean(kappas)

    # Sample standard deviation (N-1 denominator)
    std_kappa = np.std(
        kappas,
        ddof=1
    )

    print("\n" + "=" * 60)
    print(f"DIFFUSION COEFFICIENT: {ENERGY_LABEL}")
    print("=" * 60)

    for i, kappa in enumerate(kappas):

        print(
            f"Realisation {i + 1}: "
            f"{kappa:.6e} m^2/s"
        )

    print("-" * 60)

    print(
        f"Mean kappa_parallel : "
        f"{mean_kappa:.6e} m^2/s"
    )

    print(
        f"Sample std. dev.    : "
        f"{std_kappa:.6e} m^2/s"
    )

    print(
        f"Final result        : "
        f"({mean_kappa:.6e} +/- "
        f"{std_kappa:.6e}) m^2/s"
    )

    print("=" * 60)

    # ========================================================
    # SAVE NUMERICAL RESULTS
    # ========================================================

    summary_df = pd.DataFrame(plot_data)

    summary_df.to_csv(
        OUTPUT_DIR / "individual_realisation_results.csv",
        index=False
    )

    overall_df = pd.DataFrame([{
        "energy": ENERGY_LABEL,
        "fit_start_s": FIT_START_S,
        "fit_end_s": FIT_END_S,
        "number_of_realisations": 3,
        "mean_kappa_parallel_m2_s": mean_kappa,
        "sample_std_kappa_parallel_m2_s": std_kappa,
    }])

    overall_df.to_csv(
        OUTPUT_DIR / "mean_diffusion_coefficient.csv",
        index=False
    )
    
    
    # ============================================================
    # THREE-PANEL LINEAR MSD PLOT
    # ============================================================
    
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5.5),
        sharey=True
    )
    
    panel_labels = ["(a)", "(b)", "(c)"]
    
    for i, ax in enumerate(axes):
    
        # Load original MSD data
        t, msd = load_msd(INPUT_FILES[i])
    
        result = results[i]
    
        kappa = result["kappa_parallel_m2_s"]
        r2 = result["r_squared"]
    
        # Simulated MSD
        ax.plot(
            t,
            msd,
            linewidth=1.8,
            label="Simulated MSD"
        )
    
        # Individual linear fit
        ax.plot(
            result["t_fit"],
            result["fitted_msd_AU2"],
            "--",
            linewidth=2.2,
            label="Linear fit"
        )
    
        # Highlight fitting interval
        ax.axvspan(
            FIT_START_S,
            FIT_END_S,
            color="grey",
            alpha=0.12,
            label="Fitting interval"
        )
    
        # Panel label: (a), (b), (c)
        ax.text(
            0.04,
            0.95,
            panel_labels[i],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left"
        )
    
        # Diffusion coefficient text box
        kappa_text = (
            rf"$\kappa_\parallel = {kappa:.3e}$ m$^2$/s"
            "\n"
            rf"$R^2 = {r2:.5f}$"
        )
    
        ax.text(
            0.5,
            0.7,
            kappa_text,
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="black",
                alpha=0.9
            )
        )
    
        ax.set_xlabel(
            "Time (s)",
            fontsize=14
        )
    
        ax.grid(
            True,
            alpha=0.3
        )
    
        ax.legend(
            fontsize=16
            #loc="upper right"
        )
    
    axes[0].set_ylabel(
        r"$\langle \Delta z^2 \rangle$ (AU$^2$)",
        fontsize=12
    )
    
    
    
    fig.tight_layout()
    
    plot_path = (
        OUTPUT_DIR /
        "parallel_msd_three_panel_fits.png"
    )
    
    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.show()
    plt.close(fig)
    
    print(
        f"Three-panel fit plot saved: {plot_path.resolve()}"
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\nAnalysis completed successfully.")

    print(
        f"Output directory: {OUTPUT_DIR.resolve()}"
    )

    print("\nSaved files:")
    print("1. individual_realisation_results.csv")
    print("2. mean_diffusion_coefficient.csv")
    print("3. parallel_msd_three_panel_fits.png")


if __name__ == "__main__":
    main()
