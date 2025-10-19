# main.py
import atexit
import time
import os
import signal
import multiprocessing
import warnings

import numpy as np
import matplotlib

# Use a non-interactive backend for headless runs to avoid GUI hangs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configure matplotlib to use standard fonts
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]

from . import (
    data_loader,
    domain_conversion,
    wavelets,
    reflectivity,
    modeling_utils,
    visualization,
    avo_improvements,
)


def _terminate_children_on_exit(timeout=1.0):
    """Attempt to terminate any leftover multiprocessing children to avoid
    resource_tracker warnings about leaked semaphores (loky/joblib). This is a
    conservative, best-effort cleanup registered with atexit.
    """
    try:
        children = multiprocessing.active_children()
        if not children:
            return

        for p in children:
            try:
                p.terminate()
            except Exception:
                pass

        t0 = time.time()
        while any(p.is_alive() for p in children) and (time.time() - t0) < timeout:
            time.sleep(0.01)

        for p in children:
            try:
                if p.is_alive():
                    os.kill(p.pid, signal.SIGKILL)
            except Exception:
                pass
    except Exception:
        # Never raise during shutdown cleanup
        pass


atexit.register(_terminate_children_on_exit)

# Suppress the specific resource_tracker leaked semaphore warning which can be
# emitted by loky/joblib in certain environments. We keep this focused so other
# warnings still surface.
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects",
)


def main():
    """Main function to run the seismic modeling workflow."""
    # --- 1. Configuration Parameters ---
    DATA_PATH = "."
    # Map keys to the folder names in the GitHub repository
    FILE_MAP = {
        "vp": "P-wave Velocity",
        "vs": "S-wave Velocity",
        "rho": "Density",
        "facies": "Facies",
    }
    GRID_SHAPE = (150, 200, 200)  # (nx, ny, nz)
    DZ = 1.0  # Vertical spacing in meters
    DT = 0.001  # Time sampling interval in seconds (1 ms)
    F_PEAK = 30  # Wavelet peak frequency in Hz
    # Updated based on regularization study: 45° excluded due to poor performance
    # with large contrasts (Stanford VI-E has 25% Vp contrasts, violates Aki-Richards)
    # Max 15° config achieved r=0.146 vs baseline (45°) r=0.057 (2.6× improvement)
    ANGLES = [0, 15, 30]  # Optimal angle range for sharp-boundary datasets

    # --- 2. Load and Prepare Data ---
    props_depth = data_loader.load_stanfordsix_data(DATA_PATH, FILE_MAP, GRID_SHAPE)

    # Convert velocity from km/s to m/s (Stanford VI-E data is in km/s)
    print("Converting velocity units from km/s to m/s...")
    print(
        f"  Vp before: {props_depth['vp'].min():.3f} - {props_depth['vp'].max():.3f} km/s"
    )
    props_depth["vp"] = props_depth["vp"] * 1000.0
    if "vs" in props_depth:
        props_depth["vs"] = props_depth["vs"] * 1000.0
    print(
        f"  Vp after: {props_depth['vp'].min():.0f} - {props_depth['vp'].max():.0f} m/s"
    )

    # --- 3. Domain Conversion ---
    twt_irregular = domain_conversion.convert_depth_to_twt(props_depth["vp"], DZ)
    props_time, _ = domain_conversion.resample_properties_to_time(
        props_depth, twt_irregular, DT
    )

    # --- 4. Generate Wavelet ---
    wavelet = wavelets.ricker_wavelet(f_peak=F_PEAK, dt=DT)

    # --- 4.5. Pre-Flight Validity Check ---
    print("\n--- Checking Aki-Richards Linearization Validity ---")
    max_angle = max(ANGLES)
    validity = avo_improvements.check_linearization_validity(
        props_depth["vp"], props_depth["vs"], props_depth["rho"], max_angle=max_angle
    )
    avo_improvements.print_validity_report(validity)

    if validity["suggested_angles"] and validity["suggested_angles"] != ANGLES:
        print(f"\n⚠️  WARNING: Current angles {ANGLES} may not be optimal.")
        print(f"   Recommended angles: {validity['suggested_angles']}")
        print(f"   Proceeding with current configuration...\n")

    # --- 5. Run Modeling Workflows ---
    print("\n--- Starting Technique 1: AVO Modeling ---")
    _, full_stack_avo = modeling_utils.cached_avo(props_time, ANGLES, wavelet)

    print("\n--- Starting Technique 2: Acoustic Impedance Modeling ---")
    seismogram_ai = modeling_utils.cached_ai_seismogram(props_time, wavelet)
    print("Modeling complete.")

    # --- 6. Visualization ---
    print("\nGenerating 3D visualizations...")
    # Compute center slice indices from the facies cube shape (ni, nj, nk)
    ni, nj, nk = props_time["facies"].shape
    slice_indices = (ni // 2, nj // 2, nk // 2)

    # Use lightweight 2D slice plotting to avoid heavy 3D surface rendering
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Seismic Modeling Results (Time Domain)", fontsize=14)

    ax1 = fig.add_subplot(131)
    visualization.plot_2d_slices(
        ax1, props_time["facies"], slice_indices, "Facies Model", cmap="viridis"
    )

    ax2 = fig.add_subplot(132)
    visualization.plot_2d_slices(ax2, full_stack_avo, slice_indices, "AVO Full Stack")

    ax3 = fig.add_subplot(133)
    visualization.plot_2d_slices(
        ax3, seismogram_ai, slice_indices, "Acoustic Impedance Model"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(".cache", exist_ok=True)
    outfn = os.path.join(".cache", "result.png")
    plt.savefig(outfn, dpi=150)
    print(f"Saved visualization to {outfn}")


if __name__ == "__main__":
    main()
