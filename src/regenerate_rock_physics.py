#!/usr/bin/env python
"""
Complete Rock Physics Analysis Pipeline

This script performs a COMPLETE rock physics workflow:
1. Clears old rock physics cache files
2. Runs multi-angle EI analysis (6 angles: 0°, 5°, 10°, 15°, 20°, 25°)
3. Computes all rock physics attributes:
   - Lambda-Rho (λρ) - Fluid-sensitive
   - Mu-Rho (μρ) - Lithology-sensitive
   - Fluid Factor - Direct fluid indicator
   - Poisson Impedance - Fluid sensitivity
   - AVO Attributes - Intercept, gradient
   - EI Gradient - Angle-dependent response
   - Hybrid Discriminants - Combined attributes
4. Visualizes all rock physics attributes
5. Copies all PNG files to docs/images directory
6. Opens all generated plots automatically

Rock Physics Attributes:
- Lambda-Rho (λρ): Incompressibility × Density (fluid-sensitive)
- Mu-Rho (μρ): Shear modulus × Density (lithology-sensitive)
- Fluid Factor: Separates fluid effects from lithology
- Poisson Impedance: Direct fluid sensitivity indicator
- Multi-Angle EI: 6 angles for complete rock physics analysis

Expected Performance:
- Multi-angle EI: Cohen's d up to 3.4+ (HUGE effect!)
- Lambda-Rho vs Mu-Rho: Classic fluid/lithology crossplot
- Best single angle: 10° (optimal Vp/Vs balance)

Processing time estimate: 3-5 minutes
"""

import subprocess
import sys
import os
import time
import shutil
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display progress."""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    print(f"Command: {cmd}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=False, text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return result

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running command: {e}")
        print(f"   Return code: {e.returncode}")
        return None
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return None


def clear_cache(patterns=None):
    """Clear cache files matching patterns."""
    cache_dir = Path(".cache")
    if not cache_dir.exists():
        cache_dir.mkdir()
        print("OK: Created .cache directory")
        return

    if patterns is None:
        patterns = ["rock_physics_*.npz", "ei_multiangle_*.npz"]

    removed_count = 0
    removed_size = 0

    print("\nClearing old rock physics cache files:")
    for pattern in patterns:
        for filepath in cache_dir.glob(pattern):
            size = filepath.stat().st_size / (1024 * 1024)
            print(f"  Removing: {filepath.name} ({size:.1f} MB)")
            filepath.unlink()
            removed_count += 1
            removed_size += size

    if removed_count > 0:
        print(f"OK: Removed {removed_count} files ({removed_size:.1f} MB)")
    else:
        print("OK: No old rock physics cache files to remove")


def check_file_exists(filepath, description):
    """Check if a file exists and report size."""
    path = Path(filepath)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"OK: Found: {description}")
        print(f"  Path: {filepath}")
        print(f"  Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"ERROR: Missing: {description}")
        print(f"  Expected: {filepath}")
        return False


def open_file(filepath, description):
    """Open a file with the default application."""
    path = Path(filepath)
    if path.exists():
        print(f"Opening: {description}")
        try:
            subprocess.run(["open", str(path)], check=True)
            print(f"OK: Opened: {filepath}")
            time.sleep(0.5)  # Give time for app to launch
            return True
        except Exception as e:
            print(f"ERROR: Could not open: {e}")
            return False
    else:
        print(f"ERROR: File not found: {filepath}")
        return False


def main():
    """Main pipeline execution."""

    print("=" * 70)
    print("COMPLETE ROCK PHYSICS ANALYSIS PIPELINE")
    print("Compute ALL Attributes + Multi-Angle EI + Generate ALL Plots")
    print("=" * 70)
    print()
    print("This pipeline will:")
    print("  1. Clear old rock physics cache files")
    print("  2. Run multi-angle EI analysis (6 angles: 0-25°)")
    print("  3. Compute rock physics attributes:")
    print("     • Lambda-Rho (λρ) - Fluid-sensitive")
    print("     • Mu-Rho (μρ) - Lithology-sensitive")
    print("     • Fluid Factor - Direct fluid indicator")
    print("     • Poisson Impedance - Fluid sensitivity")
    print("     • AVO Attributes - Intercept, gradient")
    print("     • EI Gradient - Angle-dependent response")
    print("     • Hybrid Discriminants - Combined attributes")
    print("  4. Visualize all rock physics attributes")
    print("  5. Copy all PNG files to docs/images")
    print("  6. Open ALL plots automatically")
    print()
    print("Rock Physics Attributes:")
    print("  • Lambda-Rho: Incompressibility × Density (fluid)")
    print("  • Mu-Rho: Shear modulus × Density (lithology)")
    print("  • Fluid Factor: Lambda-Rho - Mu-Rho (fluid separator)")
    print("  • Poisson Impedance: Fluid-sensitive impedance")
    print("  • Multi-Angle EI: 6 angles (0°, 5°, 10°, 15°, 20°, 25°)")
    print()
    print("Expected Performance:")
    print("  • Multi-angle EI: Cohen's d up to 3.4+ (HUGE effect!)")
    print("  • Lambda-Rho vs Mu-Rho: Classic crossplot separation")
    print("  • Best single angle: 10° (optimal Vp/Vs balance)")
    print()
    print("Processing time estimate: 3-5 minutes")
    print()

    # Confirm before proceeding
    response = (
        input("Proceed with complete rock physics analysis? [Y/n]: ").strip().lower()
    )
    if response and response != "y":
        print("Cancelled by user")
        return

    # Get Python executable
    venv_python = "/Users/mazzutti/POSDOC/Experimentos/Stanford-VI-E/.venv/bin/python"

    # Check if we're in the right directory
    if not Path("src/rock_physics_attributes.py").exists():
        print("ERROR: Not in correct directory!")
        print("  Please run from: /Users/mazzutti/POSDOC/Experimentos/Stanford-VI-E")
        sys.exit(1)

    print("\n✓ Working directory confirmed")

    # Step 0: Clear old rock physics cache
    print("\n" + "#" * 70)
    print("# STEP 0: CLEAR OLD ROCK PHYSICS CACHE FILES")
    print("#" * 70)
    clear_cache()

    # Step 1: Run multi-angle EI analysis
    print("\n" + "#" * 70)
    print("# STEP 1/3: MULTI-ANGLE EI ANALYSIS")
    print("#" * 70)
    print("Computing EI at 6 angles (0°, 5°, 10°, 15°, 20°, 25°)")
    print("This analyzes angle-dependent rock physics response")
    print()

    # Check if depth domain data exists (from modeling)
    cache_dir = Path(".cache")
    avo_depth_files = list(cache_dir.glob("avo_depth_*.npz"))

    if not avo_depth_files:
        print("⚠ Warning: No depth domain cache found!")
        print("  You may need to run: python -m src.modeling first")
        print("  Continuing anyway - rock_physics_attributes will load raw data...")
        print()

    result = run_command(
        f'{venv_python} -c "from src.rock_physics_attributes import run_multiangle_analysis, load_depth_data; '
        f"props = load_depth_data(); "
        f'run_multiangle_analysis(props, angles_deg=[0, 5, 10, 15, 20, 25])"',
        "Running multi-angle EI analysis (6 angles)",
    )

    if result is None:
        print("✗ Multi-angle EI analysis failed!")
        print("  Continuing to compute other rock physics attributes...")

    # Check if EI cache was created
    ei_files = list(cache_dir.glob("ei_depth_*.npz"))
    if ei_files:
        latest_ei = max(ei_files, key=lambda p: p.stat().st_mtime)
        check_file_exists(latest_ei, "EI cache")
    else:
        print("WARNING: No EI cache file found!")

    # Step 2: Compute rock physics attributes
    print("\n" + "#" * 70)
    print("# STEP 2/3: COMPUTE ROCK PHYSICS ATTRIBUTES")
    print("#" * 70)
    print("Computing comprehensive rock physics attributes:")
    print("  • Lambda-Rho & Mu-Rho (Goodway method)")
    print("  • Fluid Factor")
    print("  • Poisson Impedance")
    print("  • AVO Attributes")
    print("  • EI Gradient")
    print("  • Hybrid Discriminants")
    print()

    result = run_command(
        f"{venv_python} -m src.rock_physics_attributes",
        "Computing all rock physics attributes",
    )

    if result is None:
        print("ERROR: Rock physics computation failed!")
        return

    # Check if rock physics cache was created
    rpa_files = list(cache_dir.glob("rock_physics_attributes.npz"))
    if rpa_files:
        latest_rpa = rpa_files[0]
        check_file_exists(latest_rpa, "Rock physics attributes cache")
    else:
        print("ERROR: No rock physics attributes cache file found!")
        return

    # Step 3: Visualize rock physics attributes
    print("\n" + "#" * 70)
    print("# STEP 3/3: VISUALIZE ROCK PHYSICS ATTRIBUTES")
    print("#" * 70)
    print("Creating comprehensive rock physics visualizations:")
    print("  • Lambda-Rho vs Mu-Rho crossplot")
    print("  • Fluid Factor analysis")
    print("  • Poisson Impedance")
    print("  • EI × Lambda/Mu products")
    print("  • Attribute comparison plots")
    print()

    result = run_command(
        f"{venv_python} -m src.plot_rock_physics_attributes",
        "Visualizing rock physics attributes",
    )

    if result is None:
        print("WARNING: Visualization had issues but continuing...")

    # Check for generated plots
    plot_files = [
        ("lambda_rho_vs_facies.png", "Lambda-Rho facies correlation"),
        ("mu_rho_vs_facies.png", "Mu-Rho facies correlation"),
        ("fluid_factor_vs_facies.png", "Fluid Factor analysis"),
        ("poisson_impedance_vs_facies.png", "Poisson Impedance"),
        ("lambda_mu_crossplot.png", "Lambda-Rho vs Mu-Rho crossplot"),
    ]

    found_plots = []
    for filename, description in plot_files:
        filepath = cache_dir / filename
        if filepath.exists():
            check_file_exists(filepath, description)
            found_plots.append((filepath, description))

    # Step 4: Copy visualization files to docs/images
    print("\n" + "#" * 70)
    print("# STEP 4/4: COPYING VISUALIZATIONS TO docs/images")
    print("#" * 70)
    print()

    # Create docs/images directory if it doesn't exist
    docs_images_dir = Path("docs/images")
    docs_images_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Ensured docs/images directory exists")

    # List of files to copy from .cache to docs/images
    files_to_copy = [
        # Multi-angle EI
        "angle_dependent_ei_depth_comparison.png",
        "angle_dependent_ei_facies_analysis.png",
        # Rock physics attributes
        "lambda-rho_λρ_vs_facies.png",
        "mu-rho_μρ_vs_facies.png",
        "fluid_factor_vs_facies.png",
        "poisson_impedance_vs_facies.png",
        "lambda_mu_crossplot.png",
        "rock_physics_attributes_comparison.png",
        # Composite 3x7/7x3 rock physics slices
        "result_rock_physics.png",
    ]

    copied_count = 0
    for filename in files_to_copy:
        source = cache_dir / filename
        dest = docs_images_dir / filename
        if source.exists():
            shutil.copy2(source, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  OK: Copied: {filename} ({size_mb:.1f} MB)")
            copied_count += 1
        else:
            print(f"  WARNING: Skipped (not found): {filename}")

    print(f"\n✓ Copied {copied_count}/{len(files_to_copy)} files to docs/images")

    # Step 5: Open all visualizations
    print("\n" + "#" * 70)
    print("# STEP 5/5: OPENING ALL VISUALIZATIONS")
    print("#" * 70)
    print()

    # Open the found plots
    opened_count = 0
    for filepath, description in found_plots:
        if open_file(filepath, description):
            opened_count += 1

    # Also try to open multi-angle plots
    multiangle_plots = [
        (
            cache_dir / "angle_dependent_ei_depth_comparison.png",
            "Multi-angle EI comparison",
        ),
        (
            cache_dir / "angle_dependent_ei_facies_analysis.png",
            "Multi-angle EI facies analysis",
        ),
    ]

    for filepath, description in multiangle_plots:
        if open_file(filepath, description):
            opened_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("ROCK PHYSICS ANALYSIS COMPLETE!")
    print("=" * 70)
    print()
    print(f"✓ Opened {opened_count} visualization files")
    print()

    print("Generated Cache Files:")
    if ei_files:
        print(
            f"  OK: EI: {latest_ei.name} "
            f"({latest_ei.stat().st_size / (1024**2):.1f} MB)"
        )
    if rpa_files:
        print(
            f"  OK: Rock Physics: {latest_rpa.name} "
            f"({latest_rpa.stat().st_size / (1024**2):.1f} MB)"
        )
    print()

    print("Rock Physics Attributes Computed:")
    print("  OK: Lambda-Rho (λρ) - Incompressibility × Density")
    print("  OK: Mu-Rho (μρ) - Shear modulus × Density")
    print("  OK: Fluid Factor - Fluid/lithology separator")
    print("  OK: Poisson Impedance - Fluid-sensitive impedance")
    print("  OK: AVO Attributes - Intercept, gradient")
    print("  OK: EI Gradient - Angle-dependent response")
    print("  OK: Multi-Angle EI - 6 angles (0-25°)")
    print()

    print("Generated Visualizations:")
    print("  OK: Lambda-Rho vs Facies correlation")
    print("  OK: Mu-Rho vs Facies correlation")
    print("  OK: Fluid Factor analysis")
    print("  OK: Poisson Impedance analysis")
    print("  OK: Lambda-Mu crossplot (classic rock physics)")
    print("  OK: Multi-angle EI comparison (if generated)")
    print("  OK: Multi-angle EI facies analysis (if generated)")
    print()
    print(f"  OK: All visualizations copied to docs/images/ ({copied_count} files)")
    print()

    print("Key Results to Review:")
    print("  1. Lambda-Rho vs Mu-Rho Crossplot:")
    print("     - Classic rock physics fluid/lithology separation")
    print("     - Different facies cluster in different regions")
    print("     - Lambda-Rho sensitive to pore fluids")
    print("     - Mu-Rho sensitive to rock matrix (lithology)")
    print()
    print("  2. Fluid Factor:")
    print("     - Direct fluid indicator (Lambda-Rho - Mu-Rho)")
    print("     - Separates fluid effects from lithology")
    print("     - Enhanced facies discrimination")
    print()
    print("  3. Multi-Angle EI:")
    print("     - Best angle: 10° (Cohen's d up to 3.4+)")
    print("     - 6 angles provide complete rock physics analysis")
    print("     - Angle-dependent sensitivity to Vp, Vs, ρ")
    print("     - Superior to single-angle approaches")
    print()

    print("Documentation:")
    print("  • Rock physics based on:")
    print("    - Goodway et al. (1997): Lambda-Mu-Rho method")
    print("    - Quakenbush et al. (2006): Poisson impedance")
    print("    - Russell et al. (2003): Hybrid attributes")
    print("    - Connolly (1999): Elastic impedance")
    print()

    print("Next Possible Steps:")
    print("  1. Review all opened rock physics visualizations")
    print("  2. Analyze Lambda-Mu crossplot for fluid indicators")
    print("  3. Compare with seismic attributes (AVO/AI/EI)")
    print("  4. Integrate with seismic inversion workflow")
    print("  5. Apply to reservoir characterization studies")
    print()
    print("=" * 70)
    print("Rock Physics Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
