"""
Generate multi-angle EI visualization plots.

This script creates comparison and analysis plots for multi-angle Elastic Impedance.
Requires that multi-angle EI has been computed and cached.

Usage:
    python -m src.plot_multiangle_ei
"""

import numpy as np
import os
from pathlib import Path
from .rock_physics_attributes import (
    plot_multiangle_ei_comparison,
    plot_multiangle_ei_facies_analysis,
)


def main():
    """Generate multi-angle EI plots from cached data."""

    print("=" * 70)
    print("MULTI-ANGLE EI VISUALIZATION")
    print("=" * 70)

    cache_dir = ".cache"

    # Find the most recent EI cache file
    ei_files = list(Path(cache_dir).glob("ei_depth_*.npz"))

    if not ei_files:
        print("\n❌ ERROR: No multi-angle EI cache file found!")
        print("Please run first:")
        print(
            '  python -c "from src.rock_physics_attributes import run_multiangle_analysis, load_depth_data;'
        )
        print("             props = load_depth_data();")
        print(
            '             run_multiangle_analysis(props, angles_deg=[0, 5, 10, 15, 20, 25])"'
        )
        return

    # Load the most recent file
    latest_ei_file = max(ei_files, key=lambda p: p.stat().st_mtime)
    print(f"\nLoading EI cache: {latest_ei_file.name}")

    ei_data = np.load(latest_ei_file)

    # Extract facies
    if "facies" not in ei_data:
        print("❌ ERROR: No facies data in cache file!")
        return

    facies = ei_data["facies"]
    print(f"✓ Loaded facies: {facies.shape}")

    # Extract angles and EI volumes
    if "angles" in ei_data:
        angles = ei_data["angles"]
        print(f"✓ Found angles: {list(angles)}")

        # Load individual EI volumes
        ei_volumes = []
        for angle in angles:
            key = f"ei_{int(angle)}deg"
            if key in ei_data:
                ei_volumes.append(ei_data[key])
            else:
                print(f"⚠ Warning: {key} not found in cache")

        if len(ei_volumes) == 0:
            print("❌ ERROR: No EI volumes found in cache!")
            return

        print(f"✓ Loaded {len(ei_volumes)} EI volumes")

        # Create results dict
        ei_results = {
            "angles": angles,
            "ei_volumes": ei_volumes,
            "ei_dict": {int(a): v for a, v in zip(angles, ei_volumes)},
        }
    else:
        print("❌ ERROR: No angles array in cache file!")
        return

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Multi-angle comparison
    path1 = plot_multiangle_ei_comparison(ei_results, facies, cache_dir)

    # 2. Facies analysis
    path2 = plot_multiangle_ei_facies_analysis(ei_results, facies, cache_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    if path1:
        print(f"  ✓ {path1}")
    if path2:
        print(f"  ✓ {path2}")
    print("=" * 70)


if __name__ == "__main__":
    main()
