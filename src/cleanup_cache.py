#!/usr/bin/env python3
"""
Cache cleanup utility for Stanford VI-E seismic modeling.

Removes old cache files before regenerating seismograms to avoid:
- Disk space waste from duplicate files
- Confusion between old and new naming conventions
- Stale data being used by visualization scripts

This script identifies and removes cache files that don't follow
the standardized naming convention with explicit domain suffixes
(_time_ and _depth_).
"""

import os
import glob
import sys


def identify_old_cache_files(cache_dir=".cache"):
    """
    Identify cache files using naming convention.

    Current conventions:
        - avo_time_{hash}.npz
        - avo_depth_{hash}.npz
        - ai_time_{hash}.npz
        - ai_depth_{hash}.npz
        - ei_time_{hash}.npz
        - ei_depth_{hash}.npz

    Args:
        cache_dir (str): Cache directory path

    Returns:
        list: List of old cache file paths to remove
    """
    if not os.path.exists(cache_dir):
        print(f"Cache directory '{cache_dir}' does not exist.")
        return []

    all_npz = glob.glob(os.path.join(cache_dir, "*.npz"))
    old_files = []

    for file_path in all_npz:
        filename = os.path.basename(file_path)

        # Check for old AVO naming (avo_{hash}.npz without _time_ or _depth_)
        if filename.startswith("avo_") and not (
            "_time_" in filename or "_depth_" in filename
        ):
            old_files.append(file_path)

        # Check for old AI naming (ai_{hash}.npz without _time_ or _depth_)
        elif filename.startswith("ai_") and not (
            "_time_" in filename or "_depth_" in filename
        ):
            old_files.append(file_path)

        # Check for old EI naming (ei_{hash}.npz without proper _time_/_depth_, or ei_depth_angle* format)
        elif filename.startswith("ei_") and (
            "angle" in filename or not ("_time_" in filename or "_depth_" in filename)
        ):
            old_files.append(file_path)

    return old_files


def cleanup_old_cache(cache_dir=".cache", dry_run=False):
    """
    Remove old cache files from cache directory.

    Args:
        cache_dir (str): Cache directory path
        dry_run (bool): If True, only print what would be deleted without actually deleting

    Returns:
        tuple: (number of files removed, total size freed in MB)
    """
    old_files = identify_old_cache_files(cache_dir)

    if not old_files:
        print(f"✓ No old cache files found in '{cache_dir}'")
        return 0, 0

    print(f"\n{'DRY RUN: ' if dry_run else ''}Found {len(old_files)} old cache files:")

    total_size_bytes = 0
    for file_path in old_files:
        file_size = os.path.getsize(file_path)
        total_size_bytes += file_size
        size_mb = file_size / (1024**2)
        print(f"  - {os.path.basename(file_path)} ({size_mb:.1f} MB)")

    total_size_mb = total_size_bytes / (1024**2)

    if dry_run:
        print(
            f"\nDRY RUN: Would remove {len(old_files)} files ({total_size_mb:.1f} MB)"
        )
        print("Run without --dry-run to actually delete files.")
        return 0, 0

    # Confirm deletion
    print(f"\nTotal space to be freed: {total_size_mb:.1f} MB")

    # Delete files
    removed_count = 0
    for file_path in old_files:
        try:
            os.remove(file_path)
            removed_count += 1
        except Exception as e:
            print(f"  ✗ Error removing {file_path}: {e}")

    print(
        f"\n✓ Removed {removed_count}/{len(old_files)} files ({total_size_mb:.1f} MB freed)"
    )

    return removed_count, total_size_mb


def main():
    """Command-line interface for cache cleanup."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up old cache files from Stanford VI-E seismic modeling"
    )
    parser.add_argument(
        "--cache-dir", default=".cache", help="Cache directory path (default: .cache)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CACHE CLEANUP UTILITY")
    print("=" * 70)
    print(f"Cache directory: {args.cache_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'DELETE'}")
    print("=" * 70)

    removed, size_mb = cleanup_old_cache(args.cache_dir, args.dry_run)

    if not args.dry_run and removed > 0:
        print("\n" + "=" * 70)
        print("CLEANUP COMPLETE")
        print("=" * 70)
        print(f"Removed {removed} old cache files")
        print(f"Freed {size_mb:.1f} MB of disk space")
        print("=" * 70)


if __name__ == "__main__":
    main()
