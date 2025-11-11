"""Generate interactive seismic full-stack 3D HTML visualizations.

Loads full-stack AVO seismogram volumes from the .cache directory and
writes Plotly HTML files under docs/images using the project's
PlotlyPlotter helper.

Usage: run this script from the repository root with Python 3. Example:
  python tools/generate_seismic_full_stack_plotly.py
"""
from pathlib import Path
from src.plotting.seismic_plotter import SeismicPlotter


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
OUT_DIR = ROOT / "docs" / "images"


def main() -> None:
    plotter = SeismicPlotter(cache_dir=str(CACHE), out_dir=str(OUT_DIR))

    for domain in ("time", "depth"):
        print(f"Generating {domain}-domain seismic full-stack interactive HTML...")
        generated = plotter.generate_from_caches(domain=domain)
        if generated:
            for p in generated:
                print(f"  ✓ Generated: {p}")
        else:
            print(f"  - Skipped: no cache available for domain {domain}")


if __name__ == "__main__":
    main()
