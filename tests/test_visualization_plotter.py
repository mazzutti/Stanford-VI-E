# mypy: ignore-errors
import os

import numpy as np

from src.plotting.facies_plotter import FaciesPlotter

# Use a non-interactive backend to avoid GUI/font issues in CI
os.environ.setdefault("MPLBACKEND", "Agg")


# Create a minimal Transition-like object to avoid importing the analysis module
class _T:
    def __init__(self, a, b):
        self.from_facies = a
        self.to_facies = b

    def __str__(self):
        return f"{self.from_facies}->{self.to_facies}"


def test_plotter_accepts_transition_keys():
    plotter = FaciesPlotter()
    key = _T(0, 1)
    avo_results = {
        "boundary_amps": {
            "at_boundaries": np.array([0.1, 0.2]),
            "away_from_boundaries": np.array([0.01]),
        },
        "interface_stats_summary": {key: {"count": 20, "mean": 0.5, "std": 0.1}},
        "facies_amplitudes": {0: np.array([0.1, 0.2, 0.3])},
        "separation_matrix": np.zeros((4, 4)),
    }
    fig = plotter.create_summary_plots(avo_results, cache_dir=".", domain="depth")
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
