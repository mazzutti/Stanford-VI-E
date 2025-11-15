"""Facies correlation summary figure plotting.

Provides FaciesPlotter class for creating summary plots of facies-seismic correlations.
Inherits from BasePlotter for consistency and shared utilities.
"""

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from src.plotting.helpers.base import BasePlotter

if TYPE_CHECKING:
    # Only import for type checking to avoid circular imports at runtime
    from src.analysis.models import AvoResults

logger = logging.getLogger(__name__)


class FaciesPlotter(BasePlotter):
    """Plotter for facies-correlation summary figures.

    This class isolates plotting code from the analysis logic so it can be
    swapped or mocked in tests. Inherits common utilities from BasePlotter.
    """

    def create_summary_plots(
        self,
        avo_results: "AvoResults",
        cache_dir: str,
        domain: str = "depth",
    ) -> Figure:

        self._log_debug("create_summary_plots: starting")

        fig: Figure = plt.figure(figsize=(18, 12))
        domain_label = "Depth Domain" if domain == "depth" else "Time Domain"
        fig.suptitle(
            "Quantitative Seismic-Facies Correlation Analysis:"
            f" AVO Only ({domain_label})",
            fontsize=16,
            y=0.995,
        )

        self._log_debug("create_summary_plots: plotting boundary distributions")
        # 1. AVO amplitude distribution (at boundaries vs away)
        ax1: Axes = fig.add_subplot(2, 3, 1)
        boundary_amps = getattr(avo_results, "boundary_amps", None)
        at_bounds: NDArray[Any]
        away: NDArray[Any]
        if boundary_amps is None:
            at_bounds = np.array([])
            away = np.array([])
        else:
            at_bounds = getattr(boundary_amps, "at_boundaries", np.array([]))
            away = getattr(boundary_amps, "away_from_boundaries", np.array([]))

        if hasattr(at_bounds, "size") and at_bounds.size:
            ax1.hist(
                at_bounds,
                bins=50,
                alpha=0.7,
                label="At Boundaries",
                density=True,
                color="red",
            )
        if hasattr(away, "size") and away.size:
            ax1.hist(
                away,
                bins=50,
                alpha=0.7,
                label="Away from Boundaries",
                density=True,
                color="blue",
            )
        ax1.set_xlabel("AVO Amplitude")
        ax1.set_ylabel("Density")
        ax1.set_title("AVO: Amplitude Distribution")
        # Only show legend if there are labeled artists
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        self._log_debug("create_summary_plots: plotting interface strengths")
        # 2. Reflection strength at different interface types (AVO)
        ax2: Axes = fig.add_subplot(2, 3, 2)
        rows: list[tuple[tuple[int, int], str, dict[str, Any]]] = []
        for key, stats in (
            getattr(avo_results, "interface_stats_summary", {}) or {}
        ).items():
            if stats is None or stats.get("count", 0) <= 10:
                continue
            # label: use str(key) for Transition objects rendering as "a->b"
            label = str(key)
            # derive stable ordering key (from_facies, to_facies)
            if hasattr(key, "from_facies") and hasattr(key, "to_facies"):
                order = (int(key.from_facies), int(key.to_facies))
            else:
                order = (0, 0)
            rows.append((order, label, stats))

        # sort rows for stable display
        rows.sort(key=lambda r: r[0])
        interface_types: list[str] = [r[1] for r in rows]
        avo_means: list[float] = [r[2].get("mean", 0.0) for r in rows]
        avo_stds: list[float] = [r[2].get("std", 0.0) for r in rows]

        # Convert to NumPy arrays for ArrayLike parameters (bar, yerr)
        avo_means_arr = np.asarray(avo_means)
        avo_stds_arr = np.asarray(avo_stds)

        x_pos = np.arange(len(interface_types))
        if len(x_pos):
            ax2.bar(
                x_pos,
                avo_means_arr,
                yerr=avo_stds_arr,
                alpha=0.7,
                color="steelblue",
                capsize=5,
            )
            # Convert ndarray to a list of floats to satisfy typing for Sequence[float]
            ax2.set_xticks(list(x_pos.astype(float)))
            ax2.set_xticklabels(interface_types, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Mean Amplitude")
        ax2.set_title("AVO: Reflection Strength at Interfaces")
        ax2.grid(True, alpha=0.3, axis="y")

        self._log_debug("create_summary_plots: plotting facies discrimination")
        # 3. Facies discrimination - amplitude by facies type (AVO)
        ax3: Axes = fig.add_subplot(2, 3, 3)
        facies_labels: list[str] = []
        avo_facies_data: list[NDArray[Any]] = []
        for facies_val in range(4):
            facies_data = (getattr(avo_results, "facies_amplitudes", {}) or {}).get(
                facies_val
            )
            if facies_data is not None:
                facies_labels.append(f"Facies {facies_val}")
                sampled = facies_data[:: max(1, len(facies_data) // 1000)]
                # sampled may be any array-like; coerce/annotate as NDArray for typing
                avo_facies_data.append(np.asarray(sampled))

        if avo_facies_data:
            # Use modern matplotlib boxplot API
            bp = ax3.boxplot(avo_facies_data, patch_artist=True)
            # Set tick labels explicitly
            ax3.set_xticks(range(1, len(avo_facies_data) + 1))
            ax3.set_xticklabels(facies_labels, rotation=0, fontsize=8)
            cmap_colors: NDArray[Any] = np.asarray(
                plt.get_cmap("tab10")(np.linspace(0, 0.4, len(bp["boxes"])))
            )
            for patch, color in zip(bp["boxes"], cmap_colors):
                patch.set_facecolor(color)
        ax3.set_ylabel("AVO Amplitude")
        ax3.set_title("AVO: Amplitude by Facies Type")
        ax3.grid(True, alpha=0.3, axis="y")

        logger.debug("create_summary_plots: plotting boundary vs background")
        # 4. Boundary amplitude comparison (AVO only: at vs away)
        ax4: Axes = fig.add_subplot(2, 3, 4)
        boundary_mean = np.nan
        away_mean = np.nan
        if hasattr(at_bounds, "size") and at_bounds.size:
            boundary_mean = np.mean(np.abs(at_bounds))
        if hasattr(away, "size") and away.size:
            away_mean = np.mean(np.abs(away))

        labels = ["At Boundaries", "Away from Boundaries"]
        values = [
            boundary_mean if not np.isnan(boundary_mean) else 0.0,
            away_mean if not np.isnan(away_mean) else 0.0,
        ]
        # Convert to NumPy array to satisfy ArrayLike parameter expectations
        ax4.bar(
            labels, np.asarray(values), color=["steelblue", "lightsteelblue"], alpha=0.8
        )
        ax4.set_ylabel("Mean |Amplitude|")
        ax4.set_title("Boundary vs Background Amplitude (AVO)")
        ax4.grid(True, alpha=0.3, axis="y")

        logger.debug("create_summary_plots: plotting separation matrix")
        # 5. Facies separation matrix (Cohen's d) for AVO
        ax5: Axes = fig.add_subplot(2, 3, 5)
        sep = getattr(avo_results, "separation_matrix", None)
        if sep is not None:
            ax5.imshow(sep, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)
            ax5.set_xticks([0, 1, 2, 3])
            ax5.set_yticks([0, 1, 2, 3])
            ax5.set_xticklabels(["F0", "F1", "F2", "F3"])
            ax5.set_yticklabels(["F0", "F1", "F2", "F3"])
            ax5.set_xlabel("Facies")
            ax5.set_ylabel("Facies")
            ax5.set_title("AVO: Facies Separation (Cohen's d)")
            if hasattr(sep, "shape") and sep.shape == (4, 4):
                for i in range(4):
                    for j in range(4):
                        ax5.text(
                            j,
                            i,
                            f"{sep[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        return fig
