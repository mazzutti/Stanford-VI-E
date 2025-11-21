"""Facies correlation summary figure plotting.

Provides FaciesPlotter class for creating summary plots of facies-seismic correlations.
Inherits from BasePlotter for consistency and shared utilities.
"""

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from src.plotting.helpers.base import BasePlotter

if TYPE_CHECKING:
    # Only import for type checking to avoid circular imports at runtime
    from src.analysis.models import AvoResults

logger = logging.getLogger(__name__)

# FaciesPlotter is a focused plotting helper with a compact public API and
# some methods intentionally have minimal public surface. Also some plotting
# helpers use multiple local temporaries for layout and rendering. Silence
# related stylistic warnings to keep lint focused on higher-risk issues.


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
        # `cache_dir` parameter is accepted for API consistency with other
        # plotters but not currently used here; silence lint warning.

        self._log_debug("create_summary_plots: starting")

        fig: Figure = plt.figure(figsize=(18, 12))
        domain_label = "Depth Domain" if domain == "depth" else "Time Domain"
        fig.suptitle(
            "Quantitative Seismic-Facies Correlation Analysis:"
            f" AVO Only ({domain_label})",
            fontsize=16,
            y=0.995,
        )

        # Prepare derived data needed by multiple subplots.
        self._log_debug("create_summary_plots: preparing summary data")
        data = self._prepare_summary_data(avo_results)

        self._log_debug("create_summary_plots: plotting boundary distributions")
        ax1: Axes = fig.add_subplot(2, 3, 1)
        self._plot_boundary_distributions(ax1, data)

        self._log_debug("create_summary_plots: plotting interface strengths")
        ax2: Axes = fig.add_subplot(2, 3, 2)
        self._plot_interface_strengths(ax2, data)

        self._log_debug("create_summary_plots: plotting facies discrimination")
        ax3: Axes = fig.add_subplot(2, 3, 3)
        self._plot_facies_discrimination(ax3, data)

        logger.debug("create_summary_plots: plotting boundary vs background")
        ax4: Axes = fig.add_subplot(2, 3, 4)
        self._plot_boundary_vs_background(ax4, data)

        logger.debug("create_summary_plots: plotting separation matrix")
        # 5. Facies separation matrix (Cohen's d) for AVO
        ax5: Axes = fig.add_subplot(2, 3, 5)
        self._plot_separation_matrix(ax5, data)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _prepare_summary_data(self, avo_results: "AvoResults") -> dict[str, Any]:
        """Prepare derived arrays and summary values for `create_summary_plots`.

        This extracts the data-aggregation and sampling logic so the large
        plotting method keeps fewer locals and statements.
        """
        # Delegate to smaller helpers to reduce locals per-function.
        at_bounds, away, boundary_mean, away_mean = self._prepare_boundary_stats(
            avo_results
        )
        interface_types, avo_means_arr, avo_stds_arr = self._prepare_interface_stats(
            avo_results
        )
        facies_labels, avo_facies_data = self._prepare_facies_data(avo_results)

        sep = getattr(avo_results, "separation_matrix", None)

        return {
            "at_bounds": at_bounds,
            "away": away,
            "interface_types": interface_types,
            "avo_means_arr": avo_means_arr,
            "avo_stds_arr": avo_stds_arr,
            "facies_labels": facies_labels,
            "avo_facies_data": avo_facies_data,
            "boundary_mean": boundary_mean,
            "away_mean": away_mean,
            "sep": sep,
        }

    def _prepare_boundary_stats(
        self, avo_results: "AvoResults"
    ) -> tuple[NDArray[Any], NDArray[Any], float, float]:
        """Return (at_bounds, away, boundary_mean, away_mean)."""
        boundary_amps = getattr(avo_results, "boundary_amps", None)
        if boundary_amps is None:
            at_bounds = np.array([])
            away = np.array([])
        else:
            at_bounds = getattr(boundary_amps, "at_boundaries", np.array([]))
            away = getattr(boundary_amps, "away_from_boundaries", np.array([]))

        # Use plain Python floats for the means to satisfy strict type checkers
        # (np.nan is typed as numpy.floating[...] which may not be considered
        # a plain float by some static analyzers).
        boundary_mean: float = float("nan")
        away_mean: float = float("nan")
        if hasattr(at_bounds, "size") and at_bounds.size:
            # Cast numpy results to native float for the declared return type
            boundary_mean = float(np.mean(np.abs(at_bounds)))
        if hasattr(away, "size") and away.size:
            away_mean = float(np.mean(np.abs(away)))
        return at_bounds, away, boundary_mean, away_mean

    def _prepare_interface_stats(
        self, avo_results: "AvoResults"
    ) -> tuple[list[str], NDArray[Any], NDArray[Any]]:
        """Return (interface_types, means_array, stds_array)."""
        rows: list[tuple[tuple[int, int], str, dict[str, Any]]] = []
        for key, stats in (
            getattr(avo_results, "interface_stats_summary", {}) or {}
        ).items():
            if stats is None or stats.get("count", 0) <= 10:
                continue
            label = str(key)
            if hasattr(key, "from_facies") and hasattr(key, "to_facies"):
                order = (int(key.from_facies), int(key.to_facies))
            else:
                order = (0, 0)
            rows.append((order, label, stats))

        rows.sort(key=lambda r: r[0])
        interface_types: list[str] = [r[1] for r in rows]
        avo_means: list[float] = [r[2].get("mean", 0.0) for r in rows]
        avo_stds: list[float] = [r[2].get("std", 0.0) for r in rows]
        return interface_types, np.asarray(avo_means), np.asarray(avo_stds)

    def _prepare_facies_data(
        self, avo_results: "AvoResults"
    ) -> tuple[list[str], list[NDArray[Any]]]:
        """Return (facies_labels, sampled_facies_arrays)."""
        facies_labels: list[str] = []
        avo_facies_data: list[NDArray[Any]] = []
        for facies_val in range(4):
            facies_data = (getattr(avo_results, "facies_amplitudes", {}) or {}).get(
                facies_val
            )
            if facies_data is not None:
                facies_labels.append(f"Facies {facies_val}")
                sampled = facies_data[:: max(1, len(facies_data) // 1000)]
                avo_facies_data.append(np.asarray(sampled))
        return facies_labels, avo_facies_data

    def _plot_boundary_distributions(self, ax: Axes, data: dict[str, Any]) -> None:
        at_bounds = data["at_bounds"]
        away = data["away"]

        if hasattr(at_bounds, "size") and at_bounds.size:
            ax.hist(
                at_bounds,
                bins=50,
                alpha=0.7,
                label="At Boundaries",
                density=True,
                color="red",
            )
        if hasattr(away, "size") and away.size:
            ax.hist(
                away,
                bins=50,
                alpha=0.7,
                label="Away from Boundaries",
                density=True,
                color="blue",
            )
        ax.set_xlabel("AVO Amplitude")
        ax.set_ylabel("Density")
        ax.set_title("AVO: Amplitude Distribution")
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_interface_strengths(self, ax: Axes, data: dict[str, Any]) -> None:
        interface_types = data["interface_types"]
        avo_means_arr = data["avo_means_arr"]
        avo_stds_arr = data["avo_stds_arr"]

        x_pos = np.arange(len(interface_types))
        if len(x_pos):
            ax.bar(
                x_pos,
                avo_means_arr,
                yerr=avo_stds_arr,
                alpha=0.7,
                color="steelblue",
                capsize=5,
            )
            ax.set_xticks(list(x_pos.astype(float)))
            ax.set_xticklabels(interface_types, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean Amplitude")
        ax.set_title("AVO: Reflection Strength at Interfaces")
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_facies_discrimination(self, ax: Axes, data: dict[str, Any]) -> None:
        facies_labels = data["facies_labels"]
        avo_facies_data = data["avo_facies_data"]

        if avo_facies_data:
            bp = ax.boxplot(avo_facies_data, patch_artist=True)
            ax.set_xticks(range(1, len(avo_facies_data) + 1))
            ax.set_xticklabels(facies_labels, rotation=0, fontsize=8)
            cmap_colors: NDArray[Any] = np.asarray(
                plt.get_cmap("tab10")(np.linspace(0, 0.4, len(bp["boxes"])))
            )
            for patch, color in zip(bp["boxes"], cmap_colors):
                patch.set_facecolor(color)
        ax.set_ylabel("AVO Amplitude")
        ax.set_title("AVO: Amplitude by Facies Type")
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_boundary_vs_background(self, ax: Axes, data: dict[str, Any]) -> None:
        boundary_mean = data["boundary_mean"]
        away_mean = data["away_mean"]

        labels = ["At Boundaries", "Away from Boundaries"]
        values = [
            boundary_mean if not np.isnan(boundary_mean) else 0.0,
            away_mean if not np.isnan(away_mean) else 0.0,
        ]
        ax.bar(
            labels, np.asarray(values), color=["steelblue", "lightsteelblue"], alpha=0.8
        )
        ax.set_ylabel("Mean |Amplitude|")
        ax.set_title("Boundary vs Background Amplitude (AVO)")
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_separation_matrix(self, ax: Axes, data: dict[str, Any]) -> None:
        sep = data["sep"]
        if sep is not None:
            ax.imshow(sep, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 1, 2, 3])
            ax.set_xticklabels(["F0", "F1", "F2", "F3"])
            ax.set_yticklabels(["F0", "F1", "F2", "F3"])
            ax.set_xlabel("Facies")
            ax.set_ylabel("Facies")
            ax.set_title("AVO: Facies Separation (Cohen's d)")
            if hasattr(sep, "shape") and sep.shape == (4, 4):
                for i in range(4):
                    for j in range(4):
                        ax.text(
                            j,
                            i,
                            f"{sep[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
