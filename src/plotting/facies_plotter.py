import os
import time
import tempfile

import matplotlib

# Force Agg backend immediately to avoid font/config discovery hangs on macOS
try:
    matplotlib.use("Agg")
except Exception:
    pass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Only import for type checking to avoid circular imports at runtime
    from src.analysis.models import AvoResults

import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class FaciesPlotter:
    """Plotter for facies-correlation summary figures.

    This class isolates plotting code from the analysis logic so it can be
    swapped or mocked in tests.
    """

    def create_summary_plots(
        self,
        avo_results: "AvoResults",
        cache_dir: str,
        domain: str = "depth",
    ) -> Any:

        logger.debug("create_summary_plots: starting")

        fig = plt.figure(figsize=(18, 12))
        domain_label = "Depth Domain" if domain == "depth" else "Time Domain"
        fig.suptitle(
            "Quantitative Seismic-Facies Correlation Analysis:"
            f" AVO Only ({domain_label})",
            fontsize=16,
            y=0.995,
        )

        logger.debug("create_summary_plots: plotting boundary distributions")
        # 1. AVO amplitude distribution (at boundaries vs away)
        ax1 = plt.subplot(2, 3, 1)
        boundary_amps = getattr(avo_results, "boundary_amps", None)
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

        logger.debug("create_summary_plots: plotting interface strengths")
        # 2. Reflection strength at different interface types (AVO)
        ax2 = plt.subplot(2, 3, 2)
        rows = []
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
        interface_types = [r[1] for r in rows]
        avo_means = [r[2].get("mean", 0.0) for r in rows]
        avo_stds = [r[2].get("std", 0.0) for r in rows]

        x_pos = np.arange(len(interface_types))
        if len(x_pos):
            ax2.bar(
                x_pos, avo_means, yerr=avo_stds, alpha=0.7, color="steelblue", capsize=5
            )
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(interface_types, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Mean Amplitude")
        ax2.set_title("AVO: Reflection Strength at Interfaces")
        ax2.grid(True, alpha=0.3, axis="y")

        logger.debug("create_summary_plots: plotting facies discrimination")
        # 3. Facies discrimination - amplitude by facies type (AVO)
        ax3 = plt.subplot(2, 3, 3)
        facies_labels = []
        avo_facies_data = []
        for facies_val in range(4):
            facies_data = (getattr(avo_results, "facies_amplitudes", {}) or {}).get(
                facies_val
            )
            if facies_data is not None:
                facies_labels.append(f"Facies {facies_val}")
                sampled = facies_data[:: max(1, len(facies_data) // 1000)]
                avo_facies_data.append(sampled)

        if avo_facies_data:
            # Use modern matplotlib boxplot API
            bp = ax3.boxplot(avo_facies_data, patch_artist=True)
            # Set tick labels explicitly
            ax3.set_xticks(range(1, len(avo_facies_data) + 1))
            ax3.set_xticklabels(facies_labels, rotation=0, fontsize=8)
            for patch, color in zip(
                bp["boxes"], plt.cm.tab10(np.linspace(0, 0.4, len(bp["boxes"])))
            ):
                patch.set_facecolor(color)
        ax3.set_ylabel("AVO Amplitude")
        ax3.set_title("AVO: Amplitude by Facies Type")
        ax3.grid(True, alpha=0.3, axis="y")

        logger.debug("create_summary_plots: plotting boundary vs background")
        # 4. Boundary amplitude comparison (AVO only: at vs away)
        ax4 = plt.subplot(2, 3, 4)
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
        ax4.bar(labels, values, color=["steelblue", "lightsteelblue"], alpha=0.8)
        ax4.set_ylabel("Mean |Amplitude|")
        ax4.set_title("Boundary vs Background Amplitude (AVO)")
        ax4.grid(True, alpha=0.3, axis="y")

        logger.debug("create_summary_plots: plotting separation matrix")
        # 5. Facies separation matrix (Cohen's d) for AVO
        ax5 = plt.subplot(2, 3, 5)
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

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
