from src.plotting.helpers.plot import init_plotting
from scipy.ndimage import sobel, gaussian_filter

# Initialize matplotlib and numpy for this module
plt, np = init_plotting(backend="Agg")

# depth_to_time_cube and resample_time_cube are not used in this module;
# avoid importing them

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "detect_facies_boundaries",
    "plot_seismic_with_facies_overlay",
    "plot_facies_only",
]


class FaciesOverlay:
    """Facade for facies overlay plotting helpers.

    The methods mirror the legacy top-level functions so callers can migrate
    to an instance-based API while the old function names remain available.
    """

    def detect_facies_boundaries(self, facies_slice):
        smoothed = gaussian_filter(facies_slice.astype(float), sigma=0.5)
        grad_x = sobel(smoothed, axis=0)
        grad_y = sobel(smoothed, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        boundaries = gradient_magnitude > 0.1
        return boundaries

    def plot_seismic_with_facies_overlay(
        self,
        ax,
        seismic_slice,
        facies_slice,
        title,
        k_scale=1.0,
        k_label="K",
        k_unit="",
        cmap="seismic",
        show_colorbar=True,
    ):
        nj, nk = seismic_slice.shape
        p = np.percentile(np.abs(seismic_slice), 99.5)
        vmax = float(p)
        vmin = -vmax
        if vmax == vmin:
            vmax = vmin + 1.0

        extent = [0, nj - 1, (nk - 1) * k_scale, 0]

        from src.plotting.helpers.plot import plot_helper

        im = plot_helper.imshow_with_labels(
            ax,
            seismic_slice,
            title,
            xlabel="Crossline (J)",
            k_label=k_label,
            k_unit=k_unit,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=extent,
            interpolation="bilinear",
            alpha=0.9,
            colorbar=True,
            colorbar_label="Amplitude",
            fontsize_title=12,
            fontsize_labels=10,
        )

        boundaries = self.detect_facies_boundaries(facies_slice)

        nj_facies, nk_facies = facies_slice.shape
        J = np.arange(nj_facies)
        K = np.arange(nk_facies) * k_scale
        JJ, KK = np.meshgrid(J, K, indexing="ij")

        ax.contour(
            JJ.T,
            KK.T,
            boundaries.T,
            levels=[0.5],
            colors="lime",
            linewidths=1.5,
            linestyles="solid",
            alpha=0.8,
        )

        facies_levels = [0.5, 1.5, 2.5]
        ax.contour(
            JJ.T,
            KK.T,
            facies_slice.T,
            levels=facies_levels,
            colors="yellow",
            linewidths=1.0,
            linestyles="dashed",
            alpha=0.6,
        )

        # use plot_helper for axis labeling if available
        try:
            plot_helper.set_axis_labels(
                ax,
                title,
                xlabel="Crossline (J)",
                k_label=k_label,
                k_unit=k_unit,
                fontsize_title=12,
                fontsize_labels=10,
                im=im if show_colorbar else None,
                colorbar_label="Amplitude" if show_colorbar else None,
            )
        except Exception:
            from src.plotting.helpers.plot import set_axis_labels

            set_axis_labels(
                ax,
                title,
                xlabel="Crossline (J)",
                k_label=k_label,
                k_unit=k_unit,
                fontsize_title=12,
                fontsize_labels=10,
                im=im if show_colorbar else None,
                colorbar_label="Amplitude" if show_colorbar else None,
            )

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                color="lime",
                linewidth=1.5,
                label="Facies Boundaries (detected)",
            ),
            Line2D(
                [0],
                [0],
                color="yellow",
                linewidth=1.0,
                linestyle="--",
                label="Facies Interfaces",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=8,
            framealpha=0.8,
        )

        return im

    def plot_facies_only(
        self,
        ax,
        facies_slice,
        title,
        k_scale=1.0,
        k_label="K",
        k_unit="",
    ):
        nj, nk = facies_slice.shape

        from src.plotting.helpers.plot import display_facies

        im = display_facies(
            ax,
            facies_slice,
            title,
            k_scale=k_scale,
            k_label=k_label,
            k_unit=k_unit,
            fontsize_title=12,
            fontsize_labels=10,
        )

        boundaries = self.detect_facies_boundaries(facies_slice)
        nj_facies, nk_facies = facies_slice.shape
        J = np.arange(nj_facies)
        K = np.arange(nk_facies) * k_scale
        JJ, KK = np.meshgrid(J, K, indexing="ij")

        ax.contour(
            JJ.T,
            KK.T,
            boundaries.T,
            levels=[0.5],
            colors="white",
            linewidths=2.0,
            linestyles="solid",
        )

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xlabel("Crossline (J)", fontsize=10)

        if k_unit:
            ax.set_ylabel(f"{k_label} ({k_unit})", fontsize=10)
        else:
            ax.set_ylabel(k_label, fontsize=10)

        cbar = plt.colorbar(
            im,
            ax=ax,
            ticks=[0, 1, 2, 3],
            boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5],
            pad=0.01,
        )
        cbar.set_label("Facies", fontsize=9)
        cbar.ax.set_yticklabels(
            ["Facies 0", "Facies 1", "Facies 2", "Facies 3"], fontsize=8
        )


from src.utils.facades import LazyObjectProxy


# Module-level lazy proxy for FaciesOverlay
facies_overlay = LazyObjectProxy(lambda: FaciesOverlay())


def get_facies_overlay(config: dict | None = None):
    if config is None:
        return facies_overlay
    return FaciesOverlay()


__all__.extend(["FaciesOverlay", "facies_overlay", "get_facies_overlay"])


def detect_facies_boundaries(facies_slice):
    return facies_overlay.detect_facies_boundaries(facies_slice)


def plot_seismic_with_facies_overlay(
    ax,
    seismic_slice,
    facies_slice,
    title,
    k_scale=1.0,
    k_label="K",
    k_unit="",
    cmap="seismic",
    show_colorbar=True,
):
    return facies_overlay.plot_seismic_with_facies_overlay(
        ax,
        seismic_slice,
        facies_slice,
        title,
        k_scale=k_scale,
        k_label=k_label,
        k_unit=k_unit,
        cmap=cmap,
        show_colorbar=show_colorbar,
    )


def plot_facies_only(
    ax,
    facies_slice,
    title,
    k_scale=1.0,
    k_label="K",
    k_unit="",
):
    return facies_overlay.plot_facies_only(
        ax, facies_slice, title, k_scale=k_scale, k_label=k_label, k_unit=k_unit
    )
