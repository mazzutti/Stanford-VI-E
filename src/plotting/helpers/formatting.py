"""Formatting helpers used by plotting.

Provides concise logging and header helpers for plotting and analysis
commands.
"""

from typing import Optional, Sequence
from pathlib import Path
import logging
from src.utils.facades import LazyObjectProxy

logger = logging.getLogger(__name__)

__all__ = [
    "print_header",
    "print_angle_summary",
    "print_selected_angles",
    "print_cache_info",
]


# Thin facade for formatting helpers
class FormattingHelper:
    def print_header(self, title: str):
        return print_header(title)

    def print_angle_summary(
        self,
        angles: Sequence[float],
        volumes: Sequence,
        stack=None,
        gradient=None,
    ):
        return print_angle_summary(angles, volumes, stack=stack, gradient=gradient)

    def print_selected_angles(self, selected_angles, weights):
        return print_selected_angles(selected_angles, weights)

    def print_cache_info(self, cache_file: Optional[str]):
        return print_cache_info(cache_file)


# Module-level lazy proxy using the shared LazyObjectProxy
formatting_helper = LazyObjectProxy(lambda: FormattingHelper())

__all__.extend(["FormattingHelper", "formatting_helper"])


def get_formatting_helper(config: dict | None = None):
    if config is None:
        return formatting_helper
    return FormattingHelper()


__all__.append("get_formatting_helper")


def print_header(title: str):
    logger.info("%s", "\n" + "=" * 70)
    logger.info("%s", title)
    logger.info("%s", "=" * 70)


def print_angle_summary(
    angles: Sequence[float], volumes: Sequence, stack=None, gradient=None
):
    """Log a concise per-angle summary and optional stack/gradient stats."""
    print_header("ANGLE-DEPENDENT SUMMARY")
    for angle, vol in zip(angles, volumes):
        logger.info(
            "  %5.1f° : value = [%.3e, %.3e] (mean = %.3e)",
            angle,
            vol.min(),
            vol.max(),
            vol.mean(),
        )

    if stack is not None:
        logger.info(
            "\nStack   : value = [%.3e, %.3e] (mean = %.3e)",
            stack.min(),
            stack.max(),
            stack.mean(),
        )

    if gradient is not None:
        logger.info(
            "Gradient: Δ = [%.3e, %.3e] (mean = %.3e)",
            gradient.min(),
            gradient.max(),
            gradient.mean(),
        )


def print_selected_angles(selected_angles, weights):
    logger.info("  Selected angles: %s", selected_angles)
    logger.info("  Weights: %s", weights)


def print_cache_info(cache_file: Optional[str]):
    if not cache_file:
        return
    logger.info("\n✓ Saved multi-angle results to: %s", cache_file)
    try:
        size_mb = Path(cache_file).stat().st_size / 1024**2
        logger.info("  File size: %.1f MB", size_mb)
    except Exception:
        pass
