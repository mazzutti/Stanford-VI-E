"""Formatting helpers used by plotting.

Provides concise logging and header helpers for plotting and analysis
commands.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "print_header",
    "print_angle_summary",
    "print_selected_angles",
    "print_cache_info",
    "FormattingHelper",
    "get_formatting_helper",
]

# Thin facade for formatting helpers
class FormattingHelper:
    """Helper class for formatting output."""

    def print_header(self, title: str) -> None:
        """Print a formatted header."""
        return print_header(title)

    def print_angle_summary(
        self,
        angles: Sequence[float],
        volumes: Sequence["NDArray[np.floating[Any]]"],
        stack: NDArray[np.floating[Any]] | None = None,
        gradient: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        """Print a concise per-angle summary."""
        return print_angle_summary(angles, volumes, stack=stack, gradient=gradient)

    def print_selected_angles(
        self,
        selected_angles: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
    ) -> None:
        """Print selected angles and weights."""
        return print_selected_angles(selected_angles, weights)

    def print_cache_info(self, cache_file: str | None) -> None:
        """Print cache file information."""
        return print_cache_info(cache_file)

# Module-level instance
_formatting_helper = FormattingHelper()

def get_formatting_helper() -> FormattingHelper:
    """Get the formatting helper instance.

    Returns:
        FormattingHelper instance
    """
    return _formatting_helper

__all__.append("get_formatting_helper")

def print_header(title: str) -> None:
    """Log a standardized header block for console output."""
    logger.info("%s", "\n" + "=" * 70)
    logger.info("%s", title)
    logger.info("%s", "=" * 70)

def print_angle_summary(
    angles: Sequence[float],
    volumes: Sequence[NDArray[np.floating[Any]]],
    stack: NDArray[np.floating[Any]] | None = None,
    gradient: NDArray[np.floating[Any]] | None = None,
) -> None:
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

def print_selected_angles(
    selected_angles: NDArray[np.floating[Any]], weights: NDArray[np.floating[Any]]
) -> None:
    """Log selected angles and their associated weights."""
    logger.info("  Selected angles: %s", selected_angles)
    logger.info("  Weights: %s", weights)

def print_cache_info(cache_file: str | None) -> None:
    """Log information about the saved cache file when present."""
    if not cache_file:
        return
    logger.info("\n✓ Saved multi-angle results to: %s", cache_file)
    try:
        size_mb = Path(cache_file).stat().st_size / 1024**2
        logger.info("  File size: %.1f MB", size_mb)
    except OSError:
        # Could not stat the file (missing, permission); ignore non-fatal
        pass
