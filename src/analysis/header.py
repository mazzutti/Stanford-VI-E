"""Small helper to centralize header printing for analysis pipelines.

This module now uses logging so callers can configure verbosity centrally.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)


def print_analysis_header(title: str, description_lines: List[str]) -> None:
    # Use logger.info for user-visible header output. Top-level callers can
    # configure logging to route this to stdout when desired.
    logger.info("%s", "=" * 70)
    logger.info("%s", title)
    logger.info("%s", description_lines[0] if description_lines else "")
    logger.info("%s", "=" * 70)
    logger.info("")
    for line in description_lines:
        logger.info("%s", line)
    logger.info("")


__all__ = ["print_analysis_header"]


# Thin object-oriented facade for header printing
class HeaderPrinter:
    """Facade for printing analysis headers via logging."""

    def print_analysis_header(self, title: str, description_lines: List[str]) -> None:
        return print_analysis_header(title, description_lines)


from src.utils.facades import LazyObjectProxy


# Use generic lazy proxy for consistency across modules
header_printer = LazyObjectProxy(lambda: HeaderPrinter())

__all__.extend(["HeaderPrinter", "header_printer"])


def get_header_printer(printer: HeaderPrinter | None = None) -> "HeaderPrinter":
    """Return the provided HeaderPrinter or the module-level lazy singleton."""
    return printer if printer is not None else header_printer


__all__.append("get_header_printer")
