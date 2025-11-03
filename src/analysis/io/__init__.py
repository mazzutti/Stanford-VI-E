"""I/O utilities subpackage for analysis workflows.

This subpackage contains header printing and other I/O utilities for analysis pipelines.

Public API:
    - HeaderPrinter: Formats and logs analysis pipeline headers

Example:
    >>> from src.analysis.io import HeaderPrinter
    >>> printer = HeaderPrinter()
    >>> printer("Analysis Complete", ["Results saved to output/"])
"""

from .header import (
    HeaderPrinter,
    printer,
)

__all__ = [
    "HeaderPrinter",
    "printer",
]
