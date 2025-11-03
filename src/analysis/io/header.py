"""Header printing for analysis pipelines.

This module provides HeaderPrinter for formatting and logging user-visible
headers in analysis workflows with flexible, dependency-injected configuration.

Example:
    >>> printer = HeaderPrinter()
    >>> printer.print_analysis_header(
    ...     "Analysis Complete",
    ...     ["Results saved to output/", "Duration: 2.5s"]
    ... )
    >>> # Or use shorthand
    >>> printer("Title", ["Line 1", "Line 2"])
    >>> # Or use factory methods
    >>> error_printer = HeaderPrinter.error_header()
    >>> error_printer("Critical Error", ["Details here"])

For convenience, a module-level instance is available:
    >>> from src.analysis.header import printer
    >>> printer("Title", ["Line 1", "Line 2"])
"""

from typing import Optional, Sequence, Iterator
import logging

logger = logging.getLogger(__name__)


class HeaderPrinter:
    """Formats and logs analysis pipeline headers.

    Supports customizable formatting (separator character, width) and logging
    configuration through dependency injection. Configuration is immutable
    after initialization.

    Attributes:
        log_level (int): Logging level (read-only)
        separator_width (int): Width of separator lines (read-only)
        separator_char (str): Character for separators (read-only)
    """

    def __init__(
        self,
        log_level: int = logging.INFO,
        separator_width: int = 70,
        separator_char: str = "=",
        logger_obj: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize HeaderPrinter with configuration.

        Args:
            log_level: Logging level (default: logging.INFO)
            separator_width: Width of separator lines (default: 70)
            separator_char: Character for separators (default: "=")
            logger_obj: Logger instance (default: module logger)

        Raises:
            ValueError: If separator_width < 1 or separator_char is empty
        """
        if separator_width < 1:
            raise ValueError(f"separator_width must be >= 1, got {separator_width}")
        if not separator_char:
            raise ValueError("separator_char cannot be empty")

        self._log_level = log_level
        self._separator_width = separator_width
        self._separator_char = separator_char
        self._logger = logger_obj or logger
        self._separator_cache = self._compute_separator()

    @property
    def log_level(self) -> int:
        """Get the logging level."""
        return self._log_level

    @property
    def separator_width(self) -> int:
        """Get the separator width."""
        return self._separator_width

    @property
    def separator_char(self) -> str:
        """Get the separator character."""
        return self._separator_char

    def __repr__(self) -> str:
        """Return a debugging representation."""
        return (
            f"<HeaderPrinter separator_width={self.separator_width} "
            f"separator_char={self.separator_char!r} id=0x{id(self):x}>"
        )

    def __call__(
        self, title: str, description_lines: Sequence[str] | None = None
    ) -> None:
        """Shorthand for print_analysis_header.

        Args:
            title: Main header title
            description_lines: Optional detail lines (default: None)

        Example:
            >>> printer = HeaderPrinter()
            >>> printer("Title", ["Line 1", "Line 2"])
        """
        self.print_analysis_header(title, description_lines)

    def __enter__(self) -> "HeaderPrinter":
        """Enable use as context manager for grouping headers.

        Returns:
            Self for use in with statement

        Example:
            >>> with HeaderPrinter() as printer:
            ...     printer("Section 1", ["Details"])
            ...     printer("Subsection", ["More details"])
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager gracefully.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        # No cleanup needed, but allows use as context manager

    @classmethod
    def error_header(
        cls, logger_obj: Optional[logging.Logger] = None
    ) -> "HeaderPrinter":
        """Create a HeaderPrinter for error messages.

        Uses WARNING level and '!' separator character.

        Args:
            logger_obj: Logger instance (default: module logger)

        Returns:
            HeaderPrinter configured for error messages
        """
        return cls(
            log_level=logging.WARNING,
            separator_width=70,
            separator_char="!",
            logger_obj=logger_obj,
        )

    @classmethod
    def section_header(
        cls, logger_obj: Optional[logging.Logger] = None
    ) -> "HeaderPrinter":
        """Create a HeaderPrinter for section headers.

        Uses INFO level and '-' separator character with reduced width.

        Args:
            logger_obj: Logger instance (default: module logger)

        Returns:
            HeaderPrinter configured for section headers
        """
        return cls(
            log_level=logging.INFO,
            separator_width=50,
            separator_char="-",
            logger_obj=logger_obj,
        )

    @classmethod
    def info_header(
        cls, logger_obj: Optional[logging.Logger] = None
    ) -> "HeaderPrinter":
        """Create a HeaderPrinter for info messages.

        Uses INFO level and '*' separator character.

        Args:
            logger_obj: Logger instance (default: module logger)

        Returns:
            HeaderPrinter configured for info messages
        """
        return cls(
            log_level=logging.INFO,
            separator_width=70,
            separator_char="*",
            logger_obj=logger_obj,
        )

    @classmethod
    def debug_header(
        cls, logger_obj: Optional[logging.Logger] = None
    ) -> "HeaderPrinter":
        """Create a HeaderPrinter for debug messages.

        Uses DEBUG level and '~' separator character.

        Args:
            logger_obj: Logger instance (default: module logger)

        Returns:
            HeaderPrinter configured for debug messages
        """
        return cls(
            log_level=logging.DEBUG,
            separator_width=70,
            separator_char="~",
            logger_obj=logger_obj,
        )

    def format_lines(
        self, title: str, description_lines: Sequence[str] | None = None
    ) -> Iterator[str]:
        """Generate formatted header lines without logging.

        Useful for composing headers or writing to different outputs.

        Args:
            title: Main header title
            description_lines: Optional detail lines (default: None)

        Yields:
            Formatted header lines as strings
        """
        if description_lines is None:
            description_lines = []

        separator = self._separator_cache
        yield separator
        yield title

        if description_lines:
            yield description_lines[0]

        yield separator
        yield ""

        for line in description_lines:
            yield line

        yield ""

    def format_string(
        self, title: str, description_lines: Sequence[str] | None = None
    ) -> str:
        """Get formatted header as a single string.

        Useful for writing to files or composing text output.

        Args:
            title: Main header title
            description_lines: Optional detail lines (default: None)

        Returns:
            Formatted header as a multiline string

        Example:
            >>> header_text = printer.format_string("Title", ["Line 1"])
            >>> with open("output.txt", "w") as f:
            ...     f.write(header_text)
        """
        return "\n".join(self.format_lines(title, description_lines))

    def print_analysis_header(
        self, title: str, description_lines: Sequence[str] | None = None
    ) -> None:
        """Log a formatted header block.

        Args:
            title: Main header title
            description_lines: Optional detail lines (default: None)

        Example:
            >>> printer.print_analysis_header(
            ...     "Processing",
            ...     ["Step 1 of 3", "Duration: 2.5s"]
            ... )
        """
        for line in self.format_lines(title, description_lines):
            self._log(line)

    def _make_separator(self) -> str:
        """Create a formatted separator line."""
        return self._separator_cache

    def _compute_separator(self) -> str:
        """Compute the cached separator string."""
        return self.separator_char * self.separator_width

    def _log(self, message: str) -> None:
        """Log a message at the configured level."""
        self._logger.log(self.log_level, "%s", message)


# Module-level convenience instance
printer = HeaderPrinter()

__all__ = ["HeaderPrinter", "printer"]
