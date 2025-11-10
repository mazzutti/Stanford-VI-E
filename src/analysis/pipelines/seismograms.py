#!/usr/bin/env python
"""Seismic modeling pipeline helpers.

This module provides the SeismogramAnalyzer class, a facade for coordinating
seismogram analysis and modeling tasks. The CLI entrypoint is exposed via
the `main()` method, which orchestrates the full pipeline through
`src.modeling.api.run_full_modeling`.
"""

from __future__ import annotations

from src.analysis.common import AnalysisCommon
from pathlib import Path
import time

import logging
import subprocess
from subprocess import CompletedProcess
from typing import Optional, Generator
from contextlib import contextmanager


__all__ = [
    "SeismogramAnalyzer",
]


# Object-oriented facade for seismogram utilities
class SeismogramAnalyzer:
    """Coordinates seismogram analysis and modeling tasks.

    This class provides a unified interface for running shell commands,
    managing cache, checking files, and orchestrating the seismic modeling
    pipeline. It delegates to AnalysisCommon for analysis-specific operations.
    """

    # Configuration constants
    DEFAULT_CACHE_PATTERNS = ["avo_*.npz"]
    FILE_READY_DELAY_SECONDS = 1
    BYTES_PER_MB = 1024 * 1024
    INDENT_PREFIX = "  "

    def __init__(self, analysis: Optional[AnalysisCommon] = None) -> None:
        """Initialize the SeismogramAnalyzer with an AnalysisCommon instance.

        Args:
            analysis: Optional AnalysisCommon instance for dependency injection.
                     If None, creates a new instance.
        """
        self._analysis = analysis or AnalysisCommon.instance()
        self._logger = logging.getLogger(self.__class__.__name__)

    @contextmanager
    def _timed_operation(
        self, description: str, prefix: str = ""
    ) -> Generator[None, None, None]:
        """Context manager for timing and logging operations.

        Logs operation start and completion time. Guarantees timing is logged
        even if an exception occurs.

        Args:
            description: Description of the operation for logging.
            prefix: Optional prefix for all log messages.

        Yields:
            None: The context block executes between start and end logging.
        """
        if description:
            self._logger.info("%s%s", prefix, description)
        t0 = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - t0
            self._logger.info("%s✓ Completed in %.1f seconds", prefix, elapsed)

    def run_command(
        self, cmd: str, description: str = "", prefix: str = ""
    ) -> Optional[CompletedProcess[bytes]]:
        """Run a shell command with timing and error reporting.

        Args:
            cmd: Shell command to execute. Must be non-empty.
            description: Optional description of the command for logging.
            prefix: Optional prefix for log messages.

        Returns:
            CompletedProcess if successful, None on error.

        Raises:
            ValueError: If cmd is empty or invalid.
        """
        if not cmd or not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("Command must be a non-empty string")

        try:
            with self._timed_operation(description, prefix):
                res = subprocess.run(cmd, shell=True, check=True)
            return res
        except subprocess.CalledProcessError as e:
            self._logger.error(
                "%s✗ Command failed with exit code %d: %s", prefix, e.returncode, e
            )
            return None
        except Exception as e:
            self._logger.error("%s✗ Error running command: %s", prefix, e)
            return None

    def clear_cache(
        self, patterns: Optional[list[str]] = None, prefix: str = ""
    ) -> int:
        """Clear cache files matching specified patterns.

        Args:
            patterns: List of glob patterns to match. Defaults to DEFAULT_CACHE_PATTERNS.
            prefix: Optional prefix for log messages.

        Returns:
            Number of cache files cleared.
        """
        cache_patterns = patterns or self.DEFAULT_CACHE_PATTERNS
        return self._analysis.clear_cache(patterns=cache_patterns, prefix=prefix)

    def check_file_exists(self, filepath: str, description: str) -> bool:
        """Check if a file exists and log its details.

        Args:
            filepath: Path to the file to check.
            description: Description of the file for logging.

        Returns:
            True if file exists, False otherwise.
        """
        try:
            path = Path(filepath)
            if path.exists():
                size_mb = path.stat().st_size / self.BYTES_PER_MB
                self._logger.info(
                    "\u2713 Found: %s (%.1f MB at %s)", description, size_mb, filepath
                )
                return True

            self._logger.error(
                "\u2717 Missing: %s (expected at %s)", description, filepath
            )
            return False
        except (OSError, PermissionError) as e:
            self._logger.error(
                "\u2717 Error checking %s: %s (cannot access %s)",
                description,
                e,
                filepath,
            )
            return False

    def open_file(self, filepath: str, description: str, prefix: str = "") -> bool:
        """Open a file using the analysis helper.

        Adds a short delay after opening to ensure file is fully ready.

        Args:
            filepath: Path to the file to open.
            description: Description of the file for logging.
            prefix: Optional prefix for log messages.

        Returns:
            True if file was opened successfully, False otherwise.
        """
        ok = self._analysis.open_file(filepath, description=description, prefix=prefix)
        if ok:
            try:
                time.sleep(self.FILE_READY_DELAY_SECONDS)
            except InterruptedError as e:
                self._logger.warning("File ready delay interrupted: %s", e)
        return ok

    def run(
        self,
        *,
        cache_dir: str = ".cache",
        skip_cleanup: bool = False,
        verbose: bool = False,
    ) -> bool:
        """Run the complete seismogram modeling pipeline.

        This method orchestrates the full pipeline by delegating to
        src.modeling.api.run_full_modeling. It handles cache cleanup,
        logging configuration, and error handling.

        Args:
            cache_dir: Directory for cache files. Defaults to ".cache".
            skip_cleanup: If True, skip cache cleanup before running.
            verbose: If True, enable debug logging for this module.

        Returns:
            True if pipeline completed successfully.

        Raises:
            ValueError: If cache_dir is invalid.
            RuntimeError: If the seismic modeling process fails.
        """
        if not cache_dir or not isinstance(cache_dir, str):
            raise ValueError("cache_dir must be a non-empty string")

        if verbose:
            self._logger.setLevel(logging.DEBUG)
            self._logger.debug("Debug logging enabled")

        self._logger.info(
            "Starting seismic modeling pipeline (cache_dir=%s)", cache_dir
        )

        if not skip_cleanup:
            self._logger.debug("Clearing cache before pipeline execution")
            self.clear_cache(prefix=self.INDENT_PREFIX)

        try:
            with self._timed_operation(
                "Running full modeling", prefix=self.INDENT_PREFIX
            ):
                from src.modeling.api import run_full_modeling

                run_full_modeling(
                    cache_dir=cache_dir,
                    add_avo_noise=False,
                )
            self._logger.info("✓ Seismic modeling pipeline completed successfully")
            return True
        except ImportError as e:
            self._logger.error("Failed to import modeling API: %s", e)
            raise RuntimeError(f"Failed to import modeling API: {e}") from e
        except Exception as e:
            self._logger.error("Seismic modeling failed: %s", e)
            raise RuntimeError(f"Seismic modeling failed (in-process): {e}") from e
