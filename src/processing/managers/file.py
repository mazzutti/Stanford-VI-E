"""File management utilities."""

from pathlib import Path
from typing import Optional
import logging

from src.processing.managers.resource_manager import ResourceManager


__all__ = ["FileManager"]


class FileManager(ResourceManager[Path]):
    """Manages file operations: opening and checking file existence."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Initialize file manager with no-op strategies."""
        super().__init__(resource_dir=Path("."), logger=logger)

    def open(
        self, filepath: str, description: Optional[str] = None, prefix: str = ""
    ) -> bool:
        """Open a file in a platform-friendly way.

        Prefer a pure-Python approach (`webbrowser.open`) and fall back to
        platform shell openers (`open`, `xdg-open`) if necessary.

        Args:
            filepath: Path to file to open
            description: Optional description for logging
            prefix: Prefix for log messages

        Returns:
            True if an attempt to open the file was made, False otherwise
        """
        p = Path(filepath)
        if not p.exists():
            self._log_error("%sMissing file: %s", prefix, filepath)
            return False

        # Try webbrowser which is cross-platform for file:// URLs
        try:
            import webbrowser

            webbrowser.open(f"file://{p.resolve()}")
            return True
        except Exception:
            pass

        # Fallback to platform-specific opener
        try:
            import shutil
            import subprocess

            if shutil.which("open"):
                subprocess.run(["open", str(p)], check=False)
                return True
            if shutil.which("xdg-open"):
                subprocess.run(["xdg-open", str(p)], check=False)
                return True
        except Exception:
            pass

        self._log_warning("%sCould not open file: %s", prefix, filepath)
        return False
