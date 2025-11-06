"""File management utilities."""

from pathlib import Path
from typing import Optional

from src.processing.managers.base import BaseManager

__all__ = ["FileManager"]


class FileManager(BaseManager):
    """Manages file operations: opening and checking file existence."""

    def clear(self, *args, **kwargs) -> int:
        """No-op for FileManager (placeholder for interface compliance)."""
        return 0

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

    def summarize(self, *args, **kwargs) -> None:
        """No-op for FileManager (placeholder for interface compliance)."""
        pass
