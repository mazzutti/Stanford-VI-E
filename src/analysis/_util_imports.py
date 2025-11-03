"""Re-exported utility types and modules for analysis workflows.

This module provides convenient imports of commonly-used utility types
and modules that are needed throughout the analysis package. Rather than
having individual modules import these from stdlib directly, they're
centralized here for consistency and easier maintenance.

Exported utilities:
    - Path: pathlib.Path for filesystem operations
    - os: Standard library os module
    - sys: Standard library sys module
    - time: Standard library time module for timing operations
    - shutil: Standard library shutil for file operations

Usage:
    >>> from src.analysis._util_imports import Path, os, sys, time, shutil
    >>> cache_path = Path.home() / ".cache"
    >>> import sys; print(sys.version)
"""

from pathlib import Path
import os
import sys
import time
import shutil

__all__ = ["Path", "os", "sys", "time", "shutil"]
