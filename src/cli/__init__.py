"""CLI module for seismic modeling framework.

This module provides centralized command-line interface utilities, including
argument parsing, tool registry, and logging configuration.

Key components:
    - ParserFactory: Unified argument parsing and CLI coordination
    - Tool decorator: Register functions as CLI tools
    - Logging configuration: Centralized logging setup
    - Tools: CLI tool implementations
    - Modeling: Data loading and workflow orchestration
"""

from src.cli.parsers import ParserFactory, tool
from src.cli import tools as _tools  # noqa: F401 - register tools
from src.cli import modeling  # noqa: F401 - utilities

__all__ = ["ParserFactory", "tool", "modeling"]
