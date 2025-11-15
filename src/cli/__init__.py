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

from src.cli.parsers import ParserFactory
from src.cli import tools
from src.cli import modeling

__all__ = ["ParserFactory", "tools", "modeling"]
