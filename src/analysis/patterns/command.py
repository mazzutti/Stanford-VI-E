"""Command Pattern Implementation for Analysis Operations

This module provides command encapsulation for analysis operations,
enabling undo/redo functionality, command history, and batch execution.

Patterns Used:
  - Command: Encapsulate requests as objects
  - History: Maintain command history for undo/redo

Example:
    >>> from src.analysis.patterns.command import CommandQueue, RunAnalysisCommand
    >>>
    >>> queue = CommandQueue()
    >>>
    >>> cmd1 = RunAnalysisCommand(analyzer, data)
    >>> queue.execute(cmd1)  # Execute and store in history
    >>>
    >>> queue.undo()  # Undo last command
    >>> queue.redo()  # Redo last undone command
    >>>
    >>> for cmd in queue.history:
    ...     print(cmd.description)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisCommand",
    "RunAnalysisCommand",
    "CommandQueue",
    "MacroCommand",
]


class AnalysisCommand(ABC):
    """Abstract base class for analysis commands.

    Commands encapsulate analysis operations, enabling undo/redo,
    serialization, and batch execution.
    """

    def __init__(self) -> None:
        """Initialize command"""
        self.executed = False
        self.timestamp = datetime.now()

    @abstractmethod
    def execute(self) -> Any:
        """Execute the command.

        Returns:
            Command result

        Raises:
            RuntimeError: If command execution fails
        """
        pass

    @abstractmethod
    def undo(self) -> bool:
        """Undo the command.

        Returns:
            True if undo successful, False if not undoable
        """
        pass

    @abstractmethod
    def redo(self) -> Any:
        """Redo the command (re-execute after undo).

        Returns:
            Command result
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get human-readable command description.

        Returns:
            Description string
        """
        pass

    @property
    def is_undoable(self) -> bool:
        """Check if command can be undone.

        Returns:
            True if command supports undo, False otherwise
        """
        return True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"executed={self.executed}, "
            f"timestamp={self.timestamp.isoformat()})"
        )


class RunAnalysisCommand(AnalysisCommand):
    """Command for running a single analysis operation.

    Encapsulates an analysis execution with the ability to
    undo and redo the operation.
    """

    def __init__(self, analyzer: Any, data: Dict[str, Any]) -> None:
        """Initialize run analysis command.

        Args:
            analyzer: Analyzer instance to run
            data: Input data for analysis
        """
        super().__init__()
        self.analyzer = analyzer
        self.data = data
        self.result: Optional[Any] = None
        self.previous_cache: Optional[dict[str, Any]] = None

    def execute(self) -> Any:
        """Execute the analysis.

        Returns:
            Analysis result
        """
        logger.info(f"Executing {self.description}")

        try:
            # Save cache state for undo
            if hasattr(self.analyzer, "cache"):
                self.previous_cache = dict(self.analyzer.cache)

            # Execute analysis
            self.result = self.analyzer.run(self.data)
            self.executed = True

            logger.debug(f"Successfully executed: {self.description}")
            return self.result

        except Exception as e:
            logger.error(f"Failed to execute: {self.description}: {e}")
            raise

    def undo(self) -> bool:
        """Undo the analysis by clearing results and restoring cache.

        Returns:
            True if undo successful
        """
        if not self.executed:
            logger.warning("Cannot undo command that hasn't been executed")
            return False

        logger.info(f"Undoing {self.description}")

        try:
            # Restore cache if available
            if self.previous_cache is not None and hasattr(self.analyzer, "cache"):
                self.analyzer.cache.clear()
                self.analyzer.cache.update(self.previous_cache)

            # Clear result
            self.result = None
            self.executed = False

            logger.debug(f"Successfully undone: {self.description}")
            return True

        except Exception as e:
            logger.error(f"Failed to undo: {self.description}: {e}")
            return False

    def redo(self) -> Any:
        """Redo the analysis (re-execute after undo).

        Returns:
            Analysis result
        """
        logger.info(f"Redoing {self.description}")
        return self.execute()

    @property
    def description(self) -> str:
        """Get command description."""
        data_summary = ", ".join(self.data.keys()) if self.data else "no data"
        return f"Run {self.analyzer.__class__.__name__} " f"on {data_summary}"


class MacroCommand(AnalysisCommand):
    """Composite command that executes multiple commands as one.

    Useful for grouping related commands that should be undone/redone
    together.
    """

    def __init__(
        self, name: str, commands: Optional[List[AnalysisCommand]] = None
    ) -> None:
        """Initialize macro command.

        Args:
            name: Macro name
            commands: Optional list of commands to execute
        """
        super().__init__()
        self.name = name
        self.commands = commands or []

    def add_command(self, command: AnalysisCommand) -> MacroCommand:
        """Add a command to the macro.

        Args:
            command: Command to add

        Returns:
            Self for chaining
        """
        self.commands.append(command)
        logger.debug(f"Added command to macro '{self.name}': {command.description}")
        return self

    def execute(self) -> Any:
        """Execute all commands in the macro.

        Returns:
            Result of last command
        """
        logger.info(f"Executing macro: {self.description}")

        result = None
        for command in self.commands:
            try:
                result = command.execute()
            except Exception as e:
                logger.error(f"Error executing command in macro: {e}")
                raise

        self.executed = True
        return result

    def undo(self) -> bool:
        """Undo all commands in reverse order.

        Returns:
            True if all undos successful
        """
        logger.info(f"Undoing macro: {self.description}")

        success = True
        # Undo in reverse order
        for command in reversed(self.commands):
            try:
                if not command.undo():
                    success = False
                    logger.warning(f"Failed to undo: {command.description}")
            except Exception as e:
                logger.error(f"Error undoing command in macro: {e}")
                success = False

        self.executed = False
        return success

    def redo(self) -> Any:
        """Redo all commands in order.

        Returns:
            Result of last command
        """
        logger.info(f"Redoing macro: {self.description}")

        result = None
        for command in self.commands:
            result = command.redo()

        self.executed = True
        return result

    @property
    def description(self) -> str:
        """Get macro description."""
        return f"Macro '{self.name}' ({len(self.commands)} commands)"


class CommandQueue:
    """Manages command execution, undo/redo history.

    Maintains a history of executed commands and provides
    undo/redo functionality.
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialize command queue.

        Args:
            max_history: Maximum number of commands to keep in history
        """
        self.history: List[AnalysisCommand] = []
        self.current_index: int = -1
        self.max_history = max_history

    def execute(self, command: AnalysisCommand) -> Any:
        """Execute a command and add to history.

        Args:
            command: Command to execute

        Returns:
            Command result
        """
        logger.info(f"Executing command: {command.description}")

        try:
            result = command.execute()

            # Remove any redo commands after current position
            if self.current_index < len(self.history) - 1:
                self.history = self.history[: self.current_index + 1]

            # Add command to history
            self.history.append(command)
            self.current_index += 1

            # Trim history if it exceeds max
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]
                self.current_index = len(self.history) - 1

            logger.debug(
                f"Command executed. History position: "
                f"{self.current_index + 1}/{len(self.history)}"
            )

            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise

    def undo(self) -> bool:
        """Undo the last executed command.

        Returns:
            True if undo successful
        """
        if self.current_index < 0:
            logger.warning("Nothing to undo")
            return False

        command = self.history[self.current_index]
        logger.info(f"Undoing: {command.description}")

        if command.undo():
            self.current_index -= 1
            logger.debug(
                f"Undo successful. Position: {self.current_index + 1}/{len(self.history)}"
            )
            return True
        else:
            logger.error(f"Undo failed for: {command.description}")
            return False

    def redo(self) -> bool:
        """Redo the next command that was undone.

        Returns:
            True if redo successful
        """
        if self.current_index >= len(self.history) - 1:
            logger.warning("Nothing to redo")
            return False

        next_index = self.current_index + 1
        command = self.history[next_index]
        logger.info(f"Redoing: {command.description}")

        try:
            command.redo()
            self.current_index = next_index
            logger.debug(
                f"Redo successful. Position: {self.current_index + 1}/{len(self.history)}"
            )
            return True
        except Exception as e:
            logger.error(f"Redo failed: {e}")
            return False

    def clear(self) -> None:
        """Clear all command history"""
        self.history.clear()
        self.current_index = -1
        logger.info("Command history cleared")

    @property
    def can_undo(self) -> bool:
        """Check if undo is available"""
        return self.current_index >= 0

    @property
    def can_redo(self) -> bool:
        """Check if redo is available"""
        return self.current_index < len(self.history) - 1

    @property
    def current_command(self) -> Optional[AnalysisCommand]:
        """Get the current command"""
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index]
        return None

    def get_history_summary(self) -> str:
        """Get a summary of command history.

        Returns:
            String summary of all commands
        """
        lines = [
            f"Command History ({len(self.history)} total):",
            f"Current position: {self.current_index + 1}/{len(self.history)}",
        ]

        for i, cmd in enumerate(self.history):
            marker = "→ " if i == self.current_index else "  "
            lines.append(f"{marker}{i + 1}. {cmd.description}")

        return "\n".join(lines)
