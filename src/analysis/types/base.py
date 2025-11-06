"""Abstract base classes for computational components.

This module provides core abstractions for building analysis systems:
- Computer, AnalysisSchema: Computational component abstractions

For type protocols, see protocols.py instead.
For analyzer abstractions, see analyzer.py instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from src.analysis.models import AvoResults

__all__ = [
    "Computer",
    "AnalysisSchema",
    "ComputationResult",
    # Type variables
    "T",
    "T_In",
    "T_Out",
]

# Type variables for generic constraints
T_In = TypeVar("T_In")  # Input type
T_Out = TypeVar("T_Out")  # Output type
T = TypeVar("T")  # Generic type


# ============================================================================
# Core Computational Abstractions
# ============================================================================


@dataclass
class ComputationResult(Generic[T_Out]):
    """Result wrapper for computation operations.

    Provides a structured way to return computation results with metadata
    about success/failure and any errors.
    """

    is_valid: bool
    """Whether computation succeeded."""

    data: Optional[T_Out] = None
    """Computed data (None if invalid)."""

    error_message: str = ""
    """Error message if computation failed."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about computation (performance, validation details, etc)."""


@dataclass
class AnalysisSchema:
    """Describes the input/output contract for an analyzer.

    Provides self-documenting information about what data an analyzer
    expects and what it produces.
    """

    input_fields: Dict[str, Type[Any]]
    """Required input field names and their types."""

    output_fields: Dict[str, Type[Any]]
    """Output field names and their types."""

    description: str = ""
    """Human-readable description of the analysis."""

    constraints: Dict[str, str] = field(default_factory=dict)
    """Any constraints on the analysis (e.g., 'vp >= 1000 m/s')."""


class Computer(ABC, Generic[T_In, T_Out]):
    """Abstract base for domain-specific computational components.

    Computers encapsulate specific computational tasks that transform
    input data into output data. They provide consistent interfaces for:
    - Input validation
    - Computation execution
    - Schema documentation

    This abstraction enables polymorphic treatment of different computers
    and makes them easy to compose into larger analysis pipelines.

    Type Parameters
    ----------------
    T_In
        Type of input data accepted by this computer.
    T_Out
        Type of output data produced by this computer.

    Examples
    --------
    A concrete computer implementation:

    >>> class MyComputer(Computer[np.ndarray, Dict[str, np.ndarray]]):
    ...     def compute(self, data: NDArray[np.floating[Any]]) -> Dict[str, np.ndarray]:
    ...         # Do computation
    ...         return {"result": computed_array}
    ...
    ...     def validate(self, data: NDArray[np.floating[Any]]) -> bool:
    ...         return data.shape == expected_shape
    """

    @abstractmethod
    def validate(self, inputs: T_In) -> bool:
        """Validate that inputs are suitable for computation.

        Parameters
        ----------
        inputs
            Input data to validate.

        Returns
        -------
        bool
            True if inputs are valid, False otherwise.

        Notes
        -----
        Implementation should not modify inputs. If validation fails,
        the compute() method should raise an exception with details.
        """
        pass

    @abstractmethod
    def compute(self, inputs: T_In) -> T_Out:
        """Execute the computational task.

        Parameters
        ----------
        inputs
            Validated input data.

        Returns
        -------
        T_Out
            Computed output data.

        Raises
        ------
        ValueError
            If computation fails or inputs are invalid.
        """
        pass

    @abstractmethod
    def get_schema(self) -> AnalysisSchema:
        """Return schema describing this computer's inputs/outputs.

        Returns
        -------
        AnalysisSchema
            Self-documenting schema of computation contract.
        """
        pass
