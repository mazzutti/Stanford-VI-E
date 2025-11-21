"""Abstract base classes for computational components.

This module provides core abstractions for building analysis systems:
- Computer, AnalysisSchema: Computational component abstractions

For type protocols, see protocols.py instead.
For analyzer abstractions, see analyzer.py instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

__all__ = [
    "Computer",
    "AnalysisSchema",
    "ComputationResult",
    # Type variables
    "T",
    "T_In",
    "T_Out",
]

# Type variable short names (T, T_In, T_Out) are conventional here; suppress
# naming warnings to avoid noise across many small protocol modules.

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

    data: T_Out | None = None
    """Computed data (None if invalid)."""

    error_message: str = ""
    """Error message if computation failed."""

    metadata: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    """Additional metadata about computation (performance, validation details, etc)."""

@dataclass
class AnalysisSchema:
    """Describes the input/output contract for an analyzer.

    Provides self-documenting information about what data an analyzer
    expects and what it produces.
    """

    input_fields: dict[str, type[Any]]
    """Required input field names and their types."""

    output_fields: dict[str, type[Any]]
    """Output field names and their types."""

    description: str = ""
    """Human-readable description of the analysis."""

    constraints: dict[str, str] = field(
        default_factory=lambda: cast(dict[str, str], {})
    )
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

    >>> class MyComputer(Computer[np.ndarray, dict[str, np.ndarray]]):
    ...     def compute(self, data: NDArray[np.floating[Any]]) -> dict[str, np.ndarray]:
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

    @abstractmethod
    def compute(self, *inputs: Any, **kwargs: Any) -> T_Out:
        """Execute the computational task.

        Concrete implementations may accept multiple positional arguments
        (e.g., `vp, vs, rho`) or a single structured input object. The
        base signature uses variadic arguments so subclasses can define
        explicit parameter lists without conflicting with the abstract
        method's signature.

        Parameters
        ----------
        *inputs
            Positional input arguments for the computation.

        **kwargs
            Optional keyword arguments.

        Returns
        -------
        T_Out
            Computed output data.

        Raises
        ------
        ValueError
            If computation fails or inputs are invalid.
        """

    @abstractmethod
    def get_schema(self) -> AnalysisSchema:
        """Return schema describing this computer's inputs/outputs.

        Returns
        -------
        AnalysisSchema
            Self-documenting schema of computation contract.
        """
