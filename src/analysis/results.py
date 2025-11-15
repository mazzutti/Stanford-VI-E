"""Generic result wrapper for unified result handling across analysis module.

This module provides a type-safe, generic result container that eliminates
the need for specialized result types (GradientCorrelationResult,
BoundaryAmpsResult, etc.), reducing boilerplate while maintaining type safety.

Design Pattern:
    - Generic Result[T]: Type-safe container for any analysis result
    - ResultMetadata: Standardized metadata (execution time, status, etc.)
    - Result composability: Results can be combined and transformed
    - Type preservation: Type information preserved at runtime via generics

Benefits:
    - Eliminates 8+ specialized result type classes (~300 lines)
    - Unified interface across all analyzers
    - Better composability and transformation
    - Maintains full type safety with generics

Example:
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Statistics:
    ...     mean: float
    ...     std: float
    ...
    >>> result = Result(
    ...     data=Statistics(mean=1.0, std=0.5),
    ...     name="gradient_correlation",
    ...     execution_time_ms=42.5
    ... )
    >>> assert result.data.mean == 1.0
    >>> print(result.summary())
"""

# NOTE: We will aim to remove these file-level suppressions after
# adding more precise overloads for `Result.get` and `Result.combine`.
# The overload implementations below help Pyright infer mapping types.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import (
    Generic,
    TypeVar,
    Any,
    Protocol,
    cast,
    overload,
)
from collections.abc import Callable, Mapping
from datetime import datetime
import logging

__all__ = [
    "Result",
    "ResultMetadata",
    "ResultData",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Generic data type for result content
T_co = TypeVar("T_co", covariant=True)  # Covariant for inheritance
V = TypeVar("V")


class ResultData(Protocol[T_co]):
    """Protocol for result data classes.

    Enables duck-typing for result data without explicit inheritance.
    All result data should be serializable and have a string representation.
    """

    def __repr__(self) -> str:
        """Return string representation."""
        ...


@dataclass(frozen=True)
class ResultMetadata:
    """Standardized metadata for all analysis results.

    Provides consistent tracking of result creation time, execution duration,
    status, and any error information.

    Attributes
    ----------
    name : str
        Identifying name for this result (e.g., "gradient_correlation")
    execution_time_ms : float
        Time to compute result in milliseconds
    created_at : datetime
        When result was created
    status : str
        Status indicator: "success", "partial", "warning", "error"
    error_message : str, optional
        Error details if status != "success"
    metadata : dict, optional
        Additional arbitrary metadata as key-value pairs
    """

    name: str
    execution_time_ms: float
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "success"
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))

    def is_success(self) -> bool:
        """Check if result indicates success."""
        return self.status == "success"

    def is_error(self) -> bool:
        """Check if result indicates error."""
        return self.status == "error"

    def has_warning(self) -> bool:
        """Check if result has warning status."""
        return self.status == "warning"

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


@dataclass
class Result(Generic[T]):
    """Generic result container for analysis outputs.

    Replaces multiple specialized result type classes with a single generic
    container. Maintains type safety while eliminating boilerplate.

    Type Parameters
    ---------------
    T : TypeVar
        Type of data contained in this result (e.g., Statistics, dict, array)

    Attributes
    ----------
    data : T
        The actual result data (type-safe via generics)
    metadata : ResultMetadata
        Execution metadata (timing, status, etc.)
    tags : list[str]
        Optional tags for categorizing/filtering results

    Examples
    --------
    Basic usage with dataclass:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Statistics:
        ...     mean: float
        ...     std: float
        ...
        >>> result = Result(
        ...     data=Statistics(mean=1.0, std=0.5),
        ...     metadata=ResultMetadata(
        ...         name="gradient",
        ...         execution_time_ms=10.0
        ...     )
        ... )
        >>> assert result.data.mean == 1.0

    With dictionaries:
        >>> result = Result(
        ...     data={"intercept": 0.1, "gradient": 0.05},
        ...     metadata=ResultMetadata(name="avo", execution_time_ms=5.0)
        ... )
        >>> print(result.get("intercept"))

    Result chaining:
        >>> result1 = Result(data={"x": 1}, metadata=...)
        >>> result2 = result1.transform(lambda d: {"x_doubled": d["x"] * 2})
        >>> print(result2.data)  # {"x_doubled": 2}
    """

    data: T
    metadata: ResultMetadata
    tags: list[str] = field(default_factory=lambda: cast(list[str], []))

    def __post_init__(self) -> None:
        """Validate result state after initialization."""
        if not isinstance(
            self.data, (dict, list, str, int, float, tuple)
        ) and not hasattr(self.data, "__dataclass_fields__"):
            logger.debug(
                f"Result contains non-standard data type: {type(self.data).__name__}"
            )

    @property
    def is_success(self) -> bool:
        """Check if result represents successful execution."""
        return self.metadata.is_success()

    @property
    def is_error(self) -> bool:
        """Check if result represents an error."""
        return self.metadata.is_error()

    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        return self.metadata.execution_time_ms

    @overload
    def get(
        self: Result[Mapping[str, V]], key: str, default: None = ...
    ) -> V | None: ...

    @overload
    def get(self: Result[Mapping[str, V]], key: str, default: V) -> V: ...

    @overload
    def get(self, key: str, default: Any = None) -> Any: ...

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from result data (dict-like access).

        Enables dict-like access for dict-based results.

        Parameters
        ----------
        key : str
            Key to retrieve
        default : Any, optional
            Default value if key not found

        Returns
        -------
        Any
            Value at key, or default if not found/not dict
        """
        # Overloads below allow type checkers to infer the return type when
        # `self.data` is a mapping with known value type. These overloads
        # are declared just above the method implementation.
        if isinstance(self.data, dict):
            # Use an `Any`-cast here to avoid Pyright's partially-unknown
            # mapping diagnostics while preserving runtime behavior.
            # pyright: ignore[reportUnknownMemberType]
            return cast(Any, self.data).get(key, default)
        return default

    def transform(self, func: Callable[[T], Any]) -> Result[Any]:
        """Transform result data with a function.

        Creates new Result with transformed data, preserving metadata and tags.

        Parameters
        ----------
        func : callable
            Function to apply to data

        Returns
        -------
        Result[Any]
            New result with transformed data

        Example
        -------
        >>> result = Result(data={"x": 1}, metadata=...)
        >>> doubled = result.transform(lambda d: {"x_doubled": d["x"] * 2})
        """
        try:
            transformed_data = func(self.data)
            return Result(
                data=transformed_data,
                metadata=self.metadata,
                tags=self.tags.copy(),
            )
        except Exception as e:
            logger.warning(f"Transform failed: {e}")
            raise

    def combine(self, other: Result[T]) -> Result[Any]:
        """Combine two results into one.

        Useful for merging partial results. Creates new result with combined
        metadata and combined tags.

        Parameters
        ----------
        other : Result[T]
            Other result to combine with

        Returns
        -------
        Result
            New combined result
        """

        combined_metadata = ResultMetadata(
            name=f"{self.metadata.name}+{other.metadata.name}",
            execution_time_ms=self.metadata.execution_time_ms
            + other.metadata.execution_time_ms,
            status="partial" if (self.is_error or other.is_error) else "success",
        )
        combined_tags = list(set(self.tags + other.tags))

        # Combine data if both are dicts
        combined_data: Any
        if isinstance(self.data, dict) and isinstance(other.data, dict):
            # Cast to `Any` first to avoid Pyright's partially-unknown
            # mapping checks, then coerce to concrete `dict[str, Any]`.
            # pyright: ignore[reportUnknownMemberType]
            self_any = cast(Any, self.data)
            # pyright: ignore[reportUnknownMemberType]
            other_any = cast(Any, other.data)
            self_dict: dict[str, Any] = dict(cast(dict[str, Any], self_any))
            other_dict: dict[str, Any] = dict(cast(dict[str, Any], other_any))
            combined_data = {**self_dict, **other_dict}
        else:
            # Heterogeneous fallback tuple
            # pyright: ignore[reportUnknownMemberType]
            combined_data = (self.data, other.data)

        return Result(
            data=combined_data,
            metadata=combined_metadata,
            tags=combined_tags,
        )

    def with_metadata(self, **kwargs: Any) -> Result[T]:
        """Create new result with updated metadata.

        Parameters
        ----------
        **kwargs : Any
            Metadata fields to update (name, status, error_message, etc.)

        Returns
        -------
        Result[T]
            New result with updated metadata
        """
        current_dict = asdict(self.metadata)
        current_dict.update(kwargs)
        new_metadata = ResultMetadata(**current_dict)
        return Result(
            data=self.data,
            metadata=new_metadata,
            tags=self.tags.copy(),
        )

    def with_tags(self, *tags: str) -> Result[T]:
        """Create new result with added tags.

        Parameters
        ----------
        *tags : str
            Tags to add

        Returns
        -------
        Result[T]
            New result with added tags
        """
        return Result(
            data=self.data,
            metadata=self.metadata,
            tags=list(set(self.tags + list(tags))),
        )

    def summary(self) -> str:
        """Get human-readable summary of result.

        Returns
        -------
        str
            Summary including name, status, timing, and data preview
        """
        status_icon = "✓" if self.is_success else "✗" if self.is_error else "⚠"
        data_preview = str(self.data)[:80]
        return (
            f"{status_icon} {self.metadata.name} "
            f"({self.execution_time_ms:.1f}ms) "
            f"[{self.metadata.status}]: {data_preview}..."
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation.

        Returns
        -------
        dict
            Dictionary with 'data', 'metadata', and 'tags' keys
        """

        def _serialize_data(value: Any) -> Any:
            """Return a JSON-serializable representation for `data`.

            Narrow types locally so the type checker sees concrete mapping types
            instead of `Unknown` generic dicts.
            """
            if isinstance(value, dict):
                # Cast to a concrete mapping type for returning.
                return cast(dict[str, Any], value)
            if hasattr(value, "__dataclass_fields__"):
                return asdict(value)
            return str(value)

        return {
            "data": _serialize_data(self.data),
            "metadata": self.metadata.to_dict(),
            "tags": self.tags,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Result(name={self.metadata.name!r}, "
            f"status={self.metadata.status!r}, "
            f"time={self.execution_time_ms:.1f}ms)"
        )

    def __str__(self) -> str:
        """Return human-readable string."""
        return self.summary()

    # ============================================================================


# Helper Functions for Result Creation
# ============================================================================


class MappingResult(Result[dict[str, V]], Generic[V]):
    """Specialized Result for mapping/dict-like data with precise typing.

    Use this class when a `Result` carries a dictionary-like payload and
    callers want statically-typed `get`/`combine` operations.
    """

    # Note: `get_mapping` below provides the typed mapping access.

    def get_mapping(self, key: str, default: V | None = None) -> V | None:
        """Dict-like typed access for mapping results.

        Use `get_mapping` when you have a `MappingResult[V]` and want a
        statically-typed value back (V | None). This avoids overriding
        `Result.get` which must remain flexible for heterogeneous `Result`.

        """
        data_dict: dict[str, V] = dict(self.data)
        return data_dict.get(key, default)

    def combine_mapping(self, other: MappingResult[V]) -> MappingResult[V]:
        combined_metadata = ResultMetadata(
            name=f"{self.metadata.name}+{other.metadata.name}",
            execution_time_ms=self.metadata.execution_time_ms
            + other.metadata.execution_time_ms,
            status="partial" if (self.is_error or other.is_error) else "success",
        )
        combined_tags = list(set(self.tags + other.tags))
        self_dict: dict[str, V] = dict(self.data)
        other_dict: dict[str, V] = dict(other.data)
        combined_data: dict[str, V] = {**self_dict, **other_dict}
        return MappingResult(
            data=combined_data, metadata=combined_metadata, tags=combined_tags
        )


def wrap_result(
    data: T,
    name: str = "result",
    execution_time_ms: float = 0.0,
    status: str = "success",
    tags: list[str] | None = None,
) -> Result[T]:
    """Wrap data in a Result[T] container.

    Provides a simple, convenient way to wrap computed data with metadata.

    Parameters
    ----------
    data : T
        The computed data to wrap
    name : str, optional
        Name for this result, by default "result"
    execution_time_ms : float, optional
        Execution time in milliseconds, by default 0.0
    status : str, optional
        Result status ("success", "error", "partial"), by default "success"
    tags : List[str], optional
        Tags for categorizing the result, by default None

    Returns
    -------
    Result[T]
        Wrapped result with metadata

    Example
    -------
    >>> result = wrap_result(42.0, name="computation", execution_time_ms=1.5)
    >>> assert result.is_success
    >>> assert result.data == 42.0
    """
    metadata = ResultMetadata(
        name=name,
        execution_time_ms=execution_time_ms,
        status=status,
    )
    return Result(
        data=data,
        metadata=metadata,
        tags=tags or [],
    )


def create_metadata(
    name: str = "result",
    execution_time_ms: float = 0.0,
    status: str = "success",
) -> ResultMetadata:
    """Create result metadata.

    Parameters
    ----------
    name : str, optional
        Metadata name, by default "result"
    execution_time_ms : float, optional
        Execution time in milliseconds, by default 0.0
    status : str, optional
        Result status, by default "success"

    Returns
    -------
    ResultMetadata
        Metadata object
    """
    return ResultMetadata(
        name=name,
        execution_time_ms=execution_time_ms,
        status=status,
    )
