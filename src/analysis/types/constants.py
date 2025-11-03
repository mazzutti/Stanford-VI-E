"""Type variables and domain constants for analysis workflows."""

from typing import TypeVar
from enum import Enum

__all__ = ["T", "Domain"]


# ============================================================================
# Type Variables
# ============================================================================

# Type variables enable flexible, reusable protocols that work with any type:
#   - CacheProtocol[T] works with any value type
#   - Preserves type safety in generic operations
#   - Essential for creating reusable abstractions
T = TypeVar("T")


# ============================================================================
# Domain & Visualization
# ============================================================================


class Domain(str, Enum):
    """Canonical domain values used across analysis and plotting.

    Use Domain.DEPTH and Domain.TIME. It's a ``str``-backed Enum so it
    compares equal to the raw string values (e.g. ``Domain.DEPTH == "depth"``).

    Examples:
        >>> domain = Domain.DEPTH
        >>> domain == "depth"
        True
        >>> Domain.is_valid("time")
        True
    """

    DEPTH = "depth"
    TIME = "time"

    def is_depth(self) -> bool:
        """Check if this domain is depth."""
        return self == Domain.DEPTH

    def is_time(self) -> bool:
        """Check if this domain is time."""
        return self == Domain.TIME

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string is a valid domain value.

        Args:
            value: String to validate

        Returns:
            True if value is 'depth' or 'time'
        """
        return value in {d.value for d in cls}
