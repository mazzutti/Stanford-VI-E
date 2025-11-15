"""Configuration and core domain models.

This module contains configuration classes and fundamental domain models
like transitions between facies.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from src.analysis.config_mixins import ValidatableConfigMixin
from src.analysis.validators_registry import ValidatorRegistry

__all__ = [
    "Transition",
    "FaciesCorrelationConfig",
]


@dataclass(frozen=True, slots=True)
class Transition:
    """Immutable, hashable representation of a facies transition.

    This class provides validation and convenience methods for
    representing transitions between facies.
    """

    from_facies: int
    to_facies: int

    def __post_init__(self) -> None:
        """Validate transition values are non-negative."""
        if self.from_facies < 0 or self.to_facies < 0:
            raise ValueError("Facies indices must be non-negative")

    def __str__(self) -> str:
        """Return string representation as 'from->to'."""
        return f"{self.from_facies}->{self.to_facies}"

    def __repr__(self) -> str:
        """Return detailed representation for debugging.

        Returns:
            String in format "Transition(from_facies=X, to_facies=Y)" that can be
            evaluated to recreate the object.
        """
        return f"Transition(from_facies={self.from_facies}, to_facies={self.to_facies})"

    def is_self_transition(self) -> bool:
        """Check if transition is from a facies to itself."""
        return self.from_facies == self.to_facies

    def reverse(self) -> Transition:
        """Return the reverse transition."""
        return Transition(self.to_facies, self.from_facies)

    def to_dict(self) -> dict[str, int]:
        """Convert transition to dictionary representation.

        Args:
            None (instance method)

        Returns:
            Dictionary containing:
            - from_facies: Source facies index (non-negative integer)
            - to_facies: Target facies index (non-negative integer)
        """
        return {"from_facies": self.from_facies, "to_facies": self.to_facies}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> Transition:
        """Create transition from dictionary representation.

        Args:
            data: Dictionary with required keys:
                - from_facies: Source facies index (must be non-negative)
                - to_facies: Target facies index (must be non-negative)

        Returns:
            Transition instance with data from dictionary.

        Raises:
            ValueError: If facies indices are negative.
            KeyError: If required keys are missing from input dictionary.
            TypeError: If facies values cannot be converted to integers.
        """
        return cls(
            from_facies=int(data["from_facies"]),
            to_facies=int(data["to_facies"]),
        )

    @classmethod
    def from_string_key(cls, key: str) -> Transition:
        """Create transition from string representation like '0->1'.

        Args:
            key: String in format 'from_facies->to_facies' (e.g., '0->1')

        Returns:
            Transition instance parsed from the string key.

        Raises:
            ValueError: If string format is invalid or facies indices are negative.
        """
        parts = key.split("->")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid transition string format: {key!r}. "
                "Expected format: 'from_facies->to_facies' (e.g., '0->1')"
            )
        try:
            from_facies = int(parts[0])
            to_facies = int(parts[1])
        except ValueError as e:
            raise ValueError(
                f"Invalid facies indices in transition string {key!r}: {e}"
            )
        return cls(from_facies=from_facies, to_facies=to_facies)


@dataclass
class FaciesCorrelationConfig(ValidatableConfigMixin):
    """Configuration for facies correlation analysis with validation."""

    facies_count: int = 4
    boundary_threshold: float = 0.1
    dilation_window: int = 2

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate all configuration parameters."""
        ValidatorRegistry.validate_positive(self.facies_count, "facies_count")
        ValidatorRegistry.validate_exclusive_range(
            self.boundary_threshold, 0.0, 1.0, "boundary_threshold"
        )
        ValidatorRegistry.validate_positive(self.dilation_window, "dilation_window")

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Note: Inherited from ValidatableConfigMixin
        """
        return super().is_valid()

    def to_dict(self) -> dict[str, int | float]:
        """Convert configuration to dictionary."""
        return {
            "facies_count": self.facies_count,
            "boundary_threshold": self.boundary_threshold,
            "dilation_window": self.dilation_window,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> FaciesCorrelationConfig:
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with optional keys for configuration values.

        Returns:
            FaciesCorrelationConfig instance with values from dict or defaults.
        """
        return cls(
            facies_count=int(config_dict.get("facies_count", 4)),
            boundary_threshold=float(config_dict.get("boundary_threshold", 0.1)),
            dilation_window=int(config_dict.get("dilation_window", 2)),
        )

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"FaciesCorrelationConfig(facies_count={self.facies_count}, "
            f"threshold={self.boundary_threshold}, window={self.dilation_window})"
        )
