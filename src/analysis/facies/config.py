"""Configuration for facies correlation analysis implementing AnalysisConfig interface.

This module provides the FaciesAnalysisConfig class that implements the unified
AnalysisConfig interface from src.analysis.core. It groups tunable parameters
for the facies correlation analysis pipeline and provides validation and
serialization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from src.analysis.base import AnalysisConfig


@dataclass
class FaciesAnalysisConfig(AnalysisConfig):
    """Configuration for facies correlation analysis pipeline.

    Implements the unified AnalysisConfig interface to enable polymorphic
    treatment of analyzer-specific configurations. Parameters control various
    aspects of the facies correlation analysis pipeline including boundary
    detection, amplitude extraction, and discrimination calculations.

    Attributes
    ----------
    cache_dir : str
        Directory containing precomputed cache files (default: ".cache").
        Used for AVO cache, resampling plans, and intermediate artifacts.

    facies_count : int
        Number of facies classes expected in the analysis (default: 4).
        Used for facies discrimination and separation matrix dimensions.

    boundary_threshold : float
        Threshold for boundary detection (default: 0.1).
        Controls sensitivity of facies boundary detection (0.0 < x < 1.0).

    dilation_window : int
        Window size for boundary dilation operations (default: 2).
        Controls spatial extent of boundary amplitude extraction.

    Examples
    --------
    Create default configuration:

    >>> config = FaciesAnalysisConfig()
    >>> print(config)
    FaciesAnalysisConfig(cache_dir='.cache', facies_count=4, ...)

    Create custom configuration:

    >>> config = FaciesAnalysisConfig(
    ...     cache_dir="/data/cache",
    ...     facies_count=6,
    ...     boundary_threshold=0.15,
    ...     dilation_window=3
    ... )

    Serialize to dictionary:

    >>> config_dict = config.to_dict()
    >>> print(config_dict)
    {'cache_dir': '.cache', 'facies_count': 4, ...}
    """

    cache_dir: str = ".cache"
    facies_count: int = 4
    boundary_threshold: float = 0.1
    dilation_window: int = 2

    def __post_init__(self) -> None:
        """Validate configuration parameters on instantiation.

        Raises
        ------
        ValueError
            If any parameter is outside valid range.
        """
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate all configuration parameters.

        Raises
        ------
        ValueError
            If facies_count < 1, boundary_threshold not in (0, 1), or
            dilation_window < 1.
        """
        if not self.cache_dir:
            raise ValueError("cache_dir must not be empty")

        if self.facies_count < 1:
            raise ValueError("facies_count must be at least 1")

        if not (0.0 < self.boundary_threshold < 1.0):
            raise ValueError("boundary_threshold must be between 0 and 1 (exclusive)")

        if self.dilation_window < 1:
            raise ValueError("dilation_window must be at least 1")

    def validate_inputs(self, **kwargs) -> bool:
        """Validate that required input parameters are provided.

        Implements AnalysisConfig abstract method. For facies analysis,
        typically validates that cache_dir exists or can be created.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments (unused, for interface compatibility).

        Returns
        -------
        bool
            True if all validations pass (cache_dir path is valid).

        Raises
        ------
        ValueError
            If validation fails.
        """
        from pathlib import Path

        cache_path = Path(self.cache_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return True

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns
        -------
        bool
            True if all parameters pass validation, False otherwise.
        """
        try:
            self._validate_params()
            return True
        except ValueError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Implements AnalysisConfig abstract method for serialization.

        Returns
        -------
        dict
            Dictionary representation with keys: cache_dir, facies_count,
            boundary_threshold, dilation_window.
        """
        return {
            "cache_dir": self.cache_dir,
            "facies_count": self.facies_count,
            "boundary_threshold": self.boundary_threshold,
            "dilation_window": self.dilation_window,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FaciesAnalysisConfig":
        """Create configuration from dictionary.

        Provides factory method for deserialization from dict representation.

        Parameters
        ----------
        config_dict
            Dictionary with optional keys: cache_dir, facies_count,
            boundary_threshold, dilation_window. Missing keys use defaults.

        Returns
        -------
        FaciesAnalysisConfig
            Configuration instance with values from dict or defaults.

        Raises
        ------
        ValueError
            If any provided value is invalid type or outside valid range.
        """
        return cls(
            cache_dir=str(config_dict.get("cache_dir", ".cache")),
            facies_count=int(config_dict.get("facies_count", 4)),
            boundary_threshold=float(config_dict.get("boundary_threshold", 0.1)),
            dilation_window=int(config_dict.get("dilation_window", 2)),
        )

    def __str__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            Human-readable string showing all configuration parameters.
        """
        return (
            f"FaciesAnalysisConfig("
            f"cache_dir={self.cache_dir!r}, "
            f"facies_count={self.facies_count}, "
            f"boundary_threshold={self.boundary_threshold}, "
            f"dilation_window={self.dilation_window})"
        )

    def __repr__(self) -> str:
        """Return detailed representation for debugging.

        Returns
        -------
        str
            Detailed representation showing type and all attributes.
        """
        return (
            f"{self.__class__.__name__}("
            f"cache_dir={self.cache_dir!r}, "
            f"facies_count={self.facies_count}, "
            f"boundary_threshold={self.boundary_threshold}, "
            f"dilation_window={self.dilation_window})"
        )
