"""Simplified context management for analysis pipelines.

Provides clean, maintainable context preparation instead of long
conditional chains.
"""

from typing import Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

__all__ = ["ContextBuilder", "GridContext"]


@dataclass
class GridContext:
    """Simplified grid configuration."""

    shape: tuple[int, ...]
    dz: float
    dt: float

    @classmethod
    def from_spec(cls, spec: Any) -> "GridContext":
        """Create from grid specification object."""
        shape = tuple(getattr(spec, "shape", ()))
        dz = float(getattr(spec, "dz", 0.0))
        dt = float(getattr(spec, "dt", 0.0))
        return cls(shape=shape, dz=dz, dt=dt)

    @classmethod
    def from_config(cls, config: Any) -> "GridContext":
        """Create from configuration object."""
        shape = tuple(config.grid_shape)
        dz = float(config.dz)
        dt = float(config.dt)
        return cls(shape=shape, dz=dz, dt=dt)


class ContextBuilder:
    """Build analysis context with simple, clear steps.

    Replaces complex conditional logic with clean method calls.
    """

    def __init__(self, config: Any):
        """Initialize builder with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.context: dict[str, Any] = {}

    def with_defaults(self) -> "ContextBuilder":
        """Set default values from config."""
        self.context.update(
            {
                "cache_dir": self.config.cache_dir,
                "data_path": self.config.data_path,
                "file_map": self.config.file_mapping(),
                "generate_plots": self.config.generate_plots,
                "save_npz_only": self.config.save_npz_only,
                "verbose": self.config.verbose,
            }
        )
        return self

    def with_grid(self, grid_spec: Any | None = None) -> "ContextBuilder":
        """Set grid configuration."""
        if grid_spec:
            grid_ctx = GridContext.from_spec(grid_spec)
        else:
            grid_ctx = GridContext.from_config(self.config)

        self.context["grid_spec"] = grid_spec
        self.context["grid_shape"] = grid_ctx.shape
        self.context["dz"] = grid_ctx.dz
        self.context["dt"] = grid_ctx.dt
        return self

    def with_avo_params(self) -> "ContextBuilder":
        """Set AVO analysis parameters."""
        self.context["angles_deg"] = self.config.angles_sequence()
        self.context["fluid_factor_k"] = self.config.fluid_factor_k
        return self

    def with_mode(self, mode: str = "analysis") -> "ContextBuilder":
        """Set execution mode."""
        self.context["mode"] = mode
        return self

    def merge(self, updates: dict[str, Any]) -> "ContextBuilder":
        """Merge additional context values."""
        self.context.update(updates)
        return self

    def build(self) -> dict[str, Any]:
        """Build final context dictionary."""
        return self.context.copy()


class ContextValidator:
    """Validate context contains required fields."""

    @staticmethod
    def require_rock_properties(context: dict[str, Any]) -> None:
        """Ensure rock properties are present."""
        required = ["vp", "vs", "rho"]
        missing = [k for k in required if context.get(k) is None]

        if missing:
            raise ValueError(f"Missing required rock properties: {', '.join(missing)}")

    @staticmethod
    def require_avo_params(context: dict[str, Any]) -> None:
        """Ensure AVO parameters are present."""
        angles = context.get("angles_deg", ())
        if not angles:
            raise ValueError("angles_deg must contain at least one angle")

    @staticmethod
    def require_field(context: dict[str, Any], field: str) -> None:
        """Ensure specific field is present."""
        if field not in context:
            raise ValueError(f"Required field missing: {field}")
