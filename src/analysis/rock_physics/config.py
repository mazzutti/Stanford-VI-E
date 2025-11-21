"""Configuration objects for rock physics analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.analysis.base import AnalysisConfig
from src.analysis.config_mixins import ValidatableConfigMixin
from src.analysis.rock_physics.computers import (
    DEFAULT_AVO_ANGLES_DEG,
    DEFAULT_FLUID_FACTOR_K,
)
from src.analysis.rock_physics.constants import RockPhysicsConstants
from src.analysis.validators_registry import ValidatorRegistry

# This configuration dataclass intentionally aggregates many tuning
# parameters; disable the instance-attribute lint for this data holder.


@dataclass
class RockPhysicsAnalysisConfig(AnalysisConfig, ValidatableConfigMixin):
    """Configuration for the rock physics analysis pipeline.

    Parameters
    ----------
    cache_dir
        Directory used to persist computed attribute volumes.
    data_path
        Root directory containing Stanford VI dataset artifacts.
    file_map
        Mapping between logical property names and on-disk filenames.
    grid_shape
        Dimensions of the working grid (inline, crossline, samples).
    dz
        Depth sampling increment used to construct the grid specification.
    dt
        Time sampling increment used to construct the grid specification.
    angles_deg
        Incidence angles (in degrees) used for AVO attribute computation.
    fluid_factor_k
        Scaling factor applied when computing the fluid factor volume.
    generate_plots
        Flag indicating whether plots should be generated after persistence.
    save_npz_only
        When True, skip plot generation even if ``generate_plots`` is True.
    verbose
        Enable verbose logging for the pipeline when True.
    """

    cache_dir: str = ".cache"
    data_path: str = RockPhysicsConstants.DEFAULT_DATA_PATH
    file_map: dict[str, str] = field(
        default_factory=RockPhysicsConstants.DEFAULT_FILE_MAP.copy
    )
    grid_shape: tuple[int, int, int] = RockPhysicsConstants.DEFAULT_GRID_SHAPE
    dz: float = RockPhysicsConstants.DEFAULT_DZ
    dt: float = RockPhysicsConstants.DEFAULT_DT
    angles_deg: tuple[float, ...] = DEFAULT_AVO_ANGLES_DEG
    fluid_factor_k: float = DEFAULT_FLUID_FACTOR_K
    generate_plots: bool = True
    save_npz_only: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate configuration immediately after instantiation."""
        self._validate_params()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_params(self) -> None:
        """Validate numeric and path parameters."""
        ValidatorRegistry.validate_non_empty(self.cache_dir, "cache_dir")
        ValidatorRegistry.validate_non_empty(self.data_path, "data_path")
        for idx, size in enumerate(self.grid_shape):
            ValidatorRegistry.validate_positive(size, f"grid_shape[{idx}]")
        ValidatorRegistry.validate_positive(self.dz, "dz")
        ValidatorRegistry.validate_positive(self.dt, "dt")
        ValidatorRegistry.validate_positive(self.fluid_factor_k, "fluid_factor_k")
        ValidatorRegistry.validate_positive(len(self.angles_deg), "angles_deg")

    def validate_inputs(self) -> bool:
        """Ensure configured paths exist or can be created."""
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        data_path = Path(self.data_path)
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        return True

    def with_updates(self, **overrides: Any) -> RockPhysicsAnalysisConfig:
        """Return a copy of the configuration with overrides applied."""
        data = self.to_dict()
        data.update(overrides)
        return self.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return {
            "cache_dir": self.cache_dir,
            "data_path": self.data_path,
            "file_map": dict(self.file_map),
            "grid_shape": tuple(self.grid_shape),
            "dz": self.dz,
            "dt": self.dt,
            "angles_deg": tuple(self.angles_deg),
            "fluid_factor_k": self.fluid_factor_k,
            "generate_plots": self.generate_plots,
            "save_npz_only": self.save_npz_only,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> RockPhysicsAnalysisConfig:
        """Create configuration from dictionary input."""
        return cls(
            cache_dir=str(config_dict.get("cache_dir", ".cache")),
            data_path=str(
                config_dict.get("data_path", RockPhysicsConstants.DEFAULT_DATA_PATH)
            ),
            file_map=dict(
                config_dict.get(
                    "file_map", RockPhysicsConstants.DEFAULT_FILE_MAP.copy()
                )
            ),
            grid_shape=tuple(
                config_dict.get("grid_shape", RockPhysicsConstants.DEFAULT_GRID_SHAPE)
            ),
            dz=float(config_dict.get("dz", RockPhysicsConstants.DEFAULT_DZ)),
            dt=float(config_dict.get("dt", RockPhysicsConstants.DEFAULT_DT)),
            angles_deg=tuple(config_dict.get("angles_deg", DEFAULT_AVO_ANGLES_DEG)),
            fluid_factor_k=float(
                config_dict.get("fluid_factor_k", DEFAULT_FLUID_FACTOR_K)
            ),
            generate_plots=bool(config_dict.get("generate_plots", True)),
            save_npz_only=bool(config_dict.get("save_npz_only", False)),
            verbose=bool(config_dict.get("verbose", False)),
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def angles_sequence(self) -> Sequence[float]:
        """Return incidence angles as a sequence of floats."""
        return tuple(self.angles_deg)

    def file_mapping(self) -> dict[str, str]:
        """Return a mutable copy of the file mapping."""
        return dict(self.file_map)
