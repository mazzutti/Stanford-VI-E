"""High-level orchestration for the AVO modeling pipeline.

ModelingPipeline coordinates the complete workflow:
1. Dataset loading
2. Depth-to-time resampling
3. AVO synthesis with caching

This simplified version delegates to specialized services for each step.
"""

from __future__ import annotations

from typing import Any, cast
import numpy as np
from numpy.typing import NDArray
import logging

from src.modeling.config import ModelingConfig
from src.modeling.modeling import AVOSynthesizer, SynthesisConfig
from src.modeling.model_cache import CacheManager
from src.io.utilities import load_depth_properties
from src.modeling.resampler import ResamplingService
from src.utils.quantity import Quantity

logger = logging.getLogger(__name__)

__all__ = ["ModelingPipeline"]


class ModelingPipeline:
    """Orchestrates the complete AVO modeling workflow.

    Simplified pipeline that delegates to specialized services:
    - ResamplingService: depth-to-time conversion
    - AVOSynthesizer: seismic synthesis
    - CacheManager: caching

    Uses ModelingDefaults for sensible defaults and ModelingConfig for customization.
    """

    def __init__(
        self,
        config: ModelingConfig | None = None,
        synthesizer: AVOSynthesizer | None = None,
        cache_manager: CacheManager | None = None,
    ):
        """Initialize the modeling pipeline.

        Args:
            config: ModelingConfig with all parameters; uses defaults if None
            synthesizer: Optional custom AVOSynthesizer
            cache_manager: Optional custom CacheManager
        """
        self.config = config or ModelingConfig()
        self.synthesizer = synthesizer or AVOSynthesizer()
        self.cache_manager = cache_manager or CacheManager(
            self.config.defaults.cache_dir
        )
        self.resampler = ResamplingService()

    def run(
        self,
    ) -> dict[
        str, bool | list[NDArray[np.floating[Any]]] | None | NDArray[np.floating[Any]]
    ]:
        """Execute the complete modeling pipeline.

        Uses configuration from self.config, delegating to specialized services:
        - DatasetManager: loads data
        - ResamplingService: depth-to-time conversion
        - AVOSynthesizer: AVO synthesis
        - CacheManager: result caching

        Returns:
            Dictionary with 'avo_cached', 'angle_stacks', and 'full_stack' keys
        """
        from src.io.loader import DatasetManager
        from src.processing.rock_physics.model import RockPhysicsModel

        cfg = self.config.defaults
        syn_cfg = SynthesisConfig(
            use_quality_weighting=self.config.use_quality_weighting,
            add_noise=self.config.add_noise,
            snr_db=self.config.snr_db,
        )

        # Load dataset
        logger.info("Loading dataset from %s...", cfg.data_path)
        dm = DatasetManager.from_stanfordsix(cfg.data_path, cfg.file_map, cfg.grid_spec)
        props_depth: dict[str, NDArray[np.floating[Any]] | None] = load_depth_properties(dm)
        rpm = RockPhysicsModel.from_props(props_depth, cfg.grid_spec)
        rpm.ensure_units()

        # Resample to time domain
        logger.info("Resampling to time domain...")
        props_dict: dict[str, NDArray[np.floating[Any]] | Quantity] = cast(
            dict[str, NDArray[np.floating[Any]] | Quantity], rpm.to_props_dict()
        )
        props_time = self.resampler.resample_to_time(props_dict, cfg.grid_spec)

        # Create synthetics with caching
        logger.info("Generating AVO synthetics...")

        def create_fn(
            props_unwrapped: dict[str, NDArray[np.floating[Any]]],
            angles_in: list[float],
            wavelet_in: NDArray[np.floating[Any]],
            config_in: SynthesisConfig | None,
        ) -> tuple[list[NDArray[np.floating[Any]]], NDArray[np.floating[Any]]]:
            return self.synthesizer.create_synthetics(
                cast(dict[str, NDArray[np.floating[Any]] | Quantity], props_unwrapped),
                angles_in,
                wavelet_in,
                config_in,
            )

        angle_stacks, full_stack = self.cache_manager.get_avo_synthetics(
            props_time,
            list(cfg.angles),
            cfg.wavelet,
            create_fn=create_fn,
            config=syn_cfg,
        )

        return {
            "avo_cached": True,
            "angle_stacks": angle_stacks,
            "full_stack": full_stack,
        }
