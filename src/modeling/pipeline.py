"""High-level orchestration for the AVO modeling pipeline.

ModelingPipeline coordinates the complete workflow:
1. Dataset loading
2. Depth-to-time resampling
3. AVO synthesis with caching

This simplified version delegates to specialized services for each step.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.io.utilities import load_depth_properties
from src.modeling.config import ModelingConfig
from src.modeling.model_cache import CacheManager
from src.modeling.modeling import AVOSynthesizer, SynthesisConfig, unwrap_quantity
from src.modeling.resampler import ResamplingService
from src.utils.quantity import Quantity

logger = logging.getLogger(__name__)

# Note: imports used inside methods avoid heavy import-time side effects
# and circular imports. Prefer keeping those lazy imports; when needed we
# apply per-line pylint suppression rather than a module-level disable.

# Some imports are intentionally deferred inside methods to reduce import
# time and avoid circular dependencies. Suppress import-order warnings
# for this module so pylint focuses on actionable problems.

# The modeling pipeline orchestrator uses a compact public API and contains
# procedural methods that may use several local variables for orchestration.
# Silence the stylistic warnings that are expected for high-level pipeline
# orchestration functions.

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

    def run(self) -> dict[str, Any]:
        """Execute the complete modeling pipeline.

        Uses configuration from self.config, delegating to specialized services:
        - DatasetManager: loads data
        - ResamplingService: depth-to-time conversion
        - AVOSynthesizer: AVO synthesis
        - CacheManager: result caching

        Returns:
            Dictionary with 'avo_cached', 'angle_stacks', and 'full_stack' keys
        """
        # Lazy imports to avoid import-time side-effects
        from src.io.loader import (
            DatasetManager,
        )
        from src.processing.rock_physics.model import (
            RockPhysicsModel,
        )

        cfg = self.config.defaults
        syn_cfg = SynthesisConfig(
            use_quality_weighting=self.config.use_quality_weighting,
            add_noise=self.config.add_noise,
            snr_db=self.config.snr_db,
        )

        # Load dataset
        logger.info("Loading dataset from %s...", cfg.data_path)
        dm = DatasetManager.from_stanfordsix(cfg.data_path, cfg.file_map, cfg.grid_spec)
        props_depth: dict[str, NDArray[Any] | Quantity | None] = load_depth_properties(
            dm
        )

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

        # Convert time-domain seismograms back to depth domain
        logger.info("Converting seismograms from time to depth domain...")
        # Lazy imports for resampling conversion
        from src.processing.resampling._cache import (
            get_resample_plan_cache,
        )
        from src.processing.resampling._resampler import (
            resampler_factory,
        )

        resampler = resampler_factory.get_resampler(cfg.grid_spec)
        vp_depth = props_depth["vp"]
        # Ensure we pass a raw ndarray to the cache/resampler (unwrap Quantity safely)
        if vp_depth is None:
            raise ValueError("vp not available for resampling")
        vp_arr = unwrap_quantity(vp_depth)

        plan_cache = get_resample_plan_cache()
        plan = plan_cache.get_plan(cfg.grid_spec, vp_arr)

        # Convert full stack to depth
        full_stack_depth = resampler.time_to_depth_cube(full_stack, vp_arr, plan=plan)

        # Convert angle stacks to depth
        angle_stacks_depth: list[NDArray[np.floating[Any]] | Quantity] = []
        if angle_stacks:
            for angle_stack in angle_stacks:
                angle_stack_depth = resampler.time_to_depth_cube(
                    angle_stack, vp_arr, plan=plan
                )
                angle_stacks_depth.append(angle_stack_depth)

        # Save depth-domain seismograms to cache
        # Compute the same cache key as time-domain (since they're derived from the same inputs)
        key = self.cache_manager.compute_cache_key(
            unwrap_quantity(props_time["vp"]),
            unwrap_quantity(props_time["vs"]),
            unwrap_quantity(props_time["rho"]),
            list(cfg.angles),
            cfg.wavelet,
            use_quality_weighting=syn_cfg.use_quality_weighting,
            add_noise=syn_cfg.add_noise,
            snr_db=syn_cfg.snr_db,
            noise_seed=syn_cfg.noise_seed,
        )
        depth_filename = f"avo_depth_{key}.npz"
        self.cache_manager.save_avo_synthetics(
            depth_filename,
            (
                full_stack_depth
                if not isinstance(full_stack_depth, Quantity)
                else full_stack_depth.array
            ),
            (
                [s.array if isinstance(s, Quantity) else s for s in angle_stacks_depth]
                if angle_stacks_depth
                else None
            ),
        )
        logger.info("Saved depth-domain seismograms to cache: %s", depth_filename)

        # debug: interactive 3D view of the full-stack depth-domain volume
        # Lazy debug import used only for optional interactive plotting
        from src.debug import plot_volume

        # choose a sensible isosurface level (None uses median inside the plot function)
        plot_volume(
            unwrap_quantity(cast(NDArray[Any], full_stack)),
            cmap="seismic",
            show=True,
        )

        return {
            "avo_cached": True,
            "angle_stacks": angle_stacks,
            "full_stack": full_stack,
            "angle_stacks_depth": angle_stacks_depth,
            "full_stack_depth": full_stack_depth,
        }
