"""Main orchestrator for rock physics analysis pipeline.

This module provides the RockPhysicsAnalyzer class that coordinates
all rock physics computations using composition with domain-specific
computer and analyzer classes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence, cast, TypeVar

import numpy as np
from src.io.grid import GridSpec
from src.io.loader import DatasetManager

from src.analysis.processors.types import (
    FloatingArray,
    IntegerArray,
    AttributeArrayDict,
)
from src.analysis.rock_physics.computers import (
    AVOAttributesComputer,
    FluidFactorComputer,
    LambdaMuRhoComputer,
    DEFAULT_AVO_ANGLES_DEG,
    DEFAULT_FLUID_FACTOR_K,
)
from src.analysis.rock_physics.discrimination import (
    AttributeDiscriminationAnalyzer,
    DiscriminationResult,
)

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")


class RockPhysicsConstants:
    """Central repository for rock physics computation constants and defaults.

    Consolidates all magic numbers, configuration values, and default parameters
    used throughout rock physics analysis into a single, well-documented class.
    This improves maintainability and makes tuning easier.

    Constants
    ---------
    EPSILON : float
        Small value for numerical stability in computations.
    SNR_EPSILON : float
        Small value for signal-to-noise ratio calculations.
    AVO_KEYS : frozenset
        Expected keys in AVO results dictionary.
    LAMBDA_MU_KEYS : frozenset
        Expected keys in Lamé parameter results.
    DISCRIMINATION_KEYS : frozenset
        Expected keys in discrimination analysis results.
    OUTPUT_FILENAME : str
        Default filename for saved rock physics attributes.

    Defaults
    --------
    DEFAULT_GRID_SHAPE : tuple
        Default grid dimensions for analysis.
    DEFAULT_DZ : float
        Default depth increment (in meters or units).
    DEFAULT_DT : float
        Default time increment (in seconds).
    DEFAULT_DATA_PATH : str
        Default path for data loading.
    DEFAULT_FILE_MAP : dict
        Default mapping of data fields to filenames.
    """

    # Numerical constants
    EPSILON: float = 1e-10
    SNR_EPSILON: float = 1e-10

    # Result key constants
    AVO_KEYS: frozenset[str] = frozenset(
        {"intercept", "gradient", "product", "scaled_gradient"}
    )
    LAMBDA_MU_KEYS: frozenset[str] = frozenset(
        {"lambda_rho", "mu_rho", "lambda_mu_ratio"}
    )
    DISCRIMINATION_KEYS: frozenset[str] = frozenset(
        {
            "name",
            "cohens_d",
            "pearson_r",
            "p_value",
            "snr",
            "mean_class0",
            "mean_class1",
            "std_class0",
            "std_class1",
        }
    )

    # Output configuration
    OUTPUT_FILENAME: str = "rock_physics_attributes.npz"

    # Default grid parameters
    DEFAULT_GRID_SHAPE: tuple[int, int, int] = (150, 200, 200)
    DEFAULT_DZ: float = 1.0
    DEFAULT_DT: float = 0.001

    # Default file handling
    DEFAULT_DATA_PATH: str = "."
    DEFAULT_FILE_MAP: dict[str, str] = {
        "vp": "P-wave Velocity",
        "vs": "S-wave Velocity",
        "rho": "Density",
        "facies": "Facies",
    }


class RockPhysicsAnalyzer:
    """Orchestrates rock physics attribute computation and analysis.

    This class uses composition with domain-specific analyzers to provide
    a unified interface for computing AVO attributes, Lamé parameters
    fluid factors, and discrimination analysis.
    """

    def __init__(self) -> None:
        """Initialize the analyzer with domain-specific computers."""
        self._avo_computer = AVOAttributesComputer()
        self._lambda_mu_computer = LambdaMuRhoComputer()
        self._fluid_computer = FluidFactorComputer()
        self._discrimination_analyzer = AttributeDiscriminationAnalyzer()

    @classmethod
    def from_builder(cls, builder_func: Optional[Any] = None) -> "RockPhysicsAnalyzer":
        """Create analyzer using fluent AnalysisBuilder pattern.

        This factory method enables fluent API construction of the analyzer
        with the AnalysisBuilder pattern for cleaner initialization code.

        Parameters
        ----------
        builder_func : Callable, optional
            Builder function to customize analyzer. If omitted, creates
            default RockPhysicsAnalyzer instance.

        Returns
        -------
        RockPhysicsAnalyzer
            Constructed analyzer instance

        Examples
        --------
        Using default construction::

            from src.analysis import build_rock_physics_analyzer
            analyzer = RockPhysicsAnalyzer.from_builder()

        Using custom builder::

            from src.analysis import AnalysisBuilder
            analyzer = RockPhysicsAnalyzer.from_builder(
                lambda: AnalysisBuilder()
                    .with_dependency("avo_computer", custom_computer)
                    .build())

        Or direct instantiation::

            analyzer = RockPhysicsAnalyzer()
        """
        from src.analysis.builder import build_rock_physics_analyzer

        if builder_func is None:
            return build_rock_physics_analyzer()

        return builder_func()

    def compute_avo_attributes(
        self,
        vp: FloatingArray,
        vs: FloatingArray,
        rho: FloatingArray,
        angles_deg: Sequence[float] = DEFAULT_AVO_ANGLES_DEG,
    ) -> Dict[str, FloatingArray]:
        """Compute AVO attributes (intercept, gradient, and derived volumes).

        Args:
            vp: P-wave velocity volume
            vs: S-wave velocity volume
            rho: Density volume

        Returns:
            Dict with keys: 'intercept', 'gradient', 'product', 'scaled_gradient'
        """
        return self._avo_computer.compute(vp, vs, rho, angles_deg=angles_deg)

    def compute_lambda_mu_rho(
        self, vp: FloatingArray, vs: FloatingArray, rho: FloatingArray
    ) -> Dict[str, FloatingArray]:
        """Compute Lamé parameters (lambda_rho, mu_rho, and their ratio).

        Args:
            vp: P-wave velocity volume
            vs: S-wave velocity volume
            rho: Density volume

        Returns:
            Dict with keys: 'lambda_rho', 'mu_rho', 'lambda_mu_ratio'
        """
        return self._lambda_mu_computer.compute(vp, vs, rho)

    def compute_fluid_factor(
        self,
        lambda_rho: FloatingArray,
        mu_rho: FloatingArray,
        k: float = DEFAULT_FLUID_FACTOR_K,
    ) -> FloatingArray:
        """Compute fluid-sensitive attribute (lambda_rho - k * mu_rho).

        Args:
            lambda_rho: Lambda-Rho attribute volume
            mu_rho: Mu-Rho attribute volume
            k: Tuning parameter (default 1.0 for clastic sequences)

        Returns:
            Fluid factor volume
        """
        return self._fluid_computer.compute(lambda_rho, mu_rho, k=k)

    def analyze_attribute_discrimination(
        self, attribute: FloatingArray, facies: IntegerArray, name: str = "Attribute"
    ) -> DiscriminationResult:
        """Analyze how well a single attribute discriminates facies classes.

        Args:
            attribute: Attribute values to analyze
            facies: Facies class labels
            name: Descriptive name for the attribute

        Returns:
            Dictionary with discrimination statistics (Cohen's d, correlation, SNR, etc.)
        """
        return self._discrimination_analyzer.analyze_single(
            attribute, facies, name=name
        )

    def compare_all_attributes(
        self, attribute_results: AttributeArrayDict, facies: IntegerArray
    ) -> Dict[str, DiscriminationResult]:
        """Analyze discrimination for multiple attributes simultaneously.

        Args:
            attribute_results: Dict mapping attribute names to value arrays
            facies: Facies class labels

        Returns:
            Dict mapping attribute names to their discrimination statistics
        """
        return self._discrimination_analyzer.analyze_multiple(attribute_results, facies)

    def _load_dataset_manager(
        self, data_path: str, file_map: Dict[str, str], grid_spec: GridSpec
    ) -> DatasetManager:
        """Load dataset using DatasetManagerFactory with fallback."""
        from src.analysis.types.base import DatasetManagerFactory

        try:
            # Type ignore: DatasetManagerFactory is a Protocol, but at runtime
            # it's used as a concrete implementation (mypy limitation)
            factory = cast(Any, DatasetManagerFactory())  # type: ignore[misc]
            return cast(DatasetManager, factory.create(data_path, file_map, grid_spec))
        except Exception as e:
            logger.debug(f"DatasetManagerFactory failed: {e}, using fallback")
            return DatasetManager.from_stanfordsix(data_path, file_map, grid_spec)

    def _build_attribute_results(
        self,
        avo_results: Dict[str, FloatingArray],
        lam_mu_rho: Dict[str, FloatingArray],
        fluid: Optional[FloatingArray],
    ) -> Dict[str, FloatingArray]:
        """Consolidate computed attributes into a single results dictionary."""
        # Validate AVO results
        missing_avo = RockPhysicsConstants.AVO_KEYS - set(avo_results.keys())
        if missing_avo:
            raise ValueError(f"AVO results missing expected keys: {missing_avo}")

        # Validate Lambda-Mu-Rho results
        missing_lmr = RockPhysicsConstants.LAMBDA_MU_KEYS - set(lam_mu_rho.keys())
        if missing_lmr:
            raise ValueError(
                f"Lambda-Mu-Rho results missing expected keys: {missing_lmr}"
            )

        results: Dict[str, FloatingArray] = {
            "intercept": avo_results["intercept"],
            "gradient": avo_results["gradient"],
            "product": avo_results["product"],
            "scaled_gradient": avo_results["scaled_gradient"],
            "lambda_rho": lam_mu_rho["lambda_rho"],
            "mu_rho": lam_mu_rho["mu_rho"],
            "lambda_mu_ratio": lam_mu_rho["lambda_mu_ratio"],
        }
        if fluid is not None:
            results["fluid_factor"] = fluid

        logger.debug(f"Consolidated {len(results)} attributes: {set(results.keys())}")
        return results

    def _load_and_unwrap_properties(self, dm: DatasetManager) -> tuple[
        FloatingArray | None,
        FloatingArray | None,
        FloatingArray | None,
        FloatingArray | None,
    ]:
        """Load and unwrap rock physics properties from dataset manager.

        Returns
        -------
        tuple
            Tuple of (vp, vs, rho, facies), each potentially None.
        """
        # Unwrap Quantity objects to numpy arrays if present
        vp = dm.vp.array if hasattr(dm.vp, "array") else dm.vp
        vs = dm.vs.array if hasattr(dm.vs, "array") else dm.vs
        rho = dm.rho.array if hasattr(dm.rho, "array") else dm.rho
        facies = dm.facies.array if hasattr(dm.facies, "array") else dm.facies
        return vp, vs, rho, facies

    def _get_grid_configuration(self) -> tuple[str, Dict[str, str], Any]:
        """Acquire grid configuration from plotting config or sensible defaults."""
        try:
            from src.plotting.helpers.config import PlotConfig

            # PlotConfig doesn't contain grid info, so use defaults
            # This method is legacy and should be refactored
            logger.debug("Using module-level grid configuration defaults.")
            from src.io.grid import GridSpec

            return (
                RockPhysicsConstants.DEFAULT_DATA_PATH,
                RockPhysicsConstants.DEFAULT_FILE_MAP.copy(),
                GridSpec(
                    RockPhysicsConstants.DEFAULT_GRID_SHAPE,
                    dz=RockPhysicsConstants.DEFAULT_DZ,
                    dt=RockPhysicsConstants.DEFAULT_DT,
                ),
            )
        except Exception as e:
            logger.debug(
                f"Failed to load grid configuration: {e}. "
                f"Using module-level defaults."
            )
            from src.io.grid import GridSpec

            return (
                RockPhysicsConstants.DEFAULT_DATA_PATH,
                RockPhysicsConstants.DEFAULT_FILE_MAP.copy(),
                GridSpec(
                    RockPhysicsConstants.DEFAULT_GRID_SHAPE,
                    dz=RockPhysicsConstants.DEFAULT_DZ,
                    dt=RockPhysicsConstants.DEFAULT_DT,
                ),
            )

    def _compute_all_attributes(
        self,
        vp: FloatingArray,
        vs: FloatingArray,
        rho: FloatingArray,
        angles_deg: Sequence[float],
    ) -> tuple[
        Dict[str, FloatingArray], Dict[str, FloatingArray], Optional[FloatingArray]
    ]:
        """Compute all rock physics attributes in sequence."""
        # ====== CRITICAL: AVO Attributes ======
        logger.info("Computing AVO attributes...")
        try:
            avo_results = self.compute_avo_attributes(
                vp, vs, rho, angles_deg=angles_deg
            )
            logger.debug(f"AVO computation completed: {set(avo_results.keys())}")
        except ValueError as e:
            logger.error(f"Input validation failed for AVO computation: {e}")
            raise
        except Exception:
            logger.exception("Failed computing AVO attributes")
            raise

        # ====== CRITICAL: Lambda-Mu-Rho Attributes ======
        logger.info("Computing Lambda-Mu-Rho attributes...")
        try:
            lam_mu_rho = self.compute_lambda_mu_rho(vp, vs, rho)
            logger.debug(
                f"Lambda-Mu-Rho computation completed: {set(lam_mu_rho.keys())}"
            )
        except Exception:
            logger.exception("Failed computing Lambda-Mu-Rho attributes")
            raise

        # ====== OPTIONAL: Fluid Factor (non-fatal failure) ======
        logger.info("Computing fluid factor...")
        fluid = None
        try:
            fluid = self.compute_fluid_factor(
                avo_results["lambda_rho"],
                avo_results["mu_rho"],
                k=DEFAULT_FLUID_FACTOR_K,
            )
            logger.debug("Fluid factor computation succeeded")
        except KeyError:
            logger.warning(
                "Cannot compute fluid factor: lambda_rho or mu_rho not in AVO results"
            )
        except Exception as e:
            logger.warning(f"Failed computing fluid factor (non-fatal): {e}")

        return avo_results, lam_mu_rho, fluid

    def run(
        self,
        *,
        cache_dir: str = ".cache",
        generate_plots: bool = True,
        save_npz_only: bool = False,
        angles_list: Optional[Sequence[float]] = None,
        verbose: bool = False,
    ) -> bool | str | None:
        """Programmatic entrypoint for the rock-physics pipeline."""
        if verbose:
            import logging as logging_module

            logging_module.basicConfig(
                level=logging_module.DEBUG, format="[%(levelname)s] %(message)s"
            )
        logger.info("Starting rock-physics analysis pipeline...")

        # Prepare angles for computation
        angles: Sequence[float] = (
            angles_list if angles_list is not None else DEFAULT_AVO_ANGLES_DEG
        )

        # Acquire grid configuration with fallback
        logger.info("Loading grid configuration...")
        DATA_PATH, FILE_MAP, grid_spec = self._get_grid_configuration()

        # Load properties from dataset manager
        logger.info(f"Loading dataset from {DATA_PATH}...")
        try:
            dm = self._load_dataset_manager(DATA_PATH, FILE_MAP, grid_spec)
        except Exception:
            logger.exception("Failed to load dataset manager")
            raise

        # Unwrap properties to numpy arrays
        logger.info("Extracting rock properties...")
        vp, vs, rho, facies = self._load_and_unwrap_properties(dm)

        # Compute all attributes
        logger.info("Computing rock physics attributes...")
        avo_results, lam_mu_rho, fluid = self._compute_all_attributes(
            vp, vs, rho, angles
        )

        # Consolidate results
        logger.info("Consolidating attribute results...")
        attribute_results = self._build_attribute_results(
            avo_results, lam_mu_rho, fluid
        )

        # Analyze discrimination
        logger.info("Analyzing attribute discrimination...")
        try:
            discrimination = self.compare_all_attributes(attribute_results, facies)
        except Exception:
            logger.exception("Attribute discrimination analysis failed")
            discrimination = {}

        # Save results to cache
        logger.info(f"Saving results to {cache_dir}...")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            out_fn = os.path.join(cache_dir, RockPhysicsConstants.OUTPUT_FILENAME)
            # Prepare keyword arguments for savez_compressed
            save_kwargs: Dict[str, Any] = {
                k: (v if v is not None else np.array([]))
                for k, v in attribute_results.items()
            }
            save_kwargs["discrimination"] = np.array(discrimination, dtype=object)
            np.savez_compressed(out_fn, **save_kwargs)
            logger.info(f"Saved rock physics attributes to {out_fn}")
        except Exception as e:
            logger.exception(f"Failed saving rock-physics cache: {e}")
            out_fn = None

        # Optionally trigger plotting
        if generate_plots and not save_npz_only:
            logger.info("Generating plots...")
            try:
                from src.plotting import RockPhysicsPlotter

                plotter = RockPhysicsPlotter()
                self._log_debug("Rock physics plotter instantiated")
                logger.info("Plot generation completed")
            except Exception:
                logger.exception("Rock physics plotting failed")

        logger.info("Pipeline completed successfully")
        return out_fn if out_fn is not None else True
