"""Rock physics analyzer pipeline and helpers.

Provides the :class:`RockPhysicsAnalyzer` pipeline used to compute AVO
and rock-physics attributes, analyze discrimination against facies and
persist results. This module keeps imports local where needed to avoid
heavy circular imports.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from src.analysis.processors.types import (
    AttributeArrayDict,
    FloatingArray,
    IntegerArray,
)
from src.analysis.rock_physics.computers import (
    AVOAttributesComputer,
    FluidFactorComputer,
    LambdaMuRhoComputer,
)
from src.analysis.rock_physics.config import RockPhysicsAnalysisConfig
from src.analysis.rock_physics.constants import RockPhysicsConstants
from src.analysis.rock_physics.discrimination import (
    AttributeDiscriminationAnalyzer,
    DiscriminationResult,
)
from src.core.analyzers import CompositeMixin, PipelineAnalyzer
from src.io.grid import GridSpec
from src.io.loader import DatasetManager

logger = logging.getLogger(__name__)

# This module may perform some imports in local scopes in order to avoid
# circular imports with heavy plotting or builder modules. These deferred
# imports are intentional — suppress import-order warnings with a short
# justification so pylint focuses on real issues.
# Many functions in this module are pipeline wiring and accept several
# arguments; allow the complexity checks locally for readability.


# Local generics not required in this module


@dataclass
class RockPhysicsPipelineResult:
    """Result container returned by :meth:`RockPhysicsAnalyzer.analyze`.

    Attributes
    ----------
    attributes: dict[str, FloatingArray]
        Mapping of attribute names to arrays.
    discrimination: dict[str, DiscriminationResult]
        Mapping of attribute names to discrimination results.
    output_path: str | None
        Path to written NPZ file when running in pipeline mode.
    """

    attributes: dict[str, FloatingArray]
    discrimination: dict[str, DiscriminationResult]
    output_path: str | None = None


PipelineContext = dict[str, Any]


class RockPhysicsAnalyzer(
    CompositeMixin,
    PipelineAnalyzer[RockPhysicsAnalysisConfig, RockPhysicsPipelineResult],
):
    """Analyzer that builds and runs the rock-physics computation pipeline.

    This class orchestrates loading datasets, computing AVO and
    Lambda/Mu/Rho attributes, optional fluid-factor computation, running
    discrimination analysis and persisting results when used in
    pipeline mode.
    """

    def __init__(self, config: RockPhysicsAnalysisConfig | None = None) -> None:
        super().__init__(
            config=config or RockPhysicsAnalysisConfig(), name="rock_physics"
        )
        self._avo_computer: AVOAttributesComputer | None = None
        self._lambda_mu_computer: LambdaMuRhoComputer | None = None
        self._fluid_computer: FluidFactorComputer | None = None
        self._discrimination_analyzer: AttributeDiscriminationAnalyzer | None = None
        self._last_result: RockPhysicsPipelineResult | None = None

    def _validate_config(self) -> None:
        if not isinstance(self._config, RockPhysicsAnalysisConfig):
            raise ValueError("Config must be a RockPhysicsAnalysisConfig instance")

    def _setup(self) -> None:
        self._avo_computer = AVOAttributesComputer()
        self._lambda_mu_computer = LambdaMuRhoComputer()
        self._fluid_computer = FluidFactorComputer()
        self._discrimination_analyzer = AttributeDiscriminationAnalyzer()

        self.add_sub_analyzer("avo_computer", self._avo_computer)
        self.add_sub_analyzer("lambda_mu_computer", self._lambda_mu_computer)
        self.add_sub_analyzer("fluid_computer", self._fluid_computer)
        self.add_sub_analyzer("discrimination_analyzer", self._discrimination_analyzer)

        super()._setup()

    def _create_pipeline(self) -> list[tuple[str, Any]]:
        return [
            ("prepare_context", self._stage_prepare_context),
            ("configure_logging", self._stage_configure_logging),
            ("load_dataset", self._stage_load_dataset_if_needed),
            ("compute_attributes", self._stage_compute_attributes),
            ("consolidate_results", self._stage_consolidate_results),
            ("analyze_discrimination", self._stage_analyze_discrimination),
            ("persist_results", self._stage_persist_results),
            ("generate_plots", self._stage_generate_plots),
            ("finalize", self._stage_finalize),
        ]

    def analyze(self, data: Any) -> RockPhysicsPipelineResult:
        """Run analyzer pipeline with given input mapping and return result.

        Parameters
        ----------
        data: Any
            Mapping of input parameters for the pipeline (or None).
        """
        context: PipelineContext
        if data is None:
            context = {}
        elif isinstance(data, dict):
            context = dict(cast(PipelineContext, data))
        else:
            raise TypeError(
                "RockPhysicsAnalyzer.analyze expects a mapping of input parameters"
            )

        context.setdefault("mode", "analysis")
        result = self._execute_pipeline(context)
        self._last_result = result
        return result

    def run(
        self,
        *,
        cache_dir: str = ".cache",
        generate_plots: bool = True,
        save_npz_only: bool = False,
        angles_list: Sequence[float] | None = None,
        verbose: bool = False,
    ) -> bool | str:
        """Run analyzer in pipeline mode and optionally persist results.

        Returns either the output path (when written) or boolean indicating
        whether attributes were produced.
        """
        payload: PipelineContext = {
            "mode": "pipeline",
            "cache_dir": cache_dir,
            "generate_plots": generate_plots,
            "save_npz_only": save_npz_only,
            "verbose": verbose,
        }
        if angles_list is not None:
            payload["angles_deg"] = tuple(angles_list)

        tmp: Any = self.execute(payload)
        result = cast(RockPhysicsPipelineResult, tmp)
        output_path = result.output_path
        if output_path:
            return output_path

        attributes = result.attributes
        return bool(attributes)

    @property
    def config(self) -> RockPhysicsAnalysisConfig:
        return cast(RockPhysicsAnalysisConfig, self._config)

    @classmethod
    def from_builder(cls, builder_func: Any | None = None) -> RockPhysicsAnalyzer:
        """Construct an analyzer using the project builder helper.

        When `builder_func` is omitted the default builder is used.
        """
        from src.analysis.builder import (
            build_rock_physics_analyzer,
        )

        if builder_func is None:
            return build_rock_physics_analyzer()

        return cast(RockPhysicsAnalyzer, builder_func())

    def _ensure_initialized(self) -> None:
        """Initialize the analyzer if not already initialized.

        This is a small helper used by public compute methods to lazily
        initialize dependent sub-analyzers.
        """
        if not self.is_initialized:
            self.initialize()

    def _stage_prepare_context(self, context: PipelineContext) -> PipelineContext:
        """Prepare and normalize pipeline context with defaults from config.

        Ensures grid configuration, angles, and other runtime defaults are set.
        """
        cfg = self.config

        data_path_default, file_map_default, grid_spec_default = (
            self._get_grid_configuration()
        )

        context.setdefault("cache_dir", cfg.cache_dir)
        context.setdefault("data_path", data_path_default)
        context.setdefault("file_map", file_map_default)

        grid_spec = context.setdefault("grid_spec", grid_spec_default)

        if "grid_shape" not in context:
            shape = getattr(grid_spec, "shape", None)
            if isinstance(shape, (tuple, list)):
                context["grid_shape"] = tuple(cast(Sequence[int], shape))
            else:
                context["grid_shape"] = tuple(cast(Sequence[int], cfg.grid_shape))

        if "dz" not in context:
            dz_value = getattr(grid_spec, "dz", None)
            context["dz"] = dz_value if isinstance(dz_value, (int, float)) else cfg.dz

        if "dt" not in context:
            dt_value = getattr(grid_spec, "dt", None)
            context["dt"] = dt_value if isinstance(dt_value, (int, float)) else cfg.dt

        context.setdefault("angles_deg", cfg.angles_sequence())
        context.setdefault("fluid_factor_k", cfg.fluid_factor_k)
        context.setdefault("generate_plots", cfg.generate_plots)
        context.setdefault("save_npz_only", cfg.save_npz_only)
        context.setdefault("verbose", cfg.verbose)
        context.setdefault("mode", context.get("mode", "analysis"))
        return context

    def _get_grid_configuration(self) -> tuple[str, dict[str, str], GridSpec]:
        cfg = self.config
        grid_spec = GridSpec(cfg.grid_shape, dz=cfg.dz, dt=cfg.dt)
        return cfg.data_path, cfg.file_mapping(), grid_spec

    def _stage_configure_logging(self, context: PipelineContext) -> PipelineContext:
        if context.get("verbose"):
            logging.basicConfig(
                level=logging.DEBUG, format="[%(levelname)s] %(message)s"
            )
        return context

    def _stage_load_dataset_if_needed(
        self, context: PipelineContext
    ) -> PipelineContext:
        """Ensure vp/vs/rho arrays are present in the pipeline context.

        If missing, a DatasetManager is created and properties are loaded
        and unwrapped into the context.
        """
        if all(context.get(key) is not None for key in ("vp", "vs", "rho")):
            return context

        grid_spec = context.get("grid_spec")
        if grid_spec is None:
            grid_shape = tuple(context.get("grid_shape", self.config.grid_shape))
            dz = float(context.get("dz", self.config.dz))
            dt = float(context.get("dt", self.config.dt))
            grid_shape_t = cast(tuple[int, int, int], grid_shape)
            grid_spec = GridSpec(grid_shape_t, dz=dz, dt=dt)
            context["grid_spec"] = grid_spec

        data_path = str(context["data_path"])
        file_map = dict(context["file_map"])
        dm = self._load_dataset_manager(data_path, file_map, grid_spec)
        vp, vs, rho, facies = self._load_and_unwrap_properties(dm)

        if vp is None or vs is None or rho is None:
            raise ValueError("Dataset manager did not provide required properties")

        context["dataset_manager"] = dm
        context["vp"] = vp
        context["vs"] = vs
        context["rho"] = rho
        if facies is not None:
            context.setdefault("facies", facies)
        return context

    def _stage_compute_attributes(self, context: PipelineContext) -> PipelineContext:
        """Compute AVO, lambda/mu/rho and optional fluid-factor attributes.

        The computed attribute mappings are placed into the context.
        """
        vp = context.get("vp")
        vs = context.get("vs")
        rho = context.get("rho")
        if vp is None or vs is None or rho is None:
            raise ValueError("Rock physics computation requires vp, vs, and rho arrays")

        angles = tuple(float(a) for a in context.get("angles_deg", ()))
        if not angles:
            raise ValueError("angles_deg must contain at least one angle")

        fluid_factor_k = float(
            context.get("fluid_factor_k", self.config.fluid_factor_k)
        )

        avo_results, lam_mu_rho, fluid = self._compute_all_attributes(
            cast(FloatingArray, vp),
            cast(FloatingArray, vs),
            cast(FloatingArray, rho),
            angles,
            fluid_factor_k,
        )
        context["avo_results"] = avo_results
        context["lambda_mu_results"] = lam_mu_rho
        context["fluid_factor"] = fluid
        return context

    def _stage_consolidate_results(self, context: PipelineContext) -> PipelineContext:
        """Consolidate individual computation results into attribute mapping.

        Validates required keys and returns a flat mapping of attributes.
        """
        attribute_results = self._build_attribute_results(
            cast(dict[str, FloatingArray], context["avo_results"]),
            cast(dict[str, FloatingArray], context["lambda_mu_results"]),
            cast(FloatingArray | None, context.get("fluid_factor")),
        )
        context["attribute_results"] = attribute_results
        return context

    def _stage_analyze_discrimination(
        self, context: PipelineContext
    ) -> PipelineContext:
        """Run discrimination analysis against facies when available.

        Stores an empty mapping on failure to keep pipeline non-fatal.
        """
        facies = context.get("facies")
        if facies is not None:
            try:
                context["discrimination"] = self.compare_all_attributes(
                    cast(AttributeArrayDict, context["attribute_results"]),
                    cast(IntegerArray, facies),
                )
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                logger.warning("Discrimination analysis failed (non-fatal): %s", exc)
                context["discrimination"] = {}
        else:
            context["discrimination"] = {}
        return context

    def _stage_persist_results(self, context: PipelineContext) -> PipelineContext:
        """Persist computed attributes and discrimination to a compressed NPZ.

        Only runs when pipeline `mode` equals "pipeline".
        """
        if context.get("mode") != "pipeline":
            return context

        try:
            output_path = self._persist_results(
                cache_dir=str(context["cache_dir"]),
                attribute_results=cast(
                    dict[str, FloatingArray], context["attribute_results"]
                ),
                discrimination=cast(
                    dict[str, DiscriminationResult], context.get("discrimination", {})
                ),
            )
        except (OSError, ValueError) as exc:
            logger.exception("Failed saving rock physics cache: %s", exc)
            output_path = None
        context["output_path"] = output_path
        return context

    def _stage_generate_plots(self, context: PipelineContext) -> PipelineContext:
        if context.get("mode") != "pipeline":
            return context
        if not context.get("generate_plots", True) or context.get(
            "save_npz_only", False
        ):
            return context
        try:
            from src.plotting import (
                RockPhysicsPlotter,
            )

            RockPhysicsPlotter()
            logger.debug("Rock physics plotter instantiated")
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            logger.exception("Rock physics plotting failed: %s", exc)
        return context

    def _stage_finalize(self, context: PipelineContext) -> RockPhysicsPipelineResult:
        result = RockPhysicsPipelineResult(
            attributes=cast(dict[str, FloatingArray], context["attribute_results"]),
            discrimination=cast(
                dict[str, DiscriminationResult], context.get("discrimination", {})
            ),
            output_path=cast(str | None, context.get("output_path")),
        )
        self._last_result = result
        return result

    def compute_avo_attributes(
        self,
        vp: FloatingArray,
        vs: FloatingArray,
        rho: FloatingArray,
        angles_deg: Sequence[float] | None = None,
    ) -> dict[str, FloatingArray]:
        """Compute AVO attributes using the configured AVOAttributesComputer.

        Returns a mapping of attribute name to array.
        """
        self._ensure_initialized()
        assert self._avo_computer is not None
        if angles_deg is None:
            angles_deg = self.config.angles_sequence()
        return self._avo_computer.compute(vp, vs, rho, angles_deg=angles_deg)

    def compute_lambda_mu_rho(
        self, vp: FloatingArray, vs: FloatingArray, rho: FloatingArray
    ) -> dict[str, FloatingArray]:
        """Compute lambda/mu/rho derived attributes.

        Returns a mapping of attribute name to array.
        """
        self._ensure_initialized()
        assert self._lambda_mu_computer is not None
        return self._lambda_mu_computer.compute(vp, vs, rho)

    def compute_fluid_factor(
        self,
        lambda_rho: FloatingArray,
        mu_rho: FloatingArray,
        k: float | None = None,
    ) -> FloatingArray:
        """Compute the fluid-factor attribute from lambda_rho and mu_rho.

        The optional `k` parameter overrides config fluid_factor_k.
        """
        self._ensure_initialized()
        assert self._fluid_computer is not None
        if k is None:
            k = self.config.fluid_factor_k
        return self._fluid_computer.compute(lambda_rho, mu_rho, k=k)

    def analyze_attribute_discrimination(
        self, attribute: FloatingArray, facies: IntegerArray, name: str = "Attribute"
    ) -> DiscriminationResult:
        """Analyze discrimination for a single attribute against facies.

        Returns a :class:`DiscriminationResult`.
        """
        self._ensure_initialized()
        assert self._discrimination_analyzer is not None
        return self._discrimination_analyzer.analyze_single(
            attribute, facies, name=name
        )

    def compare_all_attributes(
        self, attribute_results: AttributeArrayDict, facies: IntegerArray
    ) -> dict[str, DiscriminationResult]:
        """Analyze discrimination for multiple attributes and return results.

        The returned mapping contains per-attribute discrimination results.
        """
        self._ensure_initialized()
        assert self._discrimination_analyzer is not None
        return self._discrimination_analyzer.analyze_multiple(attribute_results, facies)

    def _load_dataset_manager(
        self, data_path: str, file_map: dict[str, str], grid_spec: GridSpec
    ) -> DatasetManager:
        return DatasetManager.from_stanfordsix(data_path, file_map, grid_spec)

    def _build_attribute_results(
        self,
        avo_results: dict[str, FloatingArray],
        lam_mu_rho: dict[str, FloatingArray],
        fluid: FloatingArray | None,
    ) -> dict[str, FloatingArray]:
        missing_avo = RockPhysicsConstants.AVO_KEYS - set(avo_results.keys())
        if missing_avo:
            raise ValueError(f"AVO results missing expected keys: {missing_avo}")

        missing_lmr = RockPhysicsConstants.LAMBDA_MU_KEYS - set(lam_mu_rho.keys())
        if missing_lmr:
            raise ValueError(
                f"Lambda-Mu-Rho results missing expected keys: {missing_lmr}"
            )

        results: dict[str, FloatingArray] = {
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

        logger.debug(
            "Consolidated %d attributes: %s", len(results), set(results.keys())
        )
        return results

    def _load_and_unwrap_properties(self, dm: DatasetManager) -> tuple[
        FloatingArray | None,
        FloatingArray | None,
        FloatingArray | None,
        IntegerArray | None,
    ]:
        def unwrap(value: Any) -> Any | None:
            if value is None:
                return None
            if hasattr(value, "array"):
                return cast(Any | None, getattr(value, "array"))
            return cast(Any | None, value)

        vp = unwrap(dm.vp)
        vs = unwrap(dm.vs)
        rho = unwrap(dm.rho)
        facies = unwrap(dm.facies)
        return vp, vs, rho, facies

    def _compute_all_attributes(
        self,
        vp: FloatingArray,
        vs: FloatingArray,
        rho: FloatingArray,
        angles_deg: Sequence[float] | None = None,
        fluid_factor_k: float | None = None,
    ) -> tuple[
        dict[str, FloatingArray],
        dict[str, FloatingArray],
        FloatingArray | None,
    ]:
        """Compute all derived attributes (AVO, lambda/mu/rho and fluid).

        This method orchestrates the lower-level compute calls and handles
        optional fluid-factor computation. Kept intentionally as a single
        entrypoint to simplify the pipeline wiring.
        """
        if angles_deg is None:
            angles_deg = self.config.angles_sequence()
        if fluid_factor_k is None:
            fluid_factor_k = self.config.fluid_factor_k

        logger.info("Computing AVO attributes...")
        avo_results = self.compute_avo_attributes(vp, vs, rho, angles_deg)

        logger.info("Computing Lambda-Mu-Rho attributes...")
        lam_mu_rho = self.compute_lambda_mu_rho(vp, vs, rho)

        logger.info("Computing fluid factor...")
        fluid = None
        expected_avo_keys = RockPhysicsConstants.AVO_KEYS
        has_required_avo = expected_avo_keys.issubset(avo_results.keys())
        has_required_lambda = {"lambda_rho", "mu_rho"}.issubset(lam_mu_rho.keys())

        if has_required_avo and has_required_lambda:
            try:
                fluid = self.compute_fluid_factor(
                    lam_mu_rho["lambda_rho"],
                    lam_mu_rho["mu_rho"],
                    k=fluid_factor_k,
                )
            except KeyError:
                logger.warning(
                    "Cannot compute fluid factor: lambda_rho or mu_rho not available"
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.warning("Failed computing fluid factor (non-fatal): %s", exc)
        else:
            logger.debug(
                "Skipping fluid factor computation; required keys missing "
                "(avo_ok=%s, lambda_ok=%s)",
                has_required_avo,
                has_required_lambda,
            )

        return avo_results, lam_mu_rho, fluid

    def _persist_results(
        self,
        *,
        cache_dir: str,
        attribute_results: dict[str, FloatingArray],
        discrimination: dict[str, DiscriminationResult],
    ) -> str | None:
        os.makedirs(cache_dir, exist_ok=True)
        out_fn = os.path.join(cache_dir, RockPhysicsConstants.OUTPUT_FILENAME)

        save_kwargs: dict[str, Any] = dict(attribute_results)
        save_kwargs["discrimination"] = np.array([discrimination], dtype=object)
        np.savez_compressed(out_fn, **save_kwargs)
        logger.info("Saved rock physics attributes to %s", out_fn)
        return out_fn
