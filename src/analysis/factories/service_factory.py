"""Service Factory Hierarchy for centralized service creation.

This module provides a factory hierarchy using the Factory Method pattern
for creating and configuring domain services. Services created through
factories have consistent initialization and configuration.

Patterns Used:
  - Factory Method: Each factory creates specific service types
  - Service Locator: ServiceLocator provides central access to factories
  - Dependency Injection: Services configured with dependencies

Example:
    >>> factory = ServiceLocator.get_cache_factory()
    >>> cache_mgr = factory.create_cache_manager(".cache")
    >>> resampler = ServiceLocator.create_resampler()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, TYPE_CHECKING, cast
from pathlib import Path

from src.analysis.cache.loader import CacheLoader
from src.analysis.types.protocols import CacheLoaderProtocol

if TYPE_CHECKING:
    from src.io.loader import DatasetManager

logger = logging.getLogger(__name__)

__all__ = [
    "ServiceFactory",
    "CacheServiceFactory",
    "ProcessorServiceFactory",
    "ComputerServiceFactory",
    "ServiceLocator",
]


class ServiceFactory(ABC):
    """Base factory for service creation.

    Provides abstract interface for service factory implementations.
    All service factories should inherit from this class to ensure
    consistent factory contract.
    """

    @abstractmethod
    def create(self, **kwargs: Any) -> Any:
        """Create a service with the given parameters.

        Parameters
        ----------
        **kwargs : Any
            Service-specific configuration parameters.

        Returns
        -------
        Any
            Created service instance.

        Raises
        ------
        TypeError
            If required parameters are missing.
        ValueError
            If parameters are invalid.
        """
        pass


class CacheServiceFactory(ServiceFactory):
    """Factory for cache-related services.

    Creates and configures cache loaders and related services
    with consistent initialization and default values.

    Example:
        >>> factory = CacheServiceFactory()
        >>> cache_loader = factory.create_cache_loader(dataset_mgr)
    """

    DEFAULT_CACHE_DIR: str = ".cache"

    def create(self, service_type: str = "loader", **kwargs: Any) -> Any:
        """Create a cache service.

        Parameters
        ----------
        service_type : str
            Type of service: 'loader'
        **kwargs : Any
            Service-specific parameters.

        Returns
        -------
        Any
            Created cache service.

        Raises
        ------
        ValueError
            If service_type is unknown.
        """
        if service_type == "loader":
            return self.create_cache_loader(**kwargs)
        else:
            raise ValueError(f"Unknown cache service type: {service_type}")

    @staticmethod
    def create_cache_loader(
        dm: DatasetManager,
    ) -> CacheLoaderProtocol:
        """Create a cache loader service.

        Parameters
        ----------
        dm : DatasetManager
            DatasetManager for loading cached data.

        Returns
        -------
        CacheLoaderProtocol
            Cache loader implementation.
        """
        logger.debug("Creating CacheLoader")

        # Build a selector that prefers an explicit cache_dir but falls back
        # to a default cache directory under the provided DatasetManager.
        def _selector(cache_dir: str, domain: str) -> str | None:
            try:
                if cache_dir:
                    return CacheLoader.default_selector(cache_dir, domain)
                # Fallback to dataset manager's data path + default cache dir
                base = Path(dm.data_path) / CacheServiceFactory.DEFAULT_CACHE_DIR
                return CacheLoader.default_selector(str(base), domain)
            except Exception:
                logger.exception(
                    "Selector wrapper failed; delegating to default selector"
                )
                return CacheLoader.default_selector(".", domain)

        # Create loader with the selector wrapper. Cast to the protocol to
        # satisfy the static checker (CacheLoader implements the protocol
        # at runtime but signatures are slightly more permissive).
        return cast(CacheLoaderProtocol, CacheLoader(selector=_selector))


class ProcessorServiceFactory(ServiceFactory):
    """Factory for processor-related services.

    Creates processors, resamplers, analysis processors, and other
    processing services with consistent initialization.

    Supports both data processors (resampler, synthesizer) and
    analysis processors (boundary detection, correlation calculation, etc).

    Example:
        >>> factory = ProcessorServiceFactory()
        >>> resampler = factory.create_resampler()
        >>> boundary_detector = factory.create_boundary_detector()
        >>> synthesizer = factory.create_synthesizer()
    """

    def create(self, service_type: str = "resampler", **kwargs: Any) -> Any:
        """Create a processor service.

        Parameters
        ----------
        service_type : str
            Type of service: 'resampler', 'synthesizer', 'boundary_detector',
            'cube_aligner', 'boundary_amp_extractor', 'gradient_calculator',
            'interface_analyzer', 'facies_discriminator'
        **kwargs : Any
            Service-specific parameters.

        Returns
        -------
        Any
            Created processor service.

        Raises
        ------
        ValueError
            If service_type is unknown.
        """
        if service_type == "resampler":
            return self.create_resampler(**kwargs)
        elif service_type == "synthesizer":
            return self.create_synthesizer(**kwargs)
        elif service_type == "boundary_detector":
            return self.create_boundary_detector()
        elif service_type == "cube_aligner":
            return self.create_cube_aligner()
        elif service_type == "boundary_amp_extractor":
            return self.create_boundary_amp_extractor(**kwargs)
        elif service_type == "gradient_calculator":
            return self.create_gradient_calculator()
        elif service_type == "interface_analyzer":
            return self.create_interface_analyzer()
        elif service_type == "facies_discriminator":
            return self.create_facies_discriminator()
        else:
            raise ValueError(f"Unknown processor service type: {service_type}")

    @staticmethod
    def create_resampler(grid_spec: Any | None = None) -> Any:
        """Create a resampler service.

        Parameters
        ----------
        grid_spec : GridSpec, optional
            Grid specification for resampling. If None, resampler must be
            configured before use.

        Returns
        -------
        Any
            Resampler implementation (DepthTimeResampler or similar).

        Raises
        ------
        ImportError
            If resampler factory module cannot be imported.
        """
        if grid_spec is not None:
            from src.processing.resampling._resampler import DepthTimeResampler

            logger.debug(f"Creating DepthTimeResampler with grid_spec: {grid_spec}")
            # DepthTimeResampler requires GridSpec at construction time
            return DepthTimeResampler(grid_spec)
        else:
            # Use factory if available
            try:
                from src.processing.resampling._resampler import resampler_factory

                logger.debug("Creating resampler via factory")
                # Prefer passing grid_spec when available; otherwise attempt
                # to call the factory without parameters. Use an Any-cast
                # to allow flexibility across different factory implementations.
                if grid_spec is not None:
                    return resampler_factory.get_resampler(grid_spec)
                return cast(Any, resampler_factory).get_resampler()
            except (ImportError, AttributeError):
                # Fallback: create empty resampler
                from src.processing.resampling._resampler import DepthTimeResampler

                logger.debug("Creating DepthTimeResampler (factory not available)")
                if grid_spec is not None:
                    return DepthTimeResampler(grid_spec)
                # Cannot create a DepthTimeResampler without a GridSpec
                raise RuntimeError(
                    "GridSpec is required to construct DepthTimeResampler when no resampler factory is available"
                )

    @staticmethod
    def create_synthesizer() -> Any:
        """Create an AVO synthesizer service.

        Returns
        -------
        Any
            AVOSynthesizer implementation.
        """
        from src.modeling.modeling import AVOSynthesizer

        logger.debug("Creating AVOSynthesizer")
        return AVOSynthesizer()

    @staticmethod
    def create_boundary_detector() -> Any:
        """Create a boundary detector processor.

        Returns
        -------
        Any
            BoundaryDetector implementation.
        """
        from src.analysis.processors import BoundaryDetector

        logger.debug("Creating BoundaryDetector")
        return BoundaryDetector()

    @staticmethod
    def create_cube_aligner() -> Any:
        """Create a cube aligner processor.

        Returns
        -------
        Any
            CubeAligner implementation.
        """
        from src.analysis.processors import CubeAligner

        logger.debug("Creating CubeAligner")
        return CubeAligner()

    @staticmethod
    def create_boundary_amp_extractor(dilation_window: int = 2) -> Any:
        """Create a boundary amplitude extractor processor.

        Parameters
        ----------
        dilation_window : int
            Dilation window for boundary zone expansion. Default is 2.

        Returns
        -------
        Any
            BoundaryAmplitudeExtractor implementation.
        """
        from src.analysis.processors import BoundaryAmplitudeExtractor

        logger.debug(
            f"Creating BoundaryAmplitudeExtractor with window={dilation_window}"
        )
        return BoundaryAmplitudeExtractor(dilation_window=dilation_window)

    @staticmethod
    def create_gradient_calculator() -> Any:
        """Create a gradient correlation calculator processor.

        Returns
        -------
        Any
            GradientCorrelationCalculator implementation.
        """
        from src.analysis.processors import GradientCorrelationCalculator

        logger.debug("Creating GradientCorrelationCalculator")
        return GradientCorrelationCalculator()

    @staticmethod
    def create_interface_analyzer() -> Any:
        """Create an interface reflection analyzer processor.

        Returns
        -------
        Any
            InterfaceReflectionAnalyzer implementation.
        """
        from src.analysis.processors import InterfaceReflectionAnalyzer

        logger.debug("Creating InterfaceReflectionAnalyzer")
        return InterfaceReflectionAnalyzer()

    @staticmethod
    def create_facies_discriminator() -> Any:
        """Create a facies discrimination calculator processor.

        Returns
        -------
        Any
            FaciesDiscriminationCalculator implementation.
        """
        from src.analysis.processors import FaciesDiscriminationCalculator

        logger.debug("Creating FaciesDiscriminationCalculator")
        return FaciesDiscriminationCalculator()


class ComputerServiceFactory(ServiceFactory):
    """Factory for computer/calculator services.

    Creates rock physics computers and attribute calculators
    with consistent initialization.

    Example:
        >>> factory = ComputerServiceFactory()
        >>> avo_computer = factory.create_avo_computer()
        >>> lambda_mu_computer = factory.create_lambda_mu_computer()
    """

    def create(self, service_type: str = "avo", **kwargs: Any) -> Any:
        """Create a computer service.

        Parameters
        ----------
        service_type : str
            Type of service: 'avo', 'lambda_mu', 'fluid_factor'
        **kwargs : Any
            Service-specific parameters.

        Returns
        -------
        Any
            Created computer service.

        Raises
        ------
        ValueError
            If service_type is unknown.
        """
        if service_type == "avo":
            return self.create_avo_computer(**kwargs)
        elif service_type == "lambda_mu":
            return self.create_lambda_mu_computer(**kwargs)
        elif service_type == "fluid_factor":
            return self.create_fluid_factor_computer(**kwargs)
        else:
            raise ValueError(f"Unknown computer service type: {service_type}")

    @staticmethod
    def create_avo_computer() -> Any:
        """Create an AVO attributes computer.

        Returns
        -------
        Any
            AVOAttributesComputer implementation.
        """
        from src.analysis.rock_physics.computers import AVOAttributesComputer

        logger.debug("Creating AVOAttributesComputer")
        return AVOAttributesComputer()

    @staticmethod
    def create_lambda_mu_computer() -> Any:
        """Create a Lamé parameters computer.

        Returns
        -------
        Any
            LambdaMuRhoComputer implementation.
        """
        from src.analysis.rock_physics.computers import LambdaMuRhoComputer

        logger.debug("Creating LambdaMuRhoComputer")
        return LambdaMuRhoComputer()

    @staticmethod
    def create_fluid_factor_computer() -> Any:
        """Create a fluid factor computer.

        Returns
        -------
        Any
            FluidFactorComputer implementation.
        """
        from src.analysis.rock_physics.computers import FluidFactorComputer

        logger.debug("Creating FluidFactorComputer")
        return FluidFactorComputer()


class ServiceLocator:
    """Service Locator providing centralized access to service factories.

    Implements the Service Locator pattern to provide a single point
    of access to all service factories. This simplifies dependency
    injection and centralizes service creation logic.

    Note:
        This is a namespace class and should not be instantiated.

    Example:
        >>> cache_mgr = ServiceLocator.create_cache_manager()
        >>> resampler = ServiceLocator.create_resampler()
        >>> avo_computer = ServiceLocator.create_avo_computer()
    """

    _factories: ClassVar[dict[str, ServiceFactory]] = {
        "cache": CacheServiceFactory(),
        "processor": ProcessorServiceFactory(),
        "computer": ComputerServiceFactory(),
    }

    def __init__(self) -> None:
        """Prevent instantiation of service locator.

        Raises
        ------
        TypeError
            Always raised to prevent instantiation.
        """
        raise TypeError(
            f"{self.__class__.__name__} is a namespace class and cannot be instantiated. "
            "Use its static methods directly."
        )

    @classmethod
    def get_cache_factory(cls) -> CacheServiceFactory:
        """Get the cache service factory.

        Returns
        -------
        CacheServiceFactory
            Factory for cache services.
        """
        return cast(CacheServiceFactory, cls._factories["cache"])

    @classmethod
    def get_processor_factory(cls) -> ProcessorServiceFactory:
        """Get the processor service factory.

        Returns
        -------
        ProcessorServiceFactory
            Factory for processor services.
        """
        return cast(ProcessorServiceFactory, cls._factories["processor"])

    @classmethod
    def get_computer_factory(cls) -> ComputerServiceFactory:
        """Get the computer service factory.

        Returns
        -------
        ComputerServiceFactory
            Factory for computer services.
        """
        return cast(ComputerServiceFactory, cls._factories["computer"])

    @classmethod
    def create_cache_loader(cls, dm: DatasetManager) -> CacheLoaderProtocol:
        """Create a cache loader service.

        Parameters
        ----------
        dm : DatasetManager
            Dataset manager instance.

        Returns
        -------
        CacheLoaderProtocol
            Cache loader implementation.
        """
        factory = cls.get_cache_factory()
        return factory.create_cache_loader(dm)

    @classmethod
    def create_resampler(cls) -> Any:
        """Create a resampler service.

        Returns
        -------
        Any
            Resampler implementation.
        """
        factory = cls.get_processor_factory()
        return factory.create_resampler()

    @classmethod
    def create_synthesizer(cls) -> Any:
        """Create a synthesizer service.

        Returns
        -------
        Any
            Synthesizer implementation.
        """
        factory = cls.get_processor_factory()
        return factory.create_synthesizer()

    @classmethod
    def create_boundary_detector(cls) -> Any:
        """Create a boundary detector processor.

        Returns
        -------
        Any
            BoundaryDetector implementation.
        """
        factory = cls.get_processor_factory()
        return factory.create_boundary_detector()

    @classmethod
    def create_cube_aligner(cls) -> Any:
        """Create a cube aligner processor.

        Returns
        -------
        Any
            CubeAligner implementation.
        """
        factory = cls.get_processor_factory()
        return factory.create_cube_aligner()

    @classmethod
    def create_boundary_amp_extractor(cls, dilation_window: int = 2) -> Any:
        """Create a boundary amplitude extractor processor.

        Parameters
        ----------
        dilation_window : int
            Dilation window for boundary zone expansion. Default is 2.

        Returns
        -------
        Any
            BoundaryAmplitudeExtractor implementation.
        """
        factory = cls.get_processor_factory()
        return factory.create_boundary_amp_extractor(dilation_window=dilation_window)

    @classmethod
    def create_gradient_calculator(cls) -> Any:
        """Create a gradient correlation calculator processor.

        Returns
        -------
        Any
            GradientCorrelationCalculator implementation.
        """
        factory = cls.get_processor_factory()
        return factory.create_gradient_calculator()

    @classmethod
    def create_interface_analyzer(cls) -> Any:
        """Create an interface reflection analyzer processor.

        Returns
        -------
        Any
            InterfaceReflectionAnalyzer implementation.
        """
        factory = cls.get_processor_factory()
        return factory.create_interface_analyzer()

    @classmethod
    def create_facies_discriminator(cls) -> Any:
        """Create a facies discrimination calculator processor.

        Returns
        -------
        Any
            FaciesDiscriminationCalculator implementation.
        """
        factory = cls.get_processor_factory()
        return factory.create_facies_discriminator()

    @classmethod
    def create_avo_computer(cls) -> Any:
        """Create an AVO computer service.

        Returns
        -------
        Any
            AVO computer implementation.
        """
        factory = cls.get_computer_factory()
        return factory.create_avo_computer()

    @classmethod
    def create_lambda_mu_computer(cls) -> Any:
        """Create a Lamé parameters computer.

        Returns
        -------
        Any
            Lamé parameters computer implementation.
        """
        factory = cls.get_computer_factory()
        return factory.create_lambda_mu_computer()

    @classmethod
    def create_fluid_factor_computer(cls) -> Any:
        """Create a fluid factor computer.

        Returns
        -------
        Any
            Fluid factor computer implementation.
        """
        factory = cls.get_computer_factory()
        return factory.create_fluid_factor_computer()
