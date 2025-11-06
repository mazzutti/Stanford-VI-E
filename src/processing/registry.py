"""Service registry for dependency injection and singleton management.

Centralizes creation and access to all processing services using a class-based
factory pattern instead of functional decorators.

This replaces the functional factory pattern in services.py with proper OOP design.

Usage:
    registry = ServiceRegistry()
    resampler = registry.resampler_service()
    hub = registry.manager_hub()
    
Or use the global singleton:
    from src.processing.registry import get_registry
    
    registry = get_registry()
    resampler = registry.resampler_service()
"""

from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from src.processing.resampling.service import ResamplerService
    from src.processing.managers import ManagerHub
    from src.processing.avo.validator import AVOValidator

__all__ = ["ServiceRegistry", "get_registry"]

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Centralized OOP service registry for dependency injection.
    
    Manages singleton creation and initialization of all processing services
    with proper dependency injection and clear initialization order.
    
    Benefits over functional approach:
    - Single source of truth for service configuration
    - Easier testing with dependency injection
    - Clear initialization order and dependencies
    - Cleaner API with method-based access
    - Better state management
    
    Attributes:
        _instances: Cache of singleton service instances
        _logger: Logger instance
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the registry.
        
        Args:
            logger: Optional logger instance
        """
        self._instances: Dict[str, Any] = {}
        self._logger = logger or logging.getLogger(self.__class__.__name__)
    
    def _get_or_create(self, key: str, factory: callable) -> Any:
        """Get cached instance or create new one via factory.
        
        Args:
            key: Service identifier
            factory: Callable that creates the service
            
        Returns:
            Service instance (singleton)
        """
        if key not in self._instances:
            self._instances[key] = factory()
            self._logger.debug(f"Created service: {key}")
        return self._instances[key]
    
    def resampler_service(self, grid_spec: Optional[Any] = None) -> "ResamplerService":
        """Get or create ResamplerService singleton.
        
        The ResamplerService wraps depth/time resampling with caching and metrics.
        
        Args:
            grid_spec: Optional GridSpec configuration. If None, uses default.
            
        Returns:
            ResamplerService instance
            
        Raises:
            ImportError: If required dependencies are not available
        """
        def factory():
            from src.processing.resampling.service import ResamplerService
            from src.io.grid import GridSpec
            
            grid = grid_spec or GridSpec((512, 512, 512))
            return ResamplerService(grid_spec=grid)
        
        return self._get_or_create("resampler_service", factory)
    
    def manager_hub(self) -> "ManagerHub":
        """Get or create ManagerHub singleton.
        
        The ManagerHub provides unified access to cache, file, and process managers.
        
        Returns:
            ManagerHub instance
            
        Raises:
            ImportError: If required dependencies are not available
        """
        def factory():
            from src.processing.managers import ManagerHub
            return ManagerHub()
        
        return self._get_or_create("manager_hub", factory)
    
    def avo_validator(self, max_angle: float = 30.0) -> "AVOValidator":
        """Get or create AVOValidator singleton.
        
        The AVOValidator performs AVO analysis and linearization checks.
        
        Args:
            max_angle: Maximum angle in degrees for AVO linearization check.
                      Default is 30.0.
            
        Returns:
            AVOValidator instance
            
        Raises:
            ImportError: If required dependencies are not available
        """
        def factory():
            from src.processing.avo.validator import AVOValidator
            return AVOValidator(max_angle=max_angle)
        
        return self._get_or_create("avo_validator", factory)
    
    def rock_physics_model(self):
        """Get RockPhysicsModel class (factory, not singleton).
        
        Returns the class itself for direct instantiation with custom parameters.
        
        Returns:
            RockPhysicsModel class
            
        Raises:
            ImportError: If required dependencies are not available
        """
        from src.processing.rock_physics.model import RockPhysicsModel
        return RockPhysicsModel
    
    def reset(self) -> None:
        """Clear all service instances (useful for testing).
        
        After calling this, subsequent calls will recreate services.
        """
        self._instances.clear()
        self._logger.info("Cleared all service instances")
    
    def summarize(self) -> None:
        """Print summary of registered services."""
        services = list(self._instances.keys())
        print(f"ServiceRegistry: {len(services)} service(s) registered")
        for service_name in sorted(services):
            service_type = type(self._instances[service_name]).__name__
            print(f"  • {service_name}: {service_type}")


# Global registry instance (lazy initialization)
_global_registry: Optional[ServiceRegistry] = None


def get_registry() -> ServiceRegistry:
    """Get the global service registry singleton.
    
    Lazily initializes the registry on first call.
    
    Returns:
        Global ServiceRegistry singleton
        
    Example:
        >>> registry = get_registry()
        >>> resampler = registry.resampler_service()
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _global_registry
    if _global_registry is not None:
        _global_registry.reset()
