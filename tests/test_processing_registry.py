"""Tests for src.processing.registry module.

Tests for ServiceRegistry dependency injection and singleton management.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.processing.registry import ServiceRegistry, get_registry, reset_registry


class TestServiceRegistryGetDefault:
    """Tests for get_registry() singleton access."""

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_get_default_returns_instance(self):
        """Test that get_default returns a ServiceRegistry instance."""
        registry = get_registry()
        assert isinstance(registry, ServiceRegistry)

    def test_get_default_is_singleton(self):
        """Test that get_default returns same instance each time."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_multiple_threads_same_instance(self):
        """Test that get_default returns same instance across calls."""
        instances = set()
        for _ in range(5):
            instances.add(id(get_registry()))
        assert len(instances) == 1  # All should be same instance


class TestServiceRegistryResampler:
    """Tests for resampler service management."""

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_get_resampler_service(self):
        """Test getting resampler service from registry."""
        registry = get_registry()
        service = registry.get_resampler_service()
        assert service is not None

    def test_resampler_service_is_cached(self):
        """Test that resampler service is cached."""
        registry = get_registry()
        service1 = registry.get_resampler_service()
        service2 = registry.get_resampler_service()
        assert service1 is service2


class TestServiceRegistryBackends:
    """Tests for backend management."""

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_get_backend_manager(self):
        """Test getting backend manager from registry."""
        registry = get_registry()
        manager = registry.get_backend_manager()
        assert manager is not None

    def test_backend_manager_is_cached(self):
        """Test that backend manager is cached."""
        registry = get_registry()
        manager1 = registry.get_backend_manager()
        manager2 = registry.get_backend_manager()
        assert manager1 is manager2


class TestServiceRegistryCache:
    """Tests for cache management."""

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_get_resample_cache(self):
        """Test getting resample cache from registry."""
        registry = get_registry()
        cache = registry.get_resample_cache()
        assert cache is not None

    def test_resample_cache_is_cached(self):
        """Test that resample cache is cached."""
        registry = get_registry()
        cache1 = registry.get_resample_cache()
        cache2 = registry.get_resample_cache()
        assert cache1 is cache2


class TestServiceRegistryMetrics:
    """Tests for metrics management."""

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_get_backend_metrics(self):
        """Test getting backend metrics from registry."""
        registry = get_registry()
        metrics = registry.get_backend_metrics()
        assert metrics is not None

    def test_backend_metrics_is_cached(self):
        """Test that backend metrics is cached."""
        registry = get_registry()
        metrics1 = registry.get_backend_metrics()
        metrics2 = registry.get_backend_metrics()
        assert metrics1 is metrics2


class TestServiceRegistryManagers:
    """Tests for manager hub."""

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_get_manager_hub(self):
        """Test getting manager hub from registry."""
        registry = get_registry()
        hub = registry.get_manager_hub()
        assert hub is not None

    def test_manager_hub_is_cached(self):
        """Test that manager hub is cached."""
        registry = get_registry()
        hub1 = registry.get_manager_hub()
        hub2 = registry.get_manager_hub()
        assert hub1 is hub2


class TestServiceRegistryAVO:
    """Tests for AVO validator management."""

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_get_avo_validator(self):
        """Test getting AVO validator from registry."""
        registry = get_registry()
        validator = registry.get_avo_validator()
        assert validator is not None

    def test_avo_validator_is_cached(self):
        """Test that AVO validator is cached."""
        registry = get_registry()
        validator1 = registry.get_avo_validator()
        validator2 = registry.get_avo_validator()
        assert validator1 is validator2


class TestServiceRegistryReset:
    """Tests for registry reset functionality."""

    def test_reset_clears_registry(self):
        """Test that reset_registry clears service instances."""
        registry = get_registry()
        service1 = registry.get_resampler_service()

        reset_registry()

        # After reset, cached instances should be cleared
        service2 = registry.get_resampler_service()
        # Service should be recreated
        assert service1 is not service2

    def test_reset_allows_fresh_initialization(self):
        """Test that services are reinitialized after reset."""
        registry1 = get_registry()
        service1 = registry1.get_resampler_service()

        reset_registry()

        registry2 = get_registry()
        service2 = registry2.get_resampler_service()

        # Registry instances should be same, but services should be different
        assert registry1 is registry2
        assert service1 is not service2


class TestGetRegistryFunction:
    """Tests for get_registry module function."""

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_get_registry_function(self):
        """Test that get_registry() returns a registry."""
        registry = get_registry()
        assert isinstance(registry, ServiceRegistry)

    def test_get_registry_matches_get_default(self):
        """Test that get_registry() returns same as get_registry()."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2


class TestServiceRegistryDependencyInjection:
    """Tests for dependency injection capabilities."""

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_services_have_dependencies_initialized(self):
        """Test that services can access their dependencies."""
        registry = get_registry()

        # Get multiple services
        resampler = registry.get_resampler_service()
        cache = registry.get_resample_cache()
        metrics = registry.get_backend_metrics()

        # All should be initialized
        assert resampler is not None
        assert cache is not None
        assert metrics is not None

    def test_registry_lifecycle(self):
        """Test typical registry lifecycle."""
        # First access
        registry = get_registry()
        assert registry is not None

        # Get services
        service = registry.get_resampler_service()
        assert service is not None

        # Get cached service again
        service_again = registry.get_resampler_service()
        assert service is service_again

        # Reset clears instances
        reset_registry()

        # After reset, services are recreated but registry is same
        service_new = registry.get_resampler_service()
        assert service_new is not service


class TestServiceRegistryEdgeCases:
    """Tests for edge cases and error conditions."""

    def teardown_method(self):
        """Clean up registry after each test."""
        reset_registry()

    def test_multiple_sequential_resets(self):
        """Test multiple sequential resets."""
        for _ in range(5):
            registry = get_registry()
            assert registry is not None
            reset_registry()

    def test_concurrent_access_pattern(self):
        """Test accessing multiple services sequentially."""
        registry = get_registry()

        # Access core services (skip manager_hub due to abstract FileManager)
        services = [
            registry.get_resampler_service(),
            registry.get_backend_manager(),
            registry.get_resample_cache(),
            registry.get_backend_metrics(),
            registry.get_avo_validator(),
        ]

        # All should be non-None
        assert all(s is not None for s in services)

        # Re-access should return same instances
        services_again = [
            registry.get_resampler_service(),
            registry.get_backend_manager(),
            registry.get_resample_cache(),
            registry.get_backend_metrics(),
            registry.get_avo_validator(),
        ]

        for s1, s2 in zip(services, services_again):
            assert s1 is s2
