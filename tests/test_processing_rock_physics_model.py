"""Tests for RockPhysicsModel in src.processing.rock_physics.model."""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.io.grid import GridSpec
from src.processing.rock_physics.model import RockPhysicsModel
from src.utils.quantity import Quantity


class TestRockPhysicsModelInitialization:
    """Test RockPhysicsModel initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with minimal parameters."""
        grid_spec = GridSpec((10, 10, 10), dz=1.0, dt=0.001)

        vp = np.ones((10, 10, 10), dtype=np.float32) * 3000.0
        model = RockPhysicsModel(
            vp=vp, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )

        assert model.vp is vp
        assert model.vs is None
        assert model.rho is None
        assert model.facies is None
        assert model.grid_spec is grid_spec

    def test_initialization_with_all_properties(self):
        """Test initialization with all properties."""
        grid_spec = GridSpec((5, 5, 5), dz=1.0, dt=0.001)

        vp = np.ones((5, 5, 5), dtype=np.float32) * 3000.0
        vs = np.ones((5, 5, 5), dtype=np.float32) * 1500.0
        rho = np.ones((5, 5, 5), dtype=np.float32) * 2300.0
        facies = np.ones((5, 5, 5), dtype=np.int32)

        model = RockPhysicsModel(
            vp=vp, vs=vs, rho=rho, facies=facies, grid_spec=grid_spec
        )

        assert np.array_equal(model.vp, vp)
        assert np.array_equal(model.vs, vs)
        assert np.array_equal(model.rho, rho)
        assert np.array_equal(model.facies, facies)

    def test_initialization_with_disk_cache(self):
        """Test initialization with disk cache."""
        grid_spec = GridSpec((5, 5, 5), dz=1.0, dt=0.001)
        disk_cache = Mock()

        model = RockPhysicsModel(
            vp=None,
            vs=None,
            rho=None,
            facies=None,
            grid_spec=grid_spec,
            disk_cache=disk_cache,
        )

        assert model.disk_cache is disk_cache

    def test_post_init_creates_cache(self):
        """Test that __post_init__ creates cache."""
        grid_spec = GridSpec((5, 5, 5), dz=1.0, dt=0.001)

        model = RockPhysicsModel(
            vp=None, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )

        assert model._cache is not None

    def test_post_init_with_disk_cache(self):
        """Test that __post_init__ uses provided disk cache."""
        grid_spec = GridSpec((5, 5, 5), dz=1.0, dt=0.001)
        disk_cache = Mock()

        model = RockPhysicsModel(
            vp=None,
            vs=None,
            rho=None,
            facies=None,
            grid_spec=grid_spec,
            disk_cache=disk_cache,
        )

        # Cache should be created with disk_cache
        assert model._cache is not None


class TestRockPhysicsModelFromProps:
    """Test RockPhysicsModel.from_props() class method."""

    def test_from_props_with_all_properties(self):
        """Test from_props with all properties."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        props = {
            "vp": np.ones((3, 3, 3), dtype=np.float32) * 3000.0,
            "vs": np.ones((3, 3, 3), dtype=np.float32) * 1500.0,
            "rho": np.ones((3, 3, 3), dtype=np.float32) * 2300.0,
            "facies": np.ones((3, 3, 3), dtype=np.int32),
        }

        model = RockPhysicsModel.from_props(props, grid_spec)

        assert model.vp is not None
        assert isinstance(model.vp, Quantity)
        assert model.vs is not None
        assert isinstance(model.vs, Quantity)
        assert model.rho is not None
        assert isinstance(model.rho, Quantity)
        assert model.facies is not None

    def test_from_props_with_partial_properties(self):
        """Test from_props with only vp."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        props = {
            "vp": np.ones((3, 3, 3), dtype=np.float32) * 3000.0,
        }

        model = RockPhysicsModel.from_props(props, grid_spec)

        assert model.vp is not None
        assert isinstance(model.vp, Quantity)
        assert model.vs is None
        assert model.rho is None
        assert model.facies is None

    def test_from_props_creates_copies(self):
        """Test that from_props creates copies of input arrays."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        vp_original = np.ones((3, 3, 3), dtype=np.float32) * 3000.0
        props = {"vp": vp_original}

        model = RockPhysicsModel.from_props(props, grid_spec)

        # Modify original array
        vp_original[0, 0, 0] = 9999.0

        # Model's vp should not be affected (it's a copy)
        assert model.vp.array[0, 0, 0] == 3000.0

    def test_from_props_empty_dict(self):
        """Test from_props with empty properties dict."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)
        props = {}

        model = RockPhysicsModel.from_props(props, grid_spec)

        assert model.vp is None
        assert model.vs is None
        assert model.rho is None
        assert model.facies is None


class TestRockPhysicsModelEnsureUnits:
    """Test RockPhysicsModel.ensure_units() method."""

    def test_ensure_units_converts_vp_to_quantity(self):
        """Test ensure_units converts vp array to Quantity."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)
        vp = np.ones((3, 3, 3), dtype=np.float32) * 3000.0

        model = RockPhysicsModel(
            vp=vp, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )
        model.ensure_units()

        assert isinstance(model.vp, Quantity)

    def test_ensure_units_converts_vs_to_quantity(self):
        """Test ensure_units converts vs array to Quantity."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)
        vs = np.ones((3, 3, 3), dtype=np.float32) * 1500.0

        model = RockPhysicsModel(
            vp=None, vs=vs, rho=None, facies=None, grid_spec=grid_spec
        )
        model.ensure_units()

        assert isinstance(model.vs, Quantity)

    def test_ensure_units_converts_rho_to_quantity(self):
        """Test ensure_units converts rho array to Quantity."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)
        rho = np.ones((3, 3, 3), dtype=np.float32) * 2300.0

        model = RockPhysicsModel(
            vp=None, vs=None, rho=rho, facies=None, grid_spec=grid_spec
        )
        model.ensure_units()

        assert isinstance(model.rho, Quantity)

    def test_ensure_units_already_quantity(self):
        """Test ensure_units handles already-Quantity properties."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)
        vp = Quantity(np.ones((3, 3, 3), dtype=np.float32) * 3000.0, "m/s")

        model = RockPhysicsModel(
            vp=vp, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )
        model.ensure_units()

        assert isinstance(model.vp, Quantity)

    def test_ensure_units_invalidates_cache(self):
        """Test ensure_units invalidates the cache."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)
        vp = np.ones((3, 3, 3), dtype=np.float32) * 3000.0

        model = RockPhysicsModel(
            vp=vp, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )

        # Mock the invalidate method
        model._cache.invalidate = Mock()

        model.ensure_units()

        model._cache.invalidate.assert_called_once()

    def test_ensure_units_with_all_properties(self):
        """Test ensure_units with vp, vs, and rho."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)
        vp = np.ones((3, 3, 3), dtype=np.float32) * 3000.0
        vs = np.ones((3, 3, 3), dtype=np.float32) * 1500.0
        rho = np.ones((3, 3, 3), dtype=np.float32) * 2300.0

        model = RockPhysicsModel(
            vp=vp, vs=vs, rho=rho, facies=None, grid_spec=grid_spec
        )
        model.ensure_units()

        assert isinstance(model.vp, Quantity)
        assert isinstance(model.vs, Quantity)
        assert isinstance(model.rho, Quantity)

    def test_ensure_units_with_none_properties(self):
        """Test ensure_units skips None properties."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        model = RockPhysicsModel(
            vp=None, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )

        # Should not raise an error
        model.ensure_units()


class TestRockPhysicsModelInvalidateCache:
    """Test RockPhysicsModel.invalidate_cache() method."""

    def test_invalidate_cache_calls_cache_invalidate(self):
        """Test invalidate_cache delegates to _cache.invalidate()."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        model = RockPhysicsModel(
            vp=None, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )
        model._cache.invalidate = Mock()

        model.invalidate_cache()

        model._cache.invalidate.assert_called_once()

    def test_invalidate_cache_multiple_calls(self):
        """Test multiple invalidate_cache calls."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        model = RockPhysicsModel(
            vp=None, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )
        model._cache.invalidate = Mock()

        model.invalidate_cache()
        model.invalidate_cache()

        assert model._cache.invalidate.call_count == 2


class TestRockPhysicsModelToPropsDict:
    """Test RockPhysicsModel.to_props_dict() method."""

    def test_to_props_dict_with_all_properties(self):
        """Test to_props_dict with all properties."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        vp = np.ones((3, 3, 3), dtype=np.float32) * 3000.0
        vs = np.ones((3, 3, 3), dtype=np.float32) * 1500.0
        rho = np.ones((3, 3, 3), dtype=np.float32) * 2300.0
        facies = np.ones((3, 3, 3), dtype=np.int32)

        model = RockPhysicsModel(
            vp=vp, vs=vs, rho=rho, facies=facies, grid_spec=grid_spec
        )
        props_dict = model.to_props_dict()

        assert "vp" in props_dict
        assert "vs" in props_dict
        assert "rho" in props_dict
        assert "facies" in props_dict

    def test_to_props_dict_with_quantity_properties(self):
        """Test to_props_dict extracts arrays from Quantity properties."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        vp = Quantity(np.ones((3, 3, 3), dtype=np.float32) * 3000.0, "m/s")
        vs = Quantity(np.ones((3, 3, 3), dtype=np.float32) * 1500.0, "m/s")

        model = RockPhysicsModel(
            vp=vp, vs=vs, rho=None, facies=None, grid_spec=grid_spec
        )
        props_dict = model.to_props_dict()

        # Should extract the underlying arrays
        assert "vp" in props_dict
        assert "vs" in props_dict
        assert isinstance(props_dict["vp"], np.ndarray)
        assert isinstance(props_dict["vs"], np.ndarray)

    def test_to_props_dict_with_partial_properties(self):
        """Test to_props_dict with only some properties."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        vp = np.ones((3, 3, 3), dtype=np.float32) * 3000.0

        model = RockPhysicsModel(
            vp=vp, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )
        props_dict = model.to_props_dict()

        assert "vp" in props_dict
        assert "vs" not in props_dict
        assert "rho" not in props_dict
        assert "facies" not in props_dict

    def test_to_props_dict_with_no_properties(self):
        """Test to_props_dict with no properties."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        model = RockPhysicsModel(
            vp=None, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )
        props_dict = model.to_props_dict()

        assert len(props_dict) == 0
        assert isinstance(props_dict, dict)

    def test_to_props_dict_values_are_arrays(self):
        """Test to_props_dict returns numpy arrays."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        vp = np.ones((3, 3, 3), dtype=np.float32) * 3000.0
        facies = np.ones((3, 3, 3), dtype=np.int32)

        model = RockPhysicsModel(
            vp=vp, vs=None, rho=None, facies=facies, grid_spec=grid_spec
        )
        props_dict = model.to_props_dict()

        assert isinstance(props_dict["vp"], np.ndarray)
        assert isinstance(props_dict["facies"], np.ndarray)


class TestRockPhysicsModelIntegration:
    """Integration tests for RockPhysicsModel."""

    def test_from_props_and_ensure_units_workflow(self):
        """Test typical workflow: from_props -> ensure_units."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        props = {
            "vp": np.ones((3, 3, 3), dtype=np.float32) * 3000.0,
            "vs": np.ones((3, 3, 3), dtype=np.float32) * 1500.0,
            "rho": np.ones((3, 3, 3), dtype=np.float32) * 2300.0,
        }

        model = RockPhysicsModel.from_props(props, grid_spec)
        model.ensure_units()

        assert isinstance(model.vp, Quantity)
        assert isinstance(model.vs, Quantity)
        assert isinstance(model.rho, Quantity)

    def test_from_props_to_props_dict_roundtrip(self):
        """Test roundtrip: from_props -> to_props_dict."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)

        original_props = {
            "vp": np.ones((3, 3, 3), dtype=np.float32) * 3000.0,
            "vs": np.ones((3, 3, 3), dtype=np.float32) * 1500.0,
        }

        model = RockPhysicsModel.from_props(original_props, grid_spec)
        exported_props = model.to_props_dict()

        # Check that properties are preserved
        assert "vp" in exported_props
        assert "vs" in exported_props
        np.testing.assert_array_almost_equal(exported_props["vp"], original_props["vp"])
        np.testing.assert_array_almost_equal(exported_props["vs"], original_props["vs"])

    def test_cache_invalidation_workflow(self):
        """Test cache invalidation workflow."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)
        vp = np.ones((3, 3, 3), dtype=np.float32) * 3000.0

        model = RockPhysicsModel(
            vp=vp, vs=None, rho=None, facies=None, grid_spec=grid_spec
        )

        # Mock cache invalidation
        model._cache.invalidate = Mock()

        # Ensure units should invalidate
        model.ensure_units()
        assert model._cache.invalidate.called

        # Reset mock
        model._cache.invalidate.reset_mock()

        # Explicit invalidate should also work
        model.invalidate_cache()
        model._cache.invalidate.assert_called_once()

    def test_model_with_disk_cache_initialization(self):
        """Test model initialization with disk cache."""
        grid_spec = GridSpec((3, 3, 3), dz=1.0, dt=0.001)
        disk_cache = Mock()

        vp = np.ones((3, 3, 3), dtype=np.float32) * 3000.0

        model = RockPhysicsModel(
            vp=vp,
            vs=None,
            rho=None,
            facies=None,
            grid_spec=grid_spec,
            disk_cache=disk_cache,
        )

        assert model.disk_cache is disk_cache
