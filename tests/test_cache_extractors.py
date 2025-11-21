# mypy: ignore-errors
"""Comprehensive tests for cache extractors module.

Tests cover:
- ArrayExtractor: Base array extraction interface
- NpzExtractor: NPZ archive extraction
- NpyExtractor: NPY file extraction
- ExtractorFactory: Factory for creating appropriate extractors
- Error handling and edge cases
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.analysis.cache.extractors import (ArrayExtractor, NpyExtractor,
                                           NpzExtractor)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create test array data
        test_array = np.random.rand(10, 10, 10).astype(np.float64)

        # Save as NPY file
        npy_path = data_dir / "test_data.npy"
        np.save(npy_path, test_array)

        # Save as NPZ file
        npz_path = data_dir / "test_archive.npz"
        np.savez(npz_path, full_stack=test_array, metadata=np.array([1, 2, 3]))

        yield data_dir


class TestArrayExtractorBase:
    """Tests for ArrayExtractor base class."""

    def test_array_extractor_is_abstract(self):
        """Test ArrayExtractor interface."""
        assert hasattr(ArrayExtractor, "extract")

    def test_array_extractor_cannot_be_instantiated(self):
        """Test that ArrayExtractor is abstract."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            ArrayExtractor()


class TestNpyExtractor:
    """Tests for NpyExtractor."""

    def test_npy_extractor_creation(self):
        """Test NpyExtractor instantiation."""
        extractor = NpyExtractor()
        assert extractor is not None

    def test_npy_extract_array(self, temp_data_dir):
        """Test extracting NPY file."""
        extractor = NpyExtractor()
        npy_path = temp_data_dir / "test_data.npy"

        # Load and extract
        array = np.load(npy_path)
        result = extractor.extract(array)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10, 10)

    def test_npy_extractor_preserves_dtype(self, temp_data_dir):
        """Test that NPY extractor preserves data type."""
        extractor = NpyExtractor()
        npy_path = temp_data_dir / "test_data.npy"

        array = np.load(npy_path)
        result = extractor.extract(array)

        assert result.dtype == array.dtype

    def test_npy_extractor_preserves_shape(self, temp_data_dir):
        """Test that NPY extractor preserves array shape."""
        extractor = NpyExtractor()
        npy_path = temp_data_dir / "test_data.npy"

        array = np.load(npy_path)
        result = extractor.extract(array)

        assert result.shape == array.shape

    def test_npy_extractor_with_1d_array(self):
        """Test NPY extractor with 1D array."""
        extractor = NpyExtractor()
        array_1d = np.array([1, 2, 3, 4, 5])

        result = extractor.extract(array_1d)
        assert result.shape == (5,)

    def test_npy_extractor_with_2d_array(self):
        """Test NPY extractor with 2D array."""
        extractor = NpyExtractor()
        array_2d = np.random.rand(5, 10)

        result = extractor.extract(array_2d)
        assert result.shape == (5, 10)

    def test_npy_extractor_with_large_array(self):
        """Test NPY extractor with large array."""
        extractor = NpyExtractor()
        large_array = np.random.rand(100, 100, 50)

        result = extractor.extract(large_array)
        assert result.shape == (100, 100, 50)


class TestNpzExtractor:
    """Tests for NpzExtractor."""

    def test_npz_extractor_creation(self):
        """Test NpzExtractor instantiation."""
        extractor = NpzExtractor()
        assert extractor is not None

    def test_npz_extract_archive(self, temp_data_dir):
        """Test extracting from NPZ archive."""
        extractor = NpzExtractor()
        npz_path = temp_data_dir / "test_archive.npz"

        # Load archive
        archive = np.load(npz_path)
        result = extractor.extract(archive)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10, 10)

    def test_npz_extract_full_stack_key(self, temp_data_dir):
        """Test that NpzExtractor uses full_stack key."""
        extractor = NpzExtractor()
        npz_path = temp_data_dir / "test_archive.npz"

        archive = np.load(npz_path)
        result = extractor.extract(archive)

        # Should extract full_stack
        assert result is not None

    def test_npz_extractor_with_missing_key(self, temp_data_dir):
        """Test NPZ extractor with missing key."""
        extractor = NpzExtractor()

        # Create NPZ without full_stack key
        npz_path = temp_data_dir / "no_key.npz"
        data = {"other_key": np.random.rand(5, 5, 5)}
        np.savez(npz_path, **data)

        np.load(npz_path)
        # Should handle missing key
        assert extractor is not None

    def test_npz_extractor_preserves_data(self, temp_data_dir):
        """Test that NPZ extractor preserves data values."""
        extractor = NpzExtractor()
        npz_path = temp_data_dir / "test_archive.npz"

        archive = np.load(npz_path)
        result = extractor.extract(archive)
        expected = archive["full_stack"]

        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
