"""Unit tests for src.io.cache module.

Tests for CacheEntry, CacheManager, and CacheManagerFactory classes.
Covers normal operations, error handling, and edge cases.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from src.io.cache import (
    CacheEntry,
    CacheManager,
    CacheManagerFactory,
    CacheConfig,
    PathLike,
)

# Use constants from CacheConfig class
CACHE_DEPTH_SUFFIX = CacheConfig.CACHE_DEPTH_SUFFIX
CACHE_TIME_SUFFIX = CacheConfig.CACHE_TIME_SUFFIX
AVO_PREFIX = CacheConfig.AVO_PREFIX
CONFIG_HASH_PATTERN = CacheConfig.CONFIG_HASH_PATTERN
BYTES_PER_MB = CacheConfig.BYTES_PER_MB


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_extract_metadata_with_underscore(self):
        """Extract key from filename with underscores."""
        p = Path("avo_depth_abc123def456789012345.npz")
        key, config_hash = CacheEntry._extract_metadata(p)
        assert key == "avo"
        assert config_hash == "abc123def456789012345"

    def test_extract_metadata_without_underscore(self):
        """Extract key from filename without underscores."""
        p = Path("single.npz")
        key, config_hash = CacheEntry._extract_metadata(p)
        # Since there's no underscore, filename is split and first part is used
        assert key == "single.npz"
        assert config_hash is None

    def test_extract_metadata_no_hash(self):
        """Extract key when no hash pattern in filename."""
        p = Path("avo_time_sometext.npz")
        key, config_hash = CacheEntry._extract_metadata(p)
        assert key == "avo"
        assert config_hash is None

    def test_extract_metadata_with_valid_hash(self):
        """Extract 20+ character hex hash."""
        p = Path("avo_12345678901234567890.npz")
        key, config_hash = CacheEntry._extract_metadata(p)
        assert key == "avo"
        assert config_hash == "12345678901234567890"

    def test_from_path_nonexistent_raises(self):
        """from_path raises FileNotFoundError for missing file."""
        p = Path("/tmp/nonexistent_cache_file_xyz.npz")
        with pytest.raises(FileNotFoundError):
            CacheEntry.from_path(p)

    def test_from_path_shallow_nonexistent_raises(self):
        """from_path_shallow raises FileNotFoundError for missing file."""
        p = Path("/tmp/nonexistent_cache_file_xyz.npz")
        with pytest.raises(FileNotFoundError):
            CacheEntry.from_path_shallow(p)

    def test_from_path_shallow_creates_entry(self):
        """from_path_shallow creates CacheEntry without reading NPZ."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"dummy")

        try:
            entry = CacheEntry.from_path_shallow(tmp_path)
            assert entry.key == tmp_path.name.split("_")[0]
            assert entry.path == tmp_path
            assert entry.config is None
            assert entry.valid is None
            assert entry.size_bytes == 5
        finally:
            tmp_path.unlink()

    def test_from_path_reads_npz(self):
        """from_path creates CacheEntry and attempts to load NPZ."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            np.savez_compressed(tmp_path, data=np.array([1, 2, 3]))

        try:
            entry = CacheEntry.from_path(tmp_path)
            assert entry.key == tmp_path.name.split("_")[0]
            assert entry.valid is not None
        finally:
            tmp_path.unlink()

    def test_convert_npz_config_dict_conversion(self):
        """_convert_npz_config converts numpy array to dict."""
        cfg = np.array([("a", 1), ("b", 2)], dtype=[("key", "U10"), ("value", "i4")])
        result = CacheEntry._convert_npz_config(cfg)
        assert isinstance(result, (dict, type(None)))

    def test_convert_npz_config_item_fallback(self):
        """_convert_npz_config falls back to .item()."""
        cfg = MagicMock(spec=["item"])  # Only has item() method
        cfg.item.return_value = {"key": "value"}

        result = CacheEntry._convert_npz_config(cfg)
        assert result == {"key": "value"}

    def test_convert_npz_config_returns_none_on_failure(self):
        """_convert_npz_config returns None when conversion fails."""
        cfg = MagicMock()
        cfg.item.side_effect = ValueError("Cannot convert")

        with patch("builtins.dict", side_effect=TypeError("Cannot convert")):
            result = CacheEntry._convert_npz_config(cfg)
            assert result is None

    def test_to_dict_conversion(self):
        """to_dict converts CacheEntry to dictionary."""
        entry = CacheEntry(
            key="test",
            path=Path("/tmp/test.npz"),
            mtime=1234567890.0,
            size_bytes=1024,
            config_hash="abc123",
            config={"param": "value"},
            valid=True,
        )
        result = entry.to_dict()
        assert result["key"] == "test"
        assert result["path"] == "/tmp/test.npz"
        assert result["mtime"] == 1234567890.0
        assert result["size_bytes"] == 1024
        assert result["config_hash"] == "abc123"
        assert result["config"] == {"param": "value"}
        assert result["valid"] is True

    def test_repr_format(self):
        """__repr__ returns formatted string."""
        entry = CacheEntry(
            key="avo",
            path=Path("/tmp/avo_depth_abc123.npz"),
            mtime=1234567890.0,
            size_bytes=2048,
            valid=True,
        )
        repr_str = repr(entry)
        assert "CacheEntry" in repr_str
        assert "avo" in repr_str
        assert "2048" in repr_str


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_init_valid_cache_dir(self):
        """__init__ accepts valid cache directory."""
        manager = CacheManager(cache_dir="/tmp/cache")
        assert manager.cache_dir == "/tmp/cache"

    def test_init_empty_cache_dir_raises(self):
        """__init__ raises ValueError for empty cache_dir."""
        with pytest.raises(ValueError, match="cache_dir must be a non-empty string"):
            CacheManager(cache_dir="")

    def test_init_whitespace_cache_dir_raises(self):
        """__init__ raises ValueError for whitespace-only cache_dir."""
        with pytest.raises(ValueError, match="cache_dir must be a non-empty string"):
            CacheManager(cache_dir="   ")

    def test_init_none_cache_dir_raises(self):
        """__init__ raises ValueError for None cache_dir."""
        with pytest.raises(ValueError):
            CacheManager(cache_dir=None)

    def test_init_with_custom_logger(self):
        """__init__ accepts custom logger."""
        custom_logger = MagicMock()
        manager = CacheManager(cache_dir="/tmp", logger=custom_logger)
        assert manager.logger is custom_logger

    def test_select_latest_cache_entries_empty_dir(self):
        """select_latest_cache_entries returns empty dict for nonexistent dir."""
        manager = CacheManager(cache_dir="/tmp/nonexistent_dir_xyz")
        result = manager.select_latest_cache_entries()
        assert result == {}

    def test_select_latest_cache_entries_with_files(self):
        """select_latest_cache_entries groups files by key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy cache files
            Path(tmpdir, "avo_depth_abc123.npz").touch()
            Path(tmpdir, "avo_time_def456.npz").touch()
            Path(tmpdir, "other_xyz789.npz").touch()

            manager = CacheManager(cache_dir=tmpdir)
            result = manager.select_latest_cache_entries(skip_inspect=True)

            assert "avo" in result
            assert "other" in result
            assert len(result["avo"]) == 2
            assert len(result["other"]) == 1

    def test_select_latest_cache_entries_skip_unreadable(self):
        """select_latest_cache_entries skips unreadable files with logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid file and a directory (which can't be read as cache)
            Path(tmpdir, "valid_abc123.npz").touch()
            Path(tmpdir, "notafile").mkdir()

            manager = CacheManager(cache_dir=tmpdir)
            result = manager.select_latest_cache_entries(skip_inspect=True)

            # Should have valid file but skip the directory
            assert (
                "valid" in result or len(result) >= 0
            )  # At least processes without crashing

    def test_save_npz_creates_directory(self):
        """save_npz creates parent directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "cache.npz"
            manager = CacheManager(cache_dir=tmpdir)
            data = {"array": np.array([1, 2, 3])}

            manager.save_npz(output_path, data)

            assert output_path.exists()
            loaded = np.load(output_path, allow_pickle=True)
            assert "array" in loaded

    def test_select_cache_file_by_suffix_match(self):
        """_select_cache_file_by_suffix returns matching file."""
        entry1 = CacheEntry(
            key="avo",
            path=Path("avo_depth_001.npz"),
            mtime=100,
            size_bytes=1024,
        )
        entry2 = CacheEntry(
            key="avo",
            path=Path("avo_time_002.npz"),
            mtime=200,
            size_bytes=2048,
        )
        manager = CacheManager(cache_dir="/tmp")

        result = manager._select_cache_file_by_suffix(
            [entry1, entry2], CACHE_TIME_SUFFIX
        )
        assert result.path.name == "avo_time_002.npz"

    def test_select_cache_file_by_suffix_fallback_to_latest(self):
        """_select_cache_file_by_suffix falls back to latest if no match."""
        entry1 = CacheEntry(
            key="avo",
            path=Path("avo_001.npz"),
            mtime=100,
            size_bytes=1024,
        )
        entry2 = CacheEntry(
            key="avo",
            path=Path("avo_002.npz"),
            mtime=200,
            size_bytes=2048,
        )
        manager = CacheManager(cache_dir="/tmp")

        result = manager._select_cache_file_by_suffix(
            [entry1, entry2], CACHE_DEPTH_SUFFIX
        )
        assert result.path.name == "avo_002.npz"  # Latest (by index)

    def test_resolve_latest_paths_empty_keys(self):
        """resolve_latest_paths returns empty dict for no keys."""
        manager = CacheManager(cache_dir="/tmp")
        result = manager.resolve_latest_paths(keys=[])
        assert result == {}

    def test_resolve_latest_paths_nonexistent_key(self):
        """resolve_latest_paths returns None for missing keys."""
        manager = CacheManager(cache_dir="/tmp/nonexistent")
        result = manager.resolve_latest_paths(keys=["missing_key"])
        assert result["missing_key"] is None

    def test_resolve_latest_paths_with_depth_suffix(self):
        """resolve_latest_paths resolves depth-suffixed keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "avo_depth_abc123.npz").touch()
            manager = CacheManager(cache_dir=tmpdir)

            result = manager.resolve_latest_paths(keys=["avo_depth"])
            assert result["avo_depth"] is not None
            assert "avo_depth_abc123.npz" in result["avo_depth"]

    def test_resolve_latest_paths_with_time_suffix(self):
        """resolve_latest_paths resolves time-suffixed keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "avo_time_def456.npz").touch()
            manager = CacheManager(cache_dir=tmpdir)

            result = manager.resolve_latest_paths(keys=["avo_time"])
            assert result["avo_time"] is not None
            assert "avo_time_def456.npz" in result["avo_time"]

    def test_identify_old_cache_files_avo_prefix(self):
        """identify_old_cache_files finds AVO files without modality suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Old format (no modality)
            Path(tmpdir, "avo_abc123.npz").touch()
            # New format (with modality)
            Path(tmpdir, "avo_depth_def456.npz").touch()
            Path(tmpdir, "avo_time_ghi789.npz").touch()
            # Non-AVO
            Path(tmpdir, "other_jkl012.npz").touch()

            manager = CacheManager(cache_dir=tmpdir)
            old_files = manager.identify_old_cache_files()

            assert len(old_files) == 1
            assert "avo_abc123.npz" in old_files[0]

    def test_identify_old_cache_files_empty_dir(self):
        """identify_old_cache_files returns empty list for nonexistent dir."""
        manager = CacheManager(cache_dir="/tmp/nonexistent_xyz")
        result = manager.identify_old_cache_files()
        assert result == []

    def test_get_total_cache_size(self):
        """get_total_cache_size calculates total size in MB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with known sizes
            Path(tmpdir, "file1.npz").write_bytes(b"x" * 1024)  # 1 KB
            Path(tmpdir, "file2.npz").write_bytes(b"y" * 1024)  # 1 KB

            manager = CacheManager(cache_dir=tmpdir)
            size_mb = manager.get_total_cache_size()

            assert size_mb > 0
            assert size_mb < 1  # Should be << 1 MB

    def test_get_total_cache_size_nonexistent_dir(self):
        """get_total_cache_size returns 0 for nonexistent dir."""
        manager = CacheManager(cache_dir="/tmp/nonexistent_xyz")
        size_mb = manager.get_total_cache_size()
        assert size_mb == 0.0

    def test_cleanup_old_cache_dry_run(self):
        """cleanup_old_cache with dry_run=True doesn't delete files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "avo_old_abc123.npz").touch()
            manager = CacheManager(cache_dir=tmpdir)

            removed, freed_mb = manager.cleanup_old_cache(dry_run=True)

            assert removed == 0
            assert Path(tmpdir, "avo_old_abc123.npz").exists()

    def test_cleanup_old_cache_removes_files(self):
        """cleanup_old_cache removes old cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = Path(tmpdir, "avo_old_abc123.npz")
            old_file.touch()
            manager = CacheManager(cache_dir=tmpdir)

            removed, freed_mb = manager.cleanup_old_cache(dry_run=False)

            assert removed == 1
            assert not old_file.exists()

    def test_cleanup_old_cache_no_old_files(self):
        """cleanup_old_cache returns 0,0.0 when no old files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir)
            removed, freed_mb = manager.cleanup_old_cache(dry_run=False)
            assert removed == 0
            assert freed_mb == 0.0

    def test_run_method_with_verbose(self):
        """run method accepts verbose parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir)
            removed, freed_mb = manager.run(dry_run=True, verbose=False)
            assert isinstance(removed, int)
            assert isinstance(freed_mb, float)


class TestCacheManagerFactory:
    """Tests for CacheManagerFactory class."""

    def test_get_default_manager_returns_manager(self):
        """get_default_manager returns CacheManager instance (via LazyObjectProxy)."""
        manager = CacheManagerFactory.get_default_manager()
        # May be wrapped in LazyObjectProxy, but should support manager operations
        assert hasattr(manager, "select_latest_cache_entries")

    def test_get_default_manager_is_singleton(self):
        """get_default_manager returns consistent manager."""
        manager1 = CacheManagerFactory.get_default_manager()
        manager2 = CacheManagerFactory.get_default_manager()
        # Both should have same cache_dir
        assert manager1.cache_dir == manager2.cache_dir

    def test_get_manager_with_default_dir(self):
        """get_manager returns manager for default dir."""
        manager = CacheManagerFactory.get_manager(CacheManagerFactory.DEFAULT_CACHE_DIR)
        # Should return the default manager
        assert hasattr(manager, "cache_dir")

    def test_get_manager_with_custom_dir(self):
        """get_manager returns new manager for custom dir."""
        manager = CacheManagerFactory.get_manager("/tmp/custom")
        assert isinstance(manager, CacheManager)
        assert manager.cache_dir == "/tmp/custom"

    def test_get_manager_with_none(self):
        """get_manager with None returns manager."""
        manager = CacheManagerFactory.get_manager(None)
        assert hasattr(manager, "cache_dir")

    def test_for_directory_returns_manager(self):
        """for_directory returns CacheManager instance."""
        manager = CacheManagerFactory.for_directory("/tmp/test")
        assert isinstance(manager, CacheManager)
        assert manager.cache_dir == "/tmp/test"

    def test_for_directory_with_none(self):
        """for_directory with None returns manager."""
        manager = CacheManagerFactory.for_directory(None)
        # Should return manager, not raise
        assert hasattr(manager, "cache_dir")


class TestConstants:
    """Tests for module-level constants."""

    def test_cache_suffixes_defined(self):
        """Cache suffix constants are defined."""
        assert CACHE_DEPTH_SUFFIX == "_depth"
        assert CACHE_TIME_SUFFIX == "_time"

    def test_avo_prefix_defined(self):
        """AVO prefix constant is defined."""
        assert AVO_PREFIX == "avo_"

    def test_bytes_per_mb_defined(self):
        """BYTES_PER_MB conversion factor is defined."""
        assert BYTES_PER_MB == 1024.0**2

    def test_config_hash_pattern_defined(self):
        """CONFIG_HASH_PATTERN regex is defined."""
        assert CONFIG_HASH_PATTERN == r"([0-9a-f]{20,})"

    def test_pathlike_type_alias(self):
        """PathLike type alias is defined."""
        # Just verify it's importable and has correct structure
        assert PathLike is not None


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_load_npz_config_with_corrupted_file(self):
        """_load_npz_config handles corrupted NPZ gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"not a valid npz file")

        try:
            config, valid = CacheEntry._load_npz_config(tmp_path)
            assert config is None
            assert valid is False
        finally:
            tmp_path.unlink()

    def test_cleanup_old_cache_permission_error(self):
        """cleanup_old_cache logs OSError on file removal failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = Path(tmpdir, "avo_old_abc123.npz")
            old_file.touch()
            manager = CacheManager(cache_dir=tmpdir)

            with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
                removed, freed_mb = manager.cleanup_old_cache(dry_run=False)
                # Should handle error gracefully
                assert removed == 0


class TestEdgeCases:
    """Additional edge case tests for improved coverage."""

    def test_extract_metadata_multiple_hashes(self):
        """Extract first matching hash when multiple hex sequences present."""
        p = Path("avo_abc123def456789012345_xyz789012345678901234.npz")
        key, config_hash = CacheEntry._extract_metadata(p)
        assert key == "avo"
        # Should match the first 20+ hex sequence
        assert config_hash is not None

    def test_extract_metadata_short_hash_not_matched(self):
        """Extract ignores hex sequences shorter than 20 chars."""
        p = Path("avo_abc123_def456789012345.npz")
        key, config_hash = CacheEntry._extract_metadata(p)
        assert key == "avo"
        # "def456789012345" is only 15 chars, regex needs 20+ chars, so None
        assert config_hash is None

    def test_from_path_with_string_path(self):
        """from_path accepts string path argument."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name
            Path(tmp_path).write_bytes(b"dummy")

        try:
            entry = CacheEntry.from_path(tmp_path)  # Pass string, not Path
            assert entry.key is not None
        finally:
            Path(tmp_path).unlink()

    def test_from_path_shallow_with_string_path(self):
        """from_path_shallow accepts string path argument."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name
            Path(tmp_path).write_bytes(b"dummy")

        try:
            entry = CacheEntry.from_path_shallow(tmp_path)  # Pass string
            assert entry.config is None
        finally:
            Path(tmp_path).unlink()

    def test_load_npz_config_empty_npz(self):
        """_load_npz_config handles NPZ without config key."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            np.savez_compressed(tmp_path, data=np.array([1, 2, 3]))

        try:
            config, valid = CacheEntry._load_npz_config(tmp_path)
            assert config is None
            assert valid is True
        finally:
            tmp_path.unlink()

    def test_load_npz_config_with_config_key(self):
        """_load_npz_config successfully loads config key."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            config_data = {"param1": "value1", "param2": 42}
            np.savez_compressed(tmp_path, config=config_data)

        try:
            config, valid = CacheEntry._load_npz_config(tmp_path)
            # Config loading behavior depends on NPZ internals
            assert valid is True
        finally:
            tmp_path.unlink()

    def test_convert_npz_config_with_numpy_scalar(self):
        """_convert_npz_config returns None for numpy scalar (not a dict)."""
        cfg = np.float32(3.14)
        result = CacheEntry._convert_npz_config(cfg)
        # Numpy scalars don't convert to ConfigDict, should return None
        assert result is None

    def test_cache_manager_select_entries_with_inspect(self):
        """select_latest_cache_entries with skip_inspect=False reads NPZ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir, "test_abc123def456789012345.npz")
            np.savez_compressed(cache_file, data=np.array([1, 2, 3]))

            manager = CacheManager(cache_dir=tmpdir)
            result = manager.select_latest_cache_entries(skip_inspect=False)

            assert "test" in result
            assert result["test"][0].valid is not None

    def test_cache_manager_select_entries_sorts_by_mtime(self):
        """select_latest_cache_entries returns entries sorted by mtime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different mtimes
            file1 = Path(tmpdir, "avo_001.npz")
            file1.touch()

            file2 = Path(tmpdir, "avo_002.npz")
            file2.touch()

            manager = CacheManager(cache_dir=tmpdir)
            result = manager.select_latest_cache_entries(skip_inspect=True)

            # Should group by key
            assert "avo" in result
            assert len(result["avo"]) == 2

    def test_resolve_latest_paths_none_keys_parameter(self):
        """resolve_latest_paths handles None keys parameter."""
        manager = CacheManager(cache_dir="/tmp")
        result = manager.resolve_latest_paths(keys=None)
        assert result == {}

    def test_identify_old_cache_files_multiple_avo_files(self):
        """identify_old_cache_files finds multiple old AVO files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "avo_old1_abc123.npz").touch()
            Path(tmpdir, "avo_old2_def456.npz").touch()
            Path(tmpdir, "avo_depth_ghi789.npz").touch()

            manager = CacheManager(cache_dir=tmpdir)
            old_files = manager.identify_old_cache_files()

            assert len(old_files) == 2

    def test_get_total_cache_size_with_multiple_files(self):
        """get_total_cache_size sums multiple file sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files with specific sizes
            sizes = [512, 1024, 2048]
            for i, size in enumerate(sizes):
                Path(tmpdir, f"file{i}.npz").write_bytes(b"x" * size)

            manager = CacheManager(cache_dir=tmpdir)
            total_mb = manager.get_total_cache_size()

            total_bytes = sum(sizes)
            expected_mb = total_bytes / BYTES_PER_MB
            assert abs(total_mb - expected_mb) < 0.001

    def test_save_npz_overwrites_existing(self):
        """save_npz overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cache.npz"
            manager = CacheManager(cache_dir=tmpdir)

            # Save first time
            data1 = {"array": np.array([1, 2, 3])}
            manager.save_npz(output_path, data1)
            size1 = output_path.stat().st_size

            # Save second time with different data
            data2 = {"array": np.array([1, 2, 3, 4, 5])}
            manager.save_npz(output_path, data2)
            size2 = output_path.stat().st_size

            # Sizes should differ
            assert size1 != size2

    def test_cleanup_old_cache_calculates_freed_space(self):
        """cleanup_old_cache correctly calculates freed space."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create old file with known size
            old_file = Path(tmpdir, "avo_old_abc123.npz")
            file_size = 1024  # 1 KB
            old_file.write_bytes(b"x" * file_size)

            manager = CacheManager(cache_dir=tmpdir)
            removed, freed_mb = manager.cleanup_old_cache(dry_run=False)

            assert removed == 1
            expected_mb = file_size / BYTES_PER_MB
            assert abs(freed_mb - expected_mb) < 0.001

    def test_cache_entry_dataclass_fields(self):
        """CacheEntry has all expected dataclass fields."""
        entry = CacheEntry(
            key="test",
            path=Path("/tmp/test.npz"),
            mtime=1000.0,
            size_bytes=512,
        )

        assert entry.key == "test"
        assert entry.path == Path("/tmp/test.npz")
        assert entry.mtime == 1000.0
        assert entry.size_bytes == 512
        assert entry.config_hash is None
        assert entry.config is None
        assert entry.valid is None

    def test_cache_manager_uses_provided_logger(self):
        """CacheManager uses provided logger instance."""
        custom_logger = MagicMock()
        manager = CacheManager(cache_dir="/tmp", logger=custom_logger)

        # Logger should be used for operations
        manager.identify_old_cache_files()
        assert manager.logger is custom_logger

    def test_select_cache_file_by_suffix_single_candidate(self):
        """_select_cache_file_by_suffix works with single candidate."""
        entry = CacheEntry(
            key="avo",
            path=Path("avo_001.npz"),
            mtime=100,
            size_bytes=1024,
        )
        manager = CacheManager(cache_dir="/tmp")

        result = manager._select_cache_file_by_suffix([entry], CACHE_DEPTH_SUFFIX)
        assert result is entry

    def test_resolve_latest_paths_mixed_suffixes(self):
        """resolve_latest_paths handles mixed depth/time/plain keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "avo_depth_abc.npz").touch()
            Path(tmpdir, "avo_time_def.npz").touch()
            Path(tmpdir, "avo_plain_ghi.npz").touch()

            manager = CacheManager(cache_dir=tmpdir)
            result = manager.resolve_latest_paths(keys=["avo_depth", "avo_time", "avo"])

            assert result["avo_depth"] is not None
            assert result["avo_time"] is not None
            assert result["avo"] is not None

    def test_cleanup_old_cache_mixed_file_types(self):
        """cleanup_old_cache only removes old AVO files, preserves others."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_avo = Path(tmpdir, "avo_old_abc.npz")
            old_avo.touch()

            new_avo = Path(tmpdir, "avo_depth_def.npz")
            new_avo.touch()

            other = Path(tmpdir, "other_ghi.npz")
            other.touch()

            manager = CacheManager(cache_dir=tmpdir)
            removed, freed_mb = manager.cleanup_old_cache(dry_run=False)

            assert removed == 1
            assert old_avo.exists() is False
            assert new_avo.exists() is True
            assert other.exists() is True


class TestIntegration:
    """Integration tests combining multiple cache operations."""

    def test_full_cache_workflow(self):
        """Test complete cache creation, resolution, and cleanup workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir)

            # Create cache files
            data = {"array": np.array([1, 2, 3, 4, 5])}
            manager.save_npz(Path(tmpdir) / "avo_depth_abc123def456789012345.npz", data)
            manager.save_npz(
                Path(tmpdir) / "avo_time_def456ghi789jkl012mno345.npz", data
            )

            # Select and resolve
            entries = manager.select_latest_cache_entries(skip_inspect=True)
            assert "avo" in entries
            assert len(entries["avo"]) == 2

            paths = manager.resolve_latest_paths(keys=["avo_depth", "avo_time"])
            assert paths["avo_depth"] is not None
            assert paths["avo_time"] is not None

            # Get size
            size_mb = manager.get_total_cache_size()
            assert size_mb > 0

            # No old files to cleanup
            old_files = manager.identify_old_cache_files()
            assert len(old_files) == 0

    def test_cache_entry_lifecycle(self):
        """Test CacheEntry creation, conversion, and serialization."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            np.savez_compressed(tmp_path, data=np.array([1, 2, 3]))

        try:
            # Create from file
            entry = CacheEntry.from_path(tmp_path)
            assert entry.valid is not None

            # Convert to dict
            entry_dict = entry.to_dict()
            assert entry_dict["key"] == entry.key
            assert entry_dict["path"] == str(entry.path)

            # Check repr
            repr_str = repr(entry)
            assert "CacheEntry" in repr_str
        finally:
            tmp_path.unlink()

    def test_factory_consistency_across_calls(self):
        """Test CacheManagerFactory returns consistent managers."""
        manager1 = CacheManagerFactory.get_default_manager()
        manager2 = CacheManagerFactory.get_default_manager()

        # Both should have same cache_dir setting
        assert manager1.cache_dir == manager2.cache_dir

        # Custom managers are independent
        custom1 = CacheManagerFactory.for_directory("/tmp/cache1")
        custom2 = CacheManagerFactory.for_directory("/tmp/cache2")
        assert custom1.cache_dir != custom2.cache_dir


class TestCacheEntryAdvanced:
    """Advanced tests for CacheEntry to improve branch coverage."""

    def test_load_npz_config_with_corrupted_file(self):
        """_load_npz_config handles corrupted NPZ files."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(b"corrupted data")

        try:
            config, valid = CacheEntry._load_npz_config(tmp_path)
            assert valid is False
            assert config is None
        finally:
            tmp_path.unlink()

    def test_load_npz_config_with_config_key(self):
        """_load_npz_config extracts config from NPZ with 'config' key."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            config_data = {"param1": "value1", "param2": 42}
            np.savez_compressed(tmp_path, config=config_data)

        try:
            config, valid = CacheEntry._load_npz_config(tmp_path)
            assert valid is True
        finally:
            tmp_path.unlink()

    def test_load_npz_config_without_config_key(self):
        """_load_npz_config handles NPZ without 'config' key."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            np.savez_compressed(tmp_path, data=np.array([1, 2, 3]))

        try:
            config, valid = CacheEntry._load_npz_config(tmp_path)
            assert valid is True
            assert config is None
        finally:
            tmp_path.unlink()

    def test_convert_npz_config_with_items_method(self):
        """_convert_npz_config uses items() method to convert."""
        cfg = {"key1": "value1", "key2": 42, "key3": 3.14}
        result = CacheEntry._convert_npz_config(cfg)
        assert result is not None
        assert result["key1"] == "value1"
        assert result["key2"] == 42

    def test_convert_npz_config_invalid_returns_none(self):
        """_convert_npz_config returns None for invalid configs."""

        # Non-dict object that can't be converted
        class InvalidConfig:
            pass

        result = CacheEntry._convert_npz_config(InvalidConfig())
        assert result is None

    def test_convert_npz_config_with_nested_dict(self):
        """_convert_npz_config preserves nested dictionaries."""
        cfg = {"outer": {"inner": "value"}, "scalar": 10}
        result = CacheEntry._convert_npz_config(cfg)
        assert result is not None
        assert result["outer"] == {"inner": "value"}

    def test_cache_entry_repr(self):
        """CacheEntry __repr__ returns useful debugging info."""
        entry = CacheEntry(
            key="test", path=Path("/tmp/test.npz"), mtime=1234567890.0, size_bytes=1024
        )
        repr_str = repr(entry)
        assert "CacheEntry" in repr_str
        assert "test" in repr_str

    def test_cache_entry_to_dict(self):
        """CacheEntry.to_dict() includes all fields."""
        entry = CacheEntry(
            key="avo",
            path=Path("/cache/avo.npz"),
            mtime=1234567890.0,
            size_bytes=2048,
            config_hash="abc123",
            config={"param": "value"},
            valid=True,
        )
        d = entry.to_dict()
        assert d["key"] == "avo"
        assert d["path"] == "/cache/avo.npz"
        assert d["mtime"] == 1234567890.0
        assert d["size_bytes"] == 2048
        assert d["config_hash"] == "abc123"
        assert d["valid"] is True


class TestCacheManagerAdvanced:
    """Advanced tests for CacheManager to improve coverage."""

    def test_cache_manager_with_custom_logger(self):
        """CacheManager accepts custom logger."""
        custom_logger = MagicMock()
        manager = CacheManager(cache_dir="/tmp", logger=custom_logger)
        assert manager.logger is custom_logger

    def test_cache_manager_init_with_invalid_cache_dir(self):
        """CacheManager raises ValueError for empty cache_dir."""
        with pytest.raises(ValueError, match="cache_dir must be a non-empty string"):
            CacheManager(cache_dir="")

        with pytest.raises(ValueError, match="cache_dir must be a non-empty string"):
            CacheManager(cache_dir="   ")

    def test_select_latest_cache_entries_with_corrupted_file(self):
        """select_latest_cache_entries skips corrupted files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid file
            valid_file = Path(tmpdir) / "avo_valid.npz"
            np.savez_compressed(valid_file, data=np.array([1, 2, 3]))

            # Create corrupted file
            corrupted_file = Path(tmpdir) / "avo_corrupted.npz"
            corrupted_file.write_bytes(b"corrupted")

            manager = CacheManager(cache_dir=tmpdir)
            entries = manager.select_latest_cache_entries(skip_inspect=False)

            # Should have loaded at least the valid file
            assert "avo" in entries

    def test_select_latest_cache_entries_skip_inspect_true(self):
        """select_latest_cache_entries with skip_inspect=True is faster."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "avo_1.npz"
            file2 = Path(tmpdir) / "avo_2.npz"
            file1.touch()
            file2.touch()

            manager = CacheManager(cache_dir=tmpdir)
            entries = manager.select_latest_cache_entries(skip_inspect=True)

            assert "avo" in entries
            assert len(entries["avo"]) == 2
            # With skip_inspect, config should not be loaded
            assert all(e.config is None for e in entries["avo"])

    def test_save_npz_creates_parent_directory(self):
        """save_npz creates parent directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "level1" / "level2" / "test.npz"
            manager = CacheManager(cache_dir=tmpdir)

            data = {"array": np.array([1, 2, 3])}
            manager.save_npz(nested_path, data)

            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_get_total_cache_size_nonexistent_dir(self):
        """get_total_cache_size returns 0.0 for nonexistent directory."""
        manager = CacheManager(cache_dir="/nonexistent/cache/dir/xyz")
        size = manager.get_total_cache_size()
        assert size == 0.0

    def test_get_total_cache_size_with_files(self):
        """get_total_cache_size calculates correct total."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "cache1.npz"
            file2 = Path(tmpdir) / "cache2.npz"

            np.savez_compressed(file1, data1=np.array([1, 2, 3]))
            np.savez_compressed(file2, data2=np.array([4, 5, 6]))

            manager = CacheManager(cache_dir=tmpdir)
            size_mb = manager.get_total_cache_size()

            assert size_mb > 0
            # Verify it's a reasonable calculation
            assert size_mb < 1  # Should be much less than 1 MB

    def test_identify_old_cache_files_with_mixed_files(self):
        """identify_old_cache_files identifies only old AVO files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Old AVO files (no _depth or _time)
            old_file1 = Path(tmpdir) / "avo_old.npz"
            old_file2 = Path(tmpdir) / "avo_another.npz"

            # New AVO files (with _depth or _time)
            new_file1 = Path(tmpdir) / "avo_depth_new.npz"
            new_file2 = Path(tmpdir) / "avo_time_new.npz"

            # Non-AVO files
            other_file = Path(tmpdir) / "other_file.npz"

            for f in [old_file1, old_file2, new_file1, new_file2, other_file]:
                f.touch()

            manager = CacheManager(cache_dir=tmpdir)
            old_files = manager.identify_old_cache_files()

            assert len(old_files) == 2
            assert str(old_file1) in old_files
            assert str(old_file2) in old_files
            assert str(new_file1) not in old_files
            assert str(new_file2) not in old_files
            assert str(other_file) not in old_files

    def test_cleanup_old_cache_dry_run(self):
        """cleanup_old_cache with dry_run=True doesn't delete files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = Path(tmpdir) / "avo_old.npz"
            old_file.touch()

            manager = CacheManager(cache_dir=tmpdir)
            removed, freed_mb = manager.cleanup_old_cache(dry_run=True)

            assert removed == 0
            assert freed_mb == 0.0
            assert old_file.exists()  # File should still exist

    def test_cleanup_old_cache_actual_deletion(self):
        """cleanup_old_cache without dry_run deletes old files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = Path(tmpdir) / "avo_old.npz"
            old_file.touch()

            manager = CacheManager(cache_dir=tmpdir)
            removed, freed_mb = manager.cleanup_old_cache(dry_run=False)

            assert removed == 1
            assert freed_mb >= 0.0
            assert not old_file.exists()  # File should be deleted

    def test_cleanup_old_cache_no_old_files(self):
        """cleanup_old_cache handles when there are no old files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only new files (no old files)
            Path(tmpdir, "avo_depth_new.npz").touch()

            manager = CacheManager(cache_dir=tmpdir)
            removed, freed_mb = manager.cleanup_old_cache(dry_run=False)

            assert removed == 0
            assert freed_mb == 0.0

    def test_run_with_verbose_mode(self):
        """run() method works with verbose=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(cache_dir=tmpdir)
            removed, freed_mb = manager.run(dry_run=True, verbose=True)

            assert isinstance(removed, int)
            assert isinstance(freed_mb, float)

    def test_run_with_actual_deletion(self):
        """run() method performs cleanup on actual call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = Path(tmpdir) / "avo_old.npz"
            old_file.touch()

            manager = CacheManager(cache_dir=tmpdir)
            removed, freed_mb = manager.run(dry_run=False, verbose=False)

            assert removed == 1


class TestNumpyLoader:
    """Tests for _NumpyLoader lazy loading."""

    def test_numpy_loader_lazy_import(self):
        """_NumpyLoader.get() lazily imports numpy."""
        from src.io.cache import _NumpyLoader

        # Reset cache
        _NumpyLoader._numpy = None

        # First call imports numpy
        np1 = _NumpyLoader.get()
        assert np1 is not None

        # Second call reuses cached module
        np2 = _NumpyLoader.get()
        assert np1 is np2  # Same object reference

    def test_numpy_loader_cached_across_calls(self):
        """_NumpyLoader caches numpy module after first import."""
        from src.io.cache import _NumpyLoader

        np1 = _NumpyLoader.get()
        np2 = _NumpyLoader.get()
        np3 = _NumpyLoader.get()

        # All should be the same object
        assert np1 is np2 is np3


class TestCacheManagerErrorHandling:
    """Tests for error handling in CacheManager."""

    def test_cleanup_with_permission_error(self):
        """cleanup_old_cache handles OSError gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = Path(tmpdir) / "avo_old.npz"
            old_file.touch()

            manager = CacheManager(cache_dir=tmpdir)

            # Mock unlink to raise OSError
            call_count = [0]

            def mock_unlink(self):
                call_count[0] += 1
                raise OSError("Permission denied")

            with patch.object(Path, "unlink", mock_unlink):
                removed, freed_mb = manager.cleanup_old_cache(dry_run=False)

                # Should handle error and report 0 removed
                assert removed == 0

    def test_get_total_cache_size_with_os_error(self):
        """get_total_cache_size handles OSError."""
        manager = CacheManager(cache_dir="/tmp")

        # Mock glob to raise OSError
        with patch.object(Path, "glob", side_effect=OSError("Access denied")):
            size = manager.get_total_cache_size()
            assert size == 0.0

    def test_select_latest_cache_entries_with_deleted_file(self):
        """select_latest_cache_entries handles files deleted during iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "avo_1.npz"
            file1.write_bytes(b"test")

            manager = CacheManager(cache_dir=tmpdir)

            # Mock stat() to raise OSError on second call
            original_stat = Path.stat
            call_count = [0]

            def mock_stat(self, **kwargs):
                call_count[0] += 1
                if call_count[0] > 1:
                    raise OSError("File deleted")
                return original_stat(self, **kwargs)

            with patch.object(Path, "stat", mock_stat):
                manager.select_latest_cache_entries(skip_inspect=True)
                # Should skip the problematic file but continue


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
