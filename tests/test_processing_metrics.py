"""Tests for src.processing.metrics module.

Tests for PlanFingerprint and BackendMetrics classes used for tracking
resampling performance and plan identification.
"""

import hashlib
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.processing.metrics import PlanFingerprint, BackendMetrics


class TestPlanFingerprintCreation:
    """Tests for PlanFingerprint creation and equality."""

    def test_create_fingerprint_basic(self):
        """Test basic fingerprint creation."""
        fp = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="abc123"
        )

        assert fp.ni == 10
        assert fp.nj == 10
        assert fp.nz == 20
        assert fp.nt == 100
        assert fp.dt == 0.004
        assert fp.uniform_twt is True
        assert fp.vp_hash == "abc123"

    def test_fingerprint_is_frozen(self):
        """Test that fingerprint is immutable."""
        fp = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="abc123"
        )

        # Should raise because dataclass is frozen
        with pytest.raises((AttributeError, TypeError)):
            fp.ni = 20

    def test_fingerprint_equality(self):
        """Test fingerprint equality comparison."""
        fp1 = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="abc123"
        )
        fp2 = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="abc123"
        )

        assert fp1 == fp2

    def test_fingerprint_inequality(self):
        """Test fingerprint inequality."""
        fp1 = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="abc123"
        )
        fp2 = PlanFingerprint(
            ni=20, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="abc123"
        )

        assert fp1 != fp2


class TestPlanFingerprintFromPlan:
    """Tests for creating fingerprint from ResamplePlan."""

    def test_from_plan_basic(self):
        """Test creating fingerprint from plan-like object."""
        # Create a mock plan object
        mock_plan = MagicMock()
        mock_plan.ni = 5
        mock_plan.nj = 5
        mock_plan.nz = 10
        mock_plan.nt = 50
        mock_plan.dt = 0.004
        mock_plan.uniform_twt = True
        mock_plan.vp_arr = np.ones((5, 5, 10)) * 3500

        fp = PlanFingerprint.from_plan(mock_plan)

        assert fp.ni == mock_plan.ni
        assert fp.nj == mock_plan.nj
        assert fp.nz == mock_plan.nz
        assert fp.nt == mock_plan.nt
        assert fp.dt == mock_plan.dt
        assert fp.uniform_twt == mock_plan.uniform_twt
        assert isinstance(fp.vp_hash, str)
        assert len(fp.vp_hash) == 32  # MD5 hash length

    def test_from_plan_consistent_hash(self):
        """Test that same plan produces consistent hash."""
        vp_arr = np.ones((5, 5, 10)) * 3500

        mock_plan = MagicMock()
        mock_plan.ni = 5
        mock_plan.nj = 5
        mock_plan.nz = 10
        mock_plan.nt = 50
        mock_plan.dt = 0.004
        mock_plan.uniform_twt = True
        mock_plan.vp_arr = vp_arr

        fp1 = PlanFingerprint.from_plan(mock_plan)
        fp2 = PlanFingerprint.from_plan(mock_plan)

        assert fp1.vp_hash == fp2.vp_hash

    def test_from_plan_different_vp_different_hash(self):
        """Test that different Vp arrays produce different hashes."""
        vp_arr1 = np.ones((5, 5, 10)) * 3000
        vp_arr2 = np.ones((5, 5, 10)) * 4000

        plan1 = MagicMock()
        plan1.ni = 5
        plan1.nj = 5
        plan1.nz = 10
        plan1.nt = 50
        plan1.dt = 0.004
        plan1.uniform_twt = True
        plan1.vp_arr = vp_arr1

        plan2 = MagicMock()
        plan2.ni = 5
        plan2.nj = 5
        plan2.nz = 10
        plan2.nt = 50
        plan2.dt = 0.004
        plan2.uniform_twt = True
        plan2.vp_arr = vp_arr2

        fp1 = PlanFingerprint.from_plan(plan1)
        fp2 = PlanFingerprint.from_plan(plan2)

        assert fp1.vp_hash != fp2.vp_hash

    def test_from_plan_small_array(self):
        """Test hashing small Vp array."""
        plan = MagicMock()
        plan.ni = 1
        plan.nj = 1
        plan.nz = 1
        plan.nt = 10
        plan.dt = 0.004
        plan.uniform_twt = True
        plan.vp_arr = np.array([[[3000.0]]])

        fp = PlanFingerprint.from_plan(plan)

        assert isinstance(fp.vp_hash, str)
        assert len(fp.vp_hash) > 0

    def test_from_plan_large_array(self):
        """Test hashing large Vp array (tests hash optimization)."""
        # Create large array to test hash sampling
        plan = MagicMock()
        plan.ni = 100
        plan.nj = 100
        plan.nz = 50
        plan.nt = 200
        plan.dt = 0.004
        plan.uniform_twt = True
        plan.vp_arr = np.random.rand(100, 100, 50) * 1000 + 3000

        fp = PlanFingerprint.from_plan(plan)

        assert isinstance(fp.vp_hash, str)
        assert len(fp.vp_hash) == 32


class TestBackendMetrics:
    """Tests for BackendMetrics class."""

    def test_metrics_init(self):
        """Test BackendMetrics initialization."""
        metrics = BackendMetrics()
        assert isinstance(metrics, BackendMetrics)

    def test_record_single_backend_run(self):
        """Test recording a single backend run."""
        metrics = BackendMetrics()
        fp = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="test"
        )

        metrics.record_selection("linear")
        metrics.record_runtime("linear", fp, 0.5)
        # Should not raise

    def test_record_multiple_runs(self):
        """Test recording multiple runs."""
        metrics = BackendMetrics()
        fp1 = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="test1"
        )
        fp2 = PlanFingerprint(
            ni=20, nj=20, nz=30, nt=150, dt=0.004, uniform_twt=True, vp_hash="test2"
        )

        metrics.record_selection("linear")
        metrics.record_runtime("linear", fp1, 0.5)
        metrics.record_selection("cubic")
        metrics.record_runtime("cubic", fp1, 0.3)
        metrics.record_selection("linear")
        metrics.record_runtime("linear", fp2, 0.7)

        # Should not raise

    def test_get_selection_count(self):
        """Test getting selection count per backend."""
        metrics = BackendMetrics()
        fp = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="test"
        )

        metrics.record_selection("linear")
        metrics.record_runtime("linear", fp, 0.5)
        metrics.record_selection("linear")
        metrics.record_runtime("linear", fp, 0.5)
        metrics.record_selection("cubic")
        metrics.record_runtime("cubic", fp, 0.3)

        # Should be able to retrieve counts
        assert metrics.get_selection_count("linear") == 2
        assert metrics.get_selection_count("cubic") == 1

    def test_metrics_summarize(self):
        """Test metrics summarization."""
        metrics = BackendMetrics()
        fp = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="test"
        )

        metrics.record_selection("linear")
        metrics.record_runtime("linear", fp, 0.5)
        metrics.record_selection("cubic")
        metrics.record_runtime("cubic", fp, 0.3)

        # Check that we can access the metrics
        assert metrics.get_selection_count("linear") == 1
        assert metrics.get_selection_count("cubic") == 1
        assert metrics.get_runtime("linear", fp) == 0.5
        assert metrics.get_runtime("cubic", fp) == 0.3


class TestBackendMetricsIntegration:
    """Integration tests for metrics tracking."""

    def test_track_multiple_backends_same_plan(self):
        """Test tracking multiple backends on same plan."""
        metrics = BackendMetrics()
        fp = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="test"
        )

        # Simulate selecting different backends
        for backend in ["linear", "cubic", "akima"]:
            metrics.record_selection(backend)
            metrics.record_runtime(backend, fp, np.random.rand() * 1.0)

        # Verify all backends were recorded
        assert metrics.get_selection_count("linear") == 1
        assert metrics.get_selection_count("cubic") == 1
        assert metrics.get_selection_count("akima") == 1

    def test_cumulative_runtime_tracking(self):
        """Test that runtimes are accumulated."""
        metrics = BackendMetrics()
        fp = PlanFingerprint(
            ni=10, nj=10, nz=20, nt=100, dt=0.004, uniform_twt=True, vp_hash="test"
        )

        metrics.record_selection("linear")
        metrics.record_runtime("linear", fp, 0.5)
        metrics.record_selection("linear")
        metrics.record_runtime("linear", fp, 0.3)

        # Combined runtime should be tracked
        total_runtime = metrics.get_runtime("linear", fp)
        assert total_runtime == 0.8
        assert metrics.get_selection_count("linear") == 2

    def test_metrics_with_empty_cache(self):
        """Test metrics operations when no runs recorded."""
        metrics = BackendMetrics()
        # Should not raise when empty
        counts = {"test": metrics.get_selection_count("linear")}
        assert isinstance(counts, dict)
