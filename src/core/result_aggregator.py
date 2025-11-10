"""Simplified result aggregation using builder pattern.

Replaces complex dictionary manipulation with clean, chainable methods.
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

__all__ = ["ResultAggregator", "ResultSummary"]


@dataclass
class ResultSummary:
    """Simple result summary container."""

    attributes: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attributes": self.attributes,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "errors": self.errors,
        }


class ResultAggregator:
    """Aggregate analysis results with clean, simple interface.

    Replaces nested dictionary operations with clear method calls.
    """

    def __init__(self):
        """Initialize aggregator."""
        self._results: Dict[str, Any] = {}

    def add_attribute(self, name: str, value: Any) -> "ResultAggregator":
        """Add a computed attribute."""
        self._results[name] = value
        return self

    def add_attributes(self, **attributes: Any) -> "ResultAggregator":
        """Add multiple attributes at once."""
        self._results.update(attributes)
        return self

    def add_from_dict(
        self, data: Dict[str, Any], prefix: Optional[str] = None
    ) -> "ResultAggregator":
        """Add attributes from dictionary, optionally with prefix."""
        items = {f"{prefix}_{k}": v for k, v in data.items()} if prefix else data
        self._results.update(items)
        return self

    def merge_results(self, *result_dicts: Dict[str, Any]) -> "ResultAggregator":
        """Merge multiple result dictionaries."""
        for result_dict in result_dicts:
            self._results.update(result_dict)
        return self

    def filter_keys(self, keys: List[str]) -> "ResultAggregator":
        """Keep only specified keys."""
        self._results = {k: v for k, v in self._results.items() if k in keys}
        return self

    def exclude_keys(self, keys: List[str]) -> "ResultAggregator":
        """Remove specified keys."""
        self._results = {k: v for k, v in self._results.items() if k not in keys}
        return self

    def transform(self, key: str, func: callable) -> "ResultAggregator":
        """Transform a specific result value."""
        if key in self._results:
            self._results[key] = func(self._results[key])
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific result."""
        return self._results.get(key, default)

    def keys(self) -> List[str]:
        """Get all result keys."""
        return list(self._results.keys())

    def build(self) -> Dict[str, Any]:
        """Build final results dictionary."""
        return self._results.copy()

    def build_summary(self) -> ResultSummary:
        """Build a structured summary separating metrics from attributes."""
        summary = ResultSummary()
        for key, value in self._results.items():
            target = summary.metrics if self._is_metric(value) else summary.attributes
            target[key] = value
        return summary

    @staticmethod
    def _is_metric(value: Any) -> bool:
        """Check if value is a numeric metric."""
        return isinstance(value, (int, float))


class ChainableDict(dict):
    """Dictionary with chainable operations for cleaner code."""

    def set(self, key: str, value: Any) -> "ChainableDict":
        """Set value and return self for chaining."""
        self[key] = value
        return self

    def set_default(self, key: str, value: Any) -> "ChainableDict":
        """Set default value and return self."""
        self.setdefault(key, value)
        return self

    def remove(self, key: str) -> "ChainableDict":
        """Remove key and return self."""
        self.pop(key, None)
        return self

    def merge(self, other: Dict[str, Any]) -> "ChainableDict":
        """Merge another dict and return self."""
        self.update(other)
        return self
