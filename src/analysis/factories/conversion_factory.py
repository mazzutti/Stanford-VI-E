"""Factory for creating ConversionStrategy instances.

Centralizes strategy creation and provides convenient access to all
available conversion strategies through a single factory interface.
"""

from __future__ import annotations

import logging
from typing import Literal, Union

from src.analysis.strategies import (
    AmplitudeConversionStrategy,
    ConversionStrategy,
    DepthConversionStrategy,
    TimeConversionStrategy,
    VelocityConversionStrategy,
)

logger = logging.getLogger(__name__)

__all__ = ["ConversionStrategyFactory"]


class ConversionStrategyFactory:
    """Factory for creating and managing ConversionStrategy instances.

    Provides singleton instances of all conversion strategies to avoid
    unnecessary object recreation.
    """

    _instance: ConversionStrategyFactory | None = None
    _strategies: dict[str, ConversionStrategy] = {}

    def __new__(cls) -> ConversionStrategyFactory:
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize factory with all available strategies."""
        if not self._initialized:
            self._strategies = {
                "velocity": VelocityConversionStrategy(),
                "time": TimeConversionStrategy(),
                "depth": DepthConversionStrategy(),
                "amplitude": AmplitudeConversionStrategy(),
            }
            self._initialized = True
            logger.debug("ConversionStrategyFactory initialized with 4 strategies")

    @classmethod
    def get_velocity_strategy(cls) -> VelocityConversionStrategy:
        """Get velocity conversion strategy.

        Returns:
            VelocityConversionStrategy instance
        """
        factory = cls()
        return factory._strategies["velocity"]

    @classmethod
    def get_time_strategy(cls) -> TimeConversionStrategy:
        """Get time conversion strategy.

        Returns:
            TimeConversionStrategy instance
        """
        factory = cls()
        return factory._strategies["time"]

    @classmethod
    def get_depth_strategy(cls) -> DepthConversionStrategy:
        """Get depth conversion strategy.

        Returns:
            DepthConversionStrategy instance
        """
        factory = cls()
        return factory._strategies["depth"]

    @classmethod
    def get_amplitude_strategy(cls) -> AmplitudeConversionStrategy:
        """Get amplitude conversion strategy.

        Returns:
            AmplitudeConversionStrategy instance
        """
        factory = cls()
        return factory._strategies["amplitude"]

    @classmethod
    def get_strategy(
        cls,
        strategy_type: Literal["velocity", "time", "depth", "amplitude"],
    ) -> ConversionStrategy:
        """Get a conversion strategy by type.

        Args:
            strategy_type: Type of conversion strategy to retrieve

        Returns:
            ConversionStrategy instance of requested type

        Raises:
            ValueError: If strategy_type is not recognized
        """
        factory = cls()
        if strategy_type not in factory._strategies:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available: {list(factory._strategies.keys())}"
            )
        return factory._strategies[strategy_type]

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List all available strategy types.

        Returns:
            List of strategy type names
        """
        factory = cls()
        return list(factory._strategies.keys())

    @classmethod
    def reset(cls) -> None:
        """Reset factory to uninitialized state (for testing)."""
        cls._instance = None
        cls._strategies = {}
