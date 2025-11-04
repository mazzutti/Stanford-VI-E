"""Rock physics computation and analysis subpackage.

This subpackage provides focused modules for rock physics attribute computation
using the composition pattern for clean separation of concerns.

Modules:
    - computers: Domain-specific computation classes (AVO, Lamé, fluid factor)
    - discrimination: Attribute discrimination analysis for facies
    - analyzer: Main orchestrator for the complete pipeline

Public API:
    - RockPhysicsAnalyzer: Main orchestrator class
    - AVOAttributesComputer: AVO intercept/gradient computation
    - LambdaMuRhoComputer: Lamé parameter computation
    - FluidFactorComputer: Fluid factor derivation
    - AttributeDiscriminationAnalyzer: Discrimination analysis

Example:
    >>> from src.analysis.rock_physics import RockPhysicsAnalyzer
    >>> analyzer = RockPhysicsAnalyzer()
    >>> avo = analyzer.compute_avo_attributes(vp, vs, rho)
    >>> lam_mu = analyzer.compute_lambda_mu_rho(vp, vs, rho)
"""

from src.analysis.rock_physics.analyzer import (
    RockPhysicsAnalyzer,
    RockPhysicsConstants,
)
from src.analysis.rock_physics.computers import (
    AVOAttributesComputer,
    LambdaMuRhoComputer,
    FluidFactorComputer,
    DEFAULT_AVO_ANGLES_DEG,
    DEFAULT_FLUID_FACTOR_K,
)
from src.analysis.rock_physics.discrimination import (
    AttributeDiscriminationAnalyzer,
)

__all__ = [
    "RockPhysicsAnalyzer",
    "RockPhysicsConstants",
    "AVOAttributesComputer",
    "LambdaMuRhoComputer",
    "FluidFactorComputer",
    "AttributeDiscriminationAnalyzer",
    "DEFAULT_AVO_ANGLES_DEG",
    "DEFAULT_FLUID_FACTOR_K",
]
