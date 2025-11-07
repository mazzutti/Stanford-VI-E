"""Constants and default parameters for rock physics analysis."""

from __future__ import annotations


class RockPhysicsConstants:
    """Central repository for rock physics computation constants and defaults.

    Consolidates all magic numbers, configuration values, and default parameters
    used throughout rock physics analysis into a single, well-documented class.
    This improves maintainability and makes tuning easier.

    Constants
    ---------
    EPSILON : float
        Small value for numerical stability in computations.
    SNR_EPSILON : float
        Small value for signal-to-noise ratio calculations.
    AVO_KEYS : frozenset
        Expected keys in AVO results dictionary.
    LAMBDA_MU_KEYS : frozenset
        Expected keys in Lamé parameter results.
    DISCRIMINATION_KEYS : frozenset
        Expected keys in discrimination analysis results.
    OUTPUT_FILENAME : str
        Default filename for saved rock physics attributes.

    Defaults
    --------
    DEFAULT_GRID_SHAPE : tuple
        Default grid dimensions for analysis.
    DEFAULT_DZ : float
        Default depth increment (in meters or units).
    DEFAULT_DT : float
        Default time increment (in seconds).
    DEFAULT_DATA_PATH : str
        Default path for data loading.
    DEFAULT_FILE_MAP : dict
        Default mapping of data fields to filenames.
    """

    # Numerical constants
    EPSILON: float = 1e-10
    SNR_EPSILON: float = 1e-10

    # Result key constants
    AVO_KEYS: frozenset[str] = frozenset(
        {"intercept", "gradient", "product", "scaled_gradient"}
    )
    LAMBDA_MU_KEYS: frozenset[str] = frozenset(
        {"lambda_rho", "mu_rho", "lambda_mu_ratio"}
    )
    DISCRIMINATION_KEYS: frozenset[str] = frozenset(
        {
            "name",
            "cohens_d",
            "pearson_r",
            "p_value",
            "snr",
            "mean_class0",
            "mean_class1",
            "std_class0",
            "std_class1",
        }
    )

    # Output configuration
    OUTPUT_FILENAME: str = "rock_physics_attributes.npz"

    # Default grid parameters
    DEFAULT_GRID_SHAPE: tuple[int, int, int] = (150, 200, 200)
    DEFAULT_DZ: float = 1.0
    DEFAULT_DT: float = 0.001

    # Default file handling
    DEFAULT_DATA_PATH: str = "."
    DEFAULT_FILE_MAP: dict[str, str] = {
        "vp": "P-wave Velocity",
        "vs": "S-wave Velocity",
        "rho": "Density",
        "facies": "Facies",
    }
