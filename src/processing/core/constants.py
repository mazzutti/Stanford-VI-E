"""Core constants used across the processing module."""

# Resampling
DEFAULT_MAX_CACHE_BYTES = 10 * 1024**3  # 10 GB default cache limit
DEFAULT_DT_MILLISECONDS = 4.0  # Default time sampling in ms


# Materials
DEFAULT_VELOCITY_THRESHOLD = 3000.0  # m/s threshold for unit detection
DEFAULT_DENSITY_THRESHOLD = 1000.0  # kg/m³ threshold for unit detection


# AVO Analysis
DEFAULT_MAX_AVO_ANGLE = 30.0  # degrees
DEFAULT_CONTRAST_THRESHOLD = 0.20  # 20% fractional contrast
DEFAULT_SUGGESTED_ANGLES = [0, 10, 15, 20, 25, 30]


# Logging
DEFAULT_LOG_LEVEL = "INFO"
