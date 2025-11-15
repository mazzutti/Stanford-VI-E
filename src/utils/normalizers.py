"""Unit normalization utilities for standardizing unit string representations.

Provides centralized unit alias handling and normalization for velocity,
density, and other physical quantities.
"""


class UnitNormalizer:
    """Normalizes unit strings to canonical forms.

    Provides centralized unit alias handling instead of scattering normalize
    logic throughout the codebase. Supports multiple equivalent unit representations
    and maps them to canonical forms.

    Examples:
        >>> normalizer = UnitNormalizer()
        >>> UnitNormalizer.normalize("m_per_s")
        'm/s'
        >>> UnitNormalizer.is_velocity("km/s")
        True
        >>> UnitNormalizer.is_density("g/cm3")
        True
    """

    # Mapping from aliases to canonical unit names
    VELOCITY_ALIASES: dict[str, str] = {
        "m/s": "m/s",
        "m_per_s": "m/s",
        "km/s": "km/s",
        "km_per_s": "km/s",
    }

    DENSITY_ALIASES: dict[str, str] = {
        "g/cc": "g/cc",
        "g/cm3": "g/cc",
        "g/cm^3": "g/cc",
        "kg/m3": "kg/m3",
        "kg/m^3": "kg/m3",
        "kg/m³": "kg/m3",
    }

    ALL_ALIASES: dict[str, str] = {**VELOCITY_ALIASES, **DENSITY_ALIASES}

    @classmethod
    def normalize(cls, unit: str) -> str:
        """Normalize unit string to canonical form.

        Parameters
        ----------
        unit : str
            Unit string to normalize (e.g., "m_per_s", "m/s")

        Returns
        -------
        str
            Canonical form of the unit (e.g., "m/s")
        """
        unit = unit.strip()
        return cls.ALL_ALIASES.get(unit, unit)

    @classmethod
    def is_velocity(cls, unit: str) -> bool:
        """Check if unit represents velocity.

        Parameters
        ----------
        unit : str
            Unit string to check

        Returns
        -------
        bool
            True if unit is a velocity unit (m/s or km/s)
        """
        norm = cls.normalize(unit)
        return norm in ("m/s", "km/s")

    @classmethod
    def is_density(cls, unit: str) -> bool:
        """Check if unit represents density.

        Parameters
        ----------
        unit : str
            Unit string to check

        Returns
        -------
        bool
            True if unit is a density unit (g/cc or kg/m3)
        """
        norm = cls.normalize(unit)
        return norm in ("g/cc", "kg/m3")
