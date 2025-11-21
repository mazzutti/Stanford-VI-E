"""Mesh / geometry helpers reused by debug plotting utilities.

These helpers are intentionally lightweight (NumPy-only) so other
high-level packages can import them without dragging heavy dependencies
or creating import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Use short coordinate variable names across this module for clarity in
# numerical code. These names are conventional in geometry/math code and
# are intentionally allowed here.


@dataclass(frozen=True)
class Coords:
    """Simple container for 1D coords and separation value."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sep: float


@dataclass(frozen=True)
class MeshGrids:
    """Container grouping mesh grids by orthogonal plane.

    Short, conventional variable names (X, Y, Z, etc.) are used throughout
    this module to mirror mathematical notation for mesh coordinates.
    """

    # Keep grids grouped into plane-specific small dataclasses to avoid
    # having too many top-level instance attributes which pylint flags.
    xy: "PlaneXY"
    xz: "PlaneXZ"
    yz: "PlaneYZ"


@dataclass(frozen=True)
class CoordsAndGrids:
    """Wrapper grouping `Coords` and `MeshGrids` together."""

    coords: Coords
    grids: MeshGrids


def compute_1d_coords_and_sep(
    shape: tuple[int, int, int], spacing: tuple[float, float, float]
) -> Coords:
    """Compute 1D coordinates `x, y, z` and a tiny separation `sep`.

    Small helper separated out to reduce local variable counts in the
    mesh-building function so static analyzers report fewer locals.
    """
    nx, ny, nz = shape

    x = np.arange(nx, dtype=float) * float(spacing[2])
    y = np.arange(ny, dtype=float) * float(spacing[1])
    z = np.arange(nz, dtype=float) * float(spacing[0])

    rng = max(
        float(x.max() - x.min() if x.size > 1 else 1.0),
        float(y.max() - y.min() if y.size > 1 else 1.0),
        float(z.max() - z.min() if z.size > 1 else 1.0),
    )
    sep = rng * 1e-6
    return Coords(x=x, y=y, z=z, sep=sep)


def compute_plane_xy(coords: Coords, idxs: tuple[int, int, int]) -> "PlaneXY":
    """Compute mesh arrays for the inline-crossline (XY) plane."""
    x, y, z = coords.x, coords.y, coords.z
    _, _, iz = idxs
    # Use short coordinate names (X, Y, Z) to match mathematical convention.
    # This is more readable in numerical code; disable invalid-name at the
    # module level rather than rename everywhere.
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.full_like(X, z[iz] + coords.sep)
    return PlaneXY(x=X, y=Y, z=Z)


def compute_plane_xz(coords: Coords, idxs: tuple[int, int, int]) -> "PlaneXZ":
    """Compute mesh arrays for the inline-depth (XZ) plane."""
    x, y, z = coords.x, coords.y, coords.z
    _, iy, _ = idxs
    X2, Z2 = np.meshgrid(x, z, indexing="ij")
    Y2 = np.full_like(X2, y[iy] - coords.sep)
    return PlaneXZ(x=X2, y=Y2, z=Z2)


def compute_plane_yz(coords: Coords, idxs: tuple[int, int, int]) -> "PlaneYZ":
    """Compute mesh arrays for the crossline-depth (YZ) plane."""
    x, y, z = coords.x, coords.y, coords.z
    ix, _, _ = idxs
    Y3, Z3 = np.meshgrid(y, z, indexing="ij")
    X3 = np.full_like(Y3, x[ix] + coords.sep)
    return PlaneYZ(x=X3, y=Y3, z=Z3)


# Use short coordinate variable names across this module for clarity in
# numerical code. These names are conventional in geometry/math code and
# are intentionally allowed here.


@dataclass(frozen=True)
class PlaneXY:
    """Mesh arrays for the inline-crossline (XY) plane."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class PlaneXZ:
    """Mesh arrays for the inline-depth (XZ) plane."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass(frozen=True)
class PlaneYZ:
    """Mesh arrays for the crossline-depth (YZ) plane."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


def compute_mesh_grids(coords: Coords, idxs: tuple[int, int, int]) -> MeshGrids:
    """Compute the various mesh grids used to build quad faces."""
    xy_plane = compute_plane_xy(coords, idxs)
    xz_plane = compute_plane_xz(coords, idxs)
    yz_plane = compute_plane_yz(coords, idxs)
    return MeshGrids(xy=xy_plane, xz=xz_plane, yz=yz_plane)


def compute_coords_and_grids(
    shape: tuple[int, int, int],
    idxs: tuple[int, int, int],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> CoordsAndGrids:
    """Compute coordinates and mesh grids and return them grouped."""
    coords = compute_1d_coords_and_sep(shape, spacing)
    grids = compute_mesh_grids(coords, idxs)
    return CoordsAndGrids(coords=coords, grids=grids)


def add_quads_to_lists(
    faces: list[list[tuple[float, float, float]]],
    face_colors: list[tuple[float, float, float, float]],
    plane: PlaneXY | PlaneXZ | PlaneYZ,
    face_color_array: np.ndarray,
) -> None:
    """Append quad faces and their RGBA colors to the provided lists."""
    Xa = plane.x
    Ya = plane.y
    Za = plane.z
    M, N = Xa.shape
    for i in range(M - 1):
        for j in range(N - 1):
            v0 = (float(Xa[i, j]), float(Ya[i, j]), float(Za[i, j]))
            v1 = (float(Xa[i + 1, j]), float(Ya[i + 1, j]), float(Za[i + 1, j]))
            v2 = (
                float(Xa[i + 1, j + 1]),
                float(Ya[i + 1, j + 1]),
                float(Za[i + 1, j + 1]),
            )
            v3 = (float(Xa[i, j + 1]), float(Ya[i, j + 1]), float(Za[i, j + 1]))
            faces.append([v0, v1, v2, v3])
            face_colors.append(tuple(face_color_array[i, j]))
