import os
import numpy as np


def load_gslib_file(filepath, grid_shape):
    """
    Reads a GSLIB data file and reshapes it to the specified 3D grid shape.

    Args:
        filepath (str): The path to the GSLIB file.
        grid_shape (tuple): The target 3D shape (nx, ny, nz).

    Returns:
        np.ndarray: A 3D NumPy array containing the data.
    """
    # Read GSLIB format (3 header lines: title, variable count, variable name)
    with open(filepath, "r") as f:
        # Skip 3 header lines
        f.readline()  # Title
        f.readline()  # Number of variables
        f.readline()  # Variable name
        # Read data
        values = [float(line.strip()) for line in f if line.strip()]

    # Reshape the 1D array into the 3D model grid.
    # The 'F' order (Fortran-style) is crucial as GSLIB data is column-major.
    data_column = np.array(values)
    return data_column.reshape(grid_shape, order="F")


def load_stanfordsix_data(data_path, file_map, grid_shape):
    """
    Loads all required Stanford VI-E data cubes from their respective GSLIB files.

    Args:
        data_path (str): The root directory for the data files.
        file_map (dict): A dictionary mapping property keys to their folder names.
        grid_shape (tuple): The target 3D shape (nx, ny, nz).

    Returns:
        dict: A dictionary of 3D NumPy arrays for each property.
    """
    data = {}
    for key, folder_name in file_map.items():
        # Construct candidate filenames to check. Prefer an exact-spaced name
        # (e.g. 'P-wave Velocity.dat') then the underscore variant used earlier.
        dir_path = os.path.join(data_path, folder_name)
        # For the P-wave and S-wave folders prefer the specific filenames used
        # in this workspace if they exist (Pvelocity.dat, Svelocity.dat).
        candidates = [f"{folder_name}.dat", f"{folder_name.replace(' ', '_')}.dat"]
        if folder_name.lower().startswith("p-wave"):
            candidates.insert(0, "Pvelocity.dat")
        if folder_name.lower().startswith("s-wave"):
            candidates.insert(0, "Svelocity.dat")

        # Also try a compacted name without spaces as a last straightforward guess
        candidates.append("".join(folder_name.split()) + ".dat")

        full_path = None
        for fn in candidates:
            candidate_path = os.path.join(dir_path, fn)
            if os.path.exists(candidate_path):
                full_path = candidate_path
                break

        # If we did not find a candidate by exact names, fall back to heuristic search
        if full_path is None:
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(
                    f"Data folder not found: {dir_path}. Please ensure you have downloaded the Stanford VI-E data."
                )

            dat_files = [f for f in os.listdir(dir_path) if f.lower().endswith(".dat")]
            if not dat_files:
                raise FileNotFoundError(
                    f"No .dat files found in expected folder: {dir_path}. Please ensure you have downloaded the Stanford VI-E data."
                )

            # Prefer files containing the property key (e.g., 'vp' -> 'pvelocity')
            candidate = None
            for f in dat_files:
                if key.lower() in f.lower():
                    candidate = f
                    break

            # Next, prefer files that contain parts of the folder name
            if candidate is None:
                folder_compact = "".join(folder_name.lower().split())
                for f in dat_files:
                    if folder_compact in f.lower().replace("_", "").replace(
                        "-", ""
                    ).replace(" ", ""):
                        candidate = f
                        break

            # Fallback: first .dat file
            if candidate is None:
                candidate = dat_files[0]

            full_path = os.path.join(dir_path, candidate)
            print(
                f"Warning: expected one of {candidates} not found. Using data file: {full_path}"
            )

        print(f"Loading {key} from {full_path}...")
        data[key] = load_gslib_file(full_path, grid_shape)

    print(f"All data loaded successfully. Grid shape: {grid_shape}")
    return data
