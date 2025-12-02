"""Interactive well editor for spatial distribution in XZ space.

This module provides an interactive matplotlib-based tool for editing well positions
in the XZ geological space. Wells can be loaded from existing mappings, visualized
on a geological heatmap background, and edited interactively with mouse and keyboard.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = ["run_interactive_well_editor"]


def extract_image_number(img_name: str) -> int:
    """Extract the numeric index from the image filename.

    Args:
        img_name: Image name like 'xz_crossline_042' or 'xz_crossline_042_highres'

    Returns:
        The extracted number (e.g., 42), or 0 if pattern doesn't match
    """
    match = re.search(r"xz_crossline_(\d+)", img_name)
    return int(match.group(1)) if match else 0


def run_interactive_well_editor(cache_dir: str = ".cache") -> bool:
    """Run the interactive well editor.

    Loads existing wells from wells_maping.npz and displays them on an interactive
    plot over the geological heatmap background. Users can click and drag wells to
    reposition them, delete wells, and save the modified positions.

    Args:
        cache_dir: Directory containing well mapping and image data

    Returns:
        True if successful, False otherwise

    Interactive Controls:
        Mouse:
            - Click a point to select it
            - Drag a point to move it
        Keyboard:
            - 's' - Save wells to wells_maping.npz
            - 'd' - Delete selected well
            - 'h' - Show help
            - 'q' - Quit (or close window)
    """
    cache_path = Path(cache_dir)

    # Check if we have existing wells_maping.npz to load from
    wells_mapping_file = cache_path / "wells_maping.npz"
    wells_file = cache_path / "column_best_images.npz"

    if wells_mapping_file.exists():
        logger.info("Loading existing wells from: %s", wells_mapping_file)
        wells_data = np.load(wells_mapping_file)
        initial_image_numbers = wells_data["image_numbers"].tolist()
        initial_columns = wells_data["columns"].tolist()
        initial_counts = wells_data["counts"].tolist()
        initial_image_names = wells_data["image_names"].tolist()
    elif wells_file.exists():
        logger.info("Loading wells from: %s", wells_file)
        wells_data = np.load(wells_file)
        initial_columns = wells_data["columns"].tolist()
        initial_image_names = wells_data["image_names"].tolist()
        initial_counts = wells_data["counts"].tolist()
        initial_image_numbers = [
            extract_image_number(name) for name in initial_image_names
        ]
    else:
        logger.error("Neither %s nor %s found!", wells_mapping_file, wells_file)
        logger.error(
            "Please run the full analysis script first to generate well positions."
        )
        return False

    # Load or generate the heatmap background
    analysis_file = cache_path / "column_analysis.npz"
    if analysis_file.exists():
        logger.info("Loading heatmap from: %s", analysis_file)
        analysis_data = np.load(analysis_file)
        all_column_counts: NDArray[np.int_] = analysis_data["column_counts"]
    else:
        logger.info("Generating heatmap from images...")
        # Generate heatmap from pyramid images
        from PIL import Image

        image_dir = cache_path / "images" / "pyramids" / "facies" / "1024x1024"
        image_files = sorted(image_dir.glob("xz_crossline_*.png"))

        if not image_files:
            logger.error("No images found in %s", image_dir)
            logger.error("Cannot generate heatmap background.")
            return False

        all_column_counts_list: list[NDArray[np.int_]] = []
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            width = img_array.shape[1]

            # Count non-black pixels in each column
            non_black_counts = np.zeros(width, dtype=int)
            for col in range(width):
                non_black_counts[col] = np.sum(np.any(img_array[:, col, :] > 0, axis=1))

            all_column_counts_list.append(non_black_counts)

        all_column_counts = np.array(all_column_counts_list)
        logger.info("Generated heatmap: %s", all_column_counts.shape)

    num_images = all_column_counts.shape[0]
    num_columns = all_column_counts.shape[1]

    logger.info("Loaded %d wells", len(initial_columns))
    logger.info("Heatmap dimensions: %d images × %d columns", num_images, num_columns)

    # Store well data in mutable structure for editing
    well_data = {
        "image_numbers": initial_image_numbers.copy(),
        "columns": initial_columns.copy(),
        "counts": initial_counts.copy(),
        "image_names": initial_image_names.copy(),
    }

    # Create the interactive plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Show the geological heatmap as background
    background = ax.imshow(
        all_column_counts.T,
        aspect="auto",
        cmap="Greys",
        interpolation="nearest",
        origin="lower",
        extent=[0, num_images, 0, num_columns],
        alpha=0.7,
    )

    # Overlay scatter plot of wells
    scatter = ax.scatter(
        well_data["image_numbers"],
        well_data["columns"],
        c=well_data["counts"],
        cmap="viridis",
        s=100,  # Larger for easier clicking
        alpha=0.9,
        edgecolors="red",
        linewidth=1.5,
        picker=True,
        pickradius=5,
    )

    # Draw grid for reference (20x20)
    num_grid_bins = 20
    for i in range(num_grid_bins + 1):
        img_boundary = i * num_images / num_grid_bins
        ax.axvline(
            x=img_boundary, color="red", linestyle="--", alpha=0.3, linewidth=0.8
        )

        col_boundary = i * num_columns / num_grid_bins
        ax.axhline(
            y=col_boundary, color="red", linestyle="--", alpha=0.3, linewidth=0.8
        )

    ax.set_xlabel("Image Number (Z-axis in XZ space)", fontsize=12)
    ax.set_ylabel("Column Position (X-axis in XZ space)", fontsize=12)
    ax.set_title(
        "Interactive Well Editor - Spatial Distribution in XZ Space",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, num_images)
    ax.set_ylim(0, num_columns)

    # Add colorbars
    fig.colorbar(
        background,
        ax=ax,
        label="Background: Pixel Density",
        location="left",
        pad=0.02,
        shrink=0.8,
    )
    fig.colorbar(
        scatter,
        ax=ax,
        label="Wells: Non-Black Pixel Count",
        location="right",
        pad=0.02,
        shrink=0.8,
    )

    # Interactive editing state
    selected_point: dict[str, Any] = {"index": None, "dragging": False}
    annotation_holder: dict[str, Any] = {"annotation": None}

    def on_pick(event: Any) -> None:
        """Handle point selection"""
        if event.artist != scatter:
            return

        ind: int = event.ind[0]
        selected_point["index"] = ind
        selected_point["dragging"] = False

        # Show annotation
        if annotation_holder["annotation"]:
            annotation_holder["annotation"].remove()

        img_num: int = well_data["image_numbers"][ind]
        col: int = well_data["columns"][ind]
        count: int = well_data["counts"][ind]
        img_name = well_data["image_names"][ind]

        annotation_holder["annotation"] = ax.annotate(
            f"Well {ind}\n{img_name}\nImg: {img_num}, Col: {col}\nPixels: {count}",
            xy=(img_num, col),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )
        fig.canvas.draw_idle()

    def on_motion(event: Any) -> None:
        """Handle point dragging"""
        if selected_point["index"] is None or not selected_point["dragging"]:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Update position
        ind = selected_point["index"]
        new_img = int(np.clip(event.xdata, 0, num_images - 1))
        new_col = int(np.clip(event.ydata, 0, num_columns - 1))

        well_data["image_numbers"][ind] = new_img
        well_data["columns"][ind] = new_col

        # Update scatter plot data
        offsets = scatter.get_offsets()
        offsets[ind] = [new_img, new_col]
        scatter.set_offsets(offsets)

        # Update annotation
        annotation = annotation_holder["annotation"]
        if annotation:
            annotation.xy = (new_img, new_col)
            annotation.set_text(
                f"Well {ind}\n{well_data['image_names'][ind]}\n"
                f"Img: {new_img}, Col: {new_col}\n"
                f"Pixels: {well_data['counts'][ind]}"
            )

        fig.canvas.draw_idle()

    def on_button_press(event: Any) -> None:
        """Start dragging"""
        if selected_point["index"] is not None and event.inaxes == ax:
            selected_point["dragging"] = True

    def on_button_release(event: Any) -> None:
        """Stop dragging"""
        if selected_point["dragging"]:
            selected_point["dragging"] = False
            idx = selected_point["index"]
            if idx is not None and isinstance(idx, int):
                logger.info(
                    "Updated well %s: Image %s, Column %s",
                    idx,
                    int(well_data["image_numbers"][idx]),
                    int(well_data["columns"][idx]),
                )

    def on_key(event: Any) -> None:
        """Handle keyboard shortcuts"""
        if event.key == "s":
            # Save to wells_maping.npz
            save_wells_mapping()
        elif event.key == "d" and selected_point["index"] is not None:
            # Delete selected point
            ind = selected_point["index"]
            well_data["image_numbers"].pop(ind)
            well_data["columns"].pop(ind)
            well_data["counts"].pop(ind)
            well_data["image_names"].pop(ind)

            # Update scatter plot
            scatter.set_offsets(
                list(zip(well_data["image_numbers"], well_data["columns"]))
            )
            scatter.set_array(np.array(well_data["counts"]))

            annotation = annotation_holder["annotation"]
            if annotation:
                annotation.remove()
                annotation_holder["annotation"] = None

            selected_point["index"] = None
            logger.info(
                "Deleted well %d. Total wells: %d", ind, len(well_data["image_numbers"])
            )
            fig.canvas.draw_idle()
        elif event.key == "h":
            # Show help
            show_help()

    def save_wells_mapping() -> None:
        """Save the current well positions to wells_maping.npz"""
        output_file = cache_path / "wells_maping.npz"

        # Create mapping: for each well, store (image_index, column_index)
        mapping: list[tuple[int, int]] = []
        for img_num, col in zip(well_data["image_numbers"], well_data["columns"]):
            mapping.append((img_num, col))

        mapping_array = np.array(mapping, dtype=np.int32)

        np.savez(
            output_file,
            wells=mapping_array,  # Shape: (n_wells, 2) where each row is (image_idx, column_idx)
            image_numbers=np.array(well_data["image_numbers"], dtype=np.int32),
            columns=np.array(well_data["columns"], dtype=np.int32),
            image_names=np.array(well_data["image_names"]),
            counts=np.array(well_data["counts"], dtype=np.int32),
        )

        logger.info("=" * 60)
        logger.info("✓ Saved %d wells to: %s", len(mapping), output_file)
        logger.info("  Format: wells array shape %s", mapping_array.shape)
        logger.info("  Each row: (image_index, column_index)")
        logger.info("=" * 60)

    def show_help() -> None:
        """Display help text"""
        help_text = """
╔════════════════════════════════════════════════════════════╗
║        Interactive Well Editor - Keyboard Shortcuts       ║
╠════════════════════════════════════════════════════════════╣
║  s - Save wells to .cache/wells_maping.npz                 ║
║  d - Delete selected well                                  ║
║  h - Show this help                                        ║
║  q - Quit (or close window)                                ║
╠════════════════════════════════════════════════════════════╣
║                    Mouse Controls:                         ║
║  • Click a point to select it                              ║
║  • Drag a point to move it                                 ║
╚════════════════════════════════════════════════════════════╝
        """
        print(help_text)

    # Connect event handlers
    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_press_event", on_button_press)
    fig.canvas.mpl_connect("button_release_event", on_button_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Add instruction text at bottom
    instruction_text = (
        "INTERACTIVE MODE: Click to select, drag to move wells | "
        "Press 's' to save, 'd' to delete, 'h' for help, 'q' to quit"
    )
    fig.text(
        0.5,
        0.02,
        instruction_text,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()

    # Show help on startup
    show_help()

    # Display the interactive plot
    plt.show()

    return True
