"""Nearest neighbor interpolator for image upsampling/downsampling.

This module provides a simple interpolator that uses nearest neighbor
resampling, serving as a baseline comparison to the NeuralSmoother.
It mirrors the NeuralSmoother API to be used as a drop-in replacement
in the generate_faciesgan_dataset tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom

from src.modeling.dataset import CustomDataset, CustomLoader


logger = logging.getLogger(__name__)


@dataclass
class InterpolatorConfig:
    """Configuration for interpolator initialization.

    This dataclass encapsulates all parameters needed to configure
    an interpolator instance, whether NeuralSmoother or NearestInterpolator.

    Attributes
    ----------
    num_classes : int | None
        Number of classes (used by NeuralSmoother, ignored by NearestInterpolator)
    steps : int
        Training steps (used by NeuralSmoother, ignored by NearestInterpolator)
    scale : float
        Scale parameter (used by NeuralSmoother, ignored by NearestInterpolator)
    upsample : int
        Upsampling factor for high-resolution rendering
    batch_size : int
        Batch size for data loading
    num_workers : int
        Number of workers for data loading
    lr : float
        Learning rate (used by NeuralSmoother, ignored by NearestInterpolator)
    scheduler_type : str
        Scheduler type: "cosine", "onecycle", "step", "plateau", or "none"
        (used by NeuralSmoother, ignored by NearestInterpolator)
    step_size : int
        Step size for step scheduler (used by NeuralSmoother, ignored by NearestInterpolator)
    gamma : float
        Gamma parameter for learning rate decay (used by NeuralSmoother, ignored by NearestInterpolator)
    patience : int
        Patience parameter for plateau scheduler (used by NeuralSmoother, ignored by NearestInterpolator)
    max_lr : float | None
        Max learning rate for onecycle scheduler (used by NeuralSmoother, ignored by NearestInterpolator)
    model_dir : str
        Model directory for saving/loading checkpoints
    force_retrain : bool
        Force retrain flag - if True, ignore cached models and retrain
        (used by NeuralSmoother, ignored by NearestInterpolator)
    """

    num_classes: int | None = None
    steps: int = 2000
    scale: float = 1.0
    upsample: int = 4
    batch_size: int = 8192
    num_workers: int = 0
    lr: float = 5e-4
    scheduler_type: str = "cosine"
    step_size: int = 100
    gamma: float = 0.1
    patience: int = 10
    max_lr: float | None = None
    model_dir: str = ".cache/models/"
    force_retrain: bool = False


@dataclass
class ProcessImageConfig:
    """Configuration for process_image method calls.

    This dataclass encapsulates parameters needed when processing an image,
    separate from the interpolator initialization configuration.

    Attributes
    ----------
    image_path : str
        Path to the input image
    image_type : str
        Type of image ("facies", "wells", "seismic")
    resolutions : list[tuple[int, int]]
        List of (height, width) tuples for output resolutions
    batch_size : int | None
        Batch size for data loading (overrides InterpolatorConfig if provided)
    num_workers : int | None
        Number of workers for data loading (overrides InterpolatorConfig if provided)
    transformers : list[Any] | None
        Optional list of transformation functions to apply to the image array
    """

    image_path: str
    image_type: str
    resolutions: list[tuple[int, int]]
    batch_size: int | None = None
    num_workers: int | None = None
    transformers: list[Any] | None = None


class BaseInterpolator:
    """Base class providing common functionality for all interpolators.

    This class provides the process_and_save method that can be used by
    both NearestInterpolator and NeuralSmoother without code duplication.

    Subclasses must define:
    - config: InterpolatorConfig attribute
    - reset(num_classes: int | None) -> None method
    - render(loader, resolutions) -> tuple method
    """

    config: InterpolatorConfig

    @staticmethod
    def create(
        image_type: str,
        config: InterpolatorConfig,
    ) -> BaseInterpolator:
        """Factory method to create the appropriate interpolator based on image type.

        Parameters
        ----------
        image_type : str
            Type of image ("facies", "wells", "seismic")
            - "facies": uses NeuralSmoother for better quality
            - "wells", "seismic": uses NearestInterpolator for speed
        config : InterpolatorConfig
            Configuration object containing all interpolator parameters.

        Returns
        -------
        BaseInterpolator
            NeuralSmoother for facies, NearestInterpolator for wells/seismic
        """
        if image_type == "facies":
            # Import here to avoid circular dependency
            from src.modeling.neural_smoother import NeuralSmoother

            logger.info(f"Creating NeuralSmoother for image type '{image_type}'")
            return NeuralSmoother(config)
        elif image_type in ("wells", "seismic"):
            logger.info(f"Creating NearestInterpolator for image type '{image_type}'")
            return NearestInterpolator(config)
        else:
            raise ValueError(
                f"Unknown image_type '{image_type}'. "
                f"Expected 'facies', 'wells', or 'seismic'."
            )

    def train(self, loader: Any) -> list[float]:
        """Train the interpolator (no-op by default).

        Parameters
        ----------
        loader : CustomLoader
            Data loader (not used by default, for API compatibility)

        Returns
        -------
        list[float]
            Empty list (no training losses)
        """
        logger.info("BaseInterpolator requires no training, skipping...")
        return []

    def reset(self, num_classes: int | None = None) -> None:
        """Reset the interpolator (no-op by default).

        Parameters
        ----------
        num_classes : int | None
            Number of classes (can be overridden by subclasses)
        """
        pass

    def _get_target_dimensions(self, loader: Any) -> tuple[int, int, int, int]:
        """Calculate native and target (super-resolution) dimensions.

        Helper method to get dimensions from loader and calculate upsampled dimensions.
        This is common logic used by both NearestInterpolator and NeuralSmoother.

        Parameters
        ----------
        loader : CustomLoader
            Data loader containing geometry information

        Returns
        -------
        tuple[int, int, int, int]
            (native_height, native_width, super_height, super_width)
        """
        native_height, native_width = loader.get_geometry()
        super_height = native_height * self.config.upsample
        super_width = native_width * self.config.upsample
        return native_height, native_width, super_height, super_width

    def render(
        self, loader: Any, resolutions: list[tuple[int, int]], image_type: str = ""
    ) -> tuple[list[NDArray[np.float32]], NDArray[np.float32] | None]:
        """Render images at specified resolutions (must be implemented by subclass).

        Parameters
        ----------
        loader : CustomLoader
            Data loader containing the original image and encoder
        resolutions : list[tuple[int, int]]
            List of (height, width) tuples for output resolutions

        Returns
        -------
        tuple[list[NDArray[np.float32]], NDArray[np.float32]]
            (smooth_imgs, high_res_img) where:
            - smooth_imgs: list of images at requested resolutions (float32, 0-1)
            - high_res_img: upsampled high-resolution image (float32, 0-1)
        """
        raise NotImplementedError("Subclasses must implement render")

    def process_image(
        self,
        config: ProcessImageConfig,
    ) -> tuple[list[NDArray[np.float32]], NDArray[np.float32] | None, list[float]]:
        """Process an image: create dataset, train, and render.

        This is the base implementation that handles the full workflow.
        Subclasses can override this if they need custom behavior.

        Parameters
        ----------
        config : ProcessImageConfig
            Configuration for processing the image

        Returns
        -------
        tuple[list[NDArray[np.float32]], NDArray[np.float32], list[float]]
            (smooth_imgs, high_res_img, losses)
        """

        # Use config values or fall back to interpolator config defaults
        batch_size = (
            config.batch_size
            if config.batch_size is not None
            else self.config.batch_size
        )
        num_workers = (
            config.num_workers
            if config.num_workers is not None
            else self.config.num_workers
        )

        # For wells and seismic (NearestInterpolator), skip training dataset creation
        # since no training is needed - only create the render dataset
        if config.image_type in ("wells", "seismic"):
            # Create dataset and loader for rendering only
            dataset_render = CustomDataset(
                str(config.image_path),
                image_type=config.image_type,
                transformers=config.transformers,
            )
            loader_render = CustomLoader(
                dataset_render,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            # No training needed for nearest neighbor interpolator
            losses: list[float] = []
        else:
            # For facies (NeuralSmoother), create both training and rendering datasets
            # Create dataset and loader for training
            dataset_train = CustomDataset(
                str(config.image_path),
                image_type=config.image_type,
                transformers=config.transformers,
            )
            loader_train = CustomLoader(
                dataset_train, batch_size=batch_size, num_workers=num_workers
            )

            # Reset and train
            num_classes = len(dataset_train.get_class_weights())
            self.reset(num_classes=num_classes)
            losses = self.train(loader_train)

            # Create dataset and loader for rendering
            dataset_render = CustomDataset(
                str(config.image_path),
                image_type=config.image_type,
                transformers=config.transformers,
            )
            loader_render = CustomLoader(
                dataset_render,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        # Render at requested resolutions
        smooth_imgs, high_res_img = self.render(
            loader_render, resolutions=config.resolutions, image_type=config.image_type
        )

        return smooth_imgs, high_res_img, losses


class NearestInterpolator(BaseInterpolator):
    """Nearest neighbor interpolator with API compatible with NeuralSmoother.

    This class provides a simple baseline interpolation method using
    nearest neighbor resampling. It implements a render() method that
    mirrors the NeuralSmoother API, making it easy to swap between
    neural and traditional interpolation methods.
    """

    def __init__(
        self,
        config: InterpolatorConfig,
    ) -> None:
        """Initialize the interpolator with configuration.

        Parameters
        ----------
        config : InterpolatorConfig
            Configuration object containing all interpolator parameters.
        """
        self.config = config
        self.trained = True
        self.loss_history: list[float] = []

    def render(
        self, loader: Any, resolutions: list[tuple[int, int]], image_type: str = ""
    ) -> tuple[list[NDArray[np.float32]], NDArray[np.float32] | None]:
        """Render images at specified resolutions using trace-wise nearest neighbor interpolation.

        Parameters
        ----------
        loader : CustomLoader
            Data loader containing the original image and encoder
        resolutions : list[tuple[int, int]]
            List of (height, width) tuples for output resolutions

        Returns
        -------
        tuple[list[NDArray[np.float32]], NDArray[np.float32]]
            (smooth_imgs, high_res_img) where:
            - smooth_imgs: list of images at requested resolutions (float32, 0-1)
            - high_res_img: upsampled high-resolution image (float32, 0-1)
        """
        smooth_imgs: list[NDArray[np.float32]] = []
        if image_type and image_type == "wells":
            # Get the first resolution (highest) image
            high_res_img = np.array(loader.get_image()).astype(np.float32) / 255.0
            smooth_imgs.append(high_res_img)

            # Get original dimensions
            height, width = high_res_img.shape[:2]

            # Find the column with the most non-black pixels (the well trace)
            non_black_counts = np.zeros(width, dtype=int)
            for col in range(width):
                non_black_counts[col] = np.sum(
                    np.any(high_res_img[:, col, :] > 0, axis=1)
                )

            well_column = int(np.argmax(non_black_counts))
            well_trace = high_res_img[:, well_column, :]  # Extract the well column

            logger.info(f"Wells: Found well column at {well_column}/{width}")

            # Process each lower resolution
            for new_h, new_w in resolutions[1:]:
                # Scale the column position proportionally
                scaled_column = int(well_column * new_w / width)
                scaled_column = min(new_w - 1, max(0, scaled_column))

                # Create output image (black background)
                downsampled = np.zeros((new_h, new_w, 3), dtype=np.float32)

                # Downsample the vertical trace using nearest neighbor
                step = height // new_h
                for i in range(new_h):
                    src_row = min(i * step, height - 1)
                    downsampled[i, scaled_column, :] = well_trace[src_row, :]

                smooth_imgs.append(downsampled)
                logger.info(
                    f"Wells: Downsampled to {new_h}x{new_w}, well at column {scaled_column}"
                )

            return smooth_imgs, high_res_img
        else:
            logger.info("Rendering with trace-wise nearest neighbor interpolation...")

            # Get dimensions using base helper method
            _, _, super_height, super_width = self._get_target_dimensions(loader)

            img_original = loader.get_image()

            img_array = np.array(img_original).astype(np.float32) / 255.0

            # Trace-wise upsampling
            high_res_img = self._upsample_image(img_array, super_height, super_width)

            # Downsample to each requested resolution

            for new_h, new_w in resolutions:
                smooth_img = self._downsample_image(high_res_img, new_h, new_w)
                smooth_imgs.append(smooth_img.astype(np.float32))

            return smooth_imgs, high_res_img.astype(np.float32)

    def _upsample_image(
        self, img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        """Upsample image using trace-wise nearest neighbor interpolation.

        Helper method that implements trace-wise (column-by-column) upsampling.

        Parameters
        ----------
        img : NDArray[np.float32]
            Input image of shape (h, w, 3)
        target_h : int
            Target height
        target_w : int
            Target width

        Returns
        -------
        NDArray[np.float32]
            Upsampled image of shape (target_h, target_w, 3)
        """
        h, w, c = img.shape

        # Step 1: Interpolate each column (trace) vertically
        upsampled_traces = np.zeros((target_h, w, c), dtype=np.float32)
        for col in range(w):
            for channel in range(c):
                # Interpolate this column vertically using nearest neighbor
                upsampled_traces[:, col, channel] = zoom(
                    img[:, col, channel], target_h / h, order=0
                )

        # Step 2: Replicate horizontally (nearest neighbor)
        output = np.zeros((target_h, target_w, c), dtype=np.float32)
        for channel in range(c):
            output[:, :, channel] = zoom(
                upsampled_traces[:, :, channel], (1, target_w / w), order=0
            )

        return output

    def _downsample_image(
        self, img: NDArray[np.float32], target_h: int, target_w: int
    ) -> NDArray[np.float32]:
        """Downsample image using trace-wise (column-by-column) interpolation.

        This implementation uses nearest neighbor interpolation applied
        trace-wise (column-by-column) for better handling of seismic data.

        Parameters
        ----------
        img : NDArray[np.float32]
            Input image of shape (h, w, 3)
        target_h : int
            Target height
        target_w : int
            Target width

        Returns
        -------
        NDArray[np.float32]
            Downsampled image of shape (target_h, target_w, 3)
        """
        h, w, c = img.shape

        # Step 1: Downsample horizontally first
        h_downsampled = np.zeros((h, target_w, c), dtype=np.float32)
        for channel in range(c):
            h_downsampled[:, :, channel] = zoom(
                img[:, :, channel], (1, target_w / w), order=0
            )

        # Step 2: Downsample each column (trace) vertically
        output = np.zeros((target_h, target_w, c), dtype=np.float32)
        for col in range(target_w):
            for channel in range(c):
                output[:, col, channel] = zoom(
                    h_downsampled[:, col, channel], target_h / h, order=0
                )

        return output
