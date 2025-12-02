"""Custom dataset and data loader for image-based neural network training.

This module provides dataset and loader implementations for processing images
with optional transformations, color encoding, and batching support.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader

from src.modeling.color_encoder import ColorEncoder
from src.modeling.utils import get_mgrid, resolve_device

logger = logging.getLogger(__name__)


class CustomDataset:
    """Lightweight dataset wrapping coordinate and label tensors for a single image.

    This simplified dataset stores one image, computes a ColorEncoder palette,
    and exposes coordinate and label tensors suitable for a DataLoader.
    """

    def __init__(
        self,
        image: str | os.PathLike[str],
        image_type: str = "facies",
        device: torch.device = resolve_device(),
        transformers: (
            list[Callable[[NDArray[np.float32]], NDArray[np.float32]]] | None
        ) = None,
    ) -> None:
        """Create a dataset from a single image path.

        Parameters
        ----------
        image: str | os.PathLike
            Path to the input image used to construct coordinates and labels.
        device: torch.device
            Device where encoder tensors will be allocated.
        transformers: list[Callable] | None
            Optional list of transformation functions to apply to the image array
            after loading. Each callable should accept and return an NDArray[np.float32].
        """

        self.device = device
        self.image_path: str = str(image)
        self.id: str = f"{Path(self.image_path).stem}"
        self.image_type: str = image_type
        self.transformers = transformers or []

        self._load_image()

    def _load_image(self) -> None:
        self.img_pil: Image.Image = Image.open(self.image_path).convert("RGB")
        # Cast to float32 BEFORE dividing so the resulting array is float32
        img_np = np.array(self.img_pil).astype(np.float32, copy=False) / 255.0

        # Apply transformers if any
        for transformer in self.transformers:
            img_np = transformer(img_np)
            self.img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

        self.h, self.w, _ = img_np.shape

        # Always construct encoder on the instance device
        self.encoder = ColorEncoder(img_np, device=self.device)
        self.coords = get_mgrid(self.h, self.w).cpu()
        # Create tensor from NumPy (preserves dtype) and move to device.
        pixels = torch.from_numpy(  # pyright: ignore
            img_np.reshape(-1, 3),
        ).to(self.device)
        # Ensure pixels are float32 on device
        if pixels.dtype != torch.float32:
            pixels = pixels.to(dtype=torch.float32)
        self.labels = self.encoder.rgb_to_labels(pixels).cpu()

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # No image switching: the dataset represents a single image.
        return self.coords[idx], self.labels[idx]

    def labels_to_rgb(self, labels: torch.Tensor) -> torch.Tensor:
        """Utility to map label indices to RGB using the attached `ColorEncoder`.

        Raises a RuntimeError if no encoder is attached.
        """
        return self.encoder.labels_to_rgb(labels)

    def get_class_weights(self) -> torch.Tensor:
        """Utility to get class weights using the attached `ColorEncoder`.

        Raises a RuntimeError if no encoder is attached.
        """
        return self.encoder.get_class_weights(self.labels)


class CustomLoader:
    """A small wrapper around torch.utils.data.DataLoader.

    Purpose:
    - Centralize DataLoader construction logic (pin_memory, multiprocessing
      context, persistent workers).
    - Provide a stable iterable with a `shutdown()` helper for tests or
      deterministic shutdown in long-running processes.

    The wrapper delegates iteration and len() to the underlying DataLoader so
    it is a drop-in replacement for code expecting a DataLoader.
    """

    def __init__(
        self,
        dataset: CustomDataset,
        *,
        batch_size: int = 8192,
        shuffle: bool = True,
        num_workers: int = 8,
        device: torch.device = resolve_device(),
        multiprocessing_context: Any | None = None,
        persistent_workers: bool = False,
    ) -> None:

        self.pin_memory: bool = True if device.type == "cuda" else False
        self.encoder = dataset.encoder
        self.dataset = dataset

        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": int(batch_size),
            "shuffle": bool(shuffle),
            "num_workers": int(num_workers),
            "pin_memory": bool(self.pin_memory),
            "persistent_workers": bool(persistent_workers),
        }
        if multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context

        # store the constructed DataLoader
        self._dl = DataLoader(**loader_kwargs)

    def __iter__(self):
        return iter(self._dl)

    def __len__(self) -> int:
        try:
            return len(self._dl)
        except Exception:
            # Some DataLoader variants may not implement __len__ reliably
            return 0

    def shutdown(self) -> None:
        """Attempt to cleanly shutdown worker processes (best-effort)."""
        try:
            # DataLoader exposes a private helper to shutdown workers; call it
            # if present to avoid leaking processes when tests create many loaders.
            shutdown = getattr(self._dl, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
        except Exception:
            pass

    def get_class_weights(self) -> torch.Tensor:
        """Utility to get class weights using the attached `ColorEncoder`.

        Raises a RuntimeError if no encoder is attached.
        """
        return self.encoder.get_class_weights(
            cast(CustomDataset, self._dl.dataset).labels
        )

    def get_geometry(self) -> tuple[int, int]:
        """Utility to get geometry (height, width) using the attached `ColorEncoder`.

        Raises a RuntimeError if no encoder is attached.
        """
        dataset = cast(CustomDataset, self._dl.dataset)
        return dataset.h, dataset.w

    def get_id(self) -> str:
        """Utility to get dataset ID using the attached `ColorEncoder`."""
        dataset = cast(CustomDataset, self._dl.dataset)
        return dataset.id

    def get_image_type(self) -> str:
        """Utility to get dataset image type using the attached `ColorEncoder`."""
        dataset = cast(CustomDataset, self._dl.dataset)
        return dataset.image_type

    def get_image(self) -> Image.Image:
        """Utility to get original image array using the attached `ColorEncoder`."""
        dataset = cast(CustomDataset, self._dl.dataset)
        return dataset.img_pil
