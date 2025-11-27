import logging
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.optim as optim
from numpy.typing import NDArray
from PIL import Image
from torch.nn import (
    GELU,
    CrossEntropyLoss,
    LayerNorm,
    Linear,
    Module,
    Sequential,
)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Module logger
logger = logging.getLogger(__name__)


def get_mgrid(height: int, width: int) -> torch.Tensor:
    """Create a flattened meshgrid of normalized coordinates in [-1, 1].

    This is the module-level variant extracted from `NeuralSmoother.get_mgrid`.
    Keeping a top-level function makes it easier for other modules to import
    and reuse the grid logic without constructing a `NeuralSmoother`.
    """
    tensors = (
        torch.linspace(-1, 1, steps=height),
        torch.linspace(-1, 1, steps=width),
    )
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    return mgrid.reshape(-1, 2)


def set_seed(seed: int = 42) -> None:
    """Set seeds for torch, numpy and python.random at module level.

    Keeping a module-level `set_seed` makes it easy for other modules to
    call it without constructing a `NeuralSmoother` instance. The
    `NeuralSmoother.set_seed` method remains as a thin wrapper that
    delegates to this function to preserve backward compatibility.
    """
    seed_int = int(seed)
    torch.manual_seed(seed_int)  # pyright: ignore
    np.random.seed(seed_int)
    random.seed(seed_int)


def resolve_device() -> torch.device:
    """Return the preferred device: MPS (Apple), CUDA, or CPU.

    Exposed at module level so other modules can determine the best device
    without constructing a `NeuralSmoother` instance.
    """
    if (
        getattr(torch.backends, "mps", None) is not None
        and getattr(torch.backends.mps, "is_available", lambda: False)()
        and getattr(torch.backends.mps, "is_built", lambda: True)()
    ):
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ColorEncoder:
    def __init__(
        self, img_array: NDArray[Any], device: torch.device = resolve_device()
    ):
        # Ensure we work with float32 NumPy arrays to avoid creating
        # torch.float64 tensors which are not supported on MPS devices.
        pixels = img_array.reshape(-1, 3).astype(np.float32, copy=False)
        self.palette = np.unique(pixels, axis=0).astype(np.float32, copy=False)
        self.num_classes = len(self.palette)
        self.device = device
        # Use torch.from_numpy to preserve dtype (float32) and avoid creating
        # float64 tensors that MPS cannot handle.
        try:
            self.palette_tensor = torch.from_numpy(  # pyright: ignore
                self.palette,
            ).to(self.device)
        except Exception:
            # Fall back to generic constructor but force float32
            self.palette_tensor = torch.tensor(self.palette, dtype=torch.float32).to(
                self.device
            )
        logger.info(f"Detected {self.num_classes} unique facies classes.")

    def rgb_to_labels(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Convert an [N,3] RGB tensor to label indices using the encoder palette.

        Args:
            img_tensor: Tensor of shape [N, 3] with RGB values (0..1) on same device
                        as the encoder.palette_tensor.
        Returns:
            torch.Tensor: Long tensor with shape [N] with the index of closest palette
                          color for each input pixel.
        """
        dists = torch.cdist(img_tensor, self.palette_tensor)
        return torch.argmin(dists, dim=1)

    def labels_to_rgb(self, label_tensor: torch.Tensor) -> torch.Tensor:
        """Map label indices back to RGB values from the palette.

        Args:
            label_tensor: Tensor of shape [N] containing class indices (ints).
        Returns:
            Tensor of shape [N, 3] with RGB colors as floats.
        """
        return self.palette_tensor[label_tensor.long()]

    def get_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Automatically calculates inverse frequency weights.
        Rare classes (thin beds) get higher weights.
        """
        # Work with torch tensors directly to avoid numpy type inference issues
        # Move labels to CPU and ensure integer dtype
        labels_cpu = labels.detach().cpu().long()

        # Use torch.bincount to get per-class counts (safe and faster)
        counts = torch.bincount(labels_cpu, minlength=self.num_classes).float()
        total = labels_cpu.numel()

        # Weight = Total / (Num_Classes * Count)
        # Add small epsilon to avoid division by zero and keep float math
        eps = 1e-6
        weights = total / (self.num_classes * (counts + eps))

        # Normalize so mean is roughly 1.0
        weights = weights / weights.mean()

        # Log rounded weights for user information
        try:
            # Prefer a numpy -> list conversion with explicit float dtype
            rounded_arr = (  # pyright: ignore
                weights.cpu().numpy().round(2).astype(float)  # pyright: ignore
            )
            rounded: list[float] = rounded_arr.tolist()
        except Exception:
            rounded = weights.tolist()  # pyright: ignore
        logger.info(f"Auto-calculated Class Weights: {rounded}")

        return weights.to(self.device).float()


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
    ) -> None:
        """Create a dataset from a single image path.

        Parameters
        ----------
        image: str | os.PathLike
            Path to the input image used to construct coordinates and labels.
        device: torch.device
            Device where encoder tensors will be allocated.
        """

        self.device = device
        self.image: str = str(image)
        self.id: str = f"{Path(self.image).stem}"
        self.image_type: str = image_type
        self._load_image()

    def _load_image(self) -> None:
        img_pil = Image.open(self.image).convert("RGB")
        # Cast to float32 BEFORE dividing so the resulting array is float32
        img_np = np.array(img_pil).astype(np.float32, copy=False) / 255.0
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


class NeuralSmoother:
    """Encapsulates utility helpers and the main training/rendering engine.

    The original module-level functions `set_seed`, `get_mgrid` and
    `train` have been moved here. Thin module-level wrappers
    below keep API compatibility."""

    def __init__(
        self,
        num_classes: int = 4,
        steps: int = 2000,
        scale: float = 1.0,
        upsample: int = 4,
        batch_size: int = 8192,
        num_workers: int = 0,
        lr: float = 5e-4,
        scheduler_type: str = "cosine",
        step_size: int = 100,
        gamma: float = 0.1,
        patience: int = 10,
        max_lr: float | None = None,
        model_dir: str = ".cache/models/",
        force_retrain: bool = False,
    ) -> None:
        # resolved device for the instance (MPS/CUDA/CPU)
        self.device: torch.device = resolve_device()
        self.num_classes: int = int(num_classes)
        self.model = ResidualMLP(num_classes=self.num_classes, scale=scale).to(
            self.device
        )

        self.criterion = CrossEntropyLoss()
        self.steps: int = int(steps)
        self.scale: float = float(scale)
        self.upsample: int = int(upsample)
        self.batch_size: int = int(batch_size)
        self.num_workers: int = int(num_workers)
        self.lr: float = float(lr)
        self.scheduler_type: str = scheduler_type
        self.step_size: int = int(step_size)
        self.gamma: float = float(gamma)
        self.patience: int = int(patience)
        self.max_lr: float | None = max_lr
        self.model_dir: Path = Path(model_dir)
        self.force_retrain: bool = bool(force_retrain)
        self.chunk_size: int = 65536
        self.trained = False
        self.loss_history = []

    def reset(self, num_classes: int | None = None) -> None:
        """Reset the NeuralSmoother to an untrained initial state.

        This recreates the model instance (with the same `num_classes` and
        `scale`) on the resolved device, clears optimizer/scheduler/criterion
        and loss history, and marks the instance as not trained. It does NOT
        remove any on-disk checkpoints; those are left intact.
        """

        self.model = ResidualMLP(
            num_classes=num_classes or self.num_classes, scale=self.scale
        ).to(self.device)

        self._init_optimizer()
        self._init_scheduler()
        self.loss_history.clear()
        self.criterion.weight = None

        # Mark as untrained
        self.trained = False

    def _state_from_checkpoint(self, state: Mapping[str, Any]) -> dict[str, Any] | None:
        """Normalize common checkpoint shapes into a dict[str, Any].

        This is intentionally short and permissive: we accept Mapping checkpoints
        that contain `model_state` or `state_dict`, mapping-like state dicts
        (name->tensor), or any iterable convertible to `dict()`.
        """
        try:
            ms = state.get("model_state", state.get("state_dict", state))
            normalized = {str(k): v for k, v in dict(ms).items()}
            return normalized
        except Exception:
            return None

    def _compile_model(self) -> None:
        """Attempt to JIT/compile the model when supported (skip on MPS)."""
        try:
            if self.device.type == "mps":
                logger.info("Skipping torch.compile() on MPS (known Inductor issues)")
            else:
                # torch.compile may not be available on all torch builds; guard it
                try:
                    self.model = torch.compile(self.model)  # pyright: ignore
                    logger.info("Model compiled with torch.compile()")
                except Exception:
                    logger.info("torch.compile() not available or failed; continuing")
        except Exception:
            logger.info("Failed checking device for compilation; continuing")

    def _restore(self, model_checkpoint_Path: Path) -> bool:
        """Try to restore model (and optional loss history) from checkpoint.

        Returns True when a checkpoint was successfully loaded (training can be
        skipped); False otherwise.
        """
        if not (model_checkpoint_Path.exists() and not self.force_retrain):
            return False

        try:
            state = torch.load(str(model_checkpoint_Path), map_location=self.device)
            ms: dict[str, Any] | None = self._state_from_checkpoint(state)
            if ms is None:
                raise RuntimeError(
                    "Unable to interpret checkpoint state as model state"
                )

            # Load model state
            self.model.load_state_dict(ms)  # pyright: ignore
            logger.info(
                f"Loaded model checkpoint from {model_checkpoint_Path}; skipping training."
            )
            try:
                self.loss_history = list(state["loss_history"])  # pyright: ignore
            except Exception:
                self.loss_history = []
            self.trained = True
            return True
        except Exception as e:
            logger.info(
                f"Failed loading checkpoint {model_checkpoint_Path}: {e}; will train from scratch."
            )
            self.trained = False
            return False

    def _init_optimizer(
        self,
    ) -> None:
        """Create optimizer and LR scheduler based on instance config.

        Returns (optimizer, scheduler). Splitting into a return form makes the
        helpers easier to unit-test and composes cleanly with the orchestrator.
        """
        # Create optimizer and delegate scheduler construction to helper
        self.optimizer: torch.optim.Optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr  # pyright: ignore
        )

    def _init_scheduler(self) -> None:
        """Construct a learning-rate scheduler based on configuration.

        Extracted from _create_optimizer_and_scheduler to isolate scheduler
        construction and avoid referencing an undefined local variable.
        """

        sched_type = self.scheduler_type
        match sched_type:

            case "step":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=max(1, int(self.step_size)),
                    gamma=self.gamma,
                )
            case "reduce":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=self.gamma,
                    patience=self.patience,
                )
            case "onecycle":
                max_lr = self.max_lr if self.max_lr is not None else self.lr * 10
                total_steps = max(1, int(self.steps))
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=max_lr, total_steps=total_steps
                )
            case _:
                t_max = max(1, int(self.steps))
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=t_max
                )

    def _init_model(self, class_weights: torch.Tensor, model_Path: Path) -> None:
        """Orchestrate model compilation, optional restore, and optimizer setup.

        This method delegates detailed work to small helpers to keep the
        responsibilities clear and testable.
        """
        # 1) Try to compile the model where supported
        self._compile_model()

        # 2) Attempt to restore model (and loss history) from checkpoint
        restored = self._restore(model_Path)

        # 3) Create optimizer and scheduler using instance config
        self._init_optimizer()
        self._init_scheduler()

        # 4) Setup training criterion (use class weights if supplied)
        self.criterion.weight = class_weights

        # Ensure loss_history exists (may have been restored)
        if (
            not hasattr(self, "loss_history")
            or self.loss_history is None  # pyright: ignore
        ):
            self.loss_history = []

        # If restore succeeded, `self.trained` was set by _restore_from_checkpoint.
        # Otherwise we ensure training will run.
        if not restored:
            self.trained = False

    def _train_loop(self, loader: CustomLoader) -> None:
        """Perform the training loop and return loss history.

        This variant reads common training parameters from the instance so
        callers needn't pass `steps`, `pin_memory` or `device` around.
        """

        self.model.train()  # pyright: ignore[reportFunctionMemberAccess]

        self.loss_history: list[float] = []
        steps = int(self.steps)
        pbar = tqdm(range(steps))
        loader_iter = iter(loader)
        for i in pbar:
            try:
                batch_coords, batch_labels = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch_coords, batch_labels = next(loader_iter)

            batch_coords = batch_coords.to(self.device, non_blocking=loader.pin_memory)
            batch_labels = batch_labels.to(self.device, non_blocking=loader.pin_memory)
            self.optimizer.zero_grad()
            logits = self.model(batch_coords)
            loss = self.criterion(logits, batch_labels)
            loss.backward()
            self.optimizer.step()  # pyright: ignore

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss.item())  # pyright: ignore
            else:
                self.scheduler.step()  # pyright: ignore

            self.loss_history.append(loss.item())
            if i % 100 == 0:
                pbar.set_description(f"Loss: {loss.item():.5f}")
        self.trained = True

    def render(
        self, loader: CustomLoader, resolutions: list[tuple[int, int]]
    ) -> tuple[list[NDArray[np.float32]], NDArray[np.float32]]:
        """Evaluate `model` at low-res, upsample probability maps, return smooth image.

        This extracts the inference strategy used in training so it can be
        reused or tested separately. Renamed from `render_high_res` to the
        simpler `render` API.
        """
        assert self.trained, "Model not trained for rendering"

        logger.info("Rendering High-Resolution Output...")
        native_height, native_width = loader.get_geometry()
        super_height, super_width = (
            native_height * self.upsample,
            native_width * self.upsample,
        )

        # new_h, new_w = int(height * self.upsample), int(width * self.upsample)
        new_h, new_w = 16, 16
        self.model.eval()  # pyright: ignore

        inference_ctx = (
            torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        )
        with inference_ctx():
            # choose a device consistent with the model if possible so we
            # don't send coords to GPU while the model is on CPU (or vice versa)
            try:
                # model is guaranteed to be non-None by the logic above
                first_param = next(self.model.parameters(), None)  # pyright: ignore
                if first_param is not None:
                    run_device = first_param.device
                else:
                    first_buf = next(self.model.buffers(), None)  # pyright: ignore
                    run_device = (
                        first_buf.device if first_buf is not None else self.device
                    )
            except Exception:
                run_device = self.device

            coords = get_mgrid(height=super_height, width=super_width).to(run_device)

            logits_chunks: list[torch.Tensor] = []
            for i in range(0, coords.shape[0], self.chunk_size):
                chunk = coords[i : i + self.chunk_size]
                logits_chunks.append(self.model(chunk))

            logits = torch.cat(logits_chunks, dim=0)  # [H*W, C]

            probs = torch.softmax(logits, dim=1)
            probs = (
                probs.reshape(super_height, super_width, -1)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            labels = torch.argmax(probs.squeeze(0), dim=0)
            labels = labels.to(self.device)

            # 7. Convert to RGB
            pred_rgb = loader.encoder.labels_to_rgb(labels)

            pred_rgb = pred_rgb.detach().cpu().reshape(super_height, super_width, 3)
            high_res_img: NDArray[  # pyright: ignore
                np.float32
            ] = pred_rgb.numpy().astype(  # pyright: ignore
                np.float32
            )

            palette = loader.encoder.palette_tensor.to(run_device).float()
            smooth_imgs: list[NDArray[np.float32]] = []

            for new_h, new_w in resolutions:

                inter_probs = F.interpolate(
                    probs,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                inter_probs = (
                    inter_probs.squeeze(0)
                    .permute(1, 2, 0)
                    .reshape(-1, inter_probs.shape[1])
                )

                inter_probs = inter_probs.to(self.device)
                pred_rgb = torch.matmul(inter_probs, palette)
                smooth_img = (  # pyright: ignore
                    pred_rgb.detach().cpu().reshape(new_h, new_w, 3)  # pyright: ignore
                )
                smooth_img = smooth_img.numpy().astype(np.float32)  # pyright: ignore
                smooth_imgs.append(smooth_img)  # pyright: ignore

        return smooth_imgs, high_res_img

    def train(self, loader: CustomLoader) -> list[float]:
        """Train the ResidualMLP using one or more training images.

        Parameters
        ----------
        train_images:
            Either a single image path (str/PathLike) or a list of image paths
            to use for training. Backwards-compatible with the previous
            single-image API.
        render_images:
            Optional list of image paths to render after training (not
            automatically saved). Rendering is performed separately via
            `ns.render(...)` by callers; this parameter is accepted for
            convenience but not used internally here.

        Returns
        -------
        list[float]
            The training loss history.
        """

        logger.info(f"Device: {self.device} | Scale: {self.scale}")

        model_file = f"{loader.get_id()}.pt"
        img_type = loader.get_image_type()
        class_weights = loader.get_class_weights()
        model_Path = self.model_dir / img_type / model_file
        model_Path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize model/optimizer/scheduler and store artifacts on self
        self._init_model(class_weights, model_Path)

        # --- C. Training ---
        logger.info(
            f"Training ResMLP for {self.steps} steps (batch_size={self.batch_size})..."
        )

        if self.trained:
            logger.info("Skipping training (checkpoint restored). Training skipped")
        else:
            self._train_loop(loader)

            try:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                state: dict[str, Any] = {
                    "model_state": self.model.state_dict(),  # pyright: ignore
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "loss_history": self.loss_history,
                    "meta": {
                        "steps": self.steps,
                        "scale": self.scale,
                        "batch_size": self.batch_size,
                    },
                }
                torch.save(state, str(model_Path))
                logger.info(f"Saved model checkpoint to {model_Path}")
            except Exception as e:
                logger.info(f"Failed to save checkpoint {model_Path}: {e}")

        return self.loss_history


# ==========================================
# 2. IMPROVED ARCHITECTURE: Residual MLP
# ==========================================
class FourierFeatureTransform(Module):
    def __init__(self, mapping_size: int = 256, scale: float = 10.0) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        # 'scale' is the "sigma". Higher = sharper/noisier. Lower = smoother/blurrier.
        # store as a buffer (not a trainable parameter) to avoid showing up in optimizer
        B = torch.randn(2, mapping_size) * scale
        self.register_buffer("B", B, persistent=True)
        # self.B = Parameter(torch.randn(2, mapping_size) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure numeric constant is a Python float so the result of the
        # multiplication with a torch.Tensor is a torch.Tensor. This helps
        # static type-checkers resolve the expression type (avoid 'Unknown').
        factor: float = float(2.0 * np.pi)
        # Use a locally-cast buffer to satisfy static type-checkers which may
        # infer an incorrect union type for registered buffers (e.g. Tensor | Module)
        B_tensor: torch.Tensor = cast(torch.Tensor, getattr(self, "B"))
        # Use torch.matmul to make the tensor operation explicit for static type checkers
        x_proj: torch.Tensor = torch.matmul(x * factor, B_tensor)
        # x_proj = (2.0 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualMLP(Module):
    """
    IMPROVEMENT 1: Residual Skip Connections + LayerNorm
    This structure is similar to the NeRF architecture.
    """

    def __init__(
        self,
        num_classes: int,
        mapping_size: int = 128,
        scale: float = 1.0,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.fourier = FourierFeatureTransform(mapping_size, scale)
        input_dim = mapping_size * 2

        # Standard layers
        self.layer1 = Sequential(
            Linear(input_dim, hidden_dim),
            LayerNorm(hidden_dim),
            GELU(),  # Smoother than ReLU
        )
        self.layer2 = Sequential(
            Linear(hidden_dim, hidden_dim), LayerNorm(hidden_dim), GELU()
        )

        # Skip connection point: We concat input_dim + hidden_dim
        self.skip_layer = Sequential(
            Linear(hidden_dim + input_dim, hidden_dim),
            LayerNorm(hidden_dim),
            GELU(),
        )

        self.layer3 = Sequential(
            Linear(hidden_dim, hidden_dim), LayerNorm(hidden_dim), GELU()
        )

        self.output = Linear(hidden_dim, num_classes)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # Embed coordinates
        x_emb = self.fourier(coords)

        # First block
        h = self.layer1(x_emb)
        h = self.layer2(h)

        # Skip connection
        h = torch.cat([h, x_emb], dim=-1)
        h = self.skip_layer(h)

        # Final block
        h = self.layer3(h)
        return self.output(h)
