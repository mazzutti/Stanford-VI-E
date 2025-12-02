import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, TypeAlias, cast

import numpy as np
import torch
import torch.optim as optim
from numpy.typing import NDArray
from torch.nn import (
    GELU,
    CrossEntropyLoss,
    LayerNorm,
    Linear,
    Module,
    Sequential,
)
import torch.nn.functional as F
from tqdm import tqdm

from src.modeling.dataset import CustomLoader
from src.modeling.interpolator import BaseInterpolator, InterpolatorConfig
from src.modeling.utils import get_mgrid, resolve_device

FileLike: TypeAlias = str | os.PathLike[str] | IO[bytes]

# Module logger
logger = logging.getLogger(__name__)


class NeuralSmoother(BaseInterpolator):
    """Encapsulates utility helpers and the main training/rendering engine.

    The original module-level functions `set_seed`, `get_mgrid` and
    `train` have been moved here. Thin module-level wrappers
    below keep API compatibility."""

    def __init__(
        self,
        config: InterpolatorConfig,
    ) -> None:
        """Initialize NeuralSmoother with configuration.

        Parameters
        ----------
        config : InterpolatorConfig
            Configuration object containing all interpolator parameters.
        """
        self.config = config
        # resolved device for the instance (MPS/CUDA/CPU)
        self.device: torch.device = resolve_device()
        self.num_classes: int = int(config.num_classes or 4)
        self.model = ResidualMLP(num_classes=self.num_classes, scale=config.scale).to(
            self.device
        )

        self.criterion = CrossEntropyLoss()
        self.steps: int = int(config.steps)
        self.scale: float = float(config.scale)
        self.upsample: int = int(config.upsample)
        self.batch_size: int = int(config.batch_size)
        self.num_workers: int = int(config.num_workers)
        self.lr: float = float(config.lr)
        self.scheduler_type: str = config.scheduler_type
        self.step_size: int = int(config.step_size)
        self.gamma: float = float(config.gamma)
        self.patience: int = int(config.patience)
        self.max_lr: float | None = config.max_lr
        self.model_dir: Path = Path(config.model_dir)
        self.force_retrain: bool = bool(config.force_retrain)
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
        if not model_checkpoint_Path.exists() or self.force_retrain:
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
        self,
        loader: CustomLoader,
        resolutions: list[tuple[int, int]],
        image_type: str = "",
    ) -> tuple[list[NDArray[np.float32]], NDArray[np.float32] | None]:
        """Evaluate `model` at low-res, upsample probability maps, return smooth image.

        This extracts the inference strategy used in training so it can be
        reused or tested separately. Renamed from `render_high_res` to the
        simpler `render` API.
        """
        assert self.trained, "Model not trained for rendering"

        logger.info("Rendering High-Resolution Output...")
        # Get dimensions using base helper method
        _, _, super_height, super_width = self._get_target_dimensions(loader)

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
