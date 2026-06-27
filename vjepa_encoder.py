"""
v2/vjepa_encoder.py — The single canonical V-JEPA interface.

NOTE: This is a 100% standalone V2 module. It does NOT import or reference any
code from V1 (no imports from calibration/, oracle/, etc.).

This module is THE ONLY way to:
  1. Load the V-JEPA ViT-Base model
  2. Transform raw frames into the V-JEPA input format
  3. Encode frames into latent representations

There is ONE transform, used EVERYWHERE (calibration, training, inference).
No separate CPU/GPU paths. No "frames already 384" shortcuts.

Transform pipeline (V-JEPA eval protocol, from Meta's source):
  1. Resize shorter side to 438, preserving aspect ratio
  2. CenterCrop to (384, 384)
  3. Scale uint8 [0,255] → float32 [0,1]
  4. Normalize with ImageNet mean/std

Precision rules:
  - TRAINING: model loaded in FP32, outputs FP32 latents
  - INFERENCE: model can be BF16 (its natural dtype), outputs are cast to FP32
    before entering the frozen oracles
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from . import config as cfg

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# The One Transform
# ═══════════════════════════════════════════════════════════════════════════

def transform_frames(
    frames: List[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """
    THE canonical V-JEPA preprocessing transform.

    This is the ONLY way to prepare frames for V-JEPA in this entire codebase.
    Follows the V-JEPA eval protocol exactly (Meta's source, vjepa2/evals/
    video_classification_frozen/utils.py lines 68-76).

    Pipeline:
      1. Resize shorter side → 438 (bilinear), preserving aspect ratio
      2. CenterCrop → (384, 384)
      3. Scale uint8 [0,255] → float32 [0,1]
      4. Normalize with ImageNet mean/std

    Args:
        frames: list of T numpy arrays, each [H, W, 3] uint8 RGB.
                H and W can be any resolution (e.g., native 1920×1080).
        device: torch device for the output tensor.

    Returns:
        Tensor [3, T, 384, 384] float32 on `device`, normalized.

    Raises:
        ValueError: if frames list is empty or frames have wrong format.
    """
    if not frames:
        raise ValueError("Empty frames list")

    T = len(frames)
    crop = cfg.IMAGE_SIZE           # 384
    shorter = cfg.SHORTER_SIDE_SIZE # 438

    processed = []
    for i, frame in enumerate(frames):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame {i}: expected [H, W, 3] uint8, got shape {frame.shape}"
            )

        # FFMPEG already scales to shorter_side=438 and center crops to 384x384.
        # Fallback in case something slips through un-cropped.
        if frame.shape[0] != crop or frame.shape[1] != crop:
            h, w = frame.shape[:2]
            if w < h:
                new_w = shorter
                new_h = int(shorter * h / w)
            else:
                new_h = shorter
                new_w = int(shorter * w / h)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rh, rw = resized.shape[:2]
            y1 = (rh - crop) // 2
            x1 = (rw - crop) // 2
            frame = resized[y1 : y1 + crop, x1 : x1 + crop]

        processed.append(frame)

    # Stack: [T, 384, 384, 3] uint8
    stacked = np.stack(processed, axis=0)

    # Move to GPU as a single bulk transfer
    tensor = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
    # tensor: [T, 384, 384, 3]

    # Step 3: Scale to [0, 1]
    tensor = tensor / 255.0

    # Step 4: Rearrange to [3, T, 384, 384] and normalize
    tensor = tensor.permute(3, 0, 1, 2)  # [3, T, H, W]

    mean = torch.tensor(cfg.IMAGENET_MEAN, device=device, dtype=torch.float32)
    std = torch.tensor(cfg.IMAGENET_STD, device=device, dtype=torch.float32)
    tensor = (tensor - mean[:, None, None, None]) / std[:, None, None, None]

    return tensor


# ═══════════════════════════════════════════════════════════════════════════
# V-JEPA Encoder
# ═══════════════════════════════════════════════════════════════════════════

class VJEPAEncoder:
    """
    Loads the V-JEPA ViT-Base model and provides a single `encode()` method.

    Usage:
        encoder = VJEPAEncoder(device="cuda:0", fp32=True)
        latents = encoder.encode(frames)  # frames: [64, H, W, 3] uint8 numpy

    The encoder is frozen (requires_grad=False, eval mode) at all times.
    It is never fine-tuned — it's a fixed feature extractor.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        fp32: bool = True,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Load V-JEPA ViT-Base encoder.

        Args:
            device: torch device string.
            fp32:   True for training (§2 mandate). False for inference (BF16 ok).
            checkpoint_path: override default checkpoint path.
        """
        self.device = torch.device(device)
        self.fp32 = fp32
        self.dtype = torch.float32 if fp32 else torch.bfloat16

        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else Path(cfg.VJEPA_CHECKPOINT)
        if not self.checkpoint_path.exists():
            logger.info(f"V-JEPA checkpoint not found locally. Downloading from {cfg.VJEPA_DOWNLOAD_URL}...")
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(cfg.VJEPA_DOWNLOAD_URL, str(self.checkpoint_path))
            logger.info("Download complete.")

        self.model = self._load_model(self.checkpoint_path)

        n_params = sum(p.numel() for p in self.model.parameters())
        dtype_name = "FP32" if fp32 else "BF16"
        logger.info(
            f"V-JEPA ViT-Base loaded: {n_params/1e6:.1f}M params, "
            f"{dtype_name}, frozen, device={device}"
        )

    def _load_model(self, checkpoint_path) -> torch.nn.Module:
        """Load and freeze the V-JEPA ViT-Base encoder."""
        # Add V-JEPA code to Python path
        vjepa_root = str(cfg.VJEPA_CODE_ROOT)
        if vjepa_root not in sys.path:
            sys.path.insert(0, vjepa_root)

        from app.vjepa_2_1.models import vision_transformer as vit_encoder

        model = vit_encoder.vit_base(
            patch_size=cfg.PATCH_SIZE,
            img_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
            num_frames=cfg.TUBELET_RAW_FRAMES,
            tubelet_size=cfg.VJEPA_TUBELET_SIZE,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )

        # Load checkpoint
        ckpt = torch.load(
            str(checkpoint_path), map_location="cpu",
            weights_only=True, mmap=True,
        )
        encoder_key = "target_encoder" if "target_encoder" in ckpt else "encoder"

        state_dict = {}
        for k, v in ckpt[encoder_key].items():
            k = k.replace("module.", "").replace("backbone.", "")
            state_dict[k] = v

        model.load_state_dict(state_dict, strict=True)
        del ckpt, state_dict
        gc.collect()

        # Move to device with correct dtype
        model = model.to(dtype=self.dtype, device=self.device)
        model.eval()

        # Freeze all weights — this model is NEVER trained
        for p in model.parameters():
            p.requires_grad = False

        return model

    def encode(
        self,
        frames: np.ndarray,
        return_on_gpu: bool = True,
    ) -> torch.Tensor:
        """
        Encode a 64-frame tubelet through V-JEPA.

        Full pipeline:
          1. transform_frames(): Resize(438) → CenterCrop(384) → normalize
          2. V-JEPA forward pass
          3. Reshape to [32, 24, 24, 768]

        Args:
            frames:       [64, H, W, 3] uint8 numpy array. H and W can be any
                          resolution — the transform handles resizing.
            return_on_gpu: if True, return tensor on GPU; if False, return on CPU.

        Returns:
            [32, 24, 24, 768] float32 tensor (always FP32, even if model is BF16).
        """
        assert frames.shape[0] == cfg.TUBELET_RAW_FRAMES, (
            f"Expected {cfg.TUBELET_RAW_FRAMES} frames, got {frames.shape[0]}"
        )
        assert frames.ndim == 4 and frames.shape[3] == 3, (
            f"Expected [T, H, W, 3] uint8, got shape {frames.shape}"
        )

        # Step 1: Transform (THE one canonical transform)
        frame_list = [frames[i] for i in range(frames.shape[0])]
        x = transform_frames(frame_list, device=self.device)
        # x: [3, 64, 384, 384] float32

        # Cast to model dtype if needed (BF16 for inference)
        if self.dtype != torch.float32:
            x = x.to(dtype=self.dtype)

        # Step 2: V-JEPA forward pass
        x = x.unsqueeze(0)  # [1, 3, 64, 384, 384]

        with torch.inference_mode():
            out = self.model(x)

        # Step 3: Reshape to [32, 24, 24, 768] and always cast back to FP32
        z = out.reshape(
            1,
            cfg.TUBELET_LATENT_FRAMES,
            cfg.LATENT_SPATIAL,
            cfg.LATENT_SPATIAL,
            cfg.EMBED_DIM,
        ).squeeze(0).float()
        # z: [32, 24, 24, 768] float32

        if not return_on_gpu:
            z = z.cpu()

        del x, out
        return z

    def encode_batch(
        self,
        frames_list: List[np.ndarray],
        return_on_gpu: bool = True,
    ) -> List[torch.Tensor]:
        """
        Encode multiple tubelets. Convenience wrapper around encode().

        Args:
            frames_list: list of [64, H, W, 3] uint8 numpy arrays.
            return_on_gpu: if True, return tensors on GPU.

        Returns:
            List of [32, 24, 24, 768] float32 tensors.
        """
        return [
            self.encode(frames, return_on_gpu=return_on_gpu)
            for frames in frames_list
        ]


# Need Path for type hints in _load_model
from pathlib import Path
