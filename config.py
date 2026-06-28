"""
v2/config.py — Single source of truth for all constants.

Every magic number in the pipeline lives here. Nothing is hardcoded elsewhere.

FP32 MANDATE (CNN compressor.md §2):
  During TRAINING, all tensors, model weights, and latents MUST be FP32.
  During INFERENCE, V-JEPA can output BF16 (its native dtype); latents
  are cast to FP32 before entering the frozen oracles.
"""

from __future__ import annotations

from pathlib import Path

# The target number of sampled tubelets per epoch across Cloud + Local for Proportional Subsampling
EPOCH_TUBELETS = 3000


# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════

import os

# Base directory (the directory containing v2, which is 'the Compression problem')
BASE_DIR = Path(__file__).parent.parent

# V-JEPA model code
VJEPA_CODE_ROOT = Path(os.environ.get("VJEPA_CODE_ROOT", BASE_DIR.parent / "vjepa" / "vjepa2"))

# V-JEPA ViT-Base checkpoint
VJEPA_CHECKPOINT = Path(os.environ.get("VJEPA_CHECKPOINT", BASE_DIR.parent / "checkpoints" / "vjepa2_1_vitb_dist_vitG_384.pt"))
VJEPA_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"

# Project root (the Compression problem/v2/)
V2_ROOT = Path(__file__).resolve().parent

# Default output directories
OUTPUT_DIR = V2_ROOT / "output"
CALIBRATION_OUTPUT_DIR = OUTPUT_DIR / "calibration"
TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training"


# ═══════════════════════════════════════════════════════════════════════════
# V-JEPA Model Architecture (ViT-Base)
# ═══════════════════════════════════════════════════════════════════════════

EMBED_DIM = 768                 # ViT-Base hidden dimension
PATCH_SIZE = 16                 # spatial patch size (384 / 16 = 24 tokens per side)
IMAGE_SIZE = 384                # V-JEPA input resolution (after transform)
VJEPA_TUBELET_SIZE = 2          # temporal patches: 2 raw frames → 1 latent frame

# Derived spatial dimensions
LATENT_SPATIAL = IMAGE_SIZE // PATCH_SIZE   # = 24 tokens per spatial axis
LATENT_TOKENS = LATENT_SPATIAL ** 2         # = 576 tokens per frame


# ═══════════════════════════════════════════════════════════════════════════
# V-JEPA Preprocessing — THE canonical transform (CNN compressor.md §4.A)
#
# This is THE ONLY way frames are preprocessed before V-JEPA. Period.
# No separate CPU/GPU paths, no shortcuts, no "already 384" assumptions.
#
# Pipeline (matching V-JEPA eval protocol from Meta's source):
#   1. Resize shorter side to SHORTER_SIDE_SIZE (438), preserving aspect ratio
#   2. CenterCrop to (IMAGE_SIZE, IMAGE_SIZE) = (384, 384)
#   3. Scale uint8 [0, 255] → float32 [0, 1]
#   4. Normalize with ImageNet mean/std
#
# Reference: vjepa2/evals/video_classification_frozen/utils.py line 68-76
#   short_side_size = int(crop_size * 256 / 224)  # = 438 for crop_size=384
#   Resize(438, bilinear) → CenterCrop(384, 384) → ClipToTensor → Normalize
# ═══════════════════════════════════════════════════════════════════════════

SHORTER_SIDE_SIZE = int(IMAGE_SIZE * 256 / 224)  # = 438
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ═══════════════════════════════════════════════════════════════════════════
# Video Decoding
#
# Videos are decoded at NATIVE resolution (no forced square resize).
# The V-JEPA transform handles all spatial resizing with proper aspect ratio.
#
# The decoder extracts frames at TARGET_FPS and outputs them as
# uint8 numpy arrays [T, H, W, 3] at whatever the video's native resolution is.
# ═══════════════════════════════════════════════════════════════════════════

TARGET_FPS = 4                  # §4.A: strictly 4fps (Δt = 250ms)


# ═══════════════════════════════════════════════════════════════════════════
# Tubelet Geometry (CNN compressor.md §4.A)
# ═══════════════════════════════════════════════════════════════════════════

TUBELET_RAW_FRAMES = 64         # 64 raw frames → 16s @ 4fps
TUBELET_LATENT_FRAMES = TUBELET_RAW_FRAMES // VJEPA_TUBELET_SIZE  # = 32


# ═══════════════════════════════════════════════════════════════════════════
# Calibration (CNN compressor.md §3)
# ═══════════════════════════════════════════════════════════════════════════

# Stage 1: Sensor Noise Floor (ε)
STAGE1_TARGET_STATIC_COUNT = 1000     # number of most-static tubelets to collect
STAGE1_PERCENTILE = 95.0              # ε = P95 of L1 diffs in static tubelets

# Stage 2: Adaptive Kinetic Gate (τ_kinetic)
STAGE2_TARGET_SAMPLE_COUNT = 5000     # unconstrained tubelets to sample
STAGE2_PERCENTILE = 20.0              # τ_kinetic = P20 of Ψ distribution

# RAFT optical flow (for finding static tubelets in Stage 1)
RAFT_MODEL_NAME = "raft_small"        # ~1M params, fits 16GB VRAM
RAFT_RESIZE = 320                     # resize frames before RAFT (GPU memory control)
RAFT_FLOW_CLIP = 50.0                 # max L2 flow magnitude cap (clip outliers)

# Cheap pre-filter: skip tubelets with mean abs pixel diff > this threshold
# Eliminates ~80% of RAFT compute on obviously-moving scenes
PRE_FILTER_DIFF_THRESHOLD = 15.0      # mean abs pixel diff (0-255 scale)

# Minimum tubelets for a stable P95 estimate (~3,100 frame pairs at 100 tubelets)
MIN_TUBELETS_FOR_STABLE_EPSILON = 100


# ═══════════════════════════════════════════════════════════════════════════
# HuggingFace & Ego10k Dataset
# ═══════════════════════════════════════════════════════════════════════════

import os
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO = "builddotai/Egocentric-10K"
HF_REPO_LATENTS = "rookierufus/ego10k-vjepa-latents"

# Stratified sampling across environment types for representative ε/τ
DEFAULT_FACTORIES = [
    "factory_001",
    "factory_002",
    "factory_003",
    "factory_010",
    "factory_020",
    "factory_030",
    "factory_040",
    "factory_050",
]

MAX_VIDEOS_PER_FACTORY = 500


# ═══════════════════════════════════════════════════════════════════════════
# Training Hyperparameters (CNN compressor.md §5)
# ═══════════════════════════════════════════════════════════════════════════

# Optimizer (§5.C)
LR = 1e-3                              # max learning rate
LR_MIN = 1e-6                          # cosine annealing floor
WEIGHT_DECAY = 1e-4                    # AdamW weight decay
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8

# Training loop
BATCH_SIZE = 8
EPOCHS = 100
PATIENCE = 10                          # early stopping patience (§5.D)
GRAD_CLIP_NORM = 10.0                  # gradient clipping max norm

# Data budget
TUBELETS_PER_EPOCH = 5000              # IID random sample per virtual epoch
TRAIN_VAL_SPLIT = 0.9                  # 90% train, 10% validation (video-level)

# Adaptive Volatility-Gated Huber Loss (§5.B)
HUBER_BETA_MIN = 0.50                  # floor for momentum
HUBER_BETA_MAX = 0.99                  # ceiling for momentum
HUBER_ALPHA = 2.0                      # volatility sensitivity
HUBER_ETA = 1e-8                       # numerical stability (NOT ε)

SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
