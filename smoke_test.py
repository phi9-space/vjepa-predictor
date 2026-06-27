"""
v2/smoke_test.py — End-to-end smoke test with detailed timing logs.

Exercises the full calibration pipeline on a tiny sample (1 factory,
5 videos, ≤ 10 tubelets) and logs wall-clock timings at every stage.

Special feature: decodes the FIRST video twice (GPU then CPU) to give
a direct side-by-side ffmpeg NVDEC vs CPU-decode comparison.

Run from the repo root:
    cd "the Compression problem"
    HF_TOKEN=... python -m v2.smoke_test
Or from the v2/ directory directly:
    python smoke_test.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── resolve imports whether run as module or script ──
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from v2 import config as cfg
from v2.dataset import decode_video, generate_tubelets, ego10k_video_stream
from v2.vjepa_encoder import transform_frames, VJEPAEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke")

# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager: prints entry/exit with wall time."""
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        log.info(f"┌─ START  {self.label}")
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t
        log.info(f"└─ DONE   {self.label}  [{self.elapsed*1000:.1f} ms]")


def gpu_sync():
    """Force CUDA to flush all pending ops before reading a timer."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# 1. ffmpeg GPU vs CPU decode comparison
# ─────────────────────────────────────────────────────────────────────────────

def _decode_with_flag(video_bytes: bytes, use_gpu: bool) -> Optional[np.ndarray]:
    """
    Minimal ffmpeg decode — same logic as decode_video() but with the
    -hwaccel cuda flag toggled so we can A/B test.
    """
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(video_bytes)

        # ffprobe to get dimensions
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", tmp],
            capture_output=True, text=True, timeout=10,
        )
        if probe.returncode != 0:
            return None
        info = json.loads(probe.stdout)["streams"][0]
        orig_w, orig_h = info["width"], info["height"]

        shorter = 480
        if orig_w < orig_h:
            ew, eh = shorter, int(orig_h * shorter / orig_w)
        else:
            eh, ew = shorter, int(orig_w * shorter / orig_h)
        if ew % 2: ew += 1
        if eh % 2: eh += 1

        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        if use_gpu:
            cmd += ["-hwaccel", "cuda"]
        cmd += [
            "-i", tmp,
            "-vf", f"fps={cfg.TARGET_FPS},scale={ew}:{eh}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]

        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0:
            return None

        raw = proc.stdout
        fb = eh * ew * 3
        n = len(raw) // fb
        if n == 0:
            return None
        return np.frombuffer(raw[:n * fb], dtype=np.uint8).reshape(n, eh, ew, 3)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def benchmark_ffmpeg_decode(video_bytes: bytes):
    """Decode one video twice and report GPU vs CPU timing."""
    log.info("")
    log.info("══════════════════════════════════════════════")
    log.info("  ffmpeg: GPU (NVDEC) vs CPU decode benchmark")
    log.info("══════════════════════════════════════════════")
    log.info(f"  Video size: {len(video_bytes)/1e6:.2f} MB")

    # GPU
    with Timer("  ffmpeg NVDEC (-hwaccel cuda)") as t_gpu:
        frames_gpu = _decode_with_flag(video_bytes, use_gpu=True)
    shape_gpu = frames_gpu.shape if frames_gpu is not None else "FAILED"

    # CPU (run twice, take second to avoid OS caching effects)
    _decode_with_flag(video_bytes, use_gpu=False)   # warm-up
    with Timer("  ffmpeg CPU (no hwaccel)") as t_cpu:
        frames_cpu = _decode_with_flag(video_bytes, use_gpu=False)
    shape_cpu = frames_cpu.shape if frames_cpu is not None else "FAILED"

    log.info(f"  GPU result: {shape_gpu}  {t_gpu.elapsed*1000:.1f} ms")
    log.info(f"  CPU result: {shape_cpu}  {t_cpu.elapsed*1000:.1f} ms")
    if frames_gpu is not None and frames_cpu is not None:
        speedup = t_cpu.elapsed / max(t_gpu.elapsed, 1e-9)
        log.info(f"  Speedup: {speedup:.2f}× (GPU vs CPU)")
    log.info("")
    return frames_gpu if frames_gpu is not None else frames_cpu


# ─────────────────────────────────────────────────────────────────────────────
# 2. transform_frames timing
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_transform(frames: np.ndarray, device: str):
    log.info("══════════════════════════════════════════════")
    log.info("  transform_frames() timing")
    log.info("══════════════════════════════════════════════")
    log.info(f"  Input: {frames.shape}  dtype={frames.dtype}")
    log.info(f"  Expected output: [3, 64, 384, 384] fp32 on {device}")

    tubelet = frames[:cfg.TUBELET_RAW_FRAMES] if frames.shape[0] >= cfg.TUBELET_RAW_FRAMES else None
    if tubelet is None:
        log.warning(f"  Not enough frames ({frames.shape[0]} < {cfg.TUBELET_RAW_FRAMES}), skipping")
        return None

    frame_list = [tubelet[i] for i in range(tubelet.shape[0])]

    # CPU resize portion (cv2, measured separately)
    import cv2
    shorter = cfg.SHORTER_SIDE_SIZE
    crop = cfg.IMAGE_SIZE

    with Timer("  cv2 resize+crop  (64 frames, CPU)"):
        processed = []
        for f in frame_list:
            h, w = f.shape[:2]
            if w < h:
                nw, nh = shorter, int(shorter * h / w)
            else:
                nh, nw = shorter, int(shorter * w / h)
            r = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
            y1, x1 = (nh - crop) // 2, (nw - crop) // 2
            processed.append(r[y1:y1+crop, x1:x1+crop])
        stacked = np.stack(processed, axis=0)

    log.info(f"  Cropped stack: {stacked.shape}")

    with Timer("  np.stack → GPU transfer (CPU→GPU)"):
        t = torch.from_numpy(stacked).to(device=device, dtype=torch.float32)
        gpu_sync()
    log.info(f"  Transfer size: {t.numel()*4/1e6:.1f} MB")

    with Timer("  /255, permute, normalize (GPU)"):
        t = t / 255.0
        t = t.permute(3, 0, 1, 2)
        mean = torch.tensor(cfg.IMAGENET_MEAN, device=device, dtype=torch.float32)
        std  = torch.tensor(cfg.IMAGENET_STD,  device=device, dtype=torch.float32)
        t = (t - mean[:, None, None, None]) / std[:, None, None, None]
        gpu_sync()
    log.info(f"  Output tensor: {tuple(t.shape)}  dtype={t.dtype}")
    log.info("")
    return tubelet


# ─────────────────────────────────────────────────────────────────────────────
# 3. RAFT timing
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_raft(frames: np.ndarray, device: str):
    log.info("══════════════════════════════════════════════")
    log.info("  RAFT optical flow timing")
    log.info("══════════════════════════════════════════════")

    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    with Timer("  load_raft()"):
        raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
        for p in raft.parameters():
            p.requires_grad = False
        gpu_sync()

    import cv2
    raft_h = raft_w = cfg.RAFT_RESIZE

    # Pre-filter timing (one tubelet)
    from v2.calibration import compute_frame_diff_score
    with Timer("  compute_frame_diff_score() [CPU pre-filter, 1 tubelet]"):
        score = compute_frame_diff_score(frames[:cfg.TUBELET_RAW_FRAMES])
    log.info(f"  Pre-filter score: {score:.2f}  threshold: {cfg.PRE_FILTER_DIFF_THRESHOLD}")

    # RAFT one frame pair
    f1 = cv2.resize(frames[0], (raft_w, raft_h), interpolation=cv2.INTER_LINEAR)
    f2 = cv2.resize(frames[1], (raft_w, raft_h), interpolation=cv2.INTER_LINEAR)

    with Timer("  CPU→GPU transfer (1 frame pair at 320p)"):
        t1 = torch.from_numpy(f1).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        t2 = torch.from_numpy(f2).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        gpu_sync()
    log.info(f"  Transfer size: {t1.numel()*4*2/1e3:.1f} KB")

    with Timer("  raft_model() forward pass (1 pair)"):
        with torch.inference_mode():
            flow = raft(t1, t2)[-1]
        gpu_sync()

    with Timer("  magnitude + .item() → GPU→CPU scalar"):
        mag = torch.sqrt(flow[0,0]**2 + flow[0,1]**2).clamp(max=cfg.RAFT_FLOW_CLIP)
        m_flow = mag.mean().item()
        gpu_sync()
    log.info(f"  M_flow: {m_flow:.4f}")

    # Full tubelet RAFT (all 63 pairs)
    from v2.calibration import compute_flow_magnitude
    with Timer(f"  compute_flow_magnitude() [full tubelet, {cfg.TUBELET_RAW_FRAMES-1} pairs]"):
        m_full = compute_flow_magnitude(raft, frames[:cfg.TUBELET_RAW_FRAMES], device=device)
        gpu_sync()
    log.info(f"  M_flow (full tubelet): {m_full:.4f}")
    log.info("")

    del raft
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# 4. V-JEPA encode timing
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_vjepa_encode(tubelet: np.ndarray, device: str):
    log.info("══════════════════════════════════════════════")
    log.info("  VJEPAEncoder.encode() timing")
    log.info("══════════════════════════════════════════════")
    log.info(f"  Input: {tubelet.shape}  dtype={tubelet.dtype}")

    with Timer("  VJEPAEncoder load (FP32)"):
        encoder = VJEPAEncoder(device=device, fp32=True)
        gpu_sync()

    # Warm-up pass (JIT, cuDNN autotune)
    log.info("  Running warm-up encode...")
    with Timer("  encode() warm-up (1st call — JIT/cuDNN overhead)"):
        z_warm = encoder.encode(tubelet, return_on_gpu=True)
        gpu_sync()
    log.info(f"  Output: {tuple(z_warm.shape)}  dtype={z_warm.dtype}")
    del z_warm
    torch.cuda.empty_cache()

    # Measured pass
    with Timer("  encode() measured (2nd call)"):
        z = encoder.encode(tubelet, return_on_gpu=True)
        gpu_sync()

    # Ψ on GPU
    from v2.calibration import compute_psi_gpu
    with Timer("  compute_psi_gpu()  [Ψ, stays on GPU]"):
        psi_gpu = compute_psi_gpu(z, epsilon=0.01)
        gpu_sync()
    log.info(f"  Ψ (gpu): {psi_gpu:.4f}")

    # GPU → CPU latent transfer
    with Timer("  z.cpu()  [GPU→CPU latent transfer]"):
        z_cpu = z.cpu()
    log.info(f"  Transfer size: {z_cpu.numel()*4/1e6:.1f} MB")

    # Ψ on CPU
    from v2.calibration import compute_psi
    with Timer("  compute_psi()  [Ψ, CPU numpy]"):
        psi_cpu = compute_psi(z_cpu.numpy(), epsilon=0.01)
    log.info(f"  Ψ (cpu): {psi_cpu:.4f}")
    log.info(f"  Ψ delta (gpu vs cpu): {abs(psi_gpu - psi_cpu):.2e}")
    log.info("")

    return encoder, z_cpu


# ─────────────────────────────────────────────────────────────────────────────
# 5. Download + decode pipeline timing
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_download_and_decode(n_videos: int = 3):
    """
    Download n_videos from HuggingFace and time each step:
      HF download → decode_video → generate_tubelets
    """
    log.info("══════════════════════════════════════════════")
    log.info("  HuggingFace download + decode pipeline")
    log.info("══════════════════════════════════════════════")

    from huggingface_hub import HfApi
    token = cfg.HF_TOKEN
    api = HfApi(token=token)

    # List files in one factory
    factory = cfg.DEFAULT_FACTORIES[0]
    log.info(f"  Factory: {factory}")

    with Timer("  HfApi.list_repo_files()"):
        all_files = list(api.list_repo_files(
            repo_id=cfg.HF_REPO, repo_type="dataset", token=token,
        ))
    factory_files = [f for f in all_files if f.startswith(factory + "/") and f.endswith(".tar")]
    log.info(f"  Found {len(factory_files)} tar files in {factory}")

    if not factory_files:
        return None

    import tarfile
    from huggingface_hub import hf_hub_download

    first_video_bytes = None
    videos_processed = 0

    for i, tar_path in enumerate(factory_files):
        if videos_processed >= n_videos:
            break

        log.info(f"\n  ── Tar {i+1}: {tar_path} ──")
        
        with Timer(f"  hf_hub_download() [Tar cache/download]"):
            local_tar = hf_hub_download(
                repo_id=cfg.HF_REPO, repo_type="dataset",
                filename=tar_path, token=token,
            )
        log.info(f"  Tar size: {os.path.getsize(local_tar)/1e6:.1f} MB")

        with Timer(f"  tarfile.open() + extract MP4s"):
            with tarfile.open(local_tar, "r") as tar:
                mp4_members = [m for m in tar.getmembers() if m.name.endswith(".mp4")]
                
                for member in mp4_members:
                    if videos_processed >= n_videos:
                        break
                        
                    log.info(f"    Extracting {member.name} ({member.size/1e6:.1f} MB)...")
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    video_bytes = f.read()
                    f.close()
                    
                    if videos_processed == 0:
                        first_video_bytes = video_bytes

                    with Timer(f"    decode_video() [GPU hwaccel]"):
                        frames = decode_video(video_bytes, shorter_side_res=480)

                    if frames is None:
                        log.warning("    decode_video returned None, skipping")
                        continue

                    log.info(f"    Decoded: {frames.shape}  dtype={frames.dtype}  "
                             f"({frames.shape[0] / cfg.TARGET_FPS:.1f}s of video)")

                    meta = {"factory_id": factory, "worker_id": tar_path.split("/")[2], "video_index": videos_processed}
                    with Timer(f"    generate_tubelets() [stride=64]"):
                        tubelets = list(generate_tubelets(frames, meta))
                    log.info(f"    Tubelets: {len(tubelets)} × {cfg.TUBELET_RAW_FRAMES} frames")
                    
                    videos_processed += 1
                    del video_bytes


    log.info("")
    return first_video_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log.info("")
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║      V2 CALIBRATION PIPELINE SMOKE TEST      ║")
    log.info("╚══════════════════════════════════════════════╝")
    log.info(f"  Device: {device}")
    if torch.cuda.is_available():
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info(f"  PyTorch: {torch.__version__}")
    log.info(f"  HF_TOKEN set: {bool(cfg.HF_TOKEN)}")
    log.info(f"  VJEPA checkpoint: {cfg.VJEPA_CHECKPOINT}")
    log.info(f"  Checkpoint exists: {cfg.VJEPA_CHECKPOINT.exists()}")
    log.info("")

    # ── 1. Download & decode timing ──
    first_video_bytes = benchmark_download_and_decode(n_videos=3)

    if first_video_bytes is None:
        log.error("No video bytes retrieved — check HF_TOKEN and network")
        return

    # ── 2. ffmpeg GPU vs CPU benchmark ──
    if torch.cuda.is_available():
        frames = benchmark_ffmpeg_decode(first_video_bytes)
    else:
        log.warning("No CUDA device — skipping GPU/CPU ffmpeg benchmark")
        frames = decode_video(first_video_bytes)

    if frames is None or frames.shape[0] < cfg.TUBELET_RAW_FRAMES:
        log.error(f"Not enough frames to form a tubelet ({frames.shape if frames is not None else 'None'})")
        return

    log.info(f"  Working frames: {frames.shape}")

    # ── 3. Transform timing ──
    tubelet = benchmark_transform(frames, device)
    if tubelet is None:
        return

    # ── 4. RAFT timing ──
    if torch.cuda.is_available():
        benchmark_raft(frames, device)
    else:
        log.warning("No CUDA — skipping RAFT benchmark")

    # ── 5. V-JEPA encode timing ──
    encoder, z_cpu = benchmark_vjepa_encode(tubelet, device)

    # ── 6. ε computation timing ──
    log.info("══════════════════════════════════════════════")
    log.info("  ε computation timing (CPU)")
    log.info("══════════════════════════════════════════════")
    log.info("  Simulating 10 static tubelets (would be 1000 in production)")

    from v2.calibration import _compute_epsilon
    fake_latents = [z_cpu.numpy() + np.random.randn(*z_cpu.shape).astype(np.float32) * 0.001
                    for _ in range(10)]
    with Timer("  _compute_epsilon() [10 tubelets, CPU numpy]"):
        eps, stats = _compute_epsilon(fake_latents)
    log.info(f"  ε = {eps:.6f}")
    log.info(f"  Stats: {json.dumps({k: round(v, 6) for k, v in stats.items() if isinstance(v, float)}, indent=4)}")

    log.info("")
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║              SMOKE TEST COMPLETE             ║")
    log.info("╚══════════════════════════════════════════════╝")


if __name__ == "__main__":
    # Load .env if present
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    main()
