"""
v2/dataset.py — Video decoding and tubelet generation.

This module handles the extraction of raw pixels from compressed video bytes.
It intentionally avoids C-level bindings like `decord` (which can crash the
Python interpreter on corrupted bitstreams or newer GPUs) in favor of isolated
ffmpeg subprocesses.

Crucially, it preserves aspect ratio during decode, delegating the final
CenterCrop to the canonical V-JEPA transform.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Iterator, Optional, Tuple

import numpy as np

from . import config as cfg

logger = logging.getLogger(__name__)


def decode_video(
    video_bytes: bytes,
    target_fps: int = cfg.TARGET_FPS,
    shorter_side_res: int = 480,
) -> Optional[np.ndarray]:
    """
    Decode an MP4 video into a sequence of RGB frames.

    Uses an isolated ffmpeg subprocess to prevent C-level segfaults on
    corrupted videos.

    GPU DECODING: Assumes NVIDIA hardware and uses NVDEC (`-hwaccel cuda`)
    to shift the decoding burden to dedicated GPU silicon. This is massively
    faster than CPU decoding and uses very little VRAM.

    To save memory while PRESERVING ASPECT RATIO, the video is resized
    such that its SHORTER side is `shorter_side_res` (e.g., 480) and the
    longer side is scaled proportionally. This safely handles both horizontal
    (e.g., 16:9) and vertical (e.g., 9:16) videos without distortion.

    Args:
        video_bytes:      raw MP4 file bytes
        target_fps:       frames per second to extract (default 4)
        shorter_side_res: target size for the shorter side (default 480)

    Returns:
        np.ndarray of shape [T, H, W, 3] uint8 RGB,
        or None if decoding fails.
    """
    if not video_bytes:
        return None

    # Write bytes to a temporary file for ffmpeg to read
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(video_bytes)
    except Exception as e:
        logger.error(f"Failed to write temp video: {e}")
        os.unlink(tmp_path)
        return None

    try:
        # First, probe the video to get its actual dimensions to calculate new dims.
        # We need this to parse the raw RGB24 bytes from ffmpeg stdout.
        proc_probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                tmp_path
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if proc_probe.returncode != 0:
            logger.debug("ffprobe failed on video")
            return None
            
        probe_info = json.loads(proc_probe.stdout)
        orig_w = probe_info["streams"][0]["width"]
        orig_h = probe_info["streams"][0]["height"]
        
        # Calculate expected dims after shorter-side scaling
        if orig_w < orig_h:
            # Vertical video: width is shorter
            expected_w = shorter_side_res
            expected_h = int(orig_h * (shorter_side_res / orig_w))
        else:
            # Horizontal or square video: height is shorter
            expected_h = shorter_side_res
            expected_w = int(orig_w * (shorter_side_res / orig_h))
            
        # ffmpeg requires even dimensions for many pixel formats
        if expected_w % 2 != 0: expected_w += 1
        if expected_h % 2 != 0: expected_h += 1

        # The scale filter string forcing the exact dimensions we calculated
        scale_filter = f"fps={target_fps},scale={expected_w}:{expected_h}"

        # Run ffmpeg to decode with NVDEC hardware acceleration
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-hwaccel", "cuda",             # NVIDIA GPU hardware decode
                "-i", tmp_path,
                "-vf", scale_filter,
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
            ],
            capture_output=True,
            timeout=120,
        )

        if proc.returncode != 0:
            logger.debug(f"ffmpeg failed: {proc.stderr.decode(errors='replace')}")
            return None

        raw_bytes = proc.stdout
        frame_bytes = expected_h * expected_w * 3
        n_frames = len(raw_bytes) // frame_bytes

        if n_frames == 0:
            return None

        frames = np.frombuffer(raw_bytes[:n_frames * frame_bytes], dtype=np.uint8)
        frames = frames.reshape(n_frames, expected_h, expected_w, 3)
        return frames

    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg decode timed out")
        return None
    except Exception as e:
        logger.error(f"ffmpeg decode error: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def generate_tubelets(
    frames: np.ndarray,
    meta: dict,
    window_size: int = cfg.TUBELET_RAW_FRAMES,
    stride: int = cfg.TUBELET_RAW_FRAMES,
) -> Iterator[Tuple[np.ndarray, dict, int]]:
    """
    Yield non-overlapping tubelets from a sequence of frames.

    Args:
        frames:      [T, H, W, 3] uint8 numpy
        meta:        video metadata dictionary to pass along
        window_size: frames per tubelet (default 64)
        stride:      frames to advance between tubelets (default 64 for non-overlapping)

    Yields:
        (tubelet_frames, meta, start_idx)
    """
    T = frames.shape[0]
    
    for start_idx in range(0, T - window_size + 1, stride):
        tubelet = frames[start_idx : start_idx + window_size]
        # Yield a copy to prevent memory leaks if the caller holds onto the tubelet
        # while the large original video array is still in memory.
        yield tubelet.copy(), meta, start_idx


# ═══════════════════════════════════════════════════════════════════════════
# HuggingFace Ego10k Stream
# ═══════════════════════════════════════════════════════════════════════════

def ego10k_video_stream(
    factories: list = None,
    max_videos_per_factory: int = cfg.MAX_VIDEOS_PER_FACTORY,
    hf_token: str = None,
    shorter_side_res: int = 480,
    shuffle_seed: int = cfg.SEED,
):
    """
    Download and yield raw decoded frames from the Ego10k HuggingFace dataset.

    DATASET STRUCTURE (builddotai/Egocentric-10K):
      The dataset stores videos inside .tar archives, not as individual .mp4 files.
      Each tar contains 1-4 MP4 videos (~200-600MB each) + JSON metadata.
      Path format: factory_001/workers/worker_001/factory001_worker001_part00.tar

    This function:
      1. Lists all .tar files for the requested factories
      2. Downloads each tar via hf_hub_download (cached by HF)
      3. Extracts .mp4 members in-memory
      4. Decodes each MP4 via decode_video()
      5. Yields (frames, meta) per video

    Args:
        factories:              list of factory IDs to sample from (default: cfg.DEFAULT_FACTORIES)
        max_videos_per_factory: max videos to pull per factory
        hf_token:               HuggingFace token (default: cfg.HF_TOKEN)
        shorter_side_res:       shorter-side decode resolution (default 480)
        shuffle_seed:           RNG seed for per-factory shuffling

    Yields:
        (frames, meta)
          frames: [T, H, W, 3] uint8 numpy (T varies by video length)
          meta:   dict with factory_id, worker_id, video_index, tar_path, mp4_name
    """
    from huggingface_hub import hf_hub_download, HfApi
    import tarfile

    token = hf_token or cfg.HF_TOKEN
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Set the HF_TOKEN environment variable."
        )

    if factories is None:
        factories = cfg.DEFAULT_FACTORIES

    api = HfApi(token=token)
    rng = np.random.default_rng(shuffle_seed)

    # List all files once (the API paginates internally)
    logger.info("Listing all files in HF repo...")
    try:
        all_files = list(api.list_repo_files(
            repo_id=cfg.HF_REPO,
            repo_type="dataset",
            token=token,
        ))
    except Exception as e:
        logger.error(f"Failed to list HF repo files: {e}")
        return

    for factory_id in factories:
        logger.info(f"Processing factory: {factory_id}")

        # Filter to .tar files belonging to this factory
        tar_files = [
            f for f in all_files
            if f.startswith(factory_id + "/") and f.endswith(".tar")
        ]

        if not tar_files:
            logger.warning(f"  No tar files found for factory {factory_id}")
            continue

        # Shuffle tars so we sample across workers
        tar_files_shuffled = list(rng.permutation(tar_files))

        videos_yielded = 0

        for tar_path in tar_files_shuffled:
            if videos_yielded >= max_videos_per_factory:
                break

            # Parse worker_id from path: factory_001/workers/worker_001/....tar
            parts = tar_path.split("/")
            worker_id = parts[2] if len(parts) >= 3 else "unknown"

            logger.info(f"  Downloading tar: {tar_path}")
            try:
                local_tar = hf_hub_download(
                    repo_id=cfg.HF_REPO,
                    repo_type="dataset",
                    filename=tar_path,
                    token=token,
                )
            except Exception as e:
                logger.warning(f"  Skip {tar_path}: download failed ({e})")
                continue

            # Extract MP4s from the tar
            try:
                with tarfile.open(local_tar, "r") as tar:
                    mp4_members = [
                        m for m in tar.getmembers()
                        if m.name.endswith(".mp4") and m.isfile()
                    ]

                    for member in mp4_members:
                        if videos_yielded >= max_videos_per_factory:
                            break

                        logger.info(
                            f"    Extracting: {member.name} "
                            f"({member.size/1e6:.1f} MB)"
                        )

                        # Extract MP4 bytes in memory
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        video_bytes = f.read()
                        f.close()

                        meta = {
                            "factory_id": factory_id,
                            "worker_id": worker_id,
                            "video_index": videos_yielded,
                            "tar_path": tar_path,
                            "mp4_name": member.name,
                        }

                        # Decode video → frames
                        frames = decode_video(
                            video_bytes,
                            shorter_side_res=shorter_side_res,
                        )
                        del video_bytes  # free ~200-600MB immediately

                        if frames is None:
                            logger.debug(f"    Skip {member.name}: decode failed")
                            continue

                        if frames.shape[0] < cfg.TUBELET_RAW_FRAMES:
                            logger.debug(
                                f"    Skip {member.name}: only {frames.shape[0]} frames "
                                f"(need ≥{cfg.TUBELET_RAW_FRAMES})"
                            )
                            continue

                        logger.info(
                            f"    Decoded: {frames.shape} "
                            f"({frames.shape[0]/cfg.TARGET_FPS:.0f}s of video)"
                        )
                        videos_yielded += 1
                        yield frames, meta

            except Exception as e:
                logger.warning(f"  Skip {tar_path}: tar extraction failed ({e})")
                continue

        logger.info(
            f"  {factory_id}: yielded {videos_yielded} videos "
            f"(target: {max_videos_per_factory})"
        )


def make_tubelet_stream_factory(
    factories: list = None,
    max_videos_per_factory: int = cfg.MAX_VIDEOS_PER_FACTORY,
    hf_token: str = None,
    shorter_side_res: int = 480,
    shuffle_seed: int = cfg.SEED,
    window_size: int = cfg.TUBELET_RAW_FRAMES,
    stride: int = cfg.TUBELET_RAW_FRAMES,
):
    """
    Build the tubelet_stream_factory callable that calibration.run_calibration()
    expects.

    This is the TOP-LEVEL entry point that ties everything together:
      HuggingFace download → decode_video() → generate_tubelets() → yield

    Returns a FACTORY (a function that returns a fresh generator). The factory
    pattern is required because Stage 1 needs to stream the dataset twice:
      - Pass 1: RAFT scan over all tubelets
      - Pass 2: re-extract the winning static tubelets for V-JEPA encoding

    Python generators are single-use, so we return a callable that creates a
    brand-new generator from scratch each time it's called.

    Usage:
        factory = make_tubelet_stream_factory(factories=["factory_001"])
        calibration.run_calibration(tubelet_stream_factory=factory)

    Args:
        factories:              factory IDs to sample (default: cfg.DEFAULT_FACTORIES)
        max_videos_per_factory: max videos per factory
        hf_token:               HuggingFace API token
        shorter_side_res:       decode shorter-side resolution (default 480)
        shuffle_seed:           RNG seed for deterministic sampling
        window_size:            frames per tubelet (default 64)
        stride:                 stride between tubelets (default 64, non-overlapping)

    Returns:
        A callable: () -> Iterator[Tuple[np.ndarray, dict, int]]
        Each call to the returned callable yields fresh (frames, meta, start_idx).
    """
    def _stream():
        """
        Inner generator: one fresh pass over the dataset.
        Called by calibration for each pass (Pass 1, Pass 2, Stage 2).
        """
        for frames, meta in ego10k_video_stream(
            factories=factories,
            max_videos_per_factory=max_videos_per_factory,
            hf_token=hf_token,
            shorter_side_res=shorter_side_res,
            shuffle_seed=shuffle_seed,
        ):
            yield from generate_tubelets(
                frames=frames,
                meta=meta,
                window_size=window_size,
                stride=stride,
            )

    # Return the factory function, not a generator.
    # Each call to _stream() gives a fresh generator from video 1.
    return _stream
