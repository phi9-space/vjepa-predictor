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
