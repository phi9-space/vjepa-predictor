"""
v2/extract_latents.py — Pure GPU Pipeline for Dataset Generation

Streams raw H.264 video from the dataset, calculates optical flow (m_flow),
encodes through V-JEPA, and pushes raw FP32 tensors to HuggingFace Parquet.

This script runs in a single pass to build a perfectly IID Master Dataset.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from . import config as cfg
from .dataset import ego10k_video_stream
from .vjepa_encoder import VJEPAEncoder
from .utils.latent_cache import LatentCache
from calibration.optical_flow import compute_flow_magnitude_batch, load_raft

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def run_extraction(target_count: int = 25000, device: str = "cuda:0", shard_idx: int = 0, num_shards: int = 1):
    logger.info("=" * 60)
    logger.info("  UNIFIED LATENT EXTRACTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  Target: {target_count} tubelets (Global)")
    logger.info(f"  Device: {device}")
    logger.info(f"  Shard:  {shard_idx + 1} of {num_shards}")
    
    encoder = VJEPAEncoder(device=device, fp32=True)
    raft_model = load_raft(device)
    cache = LatentCache(shard_size_gb=1.0)
    
    # Fast index fetch to skip previously processed latents
    cached_ids = LatentCache.fetch_index()
    
    global_encoded = len(cached_ids)
    if global_encoded >= target_count:
        logger.info(f"Already reached global target count ({global_encoded} >= {target_count}). Exiting.")
        return
        
    global_remaining = target_count - global_encoded
    local_target = (global_remaining // num_shards) + (1 if shard_idx < global_remaining % num_shards else 0)
    
    logger.info(f"Resuming extraction. {global_encoded} latents globally in cache.")
    logger.info(f"  This shard must encode: {local_target} tubelets.")
    
    # Shard the factories
    factories = cfg.DEFAULT_FACTORIES
    if num_shards > 1:
        chunk_size = len(factories) // num_shards
        start_idx = shard_idx * chunk_size
        end_idx = start_idx + chunk_size if shard_idx < num_shards - 1 else len(factories)
        factories = factories[start_idx:end_idx]
        logger.info(f"  Processing {len(factories)} factories for this shard.")
    
    t_start = time.time()
    local_encoded_this_run = 0
    
    for tubelet_frames, meta, start_idx in ego10k_video_stream(factories=factories):
        if local_encoded_this_run >= local_target:
            logger.info("Shard target reached! Stopping.")
            break
            
        t_tubelet_start = time.time()
        
        factory_id = meta.get("factory_id", "unknown")
        worker_id = meta.get("worker_id", "unknown")
        video_index = meta.get("video_index", -1)
        latent_id = (factory_id, worker_id, video_index, start_idx)
        
        if latent_id in cached_ids:
            continue
            
        # 1. Compute m_flow (optical flow magnitude)
        t_raft = time.time()
        m_flow = compute_flow_magnitude_batch(raft_model, tubelet_frames, device=device)
        t_raft_end = time.time()
        
        # 2. Compute Latent via V-JEPA
        z = encoder.encode(tubelet_frames, return_on_gpu=False)
        z_np = z.numpy()
        t_vjepa_end = time.time()
        
        # 3. Cache the Latent
        cache.add(
            factory_id=factory_id,
            worker_id=worker_id,
            video_index=video_index,
            tubelet_start=start_idx,
            latent=z_np,
            tag="iid_sample",
            m_flow=m_flow
        )
        
        cached_ids.add(latent_id)
        local_encoded_this_run += 1
        
        t_total = time.time() - t_tubelet_start
        logger.info(
            f"Encoded {local_encoded_this_run:04d}/{local_target:04d} | "
            f"Video: {factory_id}/{worker_id} | "
            f"RAFT: {t_raft_end - t_raft:.2f}s | "
            f"VJEPA: {t_vjepa_end - t_raft_end:.2f}s | "
            f"Total: {t_total:.2f}s"
        )
            
    # Flush pending parquets
    cache.close()
    logger.info("=" * 60)
    logger.info("  EXTRACTION COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    from pathlib import Path
    import os
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    import argparse
    parser = argparse.ArgumentParser(description="Extract latents to the Master Dataset")
    parser.add_argument("--target-count", type=int, default=25000, help="Total latents to ensure are in the dataset")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., cuda:0)")
    parser.add_argument("--shard-idx", type=int, default=0, help="Index of this shard (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards running in parallel")
    args = parser.parse_args()
    
    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    run_extraction(
        target_count=args.target_count, 
        device=device,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards
    )
