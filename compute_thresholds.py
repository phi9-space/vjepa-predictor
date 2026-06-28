"""
v2/compute_thresholds.py — Offline Mathematical Calibration

This script operates on the pre-computed latents stored in HuggingFace.
It mathematically isolates the sensor noise floor (ε) and calculates the
kinematic filter gate (τ_kinetic) without running V-JEPA or decoding video.
"""

import json
import logging
import time
from pathlib import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from . import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def compute_psi(z: np.ndarray, epsilon: float) -> float:
    diffs = np.abs(z[1:] - z[:-1])
    per_frame = diffs.mean(axis=(1, 2, 3))
    psi = float(np.sum(np.maximum(0.0, per_frame - epsilon)))
    return psi

def main():
    repo_id = cfg.HF_REPO_LATENTS
    token = os.environ.get("HF_TOKEN")
    
    logger.info("=" * 60)
    logger.info("  COMPUTING CALIBRATION THRESHOLDS (OFFLINE)")
    logger.info("=" * 60)
    
    t_start = time.time()
    
    # 1. Fetch metadata only (super fast columnar read)
    logger.info("  Step 1: Fetching metadata index...")
    meta_ds = load_dataset(repo_id, split="train", token=token).select_columns(
        ["factory_id", "worker_id", "video_index", "tubelet_start", "m_flow", "tag"]
    )
    
    # Extract just the m_flow scores and row indices
    logger.info("  Sorting tubelets by optical flow...")
    rows_with_flow = []
    for i, row in enumerate(meta_ds):
        # We only care about iid_sample for baseline dataset stats
        if row["tag"] == "iid_sample" and row["m_flow"] >= 0:
            rows_with_flow.append((row["m_flow"], i))
            
    if not rows_with_flow:
        logger.error("No valid 'iid_sample' rows found in dataset!")
        return
        
    rows_with_flow.sort(key=lambda x: x[0])
    
    # Top static tubelets
    target_count = min(cfg.STAGE1_TARGET_STATIC_COUNT, len(rows_with_flow))
    static_indices = set([idx for _, idx in rows_with_flow[:target_count]])
    
    logger.info(f"  Identified {len(static_indices)} static tubelets for ε computation.")
    
    # 2. Compute ε
    logger.info("  Step 2: Streaming static tubelets to calculate ε...")
    # Stream the dataset, ignoring rows we don't need
    ds = load_dataset(repo_id, split="train", streaming=True, token=token)
    
    l1_diffs = []
    static_processed = 0
    
    for i, row in enumerate(ds):
        if i in static_indices:
            # Reconstruct float32 tensor from raw bytes
            latent_bytes = row["latent_bytes"]
            z = np.frombuffer(latent_bytes, dtype=np.float32).reshape(32, 24, 24, 768)
            
            diffs = np.abs(z[1:] - z[:-1]).mean(axis=(1, 2, 3))
            l1_diffs.extend(diffs.tolist())
            static_processed += 1
            
            if static_processed >= target_count:
                break
                
    epsilon = float(np.percentile(l1_diffs, cfg.STAGE1_PERCENTILE))
    logger.info(f"  → ε = {epsilon:.6f}")
    
    # 3. Compute τ_kinetic with Online Convergence
    logger.info("  Step 3: Computing Ψ with online convergence...")
    psi_list = []
    tau_history = []
    processed = 0
    
    # Stream again from the top
    ds_full = load_dataset(repo_id, split="train", streaming=True, token=token)
    
    t_psi_start = time.time()
    for row in ds_full:
        if row["tag"] == "iid_sample":
            latent_bytes = row["latent_bytes"]
            z = np.frombuffer(latent_bytes, dtype=np.float32).reshape(32, 24, 24, 768)
            
            psi = compute_psi(z, epsilon)
            psi_list.append(psi)
            processed += 1
            
            if processed % 1000 == 0:
                current_tau = float(np.percentile(psi_list, cfg.STAGE2_PERCENTILE))
                tau_history.append(current_tau)
                logger.info(f"    [{time.time() - t_psi_start:.0f}s] {processed} samples | Current τ_kinetic = {current_tau:.6f}")
                
                # Check for convergence
                if processed >= 5000 and len(tau_history) >= 3:
                    diff1 = abs(tau_history[-1] - tau_history[-2])
                    diff2 = abs(tau_history[-2] - tau_history[-3])
                    if diff1 < 1e-4 and diff2 < 1e-4:
                        logger.info(f"    *** Convergence Reached at {processed} samples! ***")
                        break
                
    psi_arr = np.array(psi_list)
    tau_kinetic = float(np.percentile(psi_arr, cfg.STAGE2_PERCENTILE))
    
    logger.info(f"  → Final τ_kinetic = {tau_kinetic:.6f}")
    
    # Plot Convergence Graph
    output_dir = cfg.CALIBRATION_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1000, processed + 1, 1000), tau_history, marker='o', linestyle='-', color='b')
    plt.title("Online Convergence of τ_kinetic (20th Percentile)")
    plt.xlabel("Number of Samples")
    plt.ylabel("τ_kinetic")
    plt.grid(True)
    graph_path = output_dir / "tau_convergence.png"
    plt.savefig(graph_path)
    plt.close()
    logger.info(f"  Saved convergence graph to {graph_path}")
    
    # 4. Save Output
    elapsed = time.time() - t_start
    result = {
        "epsilon": epsilon,
        "tau_kinetic": tau_kinetic,
        "n_tubelets": processed,
        "elapsed_s": elapsed,
        "psi_stats": {
            "mean": float(np.mean(psi_arr)),
            "std": float(np.std(psi_arr)),
            "p20": tau_kinetic,
            "p50": float(np.percentile(psi_arr, 50)),
            "p80": float(np.percentile(psi_arr, 80)),
        }
    }
    
    output_dir = cfg.CALIBRATION_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        
    logger.info("=" * 60)
    logger.info(f"  COMPUTATION COMPLETE in {elapsed:.0f}s")
    logger.info(f"  Saved to: {out_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
