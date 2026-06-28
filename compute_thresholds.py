"""
v2/compute_thresholds.py — Offline Mathematical Calibration (Hyper-Optimized)
"""

import json
import logging
import time
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from . import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    repo_id = cfg.HF_REPO_LATENTS
    token = os.environ.get("HF_TOKEN")

    logger.info("=" * 60)
    logger.info("  COMPUTING CALIBRATION THRESHOLDS (OFFLINE)")
    logger.info("=" * 60)
    t_start = time.time()

    api = HfApi(token=token)
    logger.info(f"  Fetching file list from {repo_id}...")
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    
    logger.info(f"  Found {len(parquet_files)} parquet files. Commencing Zero-Disk-Footprint Pass...")

    # PASS 1: Epsilon Estimation
    # We need 1000 static tubelets. We'll sample the first ~150 files (approx 2800 tubelets),
    # compute their L1 diffs on the fly, and take the 1000 with the lowest m_flow.
    logger.info("  Step 1: Computing ε (Sensor Noise Floor)...")
    
    tubelet_stats = []
    files_processed = 0
    t_pass1 = time.time()
    
    for pf in parquet_files:
        if len(tubelet_stats) >= 3000:
            break
            
        local_path = hf_hub_download(
            repo_id, pf, repo_type="dataset", token=token, 
            local_dir="/tmp", local_dir_use_symlinks=False
        )
        
        table = pq.read_table(local_path)
        for batch in table.to_batches():
            d = batch.to_pydict()
            n_rows = len(d[list(d.keys())[0]])
            for i in range(n_rows):
                if d["tag"][i] == "iid_sample" and d["m_flow"][i] >= 0:
                    z = np.frombuffer(d["latent_bytes"][i], dtype=np.float32).reshape(32, 24, 24, 768)
                    diffs = np.abs(z[1:] - z[:-1]).mean(axis=(1, 2, 3))
                    tubelet_stats.append((d["m_flow"][i], diffs.tolist()))
                    
        os.remove(local_path)
        files_processed += 1
        logger.info(f"    [Pass 1: {time.time() - t_pass1:.0f}s] Processed {files_processed} files, extracted {len(tubelet_stats)} latents...")

    # Sort by m_flow and pick top 1000
    tubelet_stats.sort(key=lambda x: x[0])
    target_count = min(cfg.STAGE1_TARGET_STATIC_COUNT, len(tubelet_stats))
    best_static = tubelet_stats[:target_count]
    
    l1_diffs = []
    for _, diffs in best_static:
        l1_diffs.extend(diffs)
        
    epsilon = float(np.percentile(l1_diffs, cfg.STAGE1_PERCENTILE))
    logger.info(f"  → ε = {epsilon:.6f} (calculated from {target_count} static anchors)")

    # PASS 2: Tau Kinetic Estimation with Online Convergence
    logger.info("  Step 2: Computing Ψ with online convergence...")
    
    psi_list = []
    tau_history = []
    processed = 0
    t_pass2 = time.time()
    converged = False
    
    # Continue from where we left off
    for pf in parquet_files[files_processed:]:
        if converged:
            break
            
        local_path = hf_hub_download(
            repo_id, pf, repo_type="dataset", token=token, 
            local_dir="/tmp", local_dir_use_symlinks=False
        )
        
        table = pq.read_table(local_path)
        for batch in table.to_batches():
            if converged: break
            
            d = batch.to_pydict()
            n_rows = len(d[list(d.keys())[0]])
            for i in range(n_rows):
                if converged: break
                
                if d["tag"][i] == "iid_sample":
                    z = np.frombuffer(d["latent_bytes"][i], dtype=np.float32).reshape(32, 24, 24, 768)
                    
                    # Compute Psi
                    diffs = np.abs(z[1:] - z[:-1])
                    per_frame = diffs.mean(axis=(1, 2, 3))
                    psi = float(np.sum(np.maximum(0.0, per_frame - epsilon)))
                    psi_list.append(psi)
                    processed += 1

                    if processed % 1000 == 0:
                        current_tau = float(np.percentile(psi_list, cfg.STAGE2_PERCENTILE))
                        tau_history.append(current_tau)
                        logger.info(f"    [Pass 2: {time.time() - t_pass2:.0f}s] {processed} samples | Current τ_kinetic = {current_tau:.6f}")

                        if processed >= 5000 and len(tau_history) >= 3:
                            diff1 = abs(tau_history[-1] - tau_history[-2])
                            diff2 = abs(tau_history[-2] - tau_history[-3])
                            if diff1 < 1e-4 and diff2 < 1e-4:
                                logger.info(f"    *** Convergence Reached at {processed} samples! ***")
                                converged = True
                                
        os.remove(local_path)

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

    out_path = output_dir / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"  COMPUTATION COMPLETE in {elapsed:.0f}s")
    logger.info(f"  Saved to: {out_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
