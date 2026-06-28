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

    api = HfApi(token=token)
    logger.info(f"  Fetching file list from {repo_id}...")
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    
    # We only need a couple of files to find 1000 static tubelets and 5000 iid samples.
    # Each file has ~19,000 rows. We'll download the first 2.
    target_files = parquet_files[:2]
    logger.info(f"  Selected {len(target_files)} parquet files for localized calibration.")

    local_paths = []
    try:
        # 1. Download files directly to bypassing cache quota limits
        logger.info("  Step 1: Downloading parquet files via high-speed transfer...")
        for pf in target_files:
            logger.info(f"    Downloading {pf}...")
            # We use local_dir="/tmp" to avoid ~/.cache quota issues
            local_path = hf_hub_download(
                repo_id, 
                pf, 
                repo_type="dataset", 
                token=token, 
                local_dir="/tmp", 
                local_dir_use_symlinks=False
            )
            local_paths.append(local_path)
            
        # Load tables
        logger.info("  Loading parquet tables into memory...")
        tables = [pq.read_table(lp) for lp in local_paths]
        
        # We need to extract the data. We'll just iterate over batches.
        rows_with_flow = []
        all_rows = []
        
        logger.info("  Extracting rows...")
        for table in tables:
            for batch in table.to_batches():
                d = batch.to_pydict()
                n_rows = len(d[list(d.keys())[0]])
                for i in range(n_rows):
                    row = {k: v[i] for k, v in d.items()}
                    all_rows.append(row)
                    if row["tag"] == "iid_sample" and row["m_flow"] >= 0:
                        rows_with_flow.append((row["m_flow"], len(all_rows) - 1))
                        
        logger.info(f"  Loaded {len(all_rows)} total rows. Found {len(rows_with_flow)} valid samples for ε.")
        
        # 2. Compute ε
        logger.info("  Step 2: Computing ε (Sensor Noise Floor)...")
        rows_with_flow.sort(key=lambda x: x[0])
        target_count = min(cfg.STAGE1_TARGET_STATIC_COUNT, len(rows_with_flow))
        static_indices = set([idx for _, idx in rows_with_flow[:target_count]])
        
        l1_diffs = []
        for idx in static_indices:
            row = all_rows[idx]
            latent_bytes = row["latent_bytes"]
            z = np.frombuffer(latent_bytes, dtype=np.float32).reshape(32, 24, 24, 768)
            diffs = np.abs(z[1:] - z[:-1]).mean(axis=(1, 2, 3))
            l1_diffs.extend(diffs.tolist())
            
        epsilon = float(np.percentile(l1_diffs, cfg.STAGE1_PERCENTILE))
        logger.info(f"  → ε = {epsilon:.6f}")

        # 3. Compute τ_kinetic with Online Convergence
        logger.info("  Step 3: Computing Ψ with online convergence...")
        psi_list = []
        tau_history = []
        processed = 0

        t_psi_start = time.time()
        for row in all_rows:
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

        out_path = output_dir / "calibration.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"  COMPUTATION COMPLETE in {elapsed:.0f}s")
        logger.info(f"  Saved to: {out_path}")
        logger.info("=" * 60)

    finally:
        # Cleanup
        logger.info("  Cleaning up temporary parquet files...")
        for lp in local_paths:
            if os.path.exists(lp):
                os.remove(lp)
                logger.info(f"    Deleted {lp}")

if __name__ == "__main__":
    main()
