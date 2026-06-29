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
from huggingface_hub import HfApi, hf_hub_download, HfFileSystem
from collections import defaultdict
import concurrent.futures
from . import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def fetch_mflow(args):
    fs, path = args
    try:
        with fs.open(path, "rb") as f:
            table = pq.read_table(f, columns=["m_flow", "tag"])
            df = table.to_pandas()
            # Need row indices to know which one to pick
            df['row_idx'] = np.arange(len(df))
            valid = df[(df['tag'] == 'iid_sample') & (df['m_flow'] >= 0)]
            if valid.empty:
                return []
            return [(path, row['row_idx'], row['m_flow']) for _, row in valid.iterrows()]
    except Exception as e:
        logger.warning(f"Error reading {path}: {e}")
        return []

def main():
    repo_id = cfg.HF_REPO_LATENTS
    token = os.environ.get("HF_TOKEN")

    logger.info("=" * 60)
    logger.info("  COMPUTING CALIBRATION THRESHOLDS (OFFLINE - V2 Optimized)")
    logger.info("=" * 60)
    t_start = time.time()

    fs = HfFileSystem(token=token)
    api = HfApi(token=token)
    logger.info(f"  Fetching file list from {repo_id}...")
    
    # We can use HfFileSystem glob to get all parquets
    files = fs.glob(f"datasets/{repo_id}/data/**/*.parquet")
    logger.info(f"  Found {len(files)} parquet files. Commencing Zero-Disk-Footprint Pass 1...")

    # PASS 1: Epsilon Estimation (Global Top 1000)
    logger.info("  Step 1: Computing ε (Sensor Noise Floor) via global column projection...")
    
    t_pass1 = time.time()
    
    all_tubelets = []
    # Using thread pool to parallel fetch m_flow
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(fetch_mflow, [(fs, f) for f in files])
        for i, res in enumerate(results):
            all_tubelets.extend(res)
            if (i+1) % 200 == 0:
                logger.info(f"    [Pass 1: {time.time() - t_pass1:.0f}s] Scanned {i+1}/{len(files)} files...")
                
    logger.info(f"  Found {len(all_tubelets)} valid tubelets globally. Sorting to find top static...")
    
    # Sort by m_flow (index 2)
    all_tubelets.sort(key=lambda x: x[2])
    target_count = min(cfg.STAGE1_TARGET_STATIC_COUNT, len(all_tubelets))
    best_static = all_tubelets[:target_count]
    
    # Group by file to minimize downloads
    file_to_rows = defaultdict(list)
    for path, row_idx, m_flow in best_static:
        file_to_rows[path].append(row_idx)
        
    logger.info(f"  Top {target_count} tubelets are spread across {len(file_to_rows)} files.")
    logger.info(f"  Downloading only these {len(file_to_rows)} files to compute L1 diffs...")
    
    l1_diffs = []
    downloaded_files = 0
    
    for path, row_indices in file_to_rows.items():
        file_path_in_repo = path.split(repo_id + "/")[-1]
        try:
            local_path = hf_hub_download(
                repo_id, file_path_in_repo, repo_type="dataset", token=token, 
                local_dir="/tmp", local_dir_use_symlinks=False
            )
            table = pq.read_table(local_path)
            for row_idx in row_indices:
                d = table.take([row_idx]).to_pydict()
                z = np.frombuffer(d["latent_bytes"][0], dtype=np.float32).reshape(32, 24, 24, 768)
                diffs = np.abs(z[1:] - z[:-1]).mean(axis=(1, 2, 3))
                l1_diffs.extend(diffs.tolist())
            os.remove(local_path)
            downloaded_files += 1
        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")
            
    if not l1_diffs:
        raise RuntimeError("Failed to compute any L1 diffs. Epsilon estimation failed.")
        
    epsilon = float(np.percentile(l1_diffs, cfg.STAGE1_PERCENTILE))
    logger.info(f"  → ε = {epsilon:.6f} (calculated from {target_count} static anchors, downloaded {downloaded_files} files)")

    # PASS 2: Tau Kinetic Estimation with Online Convergence
    logger.info("  Step 2: Computing Ψ with online convergence...")
    
    psi_list = []
    tau_history = []
    processed = 0
    t_pass2 = time.time()
    converged = False
    
    parquet_files_api = [f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet")]
    
    for pf in parquet_files_api:
        if converged:
            break
            
        local_path = hf_hub_download(
            repo_id, pf, repo_type="dataset", token=token, 
            local_dir="/tmp", local_dir_use_symlinks=False
        )
        
        try:
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
        finally:
            if os.path.exists(local_path):
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
