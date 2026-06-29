import json
import logging
import time
from pathlib import Path
import os
import gc
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
import concurrent.futures
from . import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EPSILON_PLACEHOLDER = 0.636300
MAX_WORKERS = 6  # 8 workers OOM killed. 6 is the absolute sweet spot for 46GB.

def process_file(pf, repo_id, token, epsilon):
    try:
        t_start = time.time()
        local_path = hf_hub_download(
            repo_id, pf, repo_type="dataset", token=token, 
            local_dir="/tmp", local_dir_use_symlinks=False
        )
        
        table = pq.read_table(local_path)
        local_psis = []
        for batch in table.to_batches():
            d = batch.to_pydict()
            n_rows = len(d[list(d.keys())[0]])
            for i in range(n_rows):
                if d["tag"][i] == "iid_sample":
                    z = np.frombuffer(d["latent_bytes"][i], dtype=np.float32).reshape(32, 24, 24, 768)
                    diffs = np.abs(z[1:] - z[:-1])
                    per_frame = diffs.mean(axis=(1, 2, 3))
                    psi = float(np.sum(np.maximum(0.0, per_frame - epsilon)))
                    local_psis.append(psi)
            
            # Aggressive memory free
            del d, z, diffs, per_frame
            gc.collect()
                    
        # Free table and remove file
        del table
        os.remove(local_path)
        gc.collect()
        
        return local_psis
    except Exception as e:
        logger.warning(f"Error processing {pf}: {e}")
        return []

def main():
    repo_id = cfg.HF_REPO_LATENTS
    token = os.environ.get("HF_TOKEN")
    epsilon = EPSILON_PLACEHOLDER
    
    logger.info("=" * 60)
    logger.info(f"  ONLINE PARALLEL TAU ESTIMATION (ε = {epsilon})")
    logger.info("=" * 60)

    api = HfApi(token=token)
    logger.info(f"  Fetching file list from {repo_id}...")
    parquet_files = [f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.endswith(".parquet")]
    
    global_psi_list = []
    
    # Convergence Tracking
    prev_tau = 0.0
    convergence_count = 0
    next_check = 1000
    converged = False
    final_tau = 0.0
    
    logger.info(f"  Starting ThreadPoolExecutor with {MAX_WORKERS} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_pf = {executor.submit(process_file, pf, repo_id, token, epsilon): pf for pf in parquet_files}
        
        for future in concurrent.futures.as_completed(future_to_pf):
            pf = future_to_pf[future]
            try:
                local_psis = future.result()
                if local_psis:
                    global_psi_list.extend(local_psis)
                    total_samples = len(global_psi_list)
                    logger.info(f"    [Worker] Processed {pf}. Total tubelets analyzed: {total_samples}")
                    
                    if total_samples >= next_check:
                        current_tau = float(np.percentile(global_psi_list, cfg.STAGE2_PERCENTILE))
                        delta = abs(current_tau - prev_tau)
                        logger.info(f"    --- Convergence Check ({total_samples} samples) ---")
                        logger.info(f"    Current τ = {current_tau:.6f}, Prev τ = {prev_tau:.6f}, Δ = {delta:.6f}")
                        
                        if delta < 1e-4:
                            convergence_count += 1
                            logger.info(f"    Δ < 1e-4! Convergence count = {convergence_count}/3")
                        else:
                            convergence_count = 0
                            
                        prev_tau = current_tau
                        next_check += 1000
                        
                        if convergence_count >= 3:
                            logger.info("    -> ONLINE CONVERGENCE REACHED! STOPPING.")
                            final_tau = current_tau
                            converged = True
                            # Cancel remaining futures
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
            except Exception as e:
                logger.error(f"Worker for {pf} generated an exception: {e}")
                
    if not converged:
        logger.warning("Processed all files without reaching perfect convergence!")
        final_tau = float(np.percentile(global_psi_list, cfg.STAGE2_PERCENTILE))
        
    logger.info("=" * 60)
    logger.info(f"  CALIBRATION COMPLETE")
    logger.info(f"  ε (Sensor Noise Floor): {epsilon:.6f}")
    logger.info(f"  τ (Kinetic Gate)      : {final_tau:.6f}")
    logger.info("=" * 60)
    
    calib = {
        "epsilon_noise_floor": epsilon,
        "tau_kinetic_gate": final_tau,
        "samples_used_for_epsilon": 1000,
        "samples_used_for_tau": len(global_psi_list)
    }
    with open("calibration.json", "w") as f:
        json.dump(calib, f, indent=4)
    logger.info("Saved calibration.json!")

if __name__ == "__main__":
    main()
