import os
import time
from huggingface_hub import HfApi, snapshot_download

def benchmark_download():
    repo_id = os.environ.get("HF_REPO_LATENTS", "rookierufus/ego10k-vjepa-latents")
    token = os.environ.get("HF_TOKEN")
    
    print("Listing files...")
    api = HfApi(token=token)
    all_files = [f for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token) if f.endswith('.parquet')]
    
    # Select just 50 files for benchmark (~50 GB)
    subset_files = all_files[:50]
    print(f"Downloading {len(subset_files)} files (~50GB) to test throughput...")
    
    # Enable hf_transfer
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    
    start_time = time.time()
    
    local_dir = "/root/v2/data_bench"
    os.makedirs(local_dir, exist_ok=True)
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=subset_files,
        local_dir=local_dir,
        max_workers=32, # Max out the 32 cores
        token=token
    )
    
    elapsed = time.time() - start_time
    
    # Calculate downloaded size
    total_bytes = sum(os.path.getsize(os.path.join(local_dir, f)) for f in subset_files if os.path.exists(os.path.join(local_dir, f)))
    total_gb = total_bytes / (1024 ** 3)
    speed = total_gb / elapsed
    
    print(f"Downloaded {total_gb:.2f} GB in {elapsed:.2f} seconds.")
    print(f"Average Download Speed: {speed:.2f} GB/s ({speed * 8:.2f} Gbps)")

if __name__ == "__main__":
    benchmark_download()
