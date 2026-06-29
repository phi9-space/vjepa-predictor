import os
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="rookierufus/ego10k-vjepa-latents")
    parser.add_argument("--out_dir", type=str, default="/root/v2/data")
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found in environment.")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Downloading dataset {args.repo} to {args.out_dir} with {args.max_workers} workers...")

    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=args.out_dir,
        max_workers=args.max_workers,
        token=token
    )
    
    print("Download complete!")

if __name__ == "__main__":
    main()
