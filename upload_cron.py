import os
from pathlib import Path
from huggingface_hub import HfApi

checkpoint_path = Path("/root/v2/output/training/best_oracles.pt")
lock_file = Path("/root/v2/output/training/best_oracles.uploaded")

def upload():
    # Only upload if the checkpoint exists, and if it's newer than the lock file (or lock file doesn't exist)
    if checkpoint_path.exists():
        should_upload = not lock_file.exists() or os.path.getmtime(checkpoint_path) > os.path.getmtime(lock_file)
        
        if should_upload:
        try:
            token = os.environ.get("HF_TOKEN")
            repo = os.environ.get("HF_REPO_MODEL", "rookierufus/oracle-predictor-CNN-VJEPA-Ego10k")
            print(f"Uploading {checkpoint_path} to {repo}...")
            
            api = HfApi(token=token)
            api.create_repo(repo_id=repo, private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo="best_oracles.pt",
                repo_id=repo,
                repo_type="model"
            )
            
            # Create lock file to prevent duplicate uploads
            lock_file.touch()
            print("Upload complete! Lock file created.")
        except Exception as e:
            print(f"Failed to upload: {e}")

if __name__ == "__main__":
    upload()
