import os
import time
import logging
from pathlib import Path
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        # try loading from .env
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
        token = os.environ.get("HF_TOKEN")
        
    api = HfApi(token=token)
    repo_id = "rookierufus/ego10k-vjepa-latents"
    
    upload_dir = Path(__file__).parent / "output" / "training" / "upload_cache"
    if not upload_dir.exists():
        logger.info(f"No upload directory found at {upload_dir}. Nothing to do.")
        return

    parquet_files = list(upload_dir.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files to upload.")
    
    for local_path in parquet_files:
        hf_path = f"data/train/{local_path.name}"
        success = False
        while not success:
            try:
                logger.info(f"Pushing {local_path.name} to {hf_path}...")
                t0 = time.time()
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=hf_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                logger.info(f"Successfully uploaded {local_path.name} in {time.time() - t0:.1f}s.")
                local_path.unlink(missing_ok=True)
                success = True
            except Exception as e:
                logger.error(f"Failed to upload {local_path.name}: {e}. Backing off for 15s...")
                time.sleep(15)

if __name__ == "__main__":
    main()
