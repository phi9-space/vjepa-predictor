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
        
    from v2 import config as cfg
    api = HfApi(token=token)
    repo_id = cfg.HF_REPO_LATENTS
    
    upload_dir = Path("/tmp/latent_cache/master")
    if not upload_dir.exists():
        logger.info(f"No upload directory found at {upload_dir}. Nothing to do.")
        return
    
    parquet_files = list(upload_dir.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files to upload.")
    
    if not parquet_files:
        logger.info("No files to upload. Exiting.")
        return

    # Use ThreadPoolExecutor for 8x faster uploads
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def upload_single_file(p: Path):
        success = False
        while not success:
            try:
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=f"data/train/{p.name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                logger.info(f"Successfully uploaded {p.name}")
                p.unlink(missing_ok=True)
                success = True
            except Exception as e:
                logger.error(f"Failed to upload {p.name}: {e}. Retrying in 15s...")
                time.sleep(15)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(upload_single_file, p) for p in parquet_files]
        for future in as_completed(futures):
            future.result() # Will raise exceptions if any occur, but they are caught in the loop
            
    logger.info("All leftover files successfully uploaded and cleaned up!")

if __name__ == "__main__":
    main()
