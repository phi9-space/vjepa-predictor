import os
import queue
import threading
import uuid
import time
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import numpy as np
from pathlib import Path
from huggingface_hub import HfApi
import logging

logger = logging.getLogger(__name__)

class LatentCache:
    """
    A robust background-caching utility for V-JEPA latents.
    Buffers tensors locally and asynchronously pushes 1GB Parquet shards to HuggingFace.
    """

    def __init__(
        self,
        repo_id: str = "phi9-space/ego10k-vjepa-latents",
        hf_token: str = None,
        shard_size_gb: float = 1.0,
        local_cache_dir: str = "/tmp/latent_cache",
    ):
        self.repo_id = repo_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.shard_size_gb = shard_size_gb
        
        self.local_dir = Path(local_cache_dir) / "master"
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        self.api = HfApi(token=self.hf_token)
        
        # Ensure repo exists
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type="dataset")
        except Exception:
            logger.info(f"Creating HuggingFace dataset repo: {self.repo_id}")
            try:
                self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=True, exist_ok=True)
            except Exception as e:
                logger.info(f"Repo might already exist or error occurred: {e}")

        self.buffer = []
        self.current_buffer_bytes = 0
        self.max_bytes_per_shard = int(shard_size_gb * 1024 * 1024 * 1024)
        
        self.upload_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.uploader_thread = threading.Thread(target=self._uploader_loop, daemon=True)
        self.uploader_thread.start()
        
        self.shard_counter = 0

    def add(
        self,
        factory_id: str,
        worker_id: str,
        video_index: int,
        tubelet_start: int,
        latent: torch.Tensor,
        tag: str = "unknown",
        m_flow: float = -1.0,
    ):
        """
        Add a latent tensor to the buffer.
        `latent` must be a PyTorch tensor of shape [32, 24, 24, 768] (or similar).
        """
        # Ensure FP32 bytes
        if isinstance(latent, torch.Tensor):
            latent_bytes = latent.detach().cpu().to(torch.float32).numpy().tobytes()
        elif isinstance(latent, np.ndarray):
            latent_bytes = latent.astype(np.float32).tobytes()
        else:
            raise TypeError("latent must be a torch.Tensor or numpy.ndarray")

        row = {
            "factory_id": factory_id,
            "worker_id": worker_id,
            "video_index": video_index,
            "tubelet_start": tubelet_start,
            "tag": tag,
            "m_flow": float(m_flow),
            "latent_bytes": latent_bytes,
        }
        self.buffer.append(row)
        self.current_buffer_bytes += len(latent_bytes)
        
        if self.current_buffer_bytes >= self.max_bytes_per_shard:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self.buffer:
            return
            
        # Create Parquet Table
        schema = pa.schema([
            ('factory_id', pa.string()),
            ('worker_id', pa.string()),
            ('video_index', pa.int32()),
            ('tubelet_start', pa.int32()),
            ('tag', pa.string()),
            ('m_flow', pa.float32()),
            ('latent_bytes', pa.binary()),
        ])
        
        arrays = [
            pa.array([r['factory_id'] for r in self.buffer], type=pa.string()),
            pa.array([r['worker_id'] for r in self.buffer], type=pa.string()),
            pa.array([r['video_index'] for r in self.buffer], type=pa.int32()),
            pa.array([r['tubelet_start'] for r in self.buffer], type=pa.int32()),
            pa.array([r['tag'] for r in self.buffer], type=pa.string()),
            pa.array([r['m_flow'] for r in self.buffer], type=pa.float32()),
            pa.array([r['latent_bytes'] for r in self.buffer], type=pa.binary()),
        ]
        
        table = pa.Table.from_arrays(arrays, schema=schema)
        
        # Write to local disk
        shard_id = f"{uuid.uuid4().hex[:8]}_{self.shard_counter:04d}"
        local_path = self.local_dir / f"{shard_id}.parquet"
        pq.write_table(table, local_path)
        
        logger.info(f"[LatentCache] Flushed {len(self.buffer)} tubelets to {local_path.name}")
        
        # Dispatch to uploader thread
        hf_path = f"data/train/{shard_id}.parquet"
        self.upload_queue.put((local_path, hf_path))
        
        self.buffer.clear()
        self.current_buffer_bytes = 0
        self.shard_counter += 1

    def _uploader_loop(self):
        while not self.stop_event.is_set() or not self.upload_queue.empty():
            try:
                local_path, hf_path = self.upload_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                logger.info(f"[LatentCache Upload] Pushing {local_path.name} to HuggingFace...")
                t_up = time.time()
                self.api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=hf_path,
                    repo_id=self.repo_id,
                    repo_type="dataset",
                )
                logger.info(f"[LatentCache Upload] Successfully uploaded {local_path.name} in {time.time() - t_up:.1f}s.")
                # Cleanup local disk
                local_path.unlink(missing_ok=True)
                logger.debug(f"Successfully uploaded and deleted {local_path.name}")
            except Exception as e:
                logger.error(f"Failed to upload {local_path.name}: {e}. Backing off for 15s...")
                time.sleep(15)
                # Re-queue on failure
                self.upload_queue.put((local_path, hf_path))
                
            self.upload_queue.task_done()

    def close(self):
        """
        Flush remaining buffer and wait for uploads to finish.
        """
        self._flush_buffer()
        self.stop_event.set()
        
        # Wait for queue to empty
        self.upload_queue.join()
        self.uploader_thread.join()
        logger.info(f"LatentCache closed successfully.")

    @classmethod
    def fetch_index(cls, repo_id: str = "phi9-space/ego10k-vjepa-latents", hf_token: str = None) -> set:
        """
        Downloads the metadata columns from the HuggingFace Master Dataset to build a fast
        local lookup index. Used to skip V-JEPA encoding for latents that are already cached.
        """
        from datasets import load_dataset
        import huggingface_hub
        
        token = hf_token or os.environ.get("HF_TOKEN")
        api = huggingface_hub.HfApi(token=token)
        
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
        except Exception:
            # Repo doesn't exist yet, so index is empty
            return set()
            
        try:
            # Use datasets to stream only the ID columns. This is extremely fast because 
            # Parquet's columnar layout means it completely ignores the 56MB latent_bytes column.
            ds = load_dataset(repo_id, split="train", token=token)
            ds = ds.select_columns(["factory_id", "worker_id", "video_index", "tubelet_start"])
        except Exception as e:
            # No data yet
            logger.info(f"Could not load dataset (might be empty): {e}")
            return set()
            
        cached_ids = set()
        for row in ds:
            cached_ids.add((
                row["factory_id"],
                row["worker_id"],
                row["video_index"],
                row["tubelet_start"]
            ))
            
        logger.info(f"LatentCache index fetched: {len(cached_ids)} latents already cached.")
        return cached_ids
