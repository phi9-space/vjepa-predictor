import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from huggingface_hub import HfApi
import random
import logging
import json

from v2 import config as cfg

logger = logging.getLogger(__name__)

class Ego10kLatentStream(IterableDataset):
    """
    True IID Subsampler for V-JEPA latents.
    Dynamically pulls from HuggingFace, ensuring true random sampling per epoch.
    Uses lazy PyArrow parsing to keep memory footprint under 2GB per worker.
    """
    def __init__(self, split: str = "train", epsilon: float = 0.678, tau_kinetic: float = 0.001, seed: int = 42):
        super().__init__()
        self.split = split
        self.epsilon = epsilon
        self.tau_kinetic = tau_kinetic
        self.seed = seed
        self.repo = cfg.HF_REPO_LATENTS
        self.token = cfg.HF_TOKEN
        
        logger.info(f"Listing parquet files for {split} split...")
        api = HfApi(token=self.token)
        all_files = [f for f in api.list_repo_files(repo_id=self.repo, repo_type="dataset", token=self.token) if f.endswith('.parquet')]
        
        # Filter files for the split roughly using a hash.
        # Ego10k uses factory_id + video_idx for splitting. We approximate by splitting the files.
        # Actually, previous implementation checked split per-row. We'll do the same.
        self.files = all_files
        logger.info(f"Found {len(self.files)} total parquet files.")

    def _is_valid_split(self, factory_id: str, video_index: int) -> bool:
        val_hash = hash(f"{factory_id}_{video_index}") % 10
        if self.split == "train":
            return val_hash != 0
        else: # val
            return val_hash == 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Shuffle file list using the epoch seed to guarantee random order
        rng = random.Random(self.seed)
        shuffled_files = list(self.files)
        rng.shuffle(shuffled_files)
        
        # Partition files across workers to prevent duplication
        my_files = [f for i, f in enumerate(shuffled_files) if i % num_workers == worker_id]
        
        import requests
        import pyarrow.parquet as pq
        import tempfile
        import os
        
        yielded_count = 0
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        
        for filename in my_files:
            if yielded_count >= cfg.TUBELETS_PER_EPOCH:
                break
                
            url = f"https://huggingface.co/datasets/{self.repo}/resolve/main/{filename}"
            
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                try:
                    with requests.get(url, headers=headers, stream=True) as r:
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=8192*4):
                            if chunk:
                                tmp.write(chunk)
                    tmp.flush()
                except Exception as e:
                    logger.error(f"Failed to download {filename}: {e}")
                    os.unlink(tmp.name)
                    continue
                
            # Lazily iterate over batches via memory-mapped disk file
            try:
                pf = pq.ParquetFile(tmp.name, memory_map=True)
                for batch in pf.iter_batches(batch_size=32): # Process 32 rows at a time
                    if yielded_count >= cfg.TUBELETS_PER_EPOCH:
                        break
                        
                    d = batch.to_pydict()
                    for j in range(len(d['latent_bytes'])):
                        if yielded_count >= cfg.TUBELETS_PER_EPOCH:
                            break
                            
                        factory_id = d['factory_id'][j]
                        video_idx = d['video_index'][j]
                        
                        if not self._is_valid_split(factory_id, video_idx):
                            continue
                            
                        latent_bytes = d['latent_bytes'][j]
                        z_np = np.frombuffer(latent_bytes, dtype=np.float32).copy().reshape(32, 24, 24, 768)
                        z = torch.from_numpy(z_np)
                        
                        # Compute Temporal Persistence Filter
                        z_diff = torch.abs(z[1:] - z[:-1])
                        z_diff_l1 = z_diff.mean(dim=(1, 2, 3))
                        psi = torch.sum(torch.clamp(z_diff_l1 - self.epsilon, min=0.0)).item()
                        
                        if psi < self.tau_kinetic:
                            continue 
                            
                        z = z.permute(3, 0, 1, 2)
                        yield z
                        yielded_count += 1
                        
            except Exception as e:
                logger.error(f"Failed to parse {filename}: {e}")
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                import gc
                gc.collect()


def get_dataloaders(batch_size: int = 8, num_workers: int = 4, seed: int = 42):
    """
    Constructs the train and validation dataloaders.
    """
    calibration_path = cfg.CALIBRATION_OUTPUT_DIR / "calibration.json"
    if not calibration_path.exists():
        raise RuntimeError("calibration.json not found! Must run calibration first.")
        
    with open(calibration_path, "r") as f:
        calib_data = json.load(f)
        epsilon = calib_data["epsilon_noise_floor"]
        tau_kinetic = calib_data["tau_kinetic_gate"]
        
    train_ds = Ego10kLatentStream(split="train", epsilon=epsilon, tau_kinetic=tau_kinetic, seed=seed)
    val_ds = Ego10kLatentStream(split="val", epsilon=epsilon, tau_kinetic=tau_kinetic, seed=seed)
    
    # prefetch_factor ensures data is heavily buffered in RAM so the GPU is never starved
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True
    )
    
    return train_loader, val_loader
