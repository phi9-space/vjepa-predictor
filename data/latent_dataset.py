import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from huggingface_hub import HfApi
import itertools
from pathlib import Path
import pyarrow.parquet as pq
import gc
import random
import logging

from v2 import config as cfg

logger = logging.getLogger(__name__)

class SafeLocalStreamer:
    """
    Robust generator that safely streams TARGETED Parquet files from a live auto-deleting directory.
    - Captures FileNotFoundError if background uploader deletes the file before opening.
    - Captures PyArrow exceptions if the background extractor is currently mid-write.
    - Enforces strict Garbage Collection to prevent RAM exhaustion.
    """
    def __init__(self, target_files: list[Path]):
        self.target_files = target_files
        
    def __iter__(self):
        for f in self.target_files:
            try:
                # Load the entire 1GB file into RAM instantly to protect it from background deletion
                table = pq.read_table(f)
            except (FileNotFoundError, OSError):
                # The blitz uploader successfully uploaded and deleted this file before we opened it.
                continue
            except Exception as e:
                # E.g., ArrowInvalid: The extraction script is currently mid-write and the footer is missing.
                continue
                
            try:
                # Yield rows safely from RAM
                for i in range(table.num_rows):
                    row = table.slice(i, 1).to_pylist()[0]
                    yield row
            finally:
                # MANDATORY to prevent 256GB OOM crash
                del table
                gc.collect()


class Ego10kLatentStream(IterableDataset):
    """
    True IID Proportional Subsampler for V-JEPA latents.
    Dynamically pulls from HuggingFace and Local Buffer proportionally to save bandwidth.
    """
    def __init__(self, split: str = "train", epsilon: float = 0.678, tau_kinetic: float = 0.001, seed: int = 42):
        super().__init__()
        self.split = split
        self.epsilon = epsilon
        self.tau_kinetic = tau_kinetic
        self.seed = seed
        self.api = HfApi()

    def _is_valid_split(self, factory_id: str, video_index: int) -> bool:
        val_hash = hash(f"{factory_id}_{video_index}") % 10
        if self.split == "train":
            return val_hash != 0
        else: # val
            return val_hash == 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # 1. Dynamically query available files
        try:
            cloud_files = [f.rfilename for f in self.api.list_repo_tree(repo_id=cfg.HF_REPO_LATENTS, repo_type='dataset', path_in_repo='data/train')]
        except Exception as e:
            logger.warning(f"Failed to query HF API, using empty cloud list: {e}")
            cloud_files = []
            
        local_dir = Path("/tmp/latent_cache/master")
        local_files = list(local_dir.glob("*.parquet")) if local_dir.exists() else []
        
        n_cloud = len(cloud_files)
        n_local = len(local_files)
        n_total = n_cloud + n_local
        
        if n_total == 0:
            logger.warning("No files found in cloud or local!")
            return
            
        # 2. Calculate proportions
        p_cloud = n_cloud / n_total
        p_local = n_local / n_total
        
        # We assume 19 tubelets per file on average
        files_needed = max(1, int(cfg.EPOCH_TUBELETS / 19))
        
        # Mathematically round to ensure exact distribution
        target_cloud_files = int(round(files_needed * p_cloud))
        target_local_files = files_needed - target_cloud_files
        
        # Cap to available (edge case)
        target_cloud_files = min(target_cloud_files, n_cloud)
        target_local_files = min(target_local_files, n_local)
        
        logger.info(f"Epoch sampling {target_cloud_files} cloud files and {target_local_files} local files.")
        
        # 3. Randomly Subsample
        random.shuffle(cloud_files)
        random.shuffle(local_files)
        
        sampled_cloud = cloud_files[:target_cloud_files]
        sampled_local = local_files[:target_local_files]
        
        # 4. Construct the targeted streams
        hf_dataset = None
        if sampled_cloud:
            hf_dataset = load_dataset(cfg.HF_REPO_LATENTS, data_files=sampled_cloud, split="train", streaming=True)
            hf_dataset = hf_dataset.shuffle(seed=self.seed, buffer_size=1000)
            
        local_dataset = SafeLocalStreamer(sampled_local)
        
        # 5. Union the streams
        if hf_dataset is not None:
            union_stream = itertools.chain(hf_dataset, local_dataset)
        else:
            union_stream = local_dataset
            
        # Yield exactly EPOCH_TUBELETS
        yielded_count = 0
        
        for i, row in enumerate(union_stream):
            if yielded_count >= cfg.EPOCH_TUBELETS:
                break
                
            if worker_info is not None:
                if i % worker_info.num_workers != worker_info.id:
                    continue
                    
            factory_id = row['factory_id']
            video_idx = row['video_index']
            
            if not self._is_valid_split(factory_id, video_idx):
                continue
                
            # Decode the latent bytes
            latent_bytes = row['latent_bytes']
            z_np = np.frombuffer(latent_bytes, dtype=np.float32).reshape(32, 24, 24, 768)
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


def get_dataloaders(batch_size: int = 8, num_workers: int = 4, seed: int = 42):
    """
    Constructs the train and validation dataloaders.
    """
    train_ds = Ego10kLatentStream(split="train", seed=seed)
    val_ds = Ego10kLatentStream(split="val", seed=seed)
    
    # prefetch_factor ensures data is heavily buffered in RAM so the GPU is never starved
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
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
