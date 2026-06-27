import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

from v2 import config as cfg

class Ego10kLatentStream(IterableDataset):
    """
    Streaming Parquet loader for V-JEPA latents from HuggingFace.
    Implements the Temporal Persistence Filter to drop statically inert scenes on the fly.
    """
    def __init__(self, split: str = "train", epsilon: float = 0.678, tau_kinetic: float = 0.001, seed: int = 42):
        super().__init__()
        self.split = split
        self.epsilon = epsilon
        self.tau_kinetic = tau_kinetic
        
        # Load streaming dataset from HF
        # The parquet files are all in train split by default based on our upload structure
        self.hf_dataset = load_dataset(cfg.HF_REPO_LATENTS, split="train", streaming=True)
        
        # We perform a pseudo-split for validation using deterministic hashing of the tubelet_idx or just skipping
        # But wait, HF streaming datasets can be sharded.
        # For simplicity, we just use the HF `train` split and deterministically filter based on the string hash of factory_id + video_index.
        # 10% for val, 90% for train.
        
    def _is_valid_split(self, factory_id: str, video_index: int) -> bool:
        # Simple deterministic hash for 90/10 split
        val_hash = hash(f"{factory_id}_{video_index}") % 10
        if self.split == "train":
            return val_hash != 0
        else: # val
            return val_hash == 0

    def __iter__(self):
        # When using multiple DataLoader workers, we need to shard the iterable
        worker_info = torch.utils.data.get_worker_info()
        
        # In streaming mode, datasets handles multi-processing shards automatically if we split by worker
        # But let's just use modulo arithmetic on the stream if standard IterableDataset
        
        for i, row in enumerate(self.hf_dataset):
            if worker_info is not None:
                if i % worker_info.num_workers != worker_info.id:
                    continue
                    
            factory_id = row['factory_id']
            video_idx = row['video_index']
            
            if not self._is_valid_split(factory_id, video_idx):
                continue
                
            # Decode the latent bytes
            latent_bytes = row['latent_bytes']
            # Shape is [32, 24, 24, 768] float32 as saved in LatentCache
            z_np = np.frombuffer(latent_bytes, dtype=np.float32).reshape(32, 24, 24, 768)
            z = torch.from_numpy(z_np) # [32, 24, 24, 768]
            
            # Compute Temporal Persistence Filter
            # Psi = sum_{t=1}^{31} max(0, ||Z_t - Z_{t-1}||_1 - epsilon)
            # L1 diff across all spatial and channel dims per frame
            
            z_diff = torch.abs(z[1:] - z[:-1]) # [31, 24, 24, 768]
            z_diff_l1 = z_diff.mean(dim=(1, 2, 3)) # mean L1 per frame transition
            
            psi = torch.sum(torch.clamp(z_diff_l1 - self.epsilon, min=0.0)).item()
            
            if psi < self.tau_kinetic:
                continue # Drop this tubelet, it's statically inert
                
            # Permute to [C, T, H, W] for the CNN Predictor -> [768, 32, 24, 24]
            z = z.permute(3, 0, 1, 2)
            
            # Optimization window: 30 frames
            # Forward Predictor (P_theta): input t:t+29, target t+1:t+30
            # Backward Predictor (Q_theta): input t+1:t+30, target t:t+29
            
            # Since z is 32 frames long, we just use 0:30 and 1:31
            # We can return the full 32 frames and let the trainer slice it, or slice it here.
            # Slicing it here to save memory: we need 0:30 and 1:31
            
            # Actually, returning z_full [768, 32, 24, 24] is cleanest and allows the trainer to do P and Q natively
            yield z

def get_dataloaders(batch_size: int = 8, num_workers: int = 4):
    """
    Constructs the train and validation dataloaders.
    """
    train_ds = Ego10kLatentStream(split="train")
    val_ds = Ego10kLatentStream(split="val")
    
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
