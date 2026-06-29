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
    """
    def __init__(self, split: str = "train", epsilon: float = 0.678, tau_kinetic: float = 0.001, seed: int = 42):
        super().__init__()
        self.split = split
        self.epsilon = epsilon
        self.tau_kinetic = tau_kinetic
        self.seed = seed

    def _is_valid_split(self, factory_id: str, video_index: int) -> bool:
        val_hash = hash(f"{factory_id}_{video_index}") % 10
        if self.split == "train":
            return val_hash != 0
        else: # val
            return val_hash == 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # Pure streaming from HuggingFace
        hf_dataset = load_dataset(cfg.HF_REPO_LATENTS, split="train", streaming=True)
        # Aggressive shuffle buffer since pod has massive bandwidth
        buffer_seed = random.randint(0, 2**32 - 1)
        hf_dataset = hf_dataset.shuffle(seed=buffer_seed, buffer_size=5000)
            
        yielded_count = 0
        
        for i, row in enumerate(hf_dataset):
            if yielded_count >= cfg.TUBELETS_PER_EPOCH:
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
