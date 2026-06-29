import os
import time
import torch
import config as cfg
from data.latent_dataset import get_dataloaders
import traceback

def benchmark_dataloader():
    # Make sure we use the test data we just downloaded
    # But wait, latent_dataset.py hardcodes self.local_dir = "/root/v2/data/data/train"
    # So we'll symlink our benchmark data there
    
    cfg.BATCH_SIZE = 8
    
    print("Initializing DataLoader with num_workers=8...")
    try:
        train_loader, val_loader = get_dataloaders(
            batch_size=cfg.BATCH_SIZE,
            num_workers=8 # 8 workers to guarantee we stay well under 125GB
        )
        
        print("Starting DataLoader benchmark...")
        start_time = time.time()
        
        total_batches = 100
        
        # We just iterate to test PyArrow decompression throughput
        for i, batch in enumerate(train_loader):
            # batch is [8, 768, 32, 24, 24]
            # Move to device to simulate the exact VRAM transfer
            b = batch.to("cuda")
            
            if (i+1) % 10 == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (i+1) / elapsed
                print(f"Processed {i+1} batches... ({batches_per_sec:.2f} batches/sec) | GPU Mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                
            if i >= total_batches:
                break
                
        total_time = time.time() - start_time
        print(f"Finished {total_batches} batches in {total_time:.2f} seconds.")
        print(f"Final Throughput: {total_batches / total_time:.2f} batches/sec")
    except Exception as e:
        print("Error during benchmark:")
        traceback.print_exc()

if __name__ == "__main__":
    benchmark_dataloader()
