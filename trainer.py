import torch
import torch.optim as optim
import math
import logging
from pathlib import Path
from huggingface_hub import HfApi

from v2 import config as cfg
from v2.models import create_p_theta, create_q_theta
from v2.training import AdaptiveHuberLoss
from v2.data import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def evaluate(p_theta, q_theta, val_loader, loss_fn, device, steps: int = 50):
    p_theta.eval()
    q_theta.eval()
    
    total_loss = 0.0
    
    # 1. Compute global variance of the validation set to fix the Huber threshold
    logger.info("Computing global validation variance without memory buffering...")
    var_sum = 0.0
    var_count = 0
    with torch.inference_mode():
        for i, z_batch in enumerate(val_loader):
            if i >= 10: # Just sample 10 batches to estimate variance
                break
            # Compute variance directly on device for this single batch
            z_batch_device = z_batch.to(device)
            var_sum += torch.var(z_batch_device).item()
            var_count += 1
            del z_batch_device
            
    delta_val = math.sqrt((var_sum / var_count) + cfg.HUBER_ETA)
    logger.info(f"Fixed Validation Delta: {delta_val:.4f}")
    
    # 2. Compute validation loss
    logger.info("Running validation pass...")
    with torch.inference_mode():
        for i, z_batch in enumerate(val_loader):
            if i >= steps:
                break
                
            z_batch = z_batch.to(device)
            # z_batch is [B, 768, 32, 24, 24]
            
            # Forward Target Shifting
            z_in_fwd = z_batch[:, :, 0:30, :, :]
            z_target_fwd = z_batch[:, :, 1:31, :, :]
            
            # Backward Target Shifting
            z_in_rev = z_batch[:, :, 1:31, :, :]
            z_target_rev = z_batch[:, :, 0:30, :, :]
            
            z_pred_fwd = p_theta(z_in_fwd)
            z_pred_rev = q_theta(z_in_rev)
            
            loss_fwd = loss_fn(z_pred_fwd, z_target_fwd, fixed_delta=delta_val)
            loss_rev = loss_fn(z_pred_rev, z_target_rev, fixed_delta=delta_val)
            
            loss = loss_fwd + loss_rev
            total_loss += loss.item()
            
    avg_loss = total_loss / steps
    return avg_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    # Prime Directive: STRICTLY FP32. NO AMP.
    torch.backends.cudnn.allow_tf32 = False # Ensure pure FP32
    torch.backends.cuda.matmul.allow_tf32 = False
    
    # 1. Initialize Models
    p_theta = create_p_theta().to(device)
    q_theta = create_q_theta().to(device)
    
    # 2. Dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=cfg.BATCH_SIZE, num_workers=4)
    train_iter = iter(train_loader)
    
    # 3. Optimizer
    params = list(p_theta.parameters()) + list(q_theta.parameters())
    optimizer = optim.AdamW(
        params, 
        lr=cfg.LR, 
        weight_decay=cfg.WEIGHT_DECAY,
        betas=cfg.ADAM_BETAS,
        eps=cfg.ADAM_EPS
    )
    
    # 4. Loss and Scheduler
    loss_fn = AdaptiveHuberLoss().to(device)
    # Cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info("Starting Phase 2 Training (CNN Compressor)")
    for epoch in range(1, cfg.EPOCHS + 1):
        p_theta.train()
        q_theta.train()
        
        epoch_loss = 0.0
        
        # We use TUBELETS_PER_EPOCH to define a virtual "epoch" over the streaming dataset
        # Since BATCH_SIZE is 8, steps per epoch is TUBELETS_PER_EPOCH / BATCH_SIZE
        steps_per_epoch = cfg.TUBELETS_PER_EPOCH // cfg.BATCH_SIZE
        if steps_per_epoch == 0: steps_per_epoch = 100
        
        for step in range(steps_per_epoch):
            try:
                z_batch = next(train_iter)
            except StopIteration:
                logger.info("Dataset stream exhausted! Dynamically pulling latest files from HuggingFace...")
                # Recreate the dataloader with a new seed to fetch any newly uploaded parquet shards
                train_loader, val_loader = get_dataloaders(batch_size=cfg.BATCH_SIZE, num_workers=4, seed=epoch+step)
                train_iter = iter(train_loader)
                z_batch = next(train_iter)
                
            z_batch = z_batch.to(device)
            # z_batch is [B, 768, 32, 24, 24]
            
            # Target Shifting (30-frame window)
            z_in_fwd = z_batch[:, :, 0:30, :, :]
            z_target_fwd = z_batch[:, :, 1:31, :, :]
            
            z_in_rev = z_batch[:, :, 1:31, :, :]
            z_target_rev = z_batch[:, :, 0:30, :, :]
            
            optimizer.zero_grad()
            
            # Predict
            z_pred_fwd = p_theta(z_in_fwd)
            z_pred_rev = q_theta(z_in_rev)
            
            # Loss (Adaptive)
            loss_fwd = loss_fn(z_pred_fwd, z_target_fwd)
            loss_rev = loss_fn(z_pred_rev, z_target_rev)
            
            loss = loss_fwd + loss_rev
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(params, cfg.GRAD_CLIP_NORM)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % 10 == 0:
                logger.info(f"Epoch {epoch} | Step {step}/{steps_per_epoch} | Loss: {loss.item():.4f}")
                
        scheduler.step()
        
        # Validation
        val_loss = evaluate(p_theta, q_theta, val_loader, loss_fn, device, steps=20)
        logger.info(f"--- Epoch {epoch} Complete | Train Loss: {epoch_loss/steps_per_epoch:.4f} | Val Loss: {val_loss:.4f} ---")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            cfg.TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                'p_theta': p_theta.state_dict(),
                'q_theta': q_theta.state_dict(),
            }, cfg.TRAINING_OUTPUT_DIR / "best_oracles.pt")
            logger.info("Saved new best model.")
            
            # Upload to HuggingFace
            if cfg.HF_TOKEN and hasattr(cfg, 'HF_REPO_MODEL'):
                try:
                    logger.info(f"Uploading checkpoint to HuggingFace ({cfg.HF_REPO_MODEL})...")
                    api = HfApi(token=cfg.HF_TOKEN)
                    api.create_repo(repo_id=cfg.HF_REPO_MODEL, private=True, exist_ok=True)
                    api.upload_file(
                        path_or_fileobj=str(cfg.TRAINING_OUTPUT_DIR / "best_oracles.pt"),
                        path_in_repo="best_oracles.pt",
                        repo_id=cfg.HF_REPO_MODEL,
                        repo_type="model"
                    )
                    logger.info("Successfully uploaded best_oracles.pt to HuggingFace!")
                except Exception as e:
                    logger.error(f"Failed to upload model to HuggingFace: {e}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                logger.info(f"Thermodynamic Validation Plateau reached! No improvement for {cfg.PATIENCE} epochs.")
                break

if __name__ == "__main__":
    train()
