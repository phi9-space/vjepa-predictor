import torch
import torch.nn as nn
import torch.nn.functional as F

from v2 import config as cfg

class AdaptiveHuberLoss(nn.Module):
    """
    Adaptive Volatility-Gated Huber Loss.
    Maintains an Exponential Moving Average (EMA) of batch variance to dynamically
    adjust the Huber threshold, ensuring continuous L2 behavior despite abrupt
    discontinuities (e.g. violent head whips).
    """
    def __init__(self):
        super().__init__()
        
        # State buffer for EMA so it persists across forward passes and device transfers
        self.register_buffer('delta_t_minus_1', torch.tensor(1.0))
        self.register_buffer('initialized', torch.tensor(False))
        
    def forward(self, input: torch.Tensor, target: torch.Tensor, freeze_threshold: bool = False, fixed_delta: float = None) -> torch.Tensor:
        """
        Computes the adaptive Huber loss.
        
        Args:
            input: Predicted latents [B, 768, T, 24, 24]
            target: Target latents [B, 768, T, 24, 24]
            freeze_threshold: If True, uses the existing delta_t_minus_1 without updating it (used for validation if fixed_delta is None).
            fixed_delta: If provided, completely overrides the adaptive mechanism (used for decoupled validation).
            
        Returns:
            A scalar loss value.
        """
        if fixed_delta is not None:
            delta = fixed_delta
        else:
            # 1. Calculate current batch variance (standard deviation)
            sigma_t = torch.sqrt(torch.var(target) + cfg.HUBER_ETA)
            
            if not self.initialized:
                self.delta_t_minus_1.copy_(sigma_t)
                self.initialized.copy_(torch.tensor(True))
                
            if not freeze_threshold:
                # 2. Calculate relative volatility shock
                v_t = torch.abs(sigma_t - self.delta_t_minus_1) / (self.delta_t_minus_1 + cfg.HUBER_ETA)
                
                # 3. Compute adaptive momentum
                beta_t = cfg.HUBER_BETA_MIN + (cfg.HUBER_BETA_MAX - cfg.HUBER_BETA_MIN) * torch.exp(-cfg.HUBER_ALPHA * v_t)
                
                # 4. Update Huber threshold EMA
                delta_t = beta_t * self.delta_t_minus_1 + (1 - beta_t) * sigma_t
                
                # Save for next step
                self.delta_t_minus_1.copy_(delta_t)
                delta = delta_t.item()
            else:
                delta = self.delta_t_minus_1.item()
                
        # 5. Compute Huber loss per spatial-temporal coordinate
        # The sum over channels at each spatial-temporal position, returning a scalar per (t, x, y)
        # Then mean over batch, t, x, y.
        # F.huber_loss with reduction='mean' computes the mean over ALL elements.
        # Whitepaper says: "sums over the 768 channels... returning a scalar per (t,x,y). E_fwd = 1/(B*30*576) sum_{b,t,x,y} L_huber"
        # Since it sums over channels (768) and averages over B*T*H*W, the loss magnitude will be ~768x larger than default mean reduction.
        
        raw_huber = F.huber_loss(input, target, reduction='none', delta=delta)
        # Sum over channels (dim=1)
        spatial_temporal_loss = raw_huber.sum(dim=1) # [B, T, 24, 24]
        
        return spatial_temporal_loss.mean()
