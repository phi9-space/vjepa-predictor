import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPredictor(nn.Module):
    """
    The Core Differential Kinematics Oracle.
    A Depthwise-Separable 3D Convolutional Network that acts as a local numerical
    differentiator on the V-JEPA latent manifold.
    """
    def __init__(self, causal: bool = True):
        super().__init__()
        
        # Asymmetric Padding for time (Past-biased causal or Future-biased anti-causal)
        # padding format: (left, right, top, bottom, front, back)
        # spatial is unpadded here, temporal is padded by 2 on one side
        if causal:
            self.temporal_pad = nn.ConstantPad3d((0, 0, 0, 0, 2, 0), 0.0)
        else:
            self.temporal_pad = nn.ConstantPad3d((0, 0, 0, 0, 0, 2), 0.0)

        # Depthwise Conv3D (Spatial-Temporal Physics)
        # extracts raw motion dynamics without mixing channels
        self.depthwise = nn.Conv3d(
            in_channels=768,
            out_channels=768,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=(0, 1, 1), # No temporal padding here (handled by temporal_pad)
            groups=768,
            bias=True
        )
        self.activation = nn.SiLU()

        # Feed-Forward Network (Non-Linear Semantic Mixing)
        # Expands capacity to ~4.7M parameters with an inverted bottleneck
        self.pointwise = nn.Sequential(
            nn.Conv3d(in_channels=768, out_channels=3072, kernel_size=(1, 1, 1), bias=True),
            nn.SiLU(),
            nn.Conv3d(in_channels=3072, out_channels=768, kernel_size=(1, 1, 1), bias=True)
        )

        # The Retraction Map (Scalar Gated Integration)
        # Gate sees [Z_in, Delta_Z_raw] (1536 channels) and outputs 1 channel
        self.w_gate = nn.Conv3d(
            in_channels=1536,
            out_channels=1,
            kernel_size=(1, 1, 1),
            bias=True
        )
        
        # Init: normal_(std=0.01) keeps 99.7% of pre-sigmoid in [-0.03, 0.03]
        # This forces the sigmoid to ~0.5 (balanced initial blend)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.w_gate.bias, 0.0)

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        """
        z_in shape: [B, 768, T, 24, 24]
        Returns: [B, 768, T, 24, 24]
        """
        # 1. Asymmetric Temporal Padding
        x_padded = self.temporal_pad(z_in)
        
        # 2. Depthwise (Raw Physics)
        delta_z_raw = self.depthwise(x_padded)
        delta_z_raw = self.activation(delta_z_raw)
        
        # 3. Pointwise (Semantic Rotation)
        delta_z_mixed = self.pointwise(delta_z_raw)
        
        # 4. Retraction Gate
        # Concatenate Z_in and Delta_Z_raw along the channel dimension
        gate_input = torch.cat([z_in, delta_z_raw], dim=1) # [B, 1536, T, 24, 24]
        u = torch.sigmoid(self.w_gate(gate_input))         # [B, 1, T, 24, 24]
        
        # 5. Blend
        z_target = (1 - u) * z_in + u * delta_z_mixed
        
        return z_target


def create_p_theta() -> CNNPredictor:
    """Creates the Forward/Causal Predictor."""
    return CNNPredictor(causal=True)


def create_q_theta() -> CNNPredictor:
    """Creates the Backward/Anti-Causal Predictor."""
    return CNNPredictor(causal=False)
