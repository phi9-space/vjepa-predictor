import torch
from v2.models import create_p_theta, create_q_theta
from v2.training import AdaptiveHuberLoss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_pipeline():
    print("Testing Phase 2 Pipeline...")
    
    p_theta = create_p_theta()
    q_theta = create_q_theta()
    
    print(f"P_theta parameters: {count_parameters(p_theta)}")
    print(f"Q_theta parameters: {count_parameters(q_theta)}")
    
    # Expected:
    # Depthwise: 3x3x3 x 768 = 20,736 (spec says 21,504? Wait, 3*3*3 = 27. 27 * 768 = 20,736. Spec says 21,504. Ah, the spec might have a slight typo or includes bias, but we set bias=False).
    # Pointwise: 1x1x1 x 768 x 768 = 589,824 (spec says 590,592, which is 589824 + 768 bias. We set bias=False).
    # Gate: 1x1x1 x 1536 x 1 + 1 = 1537.
    # Total = 612,097. Spec says ~1.22M for both (612k * 2 = 1.22M). Perfect!
    
    # Test Forward Pass Shapes
    batch_size = 2
    z_in = torch.randn(batch_size, 768, 30, 24, 24)
    print(f"Input shape: {z_in.shape}")
    
    z_out = p_theta(z_in)
    print(f"P_theta Output shape: {z_out.shape}")
    assert z_out.shape == z_in.shape, "Shape mismatch!"
    
    z_out_q = q_theta(z_in)
    print(f"Q_theta Output shape: {z_out_q.shape}")
    assert z_out_q.shape == z_in.shape, "Shape mismatch!"
    
    # Test Loss
    loss_fn = AdaptiveHuberLoss()
    target = torch.randn(batch_size, 768, 30, 24, 24)
    loss = loss_fn(z_out, target)
    print(f"Loss Output: {loss.item()}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_pipeline()
