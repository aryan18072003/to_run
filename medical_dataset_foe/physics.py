import torch
import torch.nn as nn
import numpy as np
import deepinv as dinv

class FieldOfExperts(nn.Module):
    def __init__(self, J=10, kernel_size=7):
        super().__init__()
        """
        Stabilized Field of Experts (FoE) with Learnable Step Size.
        Formula: R(x) = alpha * log(1 + beta * (K*x)^2)
        """
        self.J = J
        
        self.conv = nn.Conv2d(1, J, kernel_size, padding=kernel_size//2, bias=False)
        
        nn.init.orthogonal_(self.conv.weight)
        
        self.log_alpha = nn.Parameter(torch.zeros(J) - 0.5) 
        self.log_beta = nn.Parameter(torch.zeros(J)) 
        self.log_global = nn.Parameter(torch.tensor(-2.0))
        
        
        self.log_step_size = nn.Parameter(torch.tensor(np.log(0.1)))

    def get_step_size(self):
        """Returns the positive step size for the physics loop."""
        return torch.exp(self.log_step_size)

    def forward(self, x):
        z = self.conv(x)
        
        alpha = torch.exp(self.log_alpha).view(1, -1, 1, 1)
        beta = torch.exp(self.log_beta).view(1, -1, 1, 1)
        glob = torch.exp(self.log_global)
        
        potential = alpha * torch.log(1 + beta * z**2)
        
        return glob * torch.mean(torch.sum(potential, dim=1))

def get_physics_operator(img_size, acceleration, center_frac, device):
    
    mask = torch.zeros((1, img_size, img_size))
    pad = (img_size - int(img_size * center_frac) + 1) // 2
    width = max(1, int(img_size * center_frac))
    mask[:, :, pad:pad + width] = 1.0
    
    num_keep = int(img_size / acceleration)
    all_cols = np.arange(img_size)
    kept_cols = np.where(mask[0, 0, :].numpy() == 1)[0]
    zero_cols = np.setdiff1d(all_cols, kept_cols)
    if len(zero_cols) > 0 and (num_keep - len(kept_cols) > 0):
        chosen = np.random.choice(zero_cols, num_keep - len(kept_cols), replace=False)
        mask[:, :, chosen] = 1.0
        
    mask = mask.to(device)
    physics = dinv.physics.MRI(mask=mask, img_size=(1, img_size, img_size), device=device)
    return physics

def robust_normalize(x):
    b = x.shape[0]
    x_flat = x.reshape(b, -1) 
    val_min = x_flat.min(1, keepdim=True)[0].view(b,1,1,1)
    val_max = x_flat.max(1, keepdim=True)[0].view(b,1,1,1)
    denom = val_max - val_min
    denom[denom < 1e-6] = 1.0
    return (x - val_min) / denom