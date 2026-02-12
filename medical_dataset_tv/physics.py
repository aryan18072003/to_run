import torch
import torch.nn as nn
import numpy as np
import deepinv as dinv

# ==========================================
#        1. PHYSICS OPERATOR FACTORY
# ==========================================
import torch
import numpy as np
import deepinv as dinv

import torch
import numpy as np
import deepinv as dinv

def get_physics_operator(img_size, acceleration, center_frac, device, modality="CT"):
    
    if modality == "CT":
        # --- CT PHYSICS ---
        if acceleration == 1:
            num_views = 180 
        else:
            num_views = int(180 / acceleration) 
            
        angles = torch.linspace(0, 180, num_views).to(device)
        
        # 1. Create the Physics Object
        physics = dinv.physics.Tomography(
            angles=angles,
            img_width=img_size,
            circle=False,
            device=device
        )

        # 2. === MANUAL NORMALIZATION FIX ===
        print(f"-> Computing Norm for Accel {acceleration}...")
        
        # Calculate the norm using a dummy input
        # Note: If DeepInv already normalized it internally, this will return ~1.0. 
        # If not, it will return ~1300 or ~22000. Either way, we are safe.
        norm_val = physics.compute_norm(torch.randn(1, 1, img_size, img_size, device=device))
        print(f"   Norm found: {norm_val:.2f}. Applying fix...")
        
        # FIX: Inherit from 'dinv.physics.Physics' (Base Class) to avoid recursion error
        class NormalizedPhysics(dinv.physics.Physics):
            def __init__(self, original_physics, norm):
                super().__init__() # Standard init
                self.original = original_physics
                self.norm_const = norm
                
            def A(self, x):
                return self.original.A(x) / self.norm_const
            
            def A_adjoint(self, y):
                return self.original.A_adjoint(y) / self.norm_const
                
            # Forward pass just calls A(x)
            def forward(self, x): 
                return self.A(x)

        # Wrap it
        physics = NormalizedPhysics(physics, norm_val)
        # ===================================
        
        return physics

    elif modality == "MRI":
        mask = torch.zeros((1, img_size, img_size))
        
        pad = (img_size - int(img_size * center_frac) + 1) // 2
        width = max(1, int(img_size * center_frac))
        mask[:, :, pad:pad + width] = 1.0
        
        num_keep = int(img_size / acceleration)
        all_cols = np.arange(img_size)
        kept_cols = np.where(mask[0, 0, :].cpu().numpy() == 1)[0]
        zero_cols = np.setdiff1d(all_cols, kept_cols)
        
        if len(zero_cols) > 0 and (num_keep - len(kept_cols) > 0):
            chosen = np.random.choice(zero_cols, num_keep - len(kept_cols), replace=False)
            mask[:, :, chosen] = 1.0
            
        mask = mask.to(device)
        physics = dinv.physics.MRI(mask=mask, img_size=(1, img_size, img_size), device=device)
        
        # Apply the same normalization logic to MRI if desired, 
        # but MRI usually defaults to decent scaling.
        return physics

    else:
        raise ValueError(f"Unsupported modality: {modality}")

# ==========================================
#        2. INNER LOSS (TV REGULARIZER)
# ==========================================
def inner_loss_func(w, theta, y, physics_op):
    """
    Total Variation Energy Function:
    E(w) = ||y - Aw||^2 + exp(theta_0) * TV(w, epsilon=exp(theta_1))
    """
    # Data Fidelity term (||y - Aw||^2)
    # y is a Sinogram for CT or K-space for MRI
    fid = torch.norm(y - physics_op(w), p=2)**2
    
    # Extract learned hyperparameters (exp ensures positivity)
    # theta[0]: log regularization weight (lambda)
    # theta[1]: log smoothing parameter (epsilon)
    reg_weight = torch.exp(theta[0].clamp(max=1.0)) 
    eps = torch.exp(theta[1].clamp(min=-12.0))
    
    # Compute image gradients
    dx = torch.roll(w, 1, 2) - w
    dy = torch.roll(w, 1, 3) - w
    
    # Isotropic Total Variation
    tv_penalty = torch.mean(torch.sqrt(dx**2 + dy**2 + eps))
    
    return fid + reg_weight * tv_penalty

# ==========================================
#        3. UTILITY FUNCTIONS
# ==========================================
# def robust_normalize(x):
#     """
#     Normalizes image magnitude to [0, 1] range safely for batches.
#     Handles potential division by zero.
#     """
#     b = x.shape[0]
#     x_flat = x.reshape(b, -1) 
    
#     val_min = x_flat.min(1, keepdim=True)[0].view(b, 1, 1, 1)
#     val_max = x_flat.max(1, keepdim=True)[0].view(b, 1, 1, 1)
    
#     denom = val_max - val_min
#     denom[denom < 1e-7] = 1.0 # Prevent NaN
    
#     return (x - val_min) / denom


# def robust_normalize(x):
#     """
#     Robust normalization using 1st and 99th percentiles.
#     This ignores extreme outliers (streak artifacts) from CT reconstruction.
#     """
#     b = x.shape[0]
#     x_flat = x.reshape(b, -1)
    
#     # Use 1% and 99% percentiles instead of min/max
#     val_min = torch.quantile(x_flat, 0.01, dim=1, keepdim=True).view(b, 1, 1, 1)
#     val_max = torch.quantile(x_flat, 0.99, dim=1, keepdim=True).view(b, 1, 1, 1)
    
#     # Clip values to this range first
#     x = torch.clamp(x, val_min, val_max)
    
#     denom = val_max - val_min
#     denom[denom < 1e-7] = 1.0
    
#     return (x - val_min) / denom

def robust_normalize(x):
    """
    Vectorized Percentile Normalization.
    Clips the top 1% of brightest pixels (artifacts) so they don't squash the signal.
    """
    # x shape: (Batch, Channels, Height, Width)
    b = x.shape[0]
    # Flatten spatial dims: (Batch, -1)
    x_flat = x.view(b, -1)
    
    # 1. Calculate 1st and 99th percentiles
    # Result shape: (Batch) -> Reshape to (Batch, 1, 1, 1) for broadcasting
    val_min = torch.quantile(x_flat, 0.01, dim=1).view(b, 1, 1, 1)
    val_max = torch.quantile(x_flat, 0.99, dim=1).view(b, 1, 1, 1)
    
    # 2. Clip values to this range (Ignores streak artifacts)
    x = torch.clamp(x, val_min, val_max)
    
    # 3. Scale to [0, 1]
    denom = val_max - val_min
    
    # Prevent division by zero
    denom = torch.where(denom > 1e-7, denom, torch.ones_like(denom))
    
    return (x - val_min) / denom