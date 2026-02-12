import os
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, random_split
from models import UNet
from dataset import MSDDataset
from physics import get_physics_operator, inner_loss_func
from hoag_utils import compute_hoag_hypergradient

# ==========================================
#        1. CONFIGURATION
# ==========================================
class Config:
    DATA_ROOT = "./"
    TASK = "Task09_Spleen"
    OUTPUT_DIR = "./results_hoag_single_op"
    MODALITY = "CT"
    
    # Dataset Splits
    SUBSET_SIZE = 100
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    IMG_SIZE = 128
    BATCH_SIZE = 4
    
    # --- SINGLE PHYSICS SETTING (SPARSE) ---
    ACCEL = 16         # 16x Acceleration (The problem)
    NOISE_SIGMA = 0.1  # 10% Noise
    CENTER_FRAC = 0.08
    
    # --- OPTIMIZATION SETTINGS ---
    INNER_STEPS = 100    
    INNER_LR = 0.02      
    
    EPOCHS = 15
    LR_UNET = 1e-3
    LR_TV = 1e-3         
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
#        2. HELPER FUNCTIONS
# ==========================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        # 1. BCE Loss (Pixel-wise accuracy)
        bce_loss = self.bce(inputs, targets)
        
        # 2. Soft Dice Loss (Global overlap)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        dice_loss = 1 - dice
        
        # Combine: 90% BCE + 10% Dice is standard
        return 0.9 * bce_loss + 0.1 * dice_loss


def robust_normalize(x):
    """
    Standardizes input statistics.
    Crucial for bridging the gap between 'Clean Training' and 'Noisy Testing'.
    """
    b = x.shape[0]
    x_flat = x.view(b, -1)
    
    # Clip outliers (1st and 99th percentile)
    val_min = torch.quantile(x_flat, 0.01, dim=1, keepdim=True).view(b, 1, 1, 1)
    val_max = torch.quantile(x_flat, 0.99, dim=1, keepdim=True).view(b, 1, 1, 1)
    x = torch.clamp(x, val_min, val_max)
    
    # Min-Max Normalize to [0, 1]
    denom = val_max - val_min
    denom = torch.where(denom > 1e-7, denom, torch.ones_like(denom))
    
    return (x - val_min) / denom

def print_progress(epoch, batch, total_batches, loss, theta, info=""):
    reg_val = torch.exp(theta[0]).item()
    eps_val = torch.exp(theta[1]).item()
    sys.stdout.write(f"\r[{info}] Ep {epoch+1} | Batch {batch+1}/{total_batches} | Loss: {loss:.4f} | Reg: {reg_val:.5f} | Smooth: {eps_val:.5f}")
    sys.stdout.flush()

def validate(model, val_loader, physics_op, theta=None, steps=0, mode="clean"):
    """
    Validation Logic matching your Handwritten Note.
    """
    model.eval()
    dice_score = 0.0
    
    for i, (img, mask) in enumerate(val_loader):
        img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
        
        # --- MODE 1: CLEAN (Upper Bound) ---
        if mode == "clean":
            x_in = robust_normalize(img) # Direct Clean Input
        
        # --- MODE 2: NOISY (Lower Bound) ---
        elif mode == "noisy":
            y_clean = physics_op(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            with torch.no_grad():
                x_recon = physics_op.A_adjoint(y)
                
            x_mag = torch.sqrt(x_recon[:,0:1]**2 + x_recon[:,1:2]**2 + 1e-8)
            x_in = robust_normalize(x_mag)

        # --- MODE 3: HOAG (Optimized Reconstruction) ---
        elif mode == "hoag":
            y_clean = physics_op(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            w = physics_op.A_adjoint(y).detach().clone()
            w.requires_grad_(True)
            optimizer_inner = torch.optim.Adam([w], lr=Config.INNER_LR)
            
            with torch.enable_grad():
                for _ in range(steps):
                    optimizer_inner.zero_grad()
                    loss = inner_loss_func(w, theta, y, physics_op)
                    loss.backward()
                    optimizer_inner.step()
                    with torch.no_grad(): w.clamp_(0.0, 1.0)
            x_recon = w.detach()
            
            x_mag = torch.sqrt(x_recon[:,0:1]**2 + x_recon[:,1:2]**2 + 1e-8)
            x_in = robust_normalize(x_mag)

        # Predict
        with torch.no_grad():
            pred = (model(x_in) > 0.5).float()
            dice_score += (2. * (pred * mask).sum()) / (pred.sum() + mask.sum() + 1e-8)
            
    return dice_score.item() / len(val_loader)

# ==========================================
#        3. MAIN EXPERIMENT
# ==========================================
def run_experiment():
    print(f"--- Starting Experiment: {Config.TASK} (Approach 1 + 2) ---")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. Data
    full_ds = MSDDataset(Config.DATA_ROOT, Config.TASK, Config.IMG_SIZE, Config.MODALITY, Config.SUBSET_SIZE)
    train_len = int(Config.TRAIN_SPLIT * len(full_ds))
    val_len   = int(Config.VAL_SPLIT * len(full_ds))
    test_len  = len(full_ds) - train_len - val_len
    train_ds, val_ds, test_ds = random_split(full_ds, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)
    
    # 2. Physics (ONLY ONE OPERATOR)
    physics = get_physics_operator(Config.IMG_SIZE, Config.ACCEL, Config.CENTER_FRAC, Config.DEVICE, modality=Config.MODALITY)
    
    loss_fn = DiceBCELoss()
    results = {}
    dummy_theta = torch.tensor([-10.0, -10.0]) 
    
    # ====================================================================
    # PHASE 1: UPPER BOUND (Pre-trained phi_b)
    # ====================================================================
    print("\n--- PHASE 1: Upper Bound (Training on Clean Ground Truth) ---")
    model_upper = UNet().to(Config.DEVICE)
    ckpt_path = os.path.join(Config.OUTPUT_DIR, "model_upper_clean.pth")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path) # Force retrain to be safe

    opt = torch.optim.Adam(model_upper.parameters(), lr=Config.LR_UNET)
    
    for ep in range(Config.EPOCHS):
        model_upper.train()
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            # DIRECT CLEAN TRAINING (No physics involved here)
            x_in = robust_normalize(img) 
            
            opt.zero_grad()
            pred = model_upper(x_in)
            loss = loss_fn(pred, mask)
            loss.backward()
            opt.step()
            
            print_progress(ep, i, len(train_loader), loss.item(), dummy_theta, "Clean Training")
    print(""); torch.save(model_upper.state_dict(), ckpt_path)

    results['Upper Bound'] = validate(model_upper, test_loader, physics, theta=dummy_theta, mode="clean")
    print(f" -> Final Upper Bound (Clean): {results['Upper Bound']:.4f}")

    # ====================================================================
    # PHASE 2: LOWER BOUND
    # ====================================================================
    print("\n--- PHASE 2: Lower Bound (Testing Clean Model on Noisy Physics) ---")
    results['Lower Bound'] = validate(model_upper, test_loader, physics, theta=dummy_theta, mode="noisy")
    print(f" -> Final Lower Bound (Noisy): {results['Lower Bound']:.4f}")

    # ====================================================================
    # PHASE 3: APPROACH 1 (HOAG)
    # Note says: Fix phi_b, Optimize theta using Inner/Outer problem
    # ====================================================================
    print("\n--- PHASE 3: Approach 1 (HOAG - Optimizing Theta Only) ---")
    
    # Fixed Critic (Clean Model)
    model_fixed = UNet().to(Config.DEVICE)
    model_fixed.load_state_dict(torch.load(ckpt_path)) 
    model_fixed.train()
    for p in model_fixed.parameters(): p.requires_grad = False
    
    theta = torch.tensor([-4.6, -5.0], device=Config.DEVICE).requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_TV)
    path_hoag = os.path.join(Config.OUTPUT_DIR, "hoag_theta.pth")

    for ep in range(Config.EPOCHS): 
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            
            # Generate Noisy Data
            y_clean = physics(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            # 1. Inner Problem (Reconstruct w*)
            w_star = physics.A_adjoint(y).detach().clone()
            w_star.requires_grad_(True)
            optimizer_inner = torch.optim.Adam([w_star], lr=Config.INNER_LR)
            for _ in range(Config.INNER_STEPS):
                optimizer_inner.zero_grad()
                l_inner = inner_loss_func(w_star, theta.detach(), y, physics)
                l_inner.backward()
                optimizer_inner.step()
                with torch.no_grad(): w_star.clamp_(0.0, 1.0)
            
            # 2. Outer Problem (Gradient w.r.t Theta)
            w_star = w_star.detach().requires_grad_(True)
            x_in = robust_normalize(torch.sqrt(w_star[:,0:1]**2 + w_star[:,1:2]**2 + 1e-8))
            val_loss = loss_fn(model_fixed(x_in), mask)
            
            opt_theta.zero_grad()
            val_loss_grad_w = autograd.grad(val_loss, w_star)[0]
            
            # --- CORRECTED CALL: Included val_loss and tol ---
            hyper_grad = compute_hoag_hypergradient(
                w_star, theta, y, physics, inner_loss_func, val_loss, val_loss_grad_w, tol=1e-3
            )
            
            theta.grad = hyper_grad.clamp(-1.0, 1.0)
            opt_theta.step()
            with torch.no_grad(): theta[0].clamp_(-9.0, -2.0); theta[1].clamp_(-12.0, -2.0)
            
            print_progress(ep, i, len(train_loader), val_loss.item(), theta, "Appr 1 (HOAG)")
        print("")
        torch.save({'theta': theta}, path_hoag)

    results['Approach 1'] = validate(model_fixed, test_loader, physics, theta, Config.INNER_STEPS, mode="hoag")
    print(f" -> Final Approach 1 Score: {results['Approach 1']:.4f}")

    # ====================================================================
    # PHASE 4: APPROACH 2 (JOINT LEARNING)
    # Optimize BOTH Theta AND U-Net simultaneously
    # ====================================================================
    print("\n--- PHASE 4: Approach 2 (Joint Learning - Theta + U-Net) ---")
    
    # 1. Start with the Clean Model
    model_joint = UNet().to(Config.DEVICE)
    model_joint.load_state_dict(torch.load(ckpt_path)) # Start from clean weights
    opt_model = torch.optim.Adam(model_joint.parameters(), lr=Config.LR_UNET)
    
    # 2. Start with the Best Theta from Phase 3
    theta = torch.load(path_hoag)['theta'].to(Config.DEVICE).requires_grad_(True)
    opt_theta = torch.optim.Adam([theta], lr=Config.LR_TV)
    
    path_joint = os.path.join(Config.OUTPUT_DIR, "model_joint.pth")
    path_theta_joint = os.path.join(Config.OUTPUT_DIR, "theta_joint.pth")

    for ep in range(Config.EPOCHS):
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            y_clean = physics(torch.cat([img, torch.zeros_like(img)], 1))
            y = y_clean + Config.NOISE_SIGMA * torch.randn_like(y_clean)
            
            # A. Inner Problem (Find w*)
            w_star = physics.A_adjoint(y).detach().clone()
            w_star.requires_grad_(True)
            optimizer_inner = torch.optim.Adam([w_star], lr=Config.INNER_LR)
            for _ in range(Config.INNER_STEPS):
                optimizer_inner.zero_grad()
                l_inner = inner_loss_func(w_star, theta.detach(), y, physics)
                l_inner.backward()
                optimizer_inner.step()
                with torch.no_grad(): w_star.clamp_(0.0, 1.0)
            
            # B. Update U-Net (Outer Step 1)
            w_fixed = w_star.detach().clone().requires_grad_(False)
            x_in = robust_normalize(torch.sqrt(w_fixed[:,0:1]**2 + w_fixed[:,1:2]**2 + 1e-8))
            
            model_joint.train()
            opt_model.zero_grad()
            loss_unet = loss_fn(model_joint(x_in), mask)
            loss_unet.backward()
            opt_model.step()
            
            # C. Update Theta (Outer Step 2 - HOAG)
            model_joint.eval() # Freeze model for hypergradient calculation
            w_star = w_star.detach().requires_grad_(True) # Re-attach gradients
            x_in = robust_normalize(torch.sqrt(w_star[:,0:1]**2 + w_star[:,1:2]**2 + 1e-8))
            
            val_loss = loss_fn(model_joint(x_in), mask)
            val_loss_grad_w = autograd.grad(val_loss, w_star)[0]
            
            opt_theta.zero_grad()
            
            # --- CORRECTED CALL: Included val_loss and tol ---
            hyper_grad = compute_hoag_hypergradient(
                w_star, theta, y, physics, inner_loss_func, val_loss, val_loss_grad_w, tol=1e-3
            )
            
            theta.grad = hyper_grad.clamp(-1.0, 1.0)
            opt_theta.step()
            with torch.no_grad(): theta[0].clamp_(-9.0, -2.0); theta[1].clamp_(-12.0, -2.0)
            
            print_progress(ep, i, len(train_loader), val_loss.item(), theta, "Appr 2 (Joint)")
        print("")
        torch.save(model_joint.state_dict(), path_joint)
        torch.save({'theta': theta}, path_theta_joint)

    results['Approach 2'] = validate(model_joint, test_loader, physics, theta, Config.INNER_STEPS, mode="hoag")
    
    print("\n=== FINAL RESULTS ===")
    print(f"1. Upper Bound: {results['Upper Bound']:.4f}")
    print(f"2. Lower Bound: {results['Lower Bound']:.4f}")
    print(f"3. Approach 1:  {results['Approach 1']:.4f}")
    print(f"4. Approach 2:  {results['Approach 2']:.4f}")

if __name__ == "__main__":
    run_experiment()