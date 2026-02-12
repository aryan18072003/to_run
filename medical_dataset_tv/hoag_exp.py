import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
from hoag_utils import compute_hoag_hypergradient

# ==========================================
# 1. DEFINE SCALAR PROBLEMS
# ==========================================

def scalar_inner_loss(w, theta, y_noisy, physics_op=None):
    """
    Inner Problem: Ridge Regression (Denoising)
    L(w, theta) = 0.5(w - y)^2 + 0.5 * exp(theta) * w^2
    """
    fidelity = 0.5 * (w - y_noisy)**2
    reg = 0.5 * torch.exp(theta) * w**2
    return fidelity + reg

def scalar_dice_loss(w, target, smooth=1.0):
    """
    Outer Problem: Soft Dice Loss for a Scalar
    Dice = (2 * w * target) / (w + target)
    Loss = 1 - Dice
    """
    # Ensure strictly positive for log stability if needed, 
    # but for simple scalar dice:
    intersection = w * target
    union = w + target
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

def get_analytical_gradient_dice(w_star, theta, y_noisy, target, smooth=1.0):
    """
    Calculates EXACT Gradient via Chain Rule for verification.
    d(Loss)/d(theta) = d(Loss)/dw * d(w*)/d(theta)
    """
    # 1. d(DiceLoss)/dw
    # Quotient Rule: (u/v)' = (u'v - uv') / v^2
    # u = 2w*t + s, v = w + t + s
    # u' = 2t, v' = 1
    u = 2 * w_star * target + smooth
    v = w_star + target + smooth
    
    d_dice_dw = (2*target*v - u*1) / (v**2)
    d_outer_w = -1.0 * d_dice_dw # Because Loss = 1 - Dice
    
    # 2. d(w*)/d(theta)
    # w* = y / (1 + exp(theta))
    # deriv = -y * exp(theta) / (1 + exp(theta))^2
    denom = (1 + torch.exp(theta))
    d_w_theta = -y_noisy * torch.exp(theta) / (denom**2)
    
    return d_outer_w * d_w_theta

# ==========================================
# 2. EXPERIMENT LOOP
# ==========================================
def run_verification():
    print("--- Running HOAG Verification (Dice Loss) ---\n")
    
    # SETUP
    # We have a noisy signal '2.0'. Target is '1.0'.
    # Inner loop tries to shrink 2.0 -> 0.0 based on theta.
    # Outer loop wants w* to stop at 1.0 (Best Dice).
    y_noisy = torch.tensor([2.0])
    target = torch.tensor([1.0])
    
    # History for plotting
    history = {
        'theta': [],
        'w_star': [],
        'hoag_grad': [],
        'true_grad': [],
        'outer_loss': []
    }
    
    # Initialize Theta (High regularization initially)
    theta = torch.tensor([2.0], requires_grad=True) 
    optimizer = torch.optim.Adam([theta], lr=0.1)
    
    # --- OPTIMIZATION LOOP ---
    for step in range(25):
        
        # A. Inner Loop (Find w*)
        # Analytical solution for Ridge Regression is known: w = y / (1+e^theta)
        # We use this directly to skip inner SGD noise for cleaner plots
        w_star = y_noisy / (1 + torch.exp(theta))
        w_star = w_star.detach().requires_grad_(True)
        
        # B. Outer Loss (Dice)
        loss_outer = scalar_dice_loss(w_star, target)
        
        # C. HOAG Gradient
        grad_outer_w = autograd.grad(loss_outer, w_star)[0]
        
        hoag_grad = compute_hoag_hypergradient(
            w_star, theta, y_noisy, None, scalar_inner_loss, loss_outer, grad_outer_w, tol=1e-5
        )
        
        # D. True Gradient (Calculus)
        true_grad = get_analytical_gradient_dice(w_star, theta, y_noisy, target)
        
        # Store Data
        history['theta'].append(theta.item())
        history['w_star'].append(w_star.item())
        history['hoag_grad'].append(hoag_grad.item())
        history['true_grad'].append(true_grad.item())
        history['outer_loss'].append(loss_outer.item())
        
        # Update
        theta.grad = hoag_grad
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Step {step}: Theta={theta.item():.2f} | w*={w_star.item():.2f} | HOAG_Grad={hoag_grad.item():.5f} | True_Grad={true_grad.item():.5f}")

    # ==========================================
    # 3. PLOTTING
    # ==========================================
    fig, ax = plt.subplots(1, 2, figsize=(18, 5))
    
    # # Plot 1: Trajectory
    # ax[0].plot(history['w_star'], label='Current w* (Prediction)', marker='o')
    # ax[0].axhline(y=1.0, color='r', linestyle='--', label='Target (1.0)')
    # ax[0].set_title("Optimization Trajectory (w*)")
    # ax[0].set_xlabel("Step")
    # ax[0].set_ylabel("Value")
    # ax[0].legend()
    # ax[0].grid(True, alpha=0.3)
    
    # Plot 2: Gradient Comparison
    ax[1].plot(history['hoag_grad'], 'b-', label='HOAG Gradient', linewidth=2, alpha=0.7)
    ax[1].plot(history['true_grad'], 'r--', label='Analytical Gradient', linewidth=2)
    ax[1].set_title("Gradient Alignment Check")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Gradient Value")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    # Plot 3: Loss Landscape
    # Generate loss landscape for range of Theta
    t_vals = np.linspace(-2, 3, 100)
    l_vals = []
    for t in t_vals:
        t_ten = torch.tensor([t])
        w_s = y_noisy / (1 + torch.exp(t_ten)) # Analytical w*
        l = scalar_dice_loss(w_s, target)
        l_vals.append(l.item())
        
    ax[0].plot(t_vals, l_vals, 'k-', alpha=0.5, label='Dice Loss Surface')
    ax[0].scatter(history['theta'], history['outer_loss'], c=np.arange(len(history['theta'])), cmap='viridis', s=50, zorder=5, label='Optimization Path')
    ax[0].set_title("Optimization on Loss Surface")
    ax[0].set_xlabel("Theta (Hyperparameter)")
    ax[0].set_ylabel("Outer Dice Loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("hoag_verification_plot.png")
    print("\n[SUCCESS] Verification Complete. Plot saved to 'hoag_verification_plot.png'")
    plt.show()

if __name__ == "__main__":
    run_verification()