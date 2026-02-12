import torch
import torch.autograd as autograd

def hessian_vector_product_exact(inner_loss_fn, w_star, theta, y, physics_op, v):
    """
    Computes (Hessian_ww * v) using Exact Autograd (Double Backpropagation).
    Mathematically: H * v = grad_w( <grad_w(Loss), v> )
    """
    # 1. Ensure w_star requires gradient for the first derivative
    # We detach it first to ensure we aren't differentiating through the optimization trace
    w_star = w_star.detach().requires_grad_(True)
    
    with torch.enable_grad():
        # First Gradient: grad_w(Loss)
        loss = inner_loss_fn(w_star, theta, y, physics_op)
        grads = autograd.grad(loss, w_star, create_graph=True)[0] # create_graph=True is CRITICAL
        
        # 2. Derivative of the Dot Product: grad_w( grads * v )
        # This trick gives us H * v exactly without forming the matrix H
        # We assume v is constant (detached) w.r.t w_star
        metric = torch.sum(grads * v)
        Hv = autograd.grad(metric, w_star, retain_graph=True)[0]
        
    return Hv

def conjugate_gradient(inner_loss_fn, w_star, theta, y, physics_op, b, max_iter=10, tol=1e-4):
    """
    Solves the linear system H * x = b for x using Conjugate Gradient.
    Corresponds to Step (ii) of the HOAG Algorithm.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.sum(r * r)

    for i in range(max_iter):
        # Check convergence condition
        if torch.sqrt(rsold) < tol:
            break
            
        # Compute Matrix-Vector Product (H * p) using Exact Autograd
        Ap = hessian_vector_product_exact(inner_loss_fn, w_star, theta, y, physics_op, p)
        
        # Damping (Tikhonov Regularization) for numerical stability
        Ap = Ap + 1e-3 * p 
        
        # Standard CG Update Steps
        alpha = rsold / (torch.sum(p * Ap) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)
        
        p = r + (rsnew / (rsold + 1e-8)) * p
        rsold = rsnew
        
    return x

def compute_hoag_hypergradient(w_star, theta, y, physics_op, inner_loss_fn, val_loss, val_loss_grad_w, tol=1e-4):
    """
    Computes the Full Hypergradient: d(Val_Loss)/d(Theta)
    
    Formula:
    grad_total = grad_direct + grad_implicit
               = d(g)/d(theta) - (d(g)/dw) * H^-1 * (d^2(h)/dw dtheta)
    
    Args:
        val_loss: The scalar validation loss (g). Needed for direct gradient.
        val_loss_grad_w: The gradient of g w.r.t w (nabla_w g).
    """
    
    # --- TERM 1: IMPLICIT GRADIENT (The "Hard" Part) ---
    # 1. Solve Linear System H * q = val_loss_grad_w (Adjoint Method)
    q = conjugate_gradient(inner_loss_fn, w_star, theta, y, physics_op, val_loss_grad_w, max_iter=20, tol=tol)
    
    # 2. Compute Cross-Derivative Product: (d/d_theta [grad_w(Inner_Loss)])^T * q
    # Trick: Compute vjp = grad_w(Inner_Loss) * q, then differentiate vjp w.r.t theta
    
    w_star_fixed = w_star.detach().requires_grad_(True)
    
    with torch.enable_grad():
        # Calculate Inner Gradient at optimum
        l_inner = inner_loss_fn(w_star_fixed, theta, y, physics_op)
        grad_w_inner = autograd.grad(l_inner, w_star_fixed, create_graph=True)[0]
    
    # The dot product of Inner Gradient and Adjoint Variable q
    vjp = torch.sum(grad_w_inner * q)
    
    # Differentiate this scalar w.r.t theta to get the Implicit Hypergradient
    # Negative sign comes from Implicit Function Theorem
    implicit_term = -autograd.grad(vjp, theta)[0]
    
    
    # --- TERM 2: DIRECT GRADIENT (The "Easy" Part) ---
    # d(val_loss)/d(theta)
    # This captures explicit dependencies of validation loss on theta (e.g., L2 penalty on params).
    # 'allow_unused=True' ensures it returns 0 instead of crashing if no dependency exists.
    
    direct_grad = autograd.grad(val_loss, theta, retain_graph=True, allow_unused=True)[0]
    
    if direct_grad is None:
        direct_grad = torch.zeros_like(theta)
        
    # --- FINAL SUM ---
    hyper_grad = direct_grad + implicit_term
    
    return hyper_grad