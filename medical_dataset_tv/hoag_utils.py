import torch
import torch.autograd as autograd

def hessian_vector_product_fd(inner_loss_fn, w, theta, y, physics_op, v, eps=1e-3):
    """
    Approximates (Hessian of inner_loss w.r.t w) * v using Finite Differences.
    This bypasses the 'cudnn_grid_sampler_backward' unimplemented error.
    Formula: Hv approx (grad(w + eps*v) - grad(w - eps*v)) / (2*eps)
    """
    # Create perturbed versions of w (Treat as new leaves)
    w_plus = (w + eps * v).detach().requires_grad_(True)
    w_minus = (w - eps * v).detach().requires_grad_(True)

    # Compute gradient at w + eps*v
    # We only need 1st derivative here, which IS implemented for CT
    with torch.enable_grad():
        loss_plus = inner_loss_fn(w_plus, theta, y, physics_op)
        grad_plus = autograd.grad(loss_plus, w_plus)[0]

    # Compute gradient at w - eps*v
    with torch.enable_grad():
        loss_minus = inner_loss_fn(w_minus, theta, y, physics_op)
        grad_minus = autograd.grad(loss_minus, w_minus)[0]

    return (grad_plus - grad_minus) / (2 * eps)

def conjugate_gradient(inner_loss_fn, w_star, theta, y, physics_op, b, max_iter=5, tol=1e-4):
    """
    Solves Hx = b using Conjugate Gradient with Finite Difference HVP.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.sum(r * r)

    for i in range(max_iter):
        # Use Finite Difference HVP instead of exact Hessian
        Ap = hessian_vector_product_fd(inner_loss_fn, w_star, theta, y, physics_op, p)
        
        # Damping for stability (Regularization)
        Ap = Ap + 1e-2 * p 
        
        alpha = rsold / (torch.sum(p * Ap) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)
        
        if torch.sqrt(rsnew.clamp(min=1e-10)) < tol:
            break
            
        p = r + (rsnew / (rsold + 1e-8)) * p
        rsold = rsnew
        
    return x

def compute_hoag_hypergradient(w_star, theta, y, physics_op, inner_loss_fn, val_loss_grad_w):
    """
    Computes Hypergradient using Implicit Function Theorem + Finite Difference HVP.
    """
    # 1. Solve Linear System using Finite Difference CG
    q = conjugate_gradient(inner_loss_fn, w_star, theta, y, physics_op, val_loss_grad_w)
    
    # 2. Compute Cross-Derivative Term: d/d_theta [grad_w(Inner_Loss) * q]
    # This part usually works with autodiff because it's a first-order derivative w.r.t theta
    # We detach w_star to treat it as a constant for this derivative
    w_star_fixed = w_star.detach().requires_grad_(True)
    
    with torch.enable_grad():
        l_inner = inner_loss_fn(w_star_fixed, theta, y, physics_op)
        grad_w = autograd.grad(l_inner, w_star_fixed, create_graph=True)[0]
    
    vector_prod = torch.sum(grad_w * q)
    
    # Negative sign comes from the Implicit Function Theorem
    hyper_grad = -autograd.grad(vector_prod, theta)[0]
    
    return hyper_grad