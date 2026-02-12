import torch
import torch.autograd as autograd

def hessian_vector_product(loss, w, v):
    """
    Computes (Hessian of inner_loss w.r.t w) * v efficiently.
    Formula: H*v = grad( dot(grad(L, w), v), w )
    This avoids creating the massive NxN Hessian matrix.
    """
    # 1. First derivative: dL/dw
    w_grad = autograd.grad(loss, w, create_graph=True)[0]
    
    # 2. Dot product with vector v
    prod = torch.sum(w_grad * v)
    
    # 3. Second derivative: d(prod)/dw
    hv = autograd.grad(prod, w, retain_graph=True)[0]
    return hv

def conjugate_gradient(inner_loss_fn, w_star, theta, y, physics_op, b, max_iter=5, tol=1e-4):
    """
    Solves the linear system Hx = b for x using Conjugate Gradient.
    H is the Hessian of the inner loss at optimal w_star.
    """
    # We need to re-compute the graph for w_star to get gradients
    with torch.enable_grad():
        loss = inner_loss_fn(w_star, theta, y, physics_op)

    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.sum(r * r)

    for i in range(max_iter):
        # Compute Matrix-Vector product H*p
        Ap = hessian_vector_product(loss, w_star, p)
        
        # Damping/Regularization to ensure stability (H + lambda*I)
        Ap = Ap + 1e-2 * p 
        
        # Standard CG steps
        alpha = rsold / (torch.sum(p * Ap) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)
        
        if torch.sqrt(rsnew) < tol:
            break
            
        p = r + (rsnew / (rsold + 1e-8)) * p
        rsold = rsnew
        
    return x

def compute_hoag_hypergradient(w_star, theta, y, physics_op, inner_loss_fn, val_loss_grad_w):
    """
    Computes the gradient of the Validation Loss w.r.t Theta (Hyperparams).
    Formula: p = - (d^2 E / d theta dw) * H^-1 * (d L_val / dw)
    """
    # 1. Solve Linear System: q = H^-1 * grad_w(Val_Loss)
    q = conjugate_gradient(inner_loss_fn, w_star, theta, y, physics_op, val_loss_grad_w)
    
    # 2. Compute Cross-Derivative: - (d/d_theta [grad_w(Inner_Loss) * q])
    # Re-build graph for autodiff
    with torch.enable_grad():
        l_inner = inner_loss_fn(w_star, theta, y, physics_op)
        grad_w = autograd.grad(l_inner, w_star, create_graph=True)[0]
    
    # Dot product
    vector_prod = torch.sum(grad_w * q)
    
    # Gradient w.r.t Theta
    hyper_grad = autograd.grad(vector_prod, theta, retain_graph=True)[0]
    
    # The implicit theorem has a negative sign
    return -hyper_grad