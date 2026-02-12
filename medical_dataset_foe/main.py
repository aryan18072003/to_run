import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import MSDDataset
from models import UNet
from physics import get_physics_operator, robust_normalize, FieldOfExperts

class Config:
    DATA_ROOT = "./"
    TASK = "Task09_Spleen"
    OUTPUT_DIR = "./results_foe_final"
    
    MODALITY = "CT"
    SUBSET_SIZE = 100
    IMG_SIZE = 128
    BATCH_SIZE = 4
    
    ACCEL = 4
    CENTER_FRAC = 0.08
    HOAG_STEPS = 40
    
    EPOCHS = 30
    LR_UNET = 1e-3
    LR_FOE  = 5e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_progress(epoch, batch, total_batches, loss, info=""):
    sys.stdout.write(f"\r[{info}] Ep {epoch+1}/{Config.EPOCHS} | Batch {batch+1}/{total_batches} | Loss: {loss:.4f}")
    sys.stdout.flush()

def validate(model, val_loader, physics_op, foe_model=None, steps=0, mode="clean"):
    model.eval()
    dice_score = 0.0
    
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            y = physics_op(torch.cat([img, torch.zeros_like(img)], 1))
            
            if mode == "clean":
                x_clean = physics_op.A_adjoint(y)
                x_mag = torch.sqrt(x_clean[:,0:1]**2 + x_clean[:,1:2]**2 + 1e-8)
                x_in = robust_normalize(x_mag)
            
            elif mode == "noisy":
                x_naive = physics_op.A_adjoint(y)
                x_mag = torch.sqrt(x_naive[:,0:1]**2 + x_naive[:,1:2]**2 + 1e-8)
                x_in = robust_normalize(x_mag)
                
            elif mode == "foe":
                w = physics_op.A_adjoint(y).detach().clone()
                w.requires_grad_(True)
                step_size = foe_model.get_step_size()
                
                for _ in range(steps):
                    with torch.enable_grad():
                        fid = torch.norm(y - physics_op(w), p=2)**2
                        w_mag = torch.sqrt(w[:,0:1]**2 + w[:,1:2]**2 + 1e-8)
                        reg = foe_model(w_mag)
                        loss = fid + reg
                        gw = autograd.grad(loss, w)[0]
                    
                    w = w - step_size * gw
                    w = w.clamp(0.0, 1.0)
                    
                    w = w.detach()
                    w.requires_grad_(True)
                
                w = w.detach()
                x_in = robust_normalize(torch.sqrt(w[:,0:1]**2 + w[:,1:2]**2 + 1e-8))

            pred = (model(x_in) > 0.5).float()
            dice_score += (2. * (pred * mask).sum()) / (pred.sum() + mask.sum() + 1e-8)
            
    return dice_score.item() / len(val_loader)

def run_experiment():
    print(f"--- Starting Experiment: {Config.TASK} on {Config.DEVICE} ---")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    full_ds = MSDDataset(Config.DATA_ROOT, Config.TASK, Config.IMG_SIZE, Config.MODALITY, Config.SUBSET_SIZE)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    physics_under = get_physics_operator(Config.IMG_SIZE, Config.ACCEL, Config.CENTER_FRAC, Config.DEVICE)
    physics_full = get_physics_operator(Config.IMG_SIZE, 1, 1.0, Config.DEVICE) 
    loss_fn = nn.BCELoss()
    results = {}

    print("\n--- PHASE 1: Upper Bound ---")
    model = UNet().to(Config.DEVICE)
    ckpt_path = os.path.join(Config.OUTPUT_DIR, "model_upper.pth")
    
    if os.path.exists(ckpt_path):
        print("-> Loading Checkpoint...")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=Config.LR_UNET)
        for ep in range(Config.EPOCHS):
            model.train()
            for i, (img, mask) in enumerate(train_loader):
                img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
                y = physics_full(torch.cat([img, torch.zeros_like(img)], 1))
                x_clean = physics_full.A_adjoint(y)
                x_in = robust_normalize(torch.sqrt(x_clean[:,0:1]**2 + x_clean[:,1:2]**2 + 1e-8))
                opt.zero_grad(); loss = loss_fn(model(x_in), mask); loss.backward(); opt.step()
                print_progress(ep, i, len(train_loader), loss.item(), "Upper Bound")
        print("")
        torch.save(model.state_dict(), ckpt_path)
    
    results['Upper Bound'] = validate(model, val_loader, physics_full, mode="clean")
    print(f" -> Final Upper Bound: {results['Upper Bound']:.4f}")

    print("\n--- PHASE 2: Lower Bound ---")
    results['Lower Bound'] = validate(model, val_loader, physics_under, mode="noisy")
    print(f" -> Final Lower Bound: {results['Lower Bound']:.4f}")

    print("\n--- PHASE 3: Approach 1 (FoE Optimization) ---")
    model_fixed = UNet().to(Config.DEVICE)
    model_fixed.load_state_dict(torch.load(ckpt_path))
    model_fixed.train() 
    for p in model_fixed.parameters(): p.requires_grad = False
    
    foe_model = FieldOfExperts(J=10).to(Config.DEVICE)
    opt_theta = torch.optim.Adam(foe_model.parameters(), lr=Config.LR_FOE)
    app1_path = os.path.join(Config.OUTPUT_DIR, "checkpoint_app1.pth")
    best_dice = 0.0

    if os.path.exists(app1_path):
        print("-> Loading Approach 1 Checkpoint...")
        foe_model.load_state_dict(torch.load(app1_path))
    else:
        for ep in range(Config.EPOCHS):
            curr_step = foe_model.get_step_size().item()
            for i, (img, mask) in enumerate(train_loader):
                img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
                y = physics_under(torch.cat([img, torch.zeros_like(img)], 1))
                
                w = physics_under.A_adjoint(y).detach().clone().requires_grad_(True)
                for _ in range(Config.HOAG_STEPS):
                    fid = torch.norm(y - physics_under(w), p=2)**2
                    w_mag = torch.sqrt(w[:,0:1]**2 + w[:,1:2]**2 + 1e-8)
                    reg = foe_model(w_mag)
                    loss_inner = fid + reg
                    gw = autograd.grad(loss_inner, w, create_graph=True)[0]
                    step = foe_model.get_step_size()
                    w = w - step * gw 
                    w = w.clamp(0.0, 1.0)
                
                x_in = robust_normalize(torch.sqrt(w[:,0:1]**2 + w[:,1:2]**2 + 1e-8))
                opt_theta.zero_grad(); loss = loss_fn(model_fixed(x_in), mask); loss.backward()
                torch.nn.utils.clip_grad_norm_(foe_model.parameters(), 1.0); opt_theta.step()
                print_progress(ep, i, len(train_loader), loss.item(), f"Appr 1 | Step: {curr_step:.4f}")
            
            print("")
            curr_dice = validate(model_fixed, val_loader, physics_under, foe_model, Config.HOAG_STEPS, mode="foe")
            if curr_dice > best_dice:
                best_dice = curr_dice
                torch.save(foe_model.state_dict(), app1_path)

    foe_model.load_state_dict(torch.load(app1_path))
    results['Approach 1'] = validate(model_fixed, val_loader, physics_under, foe_model, Config.HOAG_STEPS, mode="foe")
    print(f" -> Final Approach 1 Score: {results['Approach 1']:.4f}")

    print("\n--- PHASE 4: Approach 2 (Joint Training) ---")
    model_joint = UNet().to(Config.DEVICE)
    model_joint.load_state_dict(torch.load(ckpt_path))
    foe_joint = FieldOfExperts(J=10).to(Config.DEVICE)
    foe_joint.load_state_dict(torch.load(app1_path))
    
    opt_model = torch.optim.Adam(model_joint.parameters(), lr=Config.LR_UNET)
    opt_theta = torch.optim.Adam(foe_joint.parameters(), lr=Config.LR_FOE)
    
    app2_model = os.path.join(Config.OUTPUT_DIR, "checkpoint_app2_model.pth")
    app2_foe = os.path.join(Config.OUTPUT_DIR, "checkpoint_app2_foe.pth")
    best_dice = 0.0

    for ep in range(Config.EPOCHS):

        model_joint.train(); 
        for p in foe_joint.parameters(): p.requires_grad = False
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            y = physics_under(torch.cat([img, torch.zeros_like(img)], 1))
            
            w = physics_under.A_adjoint(y).detach().clone()
            w.requires_grad_(True)
            
            for _ in range(Config.HOAG_STEPS):
                fid = torch.norm(y - physics_under(w), p=2)**2
                reg = foe_joint(torch.sqrt(w[:,0:1]**2 + w[:,1:2]**2 + 1e-8))
                loss_inner = fid + reg
                
                if torch.is_grad_enabled():
                    gw = autograd.grad(loss_inner, w, create_graph=False)[0]
                
                w = w - foe_joint.get_step_size() * gw
                w = w.clamp(0.0, 1.0)
                w = w.detach()
                w.requires_grad_(True)
            
            x_in = robust_normalize(torch.sqrt(w[:,0:1]**2 + w[:,1:2]**2 + 1e-8))
            
            opt_model.zero_grad()
            loss = loss_fn(model_joint(x_in), mask)
            loss.backward()
            opt_model.step()
            
            print_progress(ep, i, len(train_loader), loss.item(), "Appr 2 (U-Net)")
        print("")


        model_joint.eval(); 
        for p in foe_joint.parameters(): p.requires_grad = True
        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(Config.DEVICE), mask.to(Config.DEVICE)
            y = physics_under(torch.cat([img, torch.zeros_like(img)], 1))
            
            w = physics_under.A_adjoint(y).detach().clone().requires_grad_(True)
            for _ in range(Config.HOAG_STEPS):
                fid = torch.norm(y - physics_under(w), p=2)**2
                reg = foe_joint(torch.sqrt(w[:,0:1]**2 + w[:,1:2]**2 + 1e-8))
                loss_inner = fid + reg
                gw = autograd.grad(loss_inner, w, create_graph=True)[0]
                w = w - foe_joint.get_step_size() * gw
                w = w.clamp(0.0, 1.0)
            
            x_in = robust_normalize(torch.sqrt(w[:,0:1]**2 + w[:,1:2]**2 + 1e-8))
            opt_theta.zero_grad(); loss = loss_fn(model_joint(x_in), mask); loss.backward()
            torch.nn.utils.clip_grad_norm_(foe_joint.parameters(), 1.0); opt_theta.step()
            print_progress(ep, i, len(train_loader), loss.item(), "Appr 2 (FoE)")
        print("")
        
        # Val
        curr_dice = validate(model_joint, val_loader, physics_under, foe_joint, Config.HOAG_STEPS, mode="foe")
        print(f"   -> Val Dice: {curr_dice:.4f}")
        if curr_dice > best_dice:
            best_dice = curr_dice
            torch.save(model_joint.state_dict(), app2_model)
            torch.save(foe_joint.state_dict(), app2_foe)

    results['Approach 2'] = best_dice
    print("\n=== FINAL RESULTS ===")
    print(f"1. Upper Bound: {results['Upper Bound']:.4f}")
    print(f"2. Lower Bound: {results['Lower Bound']:.4f}")
    print(f"3. Approach 1:  {results['Approach 1']:.4f}")
    print(f"4. Approach 2:  {results['Approach 2']:.4f}")

if __name__ == "__main__":
    run_experiment()