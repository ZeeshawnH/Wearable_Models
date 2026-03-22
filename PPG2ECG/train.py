"""
Complete training pipeline for PPG2ECG conditional diffusion model.

Includes:
- Noise schedule (linear beta schedule)
- Forward diffusion process
- Training loop with validation
- Checkpointing and logging
- DDIM sampling for qualitative evaluation

Usage:
    python train.py --data_path preprocessed.npz --batch_size 32 --epochs 100
"""

import os
import sys
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Import model and dataset
from ppg2ecg_unet import ConditionalUNet1D, DDIMSampler
from dataset import get_dataloaders


class DiffusionSchedule:
    """Linear beta schedule for diffusion process."""
    
    def __init__(self, beta_start=1e-6, beta_end=1e-2, num_timesteps=1000, device='cpu'):
        """
        Args:
            beta_start: Initial beta value
            beta_end: Final beta value
            num_timesteps: Number of diffusion timesteps
            device: Device to place tensors on
        """
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        
        # Register as buffers (not parameters, just cached tensors)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
    
    def register_buffer(self, name, tensor):
        """Manually register buffer (for non-module use)."""
        setattr(self, name, tensor.to(self.device))
    
    def q_sample(self, x_0, t, eps):
        """
        Forward diffusion: q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * eps
        
        Args:
            x_0: Clean signal (B, 1, L)
            t: Timestep indices (B,)
            eps: Gaussian noise (B, 1, L)
        
        Returns:
            x_t: Noisy signal at timestep t
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t]  # (B,)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]  # (B,)
        
        # Reshape for broadcasting: (B,) -> (B, 1, 1)
        sqrt_alpha = sqrt_alpha[:, None, None]
        sqrt_one_minus_alpha = sqrt_one_minus_alpha[:, None, None]
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * eps
        return x_t


class PPGECGTrainer:
    """Trainer for conditional diffusion model."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Create directories
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # Initialize model
        self.model = ConditionalUNet1D(
            signal_length=1024,
            base_channels=args.base_channels,
            time_dim=256,
            dropout=0.2,
            scale_factor=4,
            n_res_blocks=2,
        ).to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_params_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6
        print(f"[INFO] Model parameters: {n_params:,} ({n_params_mb:.2f} MB)")
        
        # Initialize diffusion schedule
        self.schedule = DiffusionSchedule(
            beta_start=1e-6,
            beta_end=1e-2,
            num_timesteps=1000,
            device=self.device,
        )
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )
        
        # Load data
        print(f"\n[INFO] Loading data from {args.data_path}...")
        self.train_loader, self.val_loader = get_dataloaders(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
        )
        
        # Logging
        self.log_file = os.path.join(args.log_dir, 'training_log.csv')
        self.log_fields = ['epoch', 'train_loss', 'train_mae', 'val_loss', 'val_mae',
                           'val_mse_recon', 'val_mae_recon', 'val_corr_recon', 'time']
        
        # Write header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_fields)
            writer.writeheader()
        
        # DDIM sampler (for qualitative evaluation)
        self.sampler = DDIMSampler(self.model, n_timesteps=1000)
        
        # Checkpoint tracking
        self.best_val_loss = float('inf')
        self.start_epoch = 0
    
    def train_epoch(self) -> tuple:
        """Train for one epoch. Returns (avg_loss, avg_mae)."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            ppg = batch['ppg'].to(self.device)  # (B, 1, L)
            ecg = batch['ecg'].to(self.device)  # (B, 1, L)
            
            B = ecg.shape[0]
            
            # Sample random timesteps and noise
            t = torch.randint(0, 1000, (B,), device=self.device)
            eps = torch.randn_like(ecg)
            
            # Forward diffusion: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * eps
            x_t = self.schedule.q_sample(ecg, t, eps)
            
            # Predict noise
            eps_pred = self.model(x_t, ppg, t)
            
            # Loss
            loss = nn.MSELoss()(eps_pred, eps)
            mae = nn.L1Loss()(eps_pred, eps)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}: "
                      f"loss={loss.item():.6f}, mae={mae.item():.6f}")
        
        avg_loss = total_loss / len(self.train_loader)
        avg_mae = total_mae / len(self.train_loader)
        
        return avg_loss, avg_mae
    
    def validate(self) -> tuple:
        """Validate on full val set. Returns (avg_loss, avg_mae)."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                ppg = batch['ppg'].to(self.device)
                ecg = batch['ecg'].to(self.device)
                
                B = ecg.shape[0]
                t = torch.randint(0, 1000, (B,), device=self.device)
                eps = torch.randn_like(ecg)
                
                x_t = self.schedule.q_sample(ecg, t, eps)
                eps_pred = self.model(x_t, ppg, t)
                
                loss = nn.MSELoss()(eps_pred, eps)
                mae = nn.L1Loss()(eps_pred, eps)
                
                total_loss += loss.item()
                total_mae += mae.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_mae = total_mae / len(self.val_loader)
        
        return avg_loss, avg_mae
    
    def compute_reconstruction_metrics(self, n_samples=4) -> dict:
        """Run DDIM sampling and compute reconstruction metrics."""
        self.model.eval()
        
        metrics = {'mse': [], 'mae': [], 'corr': []}
        
        with torch.no_grad():
            batch_idx = 0
            for batch in self.val_loader:
                if batch_idx >= 1:  # Just use first batch
                    break
                
                ppg = batch['ppg'].to(self.device)[:n_samples]
                ecg_true = batch['ecg'].to(self.device)[:n_samples]
                
                # DDIM sampling
                ecg_recon = self.sampler.sample(ppg, n_steps=50, device=str(self.device))
                
                # Compute metrics
                ecg_true_np = ecg_true.cpu().numpy()
                ecg_recon_np = ecg_recon.cpu().numpy()
                
                for i in range(n_samples):
                    true_flat = ecg_true_np[i].flatten()
                    recon_flat = ecg_recon_np[i].flatten()
                    
                    mse = np.mean((true_flat - recon_flat) ** 2)
                    mae = np.mean(np.abs(true_flat - recon_flat))
                    corr, _ = pearsonr(true_flat, recon_flat)
                    
                    metrics['mse'].append(mse)
                    metrics['mae'].append(mae)
                    metrics['corr'].append(corr)
                
                batch_idx += 1
        
        return {
            'mse': np.mean(metrics['mse']),
            'mae': np.mean(metrics['mae']),
            'corr': np.mean(metrics['corr']),
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': self.args,
        }
        
        # Save latest
        latest_path = os.path.join(self.args.checkpoint_dir, f'epoch_{epoch}.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  [BEST] Saved best model to {best_path}")
    
    def train(self):
        """Full training loop."""
        print("\n" + "="*70)
        print("PPG2ECG DIFFUSION MODEL TRAINING")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Learning rate: {self.args.lr}")
        print("="*70 + "\n")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_start = time.time()
            
            print(f"\n[Epoch {epoch+1}/{self.args.epochs}]")
            
            # Training
            print("  Training...")
            train_loss, train_mae = self.train_epoch()
            print(f"  Train loss: {train_loss:.6f}, mae: {train_mae:.6f}")
            
            # Validation
            print("  Validating...")
            val_loss, val_mae = self.validate()
            print(f"  Val loss: {val_loss:.6f}, mae: {val_mae:.6f}")
            
            # DDIM sampling every 10 epochs
            val_mse_recon = None
            val_mae_recon = None
            val_corr_recon = None
            
            if (epoch + 1) % 10 == 0:
                print("  Computing reconstruction metrics...")
                recon_metrics = self.compute_reconstruction_metrics(n_samples=4)
                val_mse_recon = recon_metrics['mse']
                val_mae_recon = recon_metrics['mae']
                val_corr_recon = recon_metrics['corr']
                print(f"    MSE: {val_mse_recon:.6f}, MAE: {val_mae_recon:.6f}, Corr: {val_corr_recon:.4f}")
            
            # Scheduler step
            self.scheduler.step()
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, is_best=is_best)
            
            # Logging
            epoch_time = time.time() - epoch_start
            log_row = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_mse_recon': val_mse_recon if val_mse_recon is not None else '',
                'val_mae_recon': val_mae_recon if val_mae_recon is not None else '',
                'val_corr_recon': val_corr_recon if val_corr_recon is not None else '',
                'time': f'{epoch_time:.1f}s',
            }
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.log_fields)
                writer.writerow(log_row)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print(f"Best model saved to: {os.path.join(self.args.checkpoint_dir, 'best_model.pt')}")
        print(f"Logs saved to: {self.log_file}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Train conditional diffusion model for PPG→ECG reconstruction"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to preprocessed.npz'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs (default: 100)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Checkpoint directory (default: checkpoints/)'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs',
        help='Log directory (default: logs/)'
    )
    parser.add_argument(
        '--base_channels',
        type=int,
        default=64,
        help='Base channel width for model (default: 64)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of data loading workers (default: 0 for Windows)'
    )
    
    args = parser.parse_args()
    
    # Verify data path
    if not Path(args.data_path).exists():
        print(f"Error: Data file not found at {args.data_path}")
        print("Run preprocess.py first to generate preprocessed.npz")
        sys.exit(1)
    
    trainer = PPGECGTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
