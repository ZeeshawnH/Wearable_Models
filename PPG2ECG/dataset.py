"""
PyTorch Dataset and DataLoader for PPG2ECG diffusion training.

Loads preprocessed.npz and provides train/val dataloaders with
on-the-fly augmentation for training set.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class PPGECGDataset(Dataset):
    """
    PyTorch Dataset for PPG→ECG reconstruction from preprocessed NPZ files.
    
    On-the-fly augmentation for training set:
    - Random amplitude scaling (0.8-1.2)
    - Random time shift (±50 samples, circular)
    - Additive Gaussian noise on PPG only
    """
    
    def __init__(self, npz_path: str, split: str = 'train', seed: int = 42):
        """
        Args:
            npz_path: Path to preprocessed.npz
            split: 'train' or 'val'
            seed: Random seed for reproducibility
        """
        assert split in ['train', 'val'], f"split must be 'train' or 'val', got {split}"
        
        self.split = split
        self.seed = seed
        np.random.seed(seed)
        
        # Load preprocessed data
        data = np.load(npz_path)
        ecg_all = data['ecg']  # (N, 1, 1024)
        ppg_all = data['ppg']  # (N, 1, 1024)
        acc_all = data['acc']  # (N, 1, 1024)
        splits = data['split']
        
        # Filter by split
        mask = splits == split
        self.ecg = ecg_all[mask].astype(np.float32)  # (N_split, 1, 1024)
        self.ppg = ppg_all[mask].astype(np.float32)
        self.acc = acc_all[mask].astype(np.float32)
        
        self.n_samples = len(self.ecg)
        
        print(f"[INFO] Loaded {self.n_samples} samples for split '{split}'")
        print(f"  ECG shape: {self.ecg.shape}")
        print(f"  PPG shape: {self.ppg.shape}")
        print(f"  ACC shape: {self.acc.shape}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def _augment(self, ppg: np.ndarray, ecg: np.ndarray, acc: np.ndarray) -> Tuple:
        """Apply augmentations (training only)."""
        
        # 1. Random amplitude scaling (affects PPG, not ECG or ACC for now)
        scale_ppg = np.random.uniform(0.8, 1.2)
        ppg = ppg * scale_ppg
        
        # 2. Random time shift (circular shift, same for all signals)
        shift = np.random.randint(-50, 51)
        if shift != 0:
            ppg = np.roll(ppg, shift, axis=-1)
            ecg = np.roll(ecg, shift, axis=-1)
            acc = np.roll(acc, shift, axis=-1)
        
        # 3. Additive Gaussian noise on PPG only
        noise_std = np.random.uniform(0, 0.05)
        if noise_std > 0:
            ppg = ppg + np.random.normal(0, noise_std, ppg.shape)
        
        return ppg, ecg, acc
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                'ppg': tensor(1, 1024) - conditioning signal
                'ecg': tensor(1, 1024) - reconstruction target (clean)
                'acc': tensor(1, 1024) - optional secondary conditioning
        """
        ppg = self.ppg[idx].copy()
        ecg = self.ecg[idx].copy()
        acc = self.acc[idx].copy()
        
        # Apply augmentations only to training set
        if self.split == 'train':
            ppg, ecg, acc = self._augment(ppg, ecg, acc)
        
        return {
            'ppg': torch.from_numpy(ppg).float(),
            'ecg': torch.from_numpy(ecg).float(),
            'acc': torch.from_numpy(acc).float(),
        }


def get_dataloaders(
    npz_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        npz_path: Path to preprocessed.npz
        batch_size: Batch size
        num_workers: Number of data loading workers (0 for Windows)
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = PPGECGDataset(npz_path, split='train')
    val_dataset = PPGECGDataset(npz_path, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"\n[INFO] DataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


# Quick test
if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # For testing, assume preprocessed.npz is in current directory
    npz_path = 'preprocessed.npz'
    
    if not Path(npz_path).exists():
        print(f"Error: {npz_path} not found. Run preprocess.py first.")
        sys.exit(1)
    
    print("Testing PPGECGDataset and DataLoaders...")
    train_loader, val_loader = get_dataloaders(npz_path, batch_size=4, num_workers=0)
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  PPG: {batch['ppg'].shape}")
    print(f"  ECG: {batch['ecg'].shape}")
    print(f"  ACC: {batch['acc'].shape}")
