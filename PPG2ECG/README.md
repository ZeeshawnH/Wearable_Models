# PPG2ECG Conditional Diffusion Pipeline

Complete training pipeline for reconstructing ECG signals from PPG conditioning signals using a conditional diffusion model.

## Files

- **`ppg2ecg_unet.py`** (existing) - ConditionalUNet1D model and DDIMSampler
- **`preprocess.py`** - Data loading, preprocessing, and windowing script
- **`dataset.py`** - PyTorch Dataset/DataLoader with on-the-fly augmentation
- **`train.py`** - Full training loop with validation, checkpointing, and logging

## Quick Start

### 1. Preprocess Data

```bash
python preprocess.py \
  --data_path "C:\Users\Zeeshawn\College\GuoResearch\data\senssmarttech-database-of-cardiovascular-signals-synchronously-recorded-by-an-electrocardiograph-phonocardiograph-photoplethysmograph-and-accelerometer-1.0.0\WFDB" \
  --demographics "C:\Users\Zeeshawn\College\GuoResearch\data\senssmarttech-database-of-cardiovascular-signals-synchronously-recorded-by-an-electrocardiograph-phonocardiograph-photoplethysmograph-and-accelerometer-1.0.0\Demographics.csv"
```

This will:

- Scan WFDB directory for ECG/PPG/ACC triplets
- Apply bandpass filters (ECG: 0.5-40Hz, PPG: 0.5-8Hz, ACC: 20Hz lowpass)
- Normalize each signal to zero mean, unit variance
- Create 1024-sample sliding windows (stride 512)
- Perform subject-level 80/20 train/val split
- Save `preprocessed.npz` containing all windows and metadata

### 2. Train Model

```bash
python train.py \
  --data_path preprocessed.npz \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4
```

This will:

- Create train/val dataloaders with on-the-fly augmentation
- Train conditional UNet with linear diffusion schedule (T=1000)
- Log training metrics to `logs/training_log.csv`
- Save best model to `checkpoints/best_model.pt`
- Save periodic checkpoints every 10 epochs
- Compute reconstruction metrics (MSE, MAE, Pearson correlation) every 10 epochs

## Data Format

### Input (WFDB Files)

- **ECG**: 4 channels → use channel 0 (lead I)
- **PPG**: 4 channels → use channel 0 (carotid 660nm)
- **ACC**: 1 channel (z-axis accelerometer)
- **Sampling rate**: 1000 Hz
- **Signal length**: 30 seconds = 30,000 samples

### Preprocessed Format (`preprocessed.npz`)

- `ecg`: shape (N, 1, 1024) float32 - reconstruction targets
- `ppg`: shape (N, 1, 1024) float32 - conditioning signals
- `acc`: shape (N, 1, 1024) float32 - optional conditioning
- `subject_ids`: shape (N,) int32 - subject ID for each window
- `recording_ids`: shape (N,) int32 - recording index
- `split`: shape (N,) str - 'train' or 'val' (subject-level)

## Augmentation (Training Only)

Applied on-the-fly in training set:

- **Amplitude scaling**: PPG × uniform(0.8, 1.2)
- **Time shift**: Circular shift by ±50 samples (same for all signals)
- **Noise**: Gaussian noise σ ~ uniform(0, 0.05) added to PPG only

## Model Architecture

**ConditionalUNet1D**:

- Input: noisy ECG x_t (B, 1, 1024), PPG conditioning (B, 1, 1024), timestep t (B,)
- Output: predicted noise (B, 1, 1024)
- Encoder: 2 residual blocks, base_channels=64
- Bottleneck: 2 residual blocks + global self-attention
- Decoder: 2 residual blocks with skip connections
- Total parameters: ~420K

## Noise Schedule

**Linear beta schedule**:

- β_start = 1e-4
- β_end = 0.02
- T = 1000 timesteps
- Forward diffusion: x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε

## Training Details

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR over all epochs
- **Loss**: MSE between predicted and true noise
- **Validation**: Forward diffusion only (no DDIM sampling)
- **Reconstruction metrics** (every 10 epochs): MSE, MAE, Pearson correlation via DDIM sampling (50 steps)

## Output Structure

```
PPG2ECG/
├── ppg2ecg_unet.py
├── preprocess.py
├── dataset.py
├── train.py
├── preprocessed.npz          (output of preprocess.py)
├── checkpoints/
│   ├── best_model.pt
│   ├── epoch_10.pt
│   ├── epoch_20.pt
│   └── ...
└── logs/
    └── training_log.csv
```

## Hyperparameters

Defaults (all configurable via argparse):

- `batch_size`: 32
- `epochs`: 100
- `lr`: 1e-4
- `base_channels`: 64
- `window_length`: 1024
- `stride`: 512
- `train_split`: 0.8 (80% train, 20% val)

## Notes

- All random seeds set to 42 for reproducibility
- Subject-level train/val split to avoid data leakage
- Windows with >5% NaN or near-zero variance are discarded
- Signals normalized per-recording (not globally)
- Supports GPU training (auto-detected, falls back to CPU)
