#!/usr/bin/env python3
"""
Run inference on a single ECG record using a saved model checkpoint.

Usage:
    python scripts/predict.py <checkpoint_path> <record_path>

Example:
    python scripts/predict.py outputs/ConvFcClassifier_20260226_143022/final_model.pth \
        /path/to/physionet/training/WFDB_CPSC2018/A0001

The record_path should point to a .hea/.mat file pair (without extension).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
from scipy.signal import resample

from utils.training import load_saved_model
from utils.helper_code import get_frequency, get_labels, get_num_samples
from utils.utils import butterworth_elgendi_rpeak


def load_single_record(record_path, lead_num=1, cycle_num=2, target_fs=500):
    """
    Load and preprocess a single ECG record into the same format
    the training pipeline uses.

    Args:
        record_path: Path to the record (without extension), e.g. .../A0001
        lead_num: Which lead to use (1-indexed)
        cycle_num: Number of cardiac cycles to extract
        target_fs: Target sampling frequency

    Returns:
        np.ndarray: Feature vector (cycle points + durations), shape (1, N)
        dict: Metadata about the record
    """
    header_file = record_path + '.hea'
    mat_file = record_path + '.mat'

    # Read header
    with open(header_file, 'r') as f:
        header = f.read()

    fs = get_frequency(header)
    num_samples = get_num_samples(header)

    # Read signal
    mat_data = loadmat(mat_file)
    # Try common key names
    for key in ['val', 'ECG', 'data', 'signal']:
        if key in mat_data:
            signal = mat_data[key].astype(np.float64)
            break
    else:
        # Use the first non-metadata key
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        signal = mat_data[keys[0]].astype(np.float64)

    # Select lead (0-indexed)
    lead_idx = lead_num - 1
    if signal.shape[0] < signal.shape[1]:
        ecg = signal[lead_idx, :]  # Shape: (leads, samples)
    else:
        ecg = signal[:, lead_idx]  # Shape: (samples, leads)

    # Detect R-peaks
    r_peaks = butterworth_elgendi_rpeak(ecg, fs)

    if len(r_peaks) < cycle_num + 1:
        raise ValueError(
            f"Not enough R-peaks ({len(r_peaks)}) to extract {cycle_num} cycles. "
            f"Need at least {cycle_num + 1} R-peaks."
        )

    # Extract cycles (same logic as custom_data_loader)
    start = r_peaks[0]
    end = r_peaks[cycle_num]
    segment = ecg[start:end]

    # Get cycle durations (in samples)
    durations = []
    for i in range(cycle_num):
        durations.append(r_peaks[i + 1] - r_peaks[i])

    # Resample segment to fixed length
    target_length = cycle_num * 250  # Same as training pipeline
    segment_resampled = resample(segment, target_length)

    # Combine into feature vector
    features = np.concatenate([segment_resampled, durations])

    # Get labels from header if available
    try:
        labels = get_labels(header)
    except Exception:
        labels = []

    metadata = {
        'sampling_frequency': fs,
        'num_samples': num_samples,
        'r_peaks_found': len(r_peaks),
        'original_labels': labels,
        'record_path': record_path,
    }

    return features.reshape(1, -1), metadata


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single ECG record")
    parser.add_argument("checkpoint", type=str, help="Path to the saved .pth checkpoint")
    parser.add_argument("record", type=str, help="Path to the ECG record (without extension)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda, mps")
    parser.add_argument("--lead-num", type=int, default=1, help="Lead number to use (1-indexed)")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, checkpoint = load_saved_model(args.checkpoint, device=args.device)

    model_name = checkpoint['model_name']
    model_params = checkpoint.get('model_params', {})
    cycle_num = model_params.get('cycle_num', 2)

    print(f"  Model: {model_name}")
    print(f"  Test accuracy (from training): {checkpoint.get('test_accuracy', 'N/A')}")
    print(f"  cycle_num={cycle_num}, lead_num={args.lead_num}")

    # Load and preprocess the record
    print(f"\nLoading record from {args.record}...")
    features, metadata = load_single_record(
        args.record, lead_num=args.lead_num, cycle_num=cycle_num
    )
    print(f"  Sampling freq: {metadata['sampling_frequency']} Hz")
    print(f"  R-peaks found: {metadata['r_peaks_found']}")
    print(f"  Original labels: {metadata['original_labels']}")

    # Scale with MinMaxScaler (fit on the single sample — for proper use,
    # you'd want to save/load the training scalers too)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Convert to tensor: (1, 1, num_features) for Conv1d
    x = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(1).to(args.device)

    # Run inference
    with torch.no_grad():
        logits = model(x, modes=('lightweight', 'moderate', 'advanced'))
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    # Map back to diagnosis code if available
    saved_args = checkpoint.get('args', {})
    code_to_label = saved_args.get('code_to_label', None)
    if code_to_label:
        label_to_code = {v: k for k, v in code_to_label.items()}
        pred_label = label_to_code.get(pred_class, f"class_{pred_class}")
    else:
        pred_label = f"class_{pred_class}"

    print(f"\n{'='*50}")
    print(f"  Prediction:  {pred_label}")
    print(f"  Confidence:  {confidence:.4f}")
    print(f"  Class index: {pred_class}")
    print(f"  All probabilities: {probs[0].cpu().numpy()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
