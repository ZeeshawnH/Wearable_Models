"""
Preprocessing pipeline for PPG2ECG diffusion model.

Loads WFDB files from SensSmartTech database, preprocesses signals,
and saves as preprocessed.npz ready for training.

Usage:
    python preprocess.py --data_path "path/to/WFDB" --demographics "path/to/Demographics.csv"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt, resample_poly
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class WFDBPreprocessor:
    """Loads and preprocesses WFDB signals for PPG→ECG diffusion training."""
    
    def __init__(self, data_path, demographics_path, window_length=1024, stride=512, seed=42):
        """
        Args:
            data_path: Path to WFDB directory
            demographics_path: Path to Demographics.csv
            window_length: Length of signal windows (default 1024 @ 1kHz = 1 sec)
            stride: Stride for sliding window (default 512)
            seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.demographics_path = Path(demographics_path)
        self.window_length = window_length
        self.stride = stride
        self.seed = seed
        
        np.random.seed(seed)
        
        # Load demographics
        self.demographics = pd.read_csv(demographics_path, skiprows=1)
        
        # Extract subject IDs for train/val split
        self.unique_subjects = sorted(self.demographics['Subject number'].unique())
        self._setup_train_val_split()
        
        # Create bandpass filters
        self.ecg_sos = self._create_filter(0.5, 40, 1000, filter_type='bandpass')
        self.ppg_sos = self._create_filter(0.5, 8, 1000, filter_type='bandpass')
        self.acc_sos = self._create_filter(0, 20, 1000, filter_type='lowpass')
        
        print(f"[INFO] Preprocessor initialized")
        print(f"  Data path: {data_path}")
        print(f"  Window length: {window_length} samples @ 1kHz = {window_length/1000:.2f}s")
        print(f"  Stride: {stride} samples")
        print(f"  Unique subjects: {len(self.unique_subjects)}")
        print(f"  Train subjects: {len(self.train_subjects)}")
        print(f"  Val subjects: {len(self.val_subjects)}")
    
    def _setup_train_val_split(self, train_ratio=0.8):
        """Setup subject-level train/val split."""
        n_train = int(len(self.unique_subjects) * train_ratio)
        self.train_subjects = set(self.unique_subjects[:n_train])
        self.val_subjects = set(self.unique_subjects[n_train:])
    
    @staticmethod
    def _create_filter(low_cutoff, high_cutoff, fs, filter_type='bandpass', order=4):
        """Create a Butterworth filter (SOS format for stability)."""
        if filter_type == 'bandpass':
            sos = butter(order, [low_cutoff, high_cutoff], btype='bandpass', fs=fs, output='sos')
        elif filter_type == 'lowpass':
            sos = butter(order, high_cutoff, btype='lowpass', fs=fs, output='sos')
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        return sos
    
    def find_matching_recordings(self):
        """Scan WFDB directory and find all ECG/PPG/ACC triplets."""
        recordings = []
        
        # Get all unique recording identifiers from the data directory
        dat_files = list(self.data_path.glob('*.dat'))
        identifiers = set()
        
        for f in dat_files:
            # Parse filename: {subject}_{time}_{modality}.dat
            parts = f.stem.split('_')
            if len(parts) >= 3:
                modality = parts[-1]
                recording_id = '_'.join(parts[:-1])  # e.g., "1_10-09-54"
                identifiers.add(recording_id)
        
        # For each identifier, check if we have ECG, PPG, ACC
        for recording_id in sorted(identifiers):
            ecg_file = self.data_path / f"{recording_id}_ecg.dat"
            ppg_file = self.data_path / f"{recording_id}_ppg.dat"
            acc_file = self.data_path / f"{recording_id}_acc.dat"
            
            ecg_hea = self.data_path / f"{recording_id}_ecg.hea"
            ppg_hea = self.data_path / f"{recording_id}_ppg.hea"
            acc_hea = self.data_path / f"{recording_id}_acc.hea"
            
            if all([f.exists() for f in [ecg_file, ppg_file, acc_file, ecg_hea, ppg_hea, acc_hea]]):
                # Extract subject ID (first part of recording_id)
                subject_id = int(recording_id.split('_')[0])
                recordings.append({
                    'recording_id': recording_id,
                    'subject_id': subject_id,
                    'ecg_file': ecg_file,
                    'ppg_file': ppg_file,
                    'acc_file': acc_file,
                })
            else:
                missing = []
                if not ecg_file.exists() or not ecg_hea.exists():
                    missing.append('ECG')
                if not ppg_file.exists() or not ppg_hea.exists():
                    missing.append('PPG')
                if not acc_file.exists() or not acc_hea.exists():
                    missing.append('ACC')
                print(f"[WARN] Skipping {recording_id}: missing {', '.join(missing)}")
        
        return recordings
    
    def load_wfdb_record(self, file_path):
        """Load WFDB record and return signal array."""
        record = wfdb.rdrecord(str(file_path.with_suffix('')))
        return record.p_signal, record.fs
    
    def preprocess_ecg(self, signal):
        """Preprocess ECG: bandpass filter + resample + normalize."""
        # signal shape: (N,) or (N, 4) for 4 leads
        if len(signal.shape) > 1:
            signal = signal[:, 0]  # Take lead I (channel 0)
        
        # Bandpass filter
        filtered = sosfiltfilt(self.ecg_sos, signal)
        
        # Resample from 1000 Hz to 125 Hz (downsample by factor of 8)
        resampled = resample_poly(filtered, up=1, down=8)
        
        # Normalize to zero mean, unit variance
        mean = np.mean(resampled)
        std = np.std(resampled)
        if std > 1e-6:
            normalized = (resampled - mean) / std
        else:
            normalized = resampled - mean
        
        return normalized, mean, std
    
    def preprocess_ppg(self, signal):
        """Preprocess PPG: bandpass filter + resample + normalize."""
        if len(signal.shape) > 1:
            signal = signal[:, 0]  # Take carotid 660nm (channel 0)
        
        # Bandpass filter
        filtered = sosfiltfilt(self.ppg_sos, signal)
        
        # Resample from 1000 Hz to 125 Hz (downsample by factor of 8)
        resampled = resample_poly(filtered, up=1, down=8)
        
        # Normalize
        mean = np.mean(resampled)
        std = np.std(resampled)
        if std > 1e-6:
            normalized = (resampled - mean) / std
        else:
            normalized = resampled - mean
        
        return normalized, mean, std
    
    def preprocess_acc(self, signal):
        """Preprocess ACC: lowpass filter + resample + normalize."""
        if len(signal.shape) > 1:
            signal = signal[:, 0]  # Take z-axis (channel 0)
        
        # Lowpass filter
        filtered = sosfiltfilt(self.acc_sos, signal)
        
        # Resample from 1000 Hz to 125 Hz (downsample by factor of 8)
        resampled = resample_poly(filtered, up=1, down=8)
        
        # Normalize
        mean = np.mean(resampled)
        std = np.std(resampled)
        if std > 1e-6:
            normalized = (resampled - mean) / std
        else:
            normalized = resampled - mean
        
        return normalized, mean, std
    
    def create_windows(self, signal, window_length, stride):
        """Slide window over signal and return list of windows."""
        windows = []
        for start in range(0, len(signal) - window_length + 1, stride):
            window = signal[start:start + window_length]
            windows.append(window)
        return windows
    
    def is_valid_window(self, signal, min_variance=1e-4, nan_threshold=0.05):
        """Check if window is valid (not too much NaN, not flatline)."""
        # Check for NaNs
        nan_ratio = np.isnan(signal).sum() / len(signal)
        if nan_ratio > nan_threshold:
            return False
        
        # Check for flatline
        var = np.nanvar(signal)
        if var < min_variance:
            return False
        
        return True
    
    def process_recording(self, recording):
        """Load, preprocess, and window a single recording."""
        recording_id = recording['recording_id']
        subject_id = recording['subject_id']
        
        try:
            # Load signals
            ecg_signal, fs_ecg = self.load_wfdb_record(recording['ecg_file'])
            ppg_signal, fs_ppg = self.load_wfdb_record(recording['ppg_file'])
            acc_signal, fs_acc = self.load_wfdb_record(recording['acc_file'])
            
            # Verify sampling rates (should all be 1000 Hz)
            assert fs_ecg == 1000, f"ECG sampling rate {fs_ecg} != 1000"
            assert fs_ppg == 1000, f"PPG sampling rate {fs_ppg} != 1000"
            assert fs_acc == 1000, f"ACC sampling rate {fs_acc} != 1000"
            
            # Preprocess signals
            ecg_processed, _, _ = self.preprocess_ecg(ecg_signal)
            ppg_processed, _, _ = self.preprocess_ppg(ppg_signal)
            acc_processed, _, _ = self.preprocess_acc(acc_signal)
            
            # Ensure all signals have same length
            min_len = min(len(ecg_processed), len(ppg_processed), len(acc_processed))
            ecg_processed = ecg_processed[:min_len]
            ppg_processed = ppg_processed[:min_len]
            acc_processed = acc_processed[:min_len]
            
            # Create windows
            ecg_windows = self.create_windows(ecg_processed, self.window_length, self.stride)
            ppg_windows = self.create_windows(ppg_processed, self.window_length, self.stride)
            acc_windows = self.create_windows(acc_processed, self.window_length, self.stride)
            
            # Filter valid windows
            valid_windows = []
            for ecg_w, ppg_w, acc_w in zip(ecg_windows, ppg_windows, acc_windows):
                if self.is_valid_window(ecg_w) and self.is_valid_window(ppg_w) and self.is_valid_window(acc_w):
                    valid_windows.append((ecg_w, ppg_w, acc_w))
            
            # Determine split
            split = 'train' if subject_id in self.train_subjects else 'val'
            
            return {
                'ecg_windows': [w[0] for w in valid_windows],
                'ppg_windows': [w[1] for w in valid_windows],
                'acc_windows': [w[2] for w in valid_windows],
                'subject_id': subject_id,
                'recording_id': recording_id,
                'split': split,
                'n_windows': len(valid_windows),
            }
        
        except Exception as e:
            print(f"[ERROR] Failed to process {recording_id}: {e}")
            return None
    
    def run(self, output_path='preprocessed.npz'):
        """Run full preprocessing pipeline."""
        print("\n" + "="*70)
        print("PPG2ECG PREPROCESSING PIPELINE")
        print("="*70)
        
        # Find recordings
        print("\n[1] Discovering recordings...")
        recordings = self.find_matching_recordings()
        print(f"[INFO] Found {len(recordings)} valid recordings")
        
        # Process all recordings
        print("\n[2] Processing recordings...")
        all_ecg = []
        all_ppg = []
        all_acc = []
        all_subject_ids = []
        all_recording_ids = []
        all_splits = []
        
        n_processed = 0
        n_skipped = 0
        
        for i, recording in enumerate(recordings):
            result = self.process_recording(recording)
            if result is not None:
                n_processed += 1
                all_ecg.extend(result['ecg_windows'])
                all_ppg.extend(result['ppg_windows'])
                all_acc.extend(result['acc_windows'])
                all_subject_ids.extend([result['subject_id']] * result['n_windows'])
                all_recording_ids.extend([i] * result['n_windows'])
                all_splits.extend([result['split']] * result['n_windows'])
                
                split_str = f"[{result['split'].upper()}]"
                print(f"  [{i+1}/{len(recordings)}] {recording['recording_id']} {split_str}: "
                      f"{result['n_windows']} windows")
            else:
                n_skipped += 1
        
        # Convert to arrays
        ecg_arr = np.array(all_ecg, dtype=np.float32)  # (N, 1024)
        ppg_arr = np.array(all_ppg, dtype=np.float32)
        acc_arr = np.array(all_acc, dtype=np.float32)
        subject_arr = np.array(all_subject_ids, dtype=np.int32)
        recording_arr = np.array(all_recording_ids, dtype=np.int32)
        split_arr = np.array(all_splits, dtype=str)
        
        # Add channel dimension
        ecg_arr = ecg_arr[:, np.newaxis, :]  # (N, 1, 1024)
        ppg_arr = ppg_arr[:, np.newaxis, :]
        acc_arr = acc_arr[:, np.newaxis, :]
        
        # Summary
        print("\n" + "="*70)
        print("PREPROCESSING SUMMARY")
        print("="*70)
        print(f"Total recordings found: {len(recordings)}")
        print(f"Total recordings processed: {n_processed}")
        print(f"Total recordings skipped: {n_skipped}")
        print(f"Total windows created: {len(all_ecg)}")
        print(f"  - Train: {np.sum(split_arr == 'train')}")
        print(f"  - Val: {np.sum(split_arr == 'val')}")
        print(f"\nOutput shapes:")
        print(f"  ECG: {ecg_arr.shape} (N, 1, 1024)")
        print(f"  PPG: {ppg_arr.shape} (N, 1, 1024)")
        print(f"  ACC: {acc_arr.shape} (N, 1, 1024)")
        print(f"\nSignal: 1 kHz sampling rate, 1024 samples = 1.024 seconds per window")
        
        # Save to npz
        print(f"\n[3] Saving to {output_path}...")
        np.savez(
            output_path,
            ecg=ecg_arr,
            ppg=ppg_arr,
            acc=acc_arr,
            subject_ids=subject_arr,
            recording_ids=recording_arr,
            split=split_arr,
        )
        print(f"[INFO] Preprocessed data saved to {output_path}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess WFDB files for PPG→ECG diffusion training"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to WFDB directory'
    )
    parser.add_argument(
        '--demographics',
        type=str,
        required=True,
        help='Path to Demographics.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='preprocessed.npz',
        help='Output file path (default: preprocessed.npz)'
    )
    parser.add_argument(
        '--window_length',
        type=int,
        default=1024,
        help='Length of signal windows in samples (default: 1024 @ 1kHz)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=512,
        help='Stride for sliding window (default: 512)'
    )
    
    args = parser.parse_args()
    
    preprocessor = WFDBPreprocessor(
        data_path=args.data_path,
        demographics_path=args.demographics,
        window_length=args.window_length,
        stride=args.stride,
    )
    
    preprocessor.run(output_path=args.output)


if __name__ == '__main__':
    main()
