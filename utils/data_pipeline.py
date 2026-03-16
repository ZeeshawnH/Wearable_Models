"""
Shared data loading pipeline for all model training scripts.
Replicates the data loading logic from main.ipynb.
"""
import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import find_subfolders
from utils.custom_data_loader import load_data


def load_and_prepare_data(
    root_directory,
    num_datasets=1,
    lead_num=1,
    cycle_num=2,
    overlap=1,
    test_size=0.2,
    val_size=0.2,
    batch_size=64,
    random_state=23,
):
    """
    Load ECG data, preprocess, split, scale, and return DataLoaders.

    Returns:
        dict: Dictionary containing:
            - train_loader, val_loader, test_loader: DataLoaders
            - num_classes: Number of classification classes
            - code_to_label: Mapping from diagnosis codes to integer labels
            - cycle_num, lead_num: Data configuration
            - shapes: Dict of tensor shapes for logging
    """
    # Load data
    dataset_paths = find_subfolders(root_directory)[0:num_datasets]
    print(f"Loading datasets from: {dataset_paths}")

    df_all, period_list = load_data(
        dataset_paths, lead_num=lead_num, max_circle=None,
        cycle_num=cycle_num, overlap=overlap
    )
    print(f"Number of cycles: {len(df_all)}")

    # Map diagnosis codes to integer labels
    all_codes = set(df_all['diagnosis'])
    code_to_label = {code: idx for idx, code in enumerate(sorted(all_codes))}
    df_all['label'] = df_all['diagnosis'].apply(lambda x: code_to_label[x])
    num_classes = len(code_to_label)
    print(f"Number of classes: {num_classes}")
    print(f"Code-to-label mapping: {code_to_label}")

    # Separate features
    cycle_columns = [col for col in df_all.columns if col.startswith('lead_point_')]
    duration_columns = [col for col in df_all.columns if col.startswith('cycle_duration_')]

    X_cycles = df_all[cycle_columns].values
    X_durations = df_all[duration_columns].values
    X = np.hstack([X_cycles, X_durations])
    y = df_all['label'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale separately
    num_cycle_points = len(cycle_columns)
    scaler_cycles = MinMaxScaler()
    scaler_durations = MinMaxScaler()

    X_train_cycles_scaled = scaler_cycles.fit_transform(X_train[:, :num_cycle_points])
    X_train_durations_scaled = scaler_durations.fit_transform(X_train[:, num_cycle_points:])
    X_test_cycles_scaled = scaler_cycles.transform(X_test[:, :num_cycle_points])
    X_test_durations_scaled = scaler_durations.transform(X_test[:, num_cycle_points:])

    X_train_scaled = np.hstack([X_train_cycles_scaled, X_train_durations_scaled])
    X_test_scaled = np.hstack([X_test_cycles_scaled, X_test_durations_scaled])

    # Train/val split
    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
        X_train_scaled, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print label distributions
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{name} distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    shapes = {
        'X_train': X_train_tensor.shape,
        'X_val': X_val_tensor.shape,
        'X_test': X_test_tensor.shape,
    }

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'num_classes': num_classes,
        'code_to_label': code_to_label,
        'cycle_num': cycle_num,
        'lead_num': lead_num,
        'shapes': shapes,
    }
