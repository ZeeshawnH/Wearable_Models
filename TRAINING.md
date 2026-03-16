# AMS_CVD — Training Scripts

## Overview

Standalone training scripts for each ECG classification model. Each script loads data from the PhysioNet Challenge 2021 dataset, trains the model, evaluates on a test set, and saves logs, metrics, plots, and checkpoints to `outputs/`.

## Prerequisites

```bash
pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib seaborn
```

## Project Structure

```
scripts/
  train_conv_fc.py              # ConvFcClassifier
  train_attention_conv_fc.py    # AttentionConvFcClassifier
  train_lightweight_conv_fc.py  # LightweightConvFcClassifier
  train_ams.py                  # SharedAdaptiveConvClassifier (AMS)
  run_all.py                    # Train all models sequentially
models/                         # Model definitions
utils/
  data_pipeline.py              # Shared data loading
  training.py                   # Shared train/eval/logging utilities
```

## Training Individual Models

All commands should be run from the project root (`AMS_CVD/`).

### ConvFcClassifier

```bash
python scripts/train_conv_fc.py
```

### AttentionConvFcClassifier

```bash
python scripts/train_attention_conv_fc.py
```

### LightweightConvFcClassifier

```bash
python scripts/train_lightweight_conv_fc.py
```

### SharedAdaptiveConvClassifier (AMS)

```bash
python scripts/train_ams.py
```

### Train All Models Sequentially

```bash
python scripts/run_all.py
```

## Quick Testing with Data Subset

Use `--subset` to train on a fraction of the data for faster iteration. This **does not** modify the training framework — it subsamples the DataLoaders at the script level only.

```bash
# Use 10% of the data
python scripts/train_conv_fc.py --subset 0.1

# Use 25% of the data with fewer epochs
python scripts/train_ams.py --subset 0.25 --epochs 50 --patience 10

# Run all models on 10% of data for a quick smoke test
python scripts/run_all.py --subset 0.1 --epochs 20 --patience 5
```

## All Available Arguments

Every training script accepts the same arguments:

| Argument | Default | Description |
|---|---|---|
| `--data-root` | *(PhysioNet path)* | Root directory of the training data |
| `--num-datasets` | `1` | Number of sub-datasets to load |
| `--cycle-num` | `2` | Number of ECG cycles per sample |
| `--lead-num` | `1` | Number of ECG leads |
| `--overlap` | `1` | Overlap between cycles |
| `--batch-size` | `64` | Training batch size |
| `--epochs` | `500` | Maximum training epochs |
| `--lr` | `0.00005` | Learning rate (Adam) |
| `--patience` | `50` | Early stopping patience |
| `--output-dir` | `outputs/` | Directory for run outputs |
| `--subset` | `1.0` | Fraction of data to use (0.0–1.0) |

### Example with custom arguments

```bash
python scripts/train_ams.py --epochs 200 --lr 0.0001 --patience 30 --batch-size 128
```

## Output Structure

Each training run creates a timestamped folder:

```
outputs/<ModelName>_<YYYYMMDD_HHMMSS>/
  ├── training.log              # Full training log (also printed to console)
  ├── metrics.csv               # Per-epoch: train_loss, val_loss, train_accuracy, val_accuracy
  ├── metrics.json              # Same metrics in JSON (for programmatic loading)
  ├── best_model.pth            # Best model checkpoint (lowest val loss)
  ├── training_history.png      # Loss + accuracy curves
  ├── classification_report.txt # Precision / recall / F1 per class
  └── confusion_matrix.png      # Confusion matrix heatmap
```

### Loading metrics for plotting

```python
import json
import pandas as pd

# From JSON
with open("outputs/ConvFcClassifier_20260226_120000/metrics.json") as f:
    metrics = json.load(f)

# From CSV
df = pd.read_csv("outputs/ConvFcClassifier_20260226_120000/metrics.csv")
df.plot(x="epoch", y=["train_loss", "val_loss"])
```
