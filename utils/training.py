"""
Training and evaluation utilities for all model training scripts.
Extracted from main.ipynb with added logging and metric persistence.
"""
import os
import json
import csv
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for scripts
import matplotlib.pyplot as plt


def setup_logging(output_dir, model_name):
    """
    Set up logging to both console and file.

    Returns:
        logging.Logger: Configured logger instance.
        str: Path to the output directory for this run.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "training.log")

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger, run_dir


def train_classifier(
    model, train_loader, val_loader, epochs=100, lr=0.0001,
    patience=10, device='cuda', logger=None, run_dir=None
):
    """
    Train a classification model with early stopping based on validation loss.
    Saves per-epoch metrics to CSV and best model checkpoint.

    Returns:
        nn.Module: The trained model with the best validation performance.
        list: Training loss history.
        list: Validation loss history.
        list: Validation accuracy history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []

    _log(logger, f"Starting training: epochs={epochs}, lr={lr}, patience={patience}, device={device}")
    _log(logger, f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    start_time = time.time()

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs, modes=('lightweight', 'moderate', 'advanced'))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels).item()
            train_total += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / train_total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()

                outputs = model(inputs, modes=('lightweight', 'moderate', 'advanced'))
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels).item()
                total_predictions += labels.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct_predictions / total_predictions
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        msg = (
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.4f}"
        )
        _log(logger, msg)

        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = model.state_dict().copy()
            epochs_no_improve = 0
            _log(logger, "  -> Validation loss improved, saving checkpoint...")
            if run_dir:
                torch.save(best_model_wts, os.path.join(run_dir, "best_model.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                _log(logger, f"  -> Early stopping triggered after {epoch + 1} epochs.")
                break

    elapsed = time.time() - start_time
    _log(logger, f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Load the best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # Save metrics to CSV
    if run_dir:
        _save_metrics_csv(run_dir, train_losses, val_losses, train_accuracies, val_accuracies)
        _save_metrics_json(run_dir, train_losses, val_losses, train_accuracies, val_accuracies)

    return model, train_losses, val_losses, val_accuracies


def evaluate_classifier(model, test_loader, device='cuda', logger=None):
    """
    Evaluate a classification model on test data.

    Returns:
        float: Mean test loss.
        float: Test accuracy.
        list: Per-sample error indicators (1 = wrong, 0 = correct).
        torch.Tensor: All model outputs concatenated.
        torch.Tensor: All true labels concatenated.
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0
    classification_errors = []
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            outputs = model(inputs, modes=('lightweight', 'moderate', 'advanced'))
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            errors = (preds != labels).float().cpu().numpy()
            classification_errors.extend(errors)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    mean_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total

    _log(logger, f"Test Loss: {mean_test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

    return mean_test_loss, test_accuracy, classification_errors, torch.cat(all_outputs, dim=0), torch.cat(all_labels, dim=0)


def plot_training_history(train_losses, val_losses, val_accuracies, model_name, run_dir=None):
    """
    Plot training/validation loss and validation accuracy curves.
    Saves to file if run_dir is provided.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_title(f'{model_name} - Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    axes[1].set_title(f'{model_name} - Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()

    if run_dir:
        plot_path = os.path.join(run_dir, "training_history.png")
        fig.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")

    plt.close(fig)


def generate_classification_report(all_outputs, all_labels, code_to_label, model_name, run_dir=None, logger=None):
    """
    Generate and save a classification report and confusion matrix.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    _, preds = torch.max(all_outputs, 1)
    preds_np = preds.numpy()
    labels_np = all_labels.numpy()

    label_to_code = {v: k for k, v in code_to_label.items()}
    target_names = [str(label_to_code[i]) for i in sorted(label_to_code.keys())]

    report = classification_report(labels_np, preds_np, target_names=target_names, zero_division=0)
    _log(logger, f"\n{model_name} Classification Report:\n{report}")

    if run_dir:
        with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        # Confusion matrix
        cm = confusion_matrix(labels_np, preds_np)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        fig.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)


def save_final_model(model, model_name, run_dir, args=None, test_loss=None, test_acc=None, logger=None):
    """
    Save the final trained model as a complete checkpoint with metadata.
    Saves both the state_dict and a full checkpoint with training config.

    The checkpoint contains:
        - model_state_dict: The model weights
        - model_name: Name of the model class
        - args: Training arguments used
        - test_loss: Final test loss
        - test_accuracy: Final test accuracy
    """
    if run_dir is None:
        _log(logger, "No run_dir specified, skipping final model save.")
        return

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
    }
    if args is not None:
        args_dict = vars(args) if hasattr(args, '__dict__') else args
        checkpoint['args'] = args_dict
        # Store model constructor params so the model can be fully reconstructed
        checkpoint['model_params'] = {
            'num_classes': args_dict.get('num_classes'),
            'target_length': args_dict.get('target_length'),
            'cycle_num': args_dict.get('cycle_num', 2),
            'lead_num': args_dict.get('lead_num', 1),
        }

    path = os.path.join(run_dir, "final_model.pth")
    torch.save(checkpoint, path)
    _log(logger, f"Final model saved to {path}")
    _log(logger, f"To reload: see utils/training.py :: load_saved_model()")


def load_saved_model(checkpoint_path, device='cpu'):
    """
    Load a saved model from a checkpoint file. Reconstructs the model
    architecture from the stored parameters — no need to know them yourself.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (str): Device to load the model on ('cpu', 'cuda', 'mps').

    Returns:
        nn.Module: The loaded model in eval mode.
        dict: The full checkpoint dict (contains args, test_accuracy, etc.).
    """
    from models import (
        ConvFcClassifier, AttentionConvFcClassifier,
        LightweightConvFcClassifier, SharedAdaptiveConvClassifier,
    )

    MODEL_MAP = {
        'ConvFcClassifier': ConvFcClassifier,
        'AttentionConvFcClassifier': AttentionConvFcClassifier,
        'LightweightConvFcClassifier': LightweightConvFcClassifier,
        'SharedAdaptiveConvClassifier': SharedAdaptiveConvClassifier,
    }

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint['model_name']
    params = checkpoint['model_params']

    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Expected one of {list(MODEL_MAP.keys())}")

    model_cls = MODEL_MAP[model_name]
    model = model_cls(
        num_classes=params['num_classes'],
        target_length=params['target_length'],
        cycle_num=params['cycle_num'],
        lead_num=params['lead_num'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


# ---- Internal helpers ----

def _log(logger, msg):
    if logger:
        logger.info(msg)
    else:
        print(msg)


def _save_metrics_csv(run_dir, train_losses, val_losses, train_accs, val_accs):
    path = os.path.join(run_dir, "metrics.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], val_losses[i], train_accs[i], val_accs[i]])


def _save_metrics_json(run_dir, train_losses, val_losses, train_accs, val_accs):
    path = os.path.join(run_dir, "metrics.json")
    data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
