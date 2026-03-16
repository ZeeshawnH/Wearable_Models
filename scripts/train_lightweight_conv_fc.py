#!/usr/bin/env python3
"""
Train LightweightConvFcClassifier on PhysioNet ECG data.
Usage: python scripts/train_lightweight_conv_fc.py [--epochs 500] [--lr 0.00005] [--patience 50]

NOTE: This model uses target_length = cycle_num * 4 (not * 8 like the others).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from models import LightweightConvFcClassifier
from utils.data_pipeline import load_and_prepare_data
from utils.training import (
    setup_logging, train_classifier, evaluate_classifier,
    plot_training_history, generate_classification_report, save_final_model,
)

MODEL_NAME = "LightweightConvFcClassifier"
DEFAULT_DATA_ROOT = "/Users/zeeshawnh/College/GuoResearch/Transformer/data/physionet.org/files/challenge-2021/1.0.3/training"


def main():
    parser = argparse.ArgumentParser(description=f"Train {MODEL_NAME}")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-datasets", type=int, default=1)
    parser.add_argument("--cycle-num", type=int, default=2)
    parser.add_argument("--lead-num", type=int, default=1)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "outputs"))
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of data to use (0.0-1.0). E.g. 0.1 for 10%%. Default: 1.0 (all data)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    logger, run_dir = setup_logging(args.output_dir, MODEL_NAME)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Device: {device}")

    # Data
    data = load_and_prepare_data(
        root_directory=args.data_root,
        num_datasets=args.num_datasets,
        lead_num=args.lead_num,
        cycle_num=args.cycle_num,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    logger.info(f"Data shapes: {data['shapes']}")

    # Subset data for quick testing
    if args.subset < 1.0:
        import torch.utils.data as tud
        for key in ['train_loader', 'val_loader', 'test_loader']:
            ds = data[key].dataset
            n = max(1, int(len(ds) * args.subset))
            idx = torch.randperm(len(ds))[:n]
            data[key] = tud.DataLoader(tud.Subset(ds, idx), batch_size=args.batch_size, shuffle=(key == 'train_loader'))
        logger.info(f"Subset {args.subset*100:.0f}%: train={len(data['train_loader'].dataset)}, val={len(data['val_loader'].dataset)}, test={len(data['test_loader'].dataset)}")

    # Model — lightweight uses cycle_num * 4
    target_length = args.cycle_num * 4
    model = LightweightConvFcClassifier(
        num_classes=data['num_classes'],
        target_length=target_length,
        cycle_num=args.cycle_num,
        lead_num=args.lead_num,
    )
    logger.info(f"Model: {MODEL_NAME}, target_length={target_length}")

    # Attach computed params to args so they get saved in the checkpoint
    args.num_classes = data['num_classes']
    args.target_length = target_length

    # Train
    model, train_losses, val_losses, val_accs = train_classifier(
        model=model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
        logger=logger,
        run_dir=run_dir,
    )

    # Evaluate
    test_loss, test_acc, errors, outputs, labels = evaluate_classifier(
        model=model,
        test_loader=data['test_loader'],
        device=device,
        logger=logger,
    )
    logger.info(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Plots & reports
    plot_training_history(train_losses, val_losses, val_accs, MODEL_NAME, run_dir)
    generate_classification_report(outputs, labels, data['code_to_label'], MODEL_NAME, run_dir, logger)

    # Save final model with metadata
    save_final_model(model, MODEL_NAME, run_dir, args=args, test_loss=test_loss, test_acc=test_acc, logger=logger)

    logger.info(f"All outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
