#!/usr/bin/env python3
"""
Run all model training scripts sequentially.
Usage: python scripts/run_all.py [--epochs 500] [--lr 0.00005] [--patience 50]

All arguments are forwarded to each individual training script.
"""
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    os.path.join(ROOT, "scripts", "train_conv_fc.py"),
    os.path.join(ROOT, "scripts", "train_attention_conv_fc.py"),
    os.path.join(ROOT, "scripts", "train_lightweight_conv_fc.py"),
    os.path.join(ROOT, "scripts", "train_ams.py"),
]


def main():
    # Forward all CLI args to each script
    extra_args = sys.argv[1:]

    results = {}
    total_start = time.time()

    for script in SCRIPTS:
        name = os.path.basename(script)
        print(f"\n{'='*60}")
        print(f"  Starting: {name}")
        print(f"{'='*60}\n")

        start = time.time()
        cmd = [sys.executable, script] + extra_args
        result = subprocess.run(cmd, cwd=ROOT)
        elapsed = time.time() - start

        status = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
        results[name] = {"status": status, "time": elapsed}
        print(f"\n  {name}: {status} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, info in results.items():
        print(f"  {name:40s} {info['status']:>10s}  ({info['time']:.1f}s)")
    print(f"  {'Total':40s} {'':>10s}  ({total_elapsed:.1f}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
