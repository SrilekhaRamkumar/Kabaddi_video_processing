#!/usr/bin/env python3
"""
Touch Classifier Training Script

Train a touch confirmation classifier on exported confirmed windows.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kabaddi.classifier import (
    TouchWindowClassifier,
    ConfirmedWindowTouchDataset,
    train_one_epoch,
    evaluate,
)


def main():
    parser = argparse.ArgumentParser(description="Train touch confirmation classifier on exported confirmed windows.")
    parser.add_argument("--dataset-root", default="../Videos/classifier_dataset")
    parser.add_argument("--output-dir", default="../models/touch_classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--include-hli", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    dataset_root = (script_dir / args.dataset_root).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")

    # Load datasets
    train_dataset = ConfirmedWindowTouchDataset(
        dataset_root,
        split="train",
        num_frames=args.num_frames,
        image_size=args.image_size,
        include_hli=args.include_hli,
    )
    val_dataset = ConfirmedWindowTouchDataset(
        dataset_root,
        split="val",
        num_frames=args.num_frames,
        image_size=args.image_size,
        include_hli=args.include_hli,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = TouchWindowClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    history = []
    best_checkpoint_path = output_dir / "best_model.pt"

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": {
                "loss": val_metrics["loss"],
                "accuracy": val_metrics["accuracy"],
                "f1": val_metrics["f1"],
            },
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.3f} f1 {train_metrics['f1']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.3f} f1 {val_metrics['f1']:.3f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "num_frames": args.num_frames,
                        "image_size": args.image_size,
                        "include_hli": args.include_hli,
                    },
                    "best_val_f1": best_f1,
                },
                best_checkpoint_path,
            )
            print(f"  → New best model saved (F1: {best_f1:.3f})")

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"\n✓ Best checkpoint saved to: {best_checkpoint_path}")
    print(f"✓ Training history saved to: {history_path}")
    print(f"✓ Best validation F1: {best_f1:.3f}")


if __name__ == "__main__":
    main()

