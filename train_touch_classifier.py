import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from touch_classifier_dataset import ConfirmedWindowTouchDataset
from touch_classifier_model import TouchWindowClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in loader:
        clips = batch["clip"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    for batch in loader:
        clips = batch["clip"].to(device)
        labels = batch["label"].to(device)
        logits = model(clips)
        loss = criterion(logits, labels)

        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "report": classification_report(all_labels, all_preds, output_dict=True, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train touch confirmation classifier on exported confirmed windows.")
    parser.add_argument("--dataset-root", default="Videos/classifier_dataset")
    parser.add_argument("--output-dir", default="Kabaddi_video_processing/models/touch_classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--include-hli", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = TouchWindowClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    history = []
    best_checkpoint_path = output_dir / "best_model.pt"

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

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
