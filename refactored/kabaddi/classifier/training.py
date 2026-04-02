"""
Touch Classifier Training

Training and evaluation functions for the touch classifier model.
"""

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model: TouchWindowClassifier model
        loader: DataLoader for training data
        optimizer: Optimizer (e.g., AdamW)
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: torch.device

    Returns:
        dict: Training metrics (loss, accuracy, f1)
    """
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
    """
    Evaluate the model on validation/test data.

    Args:
        model: TouchWindowClassifier model
        loader: DataLoader for validation/test data
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: torch.device

    Returns:
        dict: Evaluation metrics (loss, accuracy, f1, report)
    """
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

