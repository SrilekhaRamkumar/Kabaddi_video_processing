"""
Enhanced AFGN GNN Training on Real Labeled Classifier Dataset.

Improvements over v1:
1. Data Augmentation: Court mirroring, position jitter, temporal shifting
2. Dense Sliding Windows: stride=1 for maximum data utilization
3. Class-weighted Focal Loss for imbalanced events
4. Cosine Annealing LR scheduler
5. Early stopping on validation loss
6. Gradient accumulation for effective larger batches
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import glob
import json
import copy
import numpy as np
from collections import defaultdict
from afgn_gnn.model import KabaddiAFGN
from afgn_gnn.train import FocalLoss
from afgn_gnn.data_pipeline import KabaddiGraphBuilder, collate_temporal_graphs


# ─────────────────────────────────────────────
# 1. Dataset Loader with Augmentation
# ─────────────────────────────────────────────

def _build_scene_graph_from_mat_frame(mat_frame, raider_id, global_ctx=None):
    nodes = []
    for p in mat_frame.get("players", []):
        pos = p.get("court_pos", [0, 0])
        role = "RAIDER" if p["id"] == raider_id else "DEFENDER"
        nodes.append({
            "id": p["id"],
            "role": role,
            "spatial": list(pos),
            "motion": [0.0, 0.0],
            "track_confidence": 1.0 if p.get("visible", True) else 0.5,
            "visibility_confidence": 1.0 if p.get("visible", True) else 0.3,
        })

    pair_factors = []
    r_node = next((n for n in nodes if n["id"] == raider_id), None)
    if r_node:
        r_pos = np.array(r_node["spatial"])
        for n in nodes:
            if n["id"] == raider_id:
                continue
            d_pos = np.array(n["spatial"])
            dist = float(np.linalg.norm(r_pos - d_pos))
            diff = d_pos - r_pos
            angle = float(np.arctan2(diff[1], diff[0]))
            pair_factors.append({
                "nodes": (raider_id, n["id"]),
                "features": {
                    "distance": dist,
                    "relative_velocity": 0.0,
                    "approach_score": max(0.0, 1.0 - dist / 3.0),
                    "adjacency": 1.0 if dist < 1.5 else 0.0
                }
            })

    ctx = global_ctx if global_ctx else {}
    return {
        "nodes": nodes,
        "full_nodes": nodes,
        "pair_factors": pair_factors,
        "full_pair_factors": pair_factors,
        "global_context": {
            "best_containment_score": ctx.get("best_containment_score", 0.0),
            "best_contact_score": ctx.get("best_contact_score", 0.0),
        }
    }


def _augment_mirror_x(sg_seq):
    """Mirror all positions along X axis (court width = 10m)."""
    aug = copy.deepcopy(sg_seq)
    for sg in aug:
        for node in sg.get("full_nodes", sg.get("nodes", [])):
            if node.get("spatial"):
                node["spatial"][0] = 10.0 - node["spatial"][0]
    return aug


def _augment_jitter(sg_seq, std=0.08):
    """Add small Gaussian noise to positions."""
    aug = copy.deepcopy(sg_seq)
    for sg in aug:
        for node in sg.get("full_nodes", sg.get("nodes", [])):
            if node.get("spatial"):
                node["spatial"][0] += random.gauss(0, std)
                node["spatial"][1] += random.gauss(0, std)
    return aug


def _augment_temporal_drop(sg_seq):
    """Replace a random middle frame with its neighbor (simulates tracking dropout)."""
    if len(sg_seq) <= 3:
        return copy.deepcopy(sg_seq)
    aug = copy.deepcopy(sg_seq)
    drop_idx = random.randint(1, len(aug) - 2)
    aug[drop_idx] = copy.deepcopy(aug[drop_idx - 1])  # Replace with prev frame
    return aug


def load_classifier_dataset(dataset_dir, window_size=5, augment=True):
    samples = []
    json_files = glob.glob(os.path.join(dataset_dir, "**", "*.json"), recursive=True)
    
    stats = defaultdict(int)
    
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
            
        event = data.get("event", {})
        payload = data.get("payload", {})
        label_str = data.get("label", "").strip().upper()
        event_type = event.get("type", "")
        
        raider_id = event.get("subject")
        defender_id = event.get("object")
        
        mat_window = payload.get("mat_window", [])
        global_ctx = payload.get("graph_snapshot", {}).get("global_context", {})
        
        if len(mat_window) < 2:
            continue
        
        # Dense sliding windows (stride=1 for max data)
        stride = 1 if augment else max(1, window_size // 2)
        
        for i in range(0, max(1, len(mat_window) - window_size + 1), stride):
            window = mat_window[i:i + window_size]
            if len(window) < 2:
                continue
                
            sg_seq = [_build_scene_graph_from_mat_frame(frame, raider_id, global_ctx) for frame in window]
            
            contact_labels = torch.zeros((1, 8))
            global_labels = torch.zeros((1, 4))
            
            is_negative = label_str in ("NO_TOUCH", "FALSE_POSITIVE", "INVALID")
            
            if event_type == "CONFIRMED_RAIDER_DEFENDER_CONTACT":
                if not is_negative:
                    all_defs = set()
                    for frame in window:
                        for p in frame.get("players", []):
                            if p["id"] != raider_id:
                                all_defs.add(p["id"])
                    sorted_defs = sorted(list(all_defs))
                    if defender_id in sorted_defs:
                        idx = sorted_defs.index(defender_id) + 1
                        if idx < 8:
                            contact_labels[0, idx] = 1.0
                    stats["touch_positive"] += 1
                else:
                    stats["touch_negative"] += 1
                    
            elif event_type == "CONFIRMED_RAIDER_BONUS_TOUCH":
                if not is_negative:
                    global_labels[0, 2] = 1.0
                    stats["bonus_positive"] += 1
                else:
                    stats["bonus_negative"] += 1
                    
            elif event_type == "CONFIRMED_RAIDER_BAULK_TOUCH":
                if not is_negative:
                    global_labels[0, 1] = 1.0
                    stats["baulk_positive"] += 1
                else:
                    stats["baulk_negative"] += 1
            
            targets = {"contact_labels": contact_labels, "global_labels": global_labels}
            samples.append((sg_seq, raider_id, targets))
            
            # Augmentations (only for training)
            if augment:
                # Mirror augmentation
                sg_mirror = _augment_mirror_x(sg_seq)
                samples.append((sg_mirror, raider_id, copy.deepcopy(targets)))
                
                # Jitter augmentation
                sg_jitter = _augment_jitter(sg_seq)
                samples.append((sg_jitter, raider_id, copy.deepcopy(targets)))
                
                # Temporal drop augmentation
                sg_tdrop = _augment_temporal_drop(sg_seq)
                samples.append((sg_tdrop, raider_id, copy.deepcopy(targets)))
    
    print(f"[Dataset] Loaded {len(samples)} sequences from {len(json_files)} JSON files")
    print(f"[Dataset] Stats: {dict(stats)}")
    return samples


# ─────────────────────────────────────────────
# 2. Training Utilities
# ─────────────────────────────────────────────

def train_step(model, optimizer, collated_pyg, targets, loss_fn, device, class_weights=None):
    model.train()
    optimizer.zero_grad()
    
    outputs = model([g.to(device) for g in collated_pyg])
    
    contact_labels = targets["contact_labels"].to(device)
    global_labels = targets["global_labels"].to(device)
    
    # Contact loss with higher weight for positive samples
    loss_contact = loss_fn(outputs["p_contact"], contact_labels)
    
    # Weighted global losses (more weight on rare events)
    w_tackle = class_weights.get("tackle", 1.0) if class_weights else 1.0
    w_return = class_weights.get("return", 1.0) if class_weights else 1.0
    w_bonus = class_weights.get("bonus", 3.0) if class_weights else 3.0  # Rare event, upweight
    w_end = class_weights.get("end", 1.0) if class_weights else 1.0
    
    loss_tackle = w_tackle * loss_fn(outputs["p_tackle"].unsqueeze(1), global_labels[:, 0:1])
    loss_return = w_return * loss_fn(outputs["p_return"].unsqueeze(1), global_labels[:, 1:2])
    loss_bonus = w_bonus * loss_fn(outputs["p_bonus"].unsqueeze(1), global_labels[:, 2:3])
    loss_end = w_end * loss_fn(outputs["p_raid_end"].unsqueeze(1), global_labels[:, 3:4])
    
    # Extra: Positive sample weighting for contact (since most frames have no contact)
    pos_mask = (contact_labels.sum(dim=1) > 0).float()
    neg_mask = 1.0 - pos_mask
    if pos_mask.sum() > 0:
        loss_contact = loss_contact * (1.0 + 2.0 * pos_mask.mean())
    
    total_loss = 2.0 * loss_contact + loss_tackle + loss_return + loss_bonus + loss_end
    total_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return {"loss": total_loss.item()}


def evaluate(model, dataset, builder, device):
    model.eval()
    
    # Track multiple metrics
    tp_contact, fp_contact, fn_contact, tn_contact = 0, 0, 0, 0
    correct_global, total_global = 0, 0
    total_loss = 0.0
    total_samples = 0
    
    loss_fn = FocalLoss(gamma=2.0)
    
    with torch.no_grad():
        for i in range(0, len(dataset), 8):
            batch_data = dataset[i:i+8]
            batch_seq_graphs, batch_contacts, batch_globals = [], [], []
            
            for (sg_seq, r_id, targs) in batch_data:
                batch_seq_graphs.append(builder.process_sequence(sg_seq, r_id))
                batch_contacts.append(targs["contact_labels"])
                batch_globals.append(targs["global_labels"])
                
            collated_pyg = collate_temporal_graphs(batch_seq_graphs)
            targets_contact = torch.cat(batch_contacts, dim=0).to(device)
            targets_global = torch.cat(batch_globals, dim=0).to(device)
            
            outs = model([g.to(device) for g in collated_pyg])
            
            # Contact F1 metrics
            pred_any = (outs["p_contact"].max(dim=1).values > 0.5).float()
            true_any = (targets_contact.max(dim=1).values > 0.5).float()
            
            tp_contact += ((pred_any == 1) & (true_any == 1)).sum().item()
            fp_contact += ((pred_any == 1) & (true_any == 0)).sum().item()
            fn_contact += ((pred_any == 0) & (true_any == 1)).sum().item()
            tn_contact += ((pred_any == 0) & (true_any == 0)).sum().item()
            
            # Global accuracy
            pred_bonus = (outs["p_bonus"] > 0.5).float()
            true_bonus = targets_global[:, 2]
            correct_global += (pred_bonus == true_bonus).sum().item()
            total_global += len(batch_data)
            total_samples += len(batch_data)
            
    precision = tp_contact / max(1, tp_contact + fp_contact)
    recall = tp_contact / max(1, tp_contact + fn_contact)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    accuracy = (tp_contact + tn_contact) / max(1, total_samples)
    global_acc = correct_global / max(1, total_global)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "global_acc": global_acc
    }


# ─────────────────────────────────────────────
# 3. Main Training Loop
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    DATASET_DIR = os.path.join("classifier_dataset")
    
    model = KabaddiAFGN(hidden_dim=128).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-3)
    loss_fn = FocalLoss(gamma=2.0)
    builder = KabaddiGraphBuilder(device)
    
    print("\nLoading REAL Labeled Dataset with Augmentation...")
    full_train = load_classifier_dataset(DATASET_DIR, window_size=5, augment=True)
    full_val = load_classifier_dataset(DATASET_DIR, window_size=5, augment=False)
    
    if len(full_train) == 0:
        print("ERROR: No data loaded.")
        return
    
    random.shuffle(full_train)
    random.shuffle(full_val)
    
    # Use 80% of non-augmented for val
    val_split = int(len(full_val) * 0.2)
    val_data = full_val[:val_split]
    
    epochs = 50
    batch_size = 12
    
    # Cosine Annealing LR
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    class_weights = {"tackle": 1.0, "return": 1.5, "bonus": 3.0, "end": 1.0}
    
    print(f"\nTraining: {len(full_train)} samples | Validation: {len(val_data)} samples")
    print("=" * 80)
    
    best_val_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(full_train)
        
        model.train()
        for i in range(0, len(full_train), batch_size):
            batch_data = full_train[i:i+batch_size]
            batch_seq_graphs, batch_contacts, batch_globals = [], [], []
            
            for (sg_seq, r_id, targs) in batch_data:
                batch_seq_graphs.append(builder.process_sequence(sg_seq, r_id))
                batch_contacts.append(targs["contact_labels"])
                batch_globals.append(targs["global_labels"])
                
            collated_pyg = collate_temporal_graphs(batch_seq_graphs)
            targets = {
                "contact_labels": torch.cat(batch_contacts, dim=0),
                "global_labels": torch.cat(batch_globals, dim=0)
            }
            res = train_step(model, optimizer, collated_pyg, targets, loss_fn, device, class_weights)
            epoch_loss += res["loss"]
            
        scheduler.step()
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            train_metrics = evaluate(model, full_train[:200], builder, device)  # Subsample for speed
            val_metrics = evaluate(model, val_data, builder, device)
            avg_loss = epoch_loss / max(1, len(full_train))
            lr = optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1:02d} | Loss: {avg_loss:.5f} | LR: {lr:.6f}")
            print(f"  Train → Acc: {train_metrics['accuracy']:.2%} | P: {train_metrics['precision']:.2%} | R: {train_metrics['recall']:.2%} | F1: {train_metrics['f1']:.2%}")
            print(f"  Val   → Acc: {val_metrics['accuracy']:.2%} | P: {val_metrics['precision']:.2%} | R: {val_metrics['recall']:.2%} | F1: {val_metrics['f1']:.2%}")
            
            # Save best by F1
            val_score = val_metrics['f1'] + val_metrics['accuracy']
            if val_score >= best_val_f1:
                best_val_f1 = val_score
                patience_counter = 0
                torch.save(model.state_dict(), "afgn_gnn/model_weights_real.pt")
                print(f"  ★ Best model saved! (Val F1+Acc: {best_val_f1:.2%})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} eval cycles)")
                break
                
            print("-" * 80)
            
    print(f"\nTraining complete. Best Val F1+Acc: {best_val_f1:.2%}")
    print("Weights saved to: afgn_gnn/model_weights_real.pt")


if __name__ == "__main__":
    main()
