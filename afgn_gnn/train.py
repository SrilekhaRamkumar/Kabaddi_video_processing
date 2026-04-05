import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from afgn_gnn.model import KabaddiAFGN
from afgn_gnn.data_pipeline import KabaddiGraphBuilder, collate_temporal_graphs

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, valid_mask=None):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if valid_mask is not None:
            focal_loss = focal_loss * valid_mask.float()
            if self.reduction == 'mean':
                return focal_loss.sum() / (valid_mask.sum() + 1e-8)
            return focal_loss.sum()
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

def hard_negative_mining(loss, targets, negative_ratio=3.0):
    """
    loss: [N]
    targets: [N]
    """
    pos_mask = targets > 0
    neg_mask = targets == 0
    
    num_pos = pos_mask.sum().item()
    num_neg = min(neg_mask.sum().item(), int(num_pos * negative_ratio) + 1)
    
    if num_neg == 0 or num_pos == 0:
        return loss.mean()
        
    pos_loss = loss[pos_mask]
    neg_loss = loss[neg_mask]
    
    # Sort negative losses to find hard negatives
    neg_loss, _ = torch.sort(neg_loss, descending=True)
    hard_neg_loss = neg_loss[:num_neg]
    
    return (pos_loss.sum() + hard_neg_loss.sum()) / (num_pos + num_neg)

def train_step(model, optimizer, batch_seq_graphs, targets, focal_loss_fn, device):
    """
    targets: dict containing:
      - contact_labels: [batch, max_nodes]
      - global_labels: [batch, 4] -> tackle, return, bonus, raid_end
    """
    model.train()
    optimizer.zero_grad()
    
    batch_seq_graphs = [g.to(device) for g in batch_seq_graphs]
    outputs = model(batch_seq_graphs)
    
    # Unpack targets
    contact_labels = targets["contact_labels"].to(device) # shape [batch, max_nodes]
    global_labels = targets["global_labels"].to(device)   # shape [batch, 4]
    valid_mask = outputs["valid_mask"]                    # shape [batch, max_nodes]
    
    # 1. Edge-Level / Node-Level Contact Loss (Focal Loss due to high negative ratio)
    p_contact = outputs["p_contact"]
    contact_loss_raw = focal_loss_fn(p_contact, contact_labels, valid_mask=valid_mask)
    
    # Hard negative mining for contacts (optional addition, focal loss usually covers this)
    # But as requested:
    flattened_p = p_contact[valid_mask]
    flattened_t = contact_labels[valid_mask]
    bce_raw = F.binary_cross_entropy(flattened_p, flattened_t, reduction='none')
    hnm_contact_loss = hard_negative_mining(bce_raw, flattened_t)
    
    contact_loss = 0.5 * contact_loss_raw + 0.5 * hnm_contact_loss
    
    # 2. Global Event Loss (Multi-task BCE)
    # Multi-task BCE for P(tackle), P(return), P(bonus), P(raid_end)
    global_preds = torch.stack([
        outputs["p_tackle"],
        outputs["p_return"],
        outputs["p_bonus"],
        outputs["p_raid_end"]
    ], dim=1)
    
    global_loss = F.binary_cross_entropy(global_preds, global_labels, reduction='mean')
    
    # Total loss
    total_loss = contact_loss + global_loss
    total_loss.backward()
    
    # Gradient clipping for GRU stability
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    
    return {
        "loss": total_loss.item(),
        "contact_loss": contact_loss.item(),
        "global_loss": global_loss.item()
    }

def hybrid_supervision_loss_weighting(is_manual_annotation):
    """
    Weight the loss higher if it comes from manual annotations vs pseudo-labels
    from the existing heuristic engine.
    """
    return 2.0 if is_manual_annotation else 1.0

# Example dummy data loader loop
def create_dummy_batch():
    # Placeholder for dataloader providing scene_graph lists and labels
    pass
