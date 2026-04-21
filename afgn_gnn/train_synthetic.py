import torch
import torch.optim as optim
import random
import numpy as np
from afgn_gnn.model import KabaddiAFGN
from afgn_gnn.train import FocalLoss, train_step
from afgn_gnn.data_pipeline import KabaddiGraphBuilder, collate_temporal_graphs

def generate_synthetic_raid_sequence(window_size=5):
    """
    Generates a realistic synthetic sequence with heavy noise, occlusion, and missing trackers.
    """
    raider_id = 99
    defenders = [10, 11, 12, 13, 14, 15, 16]
    
    # "near_miss" tests Hard Negatives
    raid_type = random.choice(["tackle", "touch", "empty", "near_miss"])
    
    scene_graphs = []
    
    rx, ry = random.uniform(3.0, 7.0), random.uniform(2.0, 5.0)
    def_states = {d: [random.uniform(1.0, 9.0), random.uniform(1.0, 6.0)] for d in defenders}
    target_defender = random.choice(defenders) if raid_type in ["touch", "near_miss"] else None
    
    for t in range(window_size):
        # 1. Kinematic Progress
        if raid_type == "tackle":
            for d in defenders:
                dx, dy = rx - def_states[d][0], ry - def_states[d][1]
                def_states[d][0] += dx * 0.25
                def_states[d][1] += dy * 0.25
        elif raid_type == "touch" or raid_type == "near_miss":
            dx, dy = def_states[target_defender][0] - rx, def_states[target_defender][1] - ry
            rx += dx * 0.35
            ry += dy * 0.35
            # Near miss implies they get close but then Raider turns back rapidly
            if raid_type == "near_miss" and t >= window_size - 2:
                rx -= dx * 1.5
                ry -= dy * 1.5 
        else:
            rx += random.uniform(-0.8, 0.8)
            ry += random.uniform(-0.8, 0.8)

        # 2. Add Tracking Noise & Node Dropout
        nodes = []
        
        # Raider always present, but maybe noisy
        r_noisy_x = rx + random.gauss(0, 0.15)
        r_noisy_y = ry + random.gauss(0, 0.15)
        nodes.append({"id": raider_id, "spatial": [r_noisy_x, r_noisy_y], "motion": [0.1, 0.2], "track_confidence": random.uniform(0.7, 1.0)})
        
        for d in defenders:
            # 15% chance tracker loses this defender in this frame (Node Dropout)
            if random.random() < 0.15:
                continue
                
            d_noisy_x = def_states[d][0] + random.gauss(0, 0.2)
            d_noisy_y = def_states[d][1] + random.gauss(0, 0.2)
            nodes.append({"id": d, "spatial": [d_noisy_x, d_noisy_y], "motion": [0.0, 0.0], "track_confidence": random.uniform(0.4, 0.95)})
            
        # 3. Simulate Edge Extraction
        pair_factors = []
        for d in defenders:
            dist = np.linalg.norm(np.array([rx, ry]) - np.array(def_states[d]))
            pair_factors.append({
                "nodes": (raider_id, d),
                "features": {
                    "distance": dist + random.gauss(0, 0.2), # Noisy edges
                    "relative_velocity": random.uniform(0.0, 2.0),
                    "approach_score": max(0.0, 1.0 - dist/2),
                    "adjacency": 1.0 if dist < 1.0 else 0.0
                }
            })
            
        global_ctx = {
            "best_containment_score": random.uniform(0.6, 1.0) if raid_type == "tackle" else random.uniform(0.0, 0.4),
            "best_contact_score": random.uniform(0.7, 1.0) if raid_type in ["tackle", "touch"] else random.uniform(0.0, 0.5)
        }
            
        scene_graphs.append({
            "full_nodes": nodes,
            "full_pair_factors": pair_factors,
            "global_context": global_ctx
        })
        
    # Labels
    contact_labels = torch.zeros((1, 8))
    if raid_type == "touch":
        target_idx = sorted(list(defenders)).index(target_defender) + 1
        contact_labels[0, target_idx] = 1.0
        
    global_labels = torch.zeros((1, 4))
    if raid_type == "tackle":
        global_labels[0, 0] = 1.0 # tackle
        global_labels[0, 3] = 1.0 # end
    elif raid_type == "touch":
        global_labels[0, 1] = 1.0 # return
    # near_miss counts as empty
        
    targets = {
        "contact_labels": contact_labels,
        "global_labels": global_labels
    }
        
    return scene_graphs, raider_id, targets

def evaluate(model, dataset, builder, device):
    model.eval()
    correct_tackle, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), 8):
            batch_data = dataset[i:i+8]
            batch_seq_graphs, batch_globals = [], []
            for (sg_seq, r_id, targs) in batch_data:
                batch_seq_graphs.append(builder.process_sequence(sg_seq, r_id))
                batch_globals.append(targs["global_labels"])
                
            collated_pyg = collate_temporal_graphs(batch_seq_graphs)
            targets_global = torch.cat(batch_globals, dim=0).to(device)
            
            outs = model([g.to(device) for g in collated_pyg])
            preds = (outs["p_tackle"] > 0.5).float()
            
            correct_tackle += (preds == targets_global[:, 0]).sum().item()
            total += len(batch_data)
            
    return correct_tackle / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = KabaddiAFGN(hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    focal_loss_fn = FocalLoss()
    builder = KabaddiGraphBuilder(device)
    
    print("Generating Noisy Synthetic Dataset...")
    full_data = [generate_synthetic_raid_sequence() for _ in range(500)]
    
    # Train/Val Split (80/20)
    train_data = full_data[:400]
    val_data = full_data[400:]
    
    epochs = 40
    batch_size = 16
    
    print(f"Training on {len(train_data)} samples, Validating on {len(val_data)} samples.")
    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(train_data)
        
        # Train Loop
        model.train()
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
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
            res = train_step(model, optimizer, collated_pyg, targets, focal_loss_fn, device)
            epoch_loss += res["loss"]
            
        if (epoch+1) % 10 == 0:
            train_acc = evaluate(model, train_data, builder, device)
            val_acc = evaluate(model, val_data, builder, device)
            print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss/len(train_data):.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
            
    torch.save(model.state_dict(), "afgn_gnn/model_weights_synthetic.pt")
    print("Weights saved to afgn_gnn/model_weights_synthetic.pt")

if __name__ == "__main__":
    main()
