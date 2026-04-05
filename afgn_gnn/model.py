import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool, global_add_pool

class SpatialGNN(nn.Module):
    """
    Enhanced Spatial GNN with 3 message-passing layers, residual connections,
    attention-weighted readout, and richer feature encoding.
    """
    def __init__(self, node_dim=10, edge_dim=6, global_dim=5, hidden_dim=128):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3 GNN Layers with TransformerConv (multi-head attention on edges)
        self.conv1 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        self.conv3 = TransformerConv(hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim, dropout=0.1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(0.15)

    def forward(self, x, edge_index, edge_attr, u, batch):
        x = self.node_mlp(x)
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = self.edge_mlp(edge_attr)
        else:
            edge_attr = torch.zeros(0, x.size(-1), device=x.device)
            
        u = self.global_mlp(u)

        # Layer 1 with residual
        x1 = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x + self.dropout(F.gelu(x1)))
        
        # Layer 2 with residual
        x2 = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x + self.dropout(F.gelu(x2)))
        
        # Layer 3 with residual
        x3 = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x + self.dropout(F.gelu(x3)))
        
        # Incorporate global context via gating
        u_expanded = u[batch]
        gate = torch.sigmoid(u_expanded)
        x = x * gate + u_expanded * (1 - gate)
        
        return x, u


class KabaddiAFGN(nn.Module):
    """
    Enhanced AFGN with deeper GNN, 2-layer GRU, attention pooling,
    and separate classification heads with skip connections.
    """
    def __init__(self, node_dim=10, edge_dim=6, global_dim=5, hidden_dim=128, gru_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.spatial = SpatialGNN(node_dim, edge_dim, global_dim, hidden_dim)
        
        # Temporal Modules (2-layer GRU with dropout)
        self.node_gru = nn.GRU(hidden_dim, hidden_dim, gru_layers, batch_first=True, dropout=0.1)
        self.global_gru = nn.GRU(hidden_dim * 2, hidden_dim, gru_layers, batch_first=True, dropout=0.1)
        
        # Contact Head (per-node) with skip connection
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for skip connection
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Global Head with stronger capacity
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for skip connection
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 4),
            nn.Sigmoid()
        )
        
        # Attention pooling for aggregating node embeddings to global
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq_graphs):
        T = len(seq_graphs)
        batch_size = seq_graphs[0].u.size(0)
        max_nodes = 8
        
        device = seq_graphs[0].x.device
        
        node_embs_seq = torch.zeros(batch_size, max_nodes, T, self.hidden_dim, device=device)
        u_embs_seq = torch.zeros(batch_size, T, self.hidden_dim, device=device)
        valid_node_mask = torch.zeros(batch_size, max_nodes, T, device=device, dtype=torch.bool)
        
        # Store first-frame embeddings for skip connections
        node_embs_first = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device)
        
        for t, graph in enumerate(seq_graphs):
            x, edge_index, edge_attr, u, batch_idx = graph.x, graph.edge_index, graph.edge_attr, graph.u, graph.batch
            x_emb, u_emb = self.spatial(x, edge_index, edge_attr, u, batch_idx)
            
            node_idx = graph.node_idx
            
            node_embs_seq[batch_idx, node_idx, t, :] = x_emb
            valid_node_mask[batch_idx, node_idx, t] = True
            u_embs_seq[:, t, :] = u_emb
            
            if t == 0:
                node_embs_first[batch_idx, node_idx, :] = x_emb
            
        # Temporal processing - Nodes
        node_embs_flat = node_embs_seq.view(batch_size * max_nodes, T, self.hidden_dim)
        node_out, _ = self.node_gru(node_embs_flat)
        
        node_final = node_out[:, -1, :]
        
        # Skip connection: concat first + last frame embeddings
        node_first_flat = node_embs_first.view(batch_size * max_nodes, self.hidden_dim)
        node_combined = torch.cat([node_final, node_first_flat], dim=-1)
        
        # P(contact_i) with skip
        p_contact = self.contact_head(node_combined).view(batch_size, max_nodes)
        
        final_valid_mask = valid_node_mask[:, :, -1]
        p_contact = p_contact * final_valid_mask.float()
        
        # Attention-weighted node pooling for global context
        node_final_reshaped = node_final.view(batch_size, max_nodes, self.hidden_dim)
        attn_weights = self.attn_pool(node_final_reshaped).squeeze(-1)  # [batch, max_nodes]
        attn_weights = attn_weights.masked_fill(~final_valid_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)  # [batch, max_nodes, 1]
        contact_aggregate = (node_final_reshaped * attn_weights).sum(dim=1)  # [batch, hidden_dim]
        
        # Temporal processing - Global
        u_out, _ = self.global_gru(
            torch.cat([
                u_embs_seq, 
                node_out.view(batch_size, max_nodes, T, self.hidden_dim).max(dim=1)[0]
            ], dim=-1)
        )
        u_final = u_out[:, -1, :]
        
        # Skip connection for global: concat temporal output with attention-pooled nodes
        global_combined = torch.cat([u_final, contact_aggregate], dim=-1)
        
        global_probs = self.global_head(global_combined)
        
        return {
            "p_contact": p_contact,
            "p_tackle": global_probs[:, 0],
            "p_return": global_probs[:, 1],
            "p_bonus": global_probs[:, 2],
            "p_raid_end": global_probs[:, 3],
            "valid_mask": final_valid_mask
        }
