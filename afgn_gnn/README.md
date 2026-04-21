# AFGN Reasoning Engine - Factor Encoding Strategies

This module contains a purely learnable Graph Neural Network (GNN) and GRU-based reasoning system. It replaces heuristic/rule-based logic in estimating events in Kabaddi raids.

The underlying AFGN (Action Factor Graph Network) relies on defining interacting variables via variables and factors.

### 1. Unary Factors (Node Features)
Unary factors encode the independent states of actors (nodes) in the graph without reference to others.
- **Features Extracted:** Role (is_raider binary flag), normalized position in court (x, y), velocity vector (vx, vy), and visibility/tracking confidence.
- **Encoding Mechanism:** We apply an initial Multi-Layer Perceptron (`node_mlp` in `SpatialGNN`) to embed the node feature vector dimensions into a high-dimensional continuous latent space. The role flag sharply differentiates raiders and defenders before message passing.

### 2. Pairwise Factors (Edge Features)
Pairwise factors encode direct structural or physical relationships between two nodes (e.g., Raider-Defender distances, approaches).
- **Features Extracted:** Distance, relative velocity, approach scores (angle matching velocity vectors), and adjacency flags.
- **Encoding Mechanism:** Edges representing pairwise interactions have dedicated feature vectors that pass through an `edge_mlp`. PyTorch Geometric's `TransformerConv` handles edge conditioning by incorporating these embeddings into the dynamic attention step during message passing. This avoids rigid distance thresholds by letting the network learn the soft spatial importance of an edge.

### 3. Higher-Order Factors (Global Context)
Higher-order factors encode complex subset relationships that cannot be captured pairwise, such as "defensive containment" or "overall geometric pressure".
- **Features Extracted:** Derived spatial metrics like "defenders on court", bounding box "containment scores" (how well defenders surround the raider), and abstract pressure scalars.
- **Encoding Mechanism:** We treat this as a global graph property vector `u`. An MLP projects these into the latent dimension. In our `SpatialGNN` architecture, `u` is added to node representations globally, acting as a contextual conditioning bias before temporal processing. 
Higher-order structural dependencies (like predicting tackle based on multiple contacts) are handled during pooling, where `max` aggregates of the updated node bindings are passed into the global GRU.

### Temporal Factor Flow (GRU)
Once the spatial GNN produces node embeddings, they are structurally aligned across time $T$. The node sequence is passed to `node_gru` to generate soft predictions for events like "Raider_i_Contact". These temporally aware representations are then pooled again to inject into the `global_gru`, mirroring a hierarchical dependency factor. No thresholds exist internally; raw $P(event)$ distributions are evaluated solely during loss functions, or deferred until hard logic decisions must be made.
