"""Quick validation - writes results to a JSON file."""
import torch, random, json
from afgn_gnn.model import KabaddiAFGN
from afgn_gnn.data_pipeline import KabaddiGraphBuilder
from afgn_gnn.train_real import load_classifier_dataset, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KabaddiAFGN(hidden_dim=128).to(device)
model.load_state_dict(torch.load("afgn_gnn/model_weights_real.pt", map_location=device))
builder = KabaddiGraphBuilder(device)

all_data = load_classifier_dataset("classifier_dataset", window_size=5, augment=False)
random.seed(42)
random.shuffle(all_data)
split = int(len(all_data) * 0.8)

train_m = evaluate(model, all_data[:split], builder, device)
val_m = evaluate(model, all_data[split:], builder, device)

results = {"train": train_m, "val": val_m, "total_samples": len(all_data), "train_size": split, "val_size": len(all_data) - split}
with open("afgn_gnn/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("Done. Check afgn_gnn/metrics.json")
