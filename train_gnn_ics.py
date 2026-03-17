import uproot
import awkward as ak
import numpy as np
import torch
import torch.nn as nn
import gc

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

ROOT_FILE = "ICS_training_data_11_00829.root"
TREE_NAME = "ParticleTree"

R_NEIGHBOR = 0.3
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3

print("Loading ROOT file...")
file = uproot.open(ROOT_FILE)
tree = file[TREE_NAME]

pt     = tree["pt"].array(library="ak")
eta    = tree["eta"].array(library="ak")
phi    = tree["phi"].array(library="ak")
rho    = tree["rho"].array(library="ak")
pt_ics = tree["pt_ics"].array(library="ak")

n_events = len(pt)
print("Events:", n_events)

def build_edges(eta, phi, R):
    coords = np.vstack((eta, phi)).T
    tree = cKDTree(coords)
    pairs = tree.query_pairs(R)
    edges = []
    for i, j in pairs:
        edges.append([i, j])
        edges.append([j, i])
    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edges).T

print("Building graphs...")
graphs = []

for i in range(n_events):
    p     = np.array(pt[i])
    e     = np.array(eta[i])
    ph    = np.array(phi[i])
    rh    = np.array(rho[i])
    p_ics = np.array(pt_ics[i])

    N = len(p)
    if N < 2:
        continue

    x = np.stack([p, e, ph, rh], axis=1)
    edge_index = build_edges(e, ph, R_NEIGHBOR)
    graph = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(p_ics, dtype=torch.float32)
    )
    graphs.append(graph)

print("Total graphs:", len(graphs))

train_graphs, test_graphs = train_test_split(graphs, test_size=0.1, random_state=42)

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE)

print("Train graphs:", len(train_graphs))
print("Test graphs: ", len(test_graphs))


class ICS_GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(4, 64)
        self.conv2 = SAGEConv(64, 64)
        self.conv3 = SAGEConv(64, 32)
        self.mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        # No clamp during training — gradients must flow freely.
        # The physical constraint (output <= input pT) is taught via the
        # pT-weighted excess penalty in the loss, and hard-clamped at
        # inference in ICS_GNN_TS.
        out = self.mlp(x).squeeze(-1)
        return out


class ICS_GNN_TS(nn.Module):
    def __init__(self, model: ICS_GNN):
        super().__init__()
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.conv3 = model.conv3
        self.mlp   = model.mlp

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        pt_input = x[:, 0]
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = torch.relu(self.conv3(h, edge_index))
        out = self.mlp(h).squeeze(-1)
        # Hard clamp at inference: predicted pT cannot exceed raw input pT.
        # By convergence the model rarely overshoots thanks to the penalty,
        # but this catches any remaining edge cases.
        out = torch.min(out, pt_input)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ICS_GNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print("Using device:", device)
print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    # Penalty weight ramps from 0 to 10 over the first 5 epochs so the model
    # first learns the basic subtraction task before the physical constraint
    # is fully enforced.
    penalty_weight = min(10.0, (epoch / 5) * 10.0)

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)

        pt_input = batch.x[:, 0]

        # Excess: how much each prediction overshoots the input pT.
        # Zero for correct predictions, positive for overshoots.
        excess = torch.clamp(pred - pt_input, min=0.0)

        # Weight the penalty by (1 + pt) so high-pT particles where the
        # model has weak supervision since ICS doesn't touch them are
        # penalized much more heavily for overshooting:
        #   pt = 0.5 GeV -> weight = 1.5x
        #   pt = 4.0 GeV -> weight = 5.0x
        #   pt = 10  GeV -> weight = 11x
        penalty = ((excess ** 2) * (1.0 + pt_input)).mean()

        loss = loss_fn(pred, batch.y) + penalty_weight * penalty

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        del batch, pred, loss
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.6f}  Penalty weight: {penalty_weight:.1f}")

model.eval()

pt_vals     = []
pt_ics_vals = []
for g in test_graphs:
    x = g.x.detach().cpu().numpy()
    pt_vals.append(x[:, 0])
    pt_ics_vals.append(g.y.detach().cpu().numpy())

pt_vals     = np.concatenate(pt_vals)
pt_ics_vals = np.concatenate(pt_ics_vals)

torch.cuda.empty_cache()
preds_all = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        preds = model(batch).cpu().numpy()
        preds_all.append(preds)
        del batch, preds
        torch.cuda.empty_cache()

preds_all = np.concatenate(preds_all)
gc.collect()

plt.figure(figsize=(8, 6))
plt.hist(
    [pt_vals, pt_ics_vals, preds_all],
    bins=100, range=(0, 10),
    label=['pt input', 'pt_ics target', 'predictions'],
    color=['C0', 'C1', 'C2'], alpha=0.6
)
plt.xlabel("pT")
plt.ylabel("Count")
plt.yscale('log')
plt.title("GNN pT predictions vs target")
plt.legend()
plt.tight_layout()
plt.show()

ts_model = ICS_GNN_TS(model.cpu()).eval()
scripted_model = torch.jit.script(ts_model)
scripted_model.save("gnn_model.pt")
print("TorchScript model saved as gnn_model.pt!")