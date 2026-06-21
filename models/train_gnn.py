import csv
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_model import ReasoningGAT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_PATH = os.path.join(ROOT, "graphs", "reasoning_graphs_augmented.pkl")
MODEL_SAVE_PATH = os.path.join(ROOT, "models", "gat_best.pt")
RESULTS_DIR = os.path.join(ROOT, "results")

EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 64
TRAIN_RATIO = 0.8
CLASS_WEIGHTS = [1.0, 1.4]   # index 0 = distractor (majority), index 1 = gold (minority)
PATIENCE = 15                 # early stopping: epochs without val F1 improvement


def nx_to_pyg(g) -> Data:
    """Convert a NetworkX graph to a PyG Data object with node-level labels.

    g.nodes[n]["label"] must exist (1 = gold supporting fact, 0 = distractor).
    y shape: [num_nodes] — one label per node, not per graph.
    """
    node_ids = sorted(g.nodes())
    x = torch.tensor(
        np.stack([g.nodes[n]["embedding"] for n in node_ids]),
        dtype=torch.float,
    )
    node_map = {n: i for i, n in enumerate(node_ids)}
    edges = [(node_map[u], node_map[v]) for u, v in g.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([g.nodes[n]["label"] for n in node_ids], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor):
    """Overall acc, per-class acc, and binary F1/precision/recall for class 1."""
    acc = (preds == targets).float().mean().item()

    mask0 = targets == 0
    mask1 = targets == 1
    acc_0 = (preds[mask0] == 0).float().mean().item() if mask0.any() else float("nan")
    acc_1 = (preds[mask1] == 1).float().mean().item() if mask1.any() else float("nan")

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return acc, acc_0, acc_1, f1, precision, recall


def train_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    total_loss = total_nodes = 0
    all_preds, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)          # [total_nodes, 2]
        loss = F.nll_loss(out, batch.y, weight=class_weights)
        loss.backward()
        optimizer.step()

        n = batch.y.size(0)                             # total nodes in this batch
        total_loss  += loss.item() * n
        total_nodes += n
        all_preds.append(out.argmax(dim=-1).cpu())
        all_targets.append(batch.y.cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return (total_loss / total_nodes,) + compute_metrics(preds, targets)


@torch.no_grad()
def eval_epoch(model, loader, device, class_weights):
    model.eval()
    total_loss = total_nodes = 0
    all_preds, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)          # [total_nodes, 2]
        loss = F.nll_loss(out, batch.y, weight=class_weights)

        n = batch.y.size(0)
        total_loss  += loss.item() * n
        total_nodes += n
        all_preds.append(out.argmax(dim=-1).cpu())
        all_targets.append(batch.y.cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return (total_loss / total_nodes,) + compute_metrics(preds, targets)


def compute_full_report(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Per-class precision, recall, F1, and support for classes 0 and 1."""
    report = {}
    for cls in [0, 1]:
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f  = 2 * p * r / (p + r + 1e-9)
        report[cls] = {"precision": p, "recall": r, "f1": f,
                       "support": int((targets == cls).sum().item())}
    return report


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading graphs from {GRAPHS_PATH} ...")
    with open(GRAPHS_PATH, "rb") as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs.")

    print("Converting to PyG Data objects ...")
    dataset = [nx_to_pyg(g) for g in graphs]

    # Report node-level label distribution
    all_labels = torch.cat([d.y for d in dataset])
    n_gold = (all_labels == 1).sum().item()
    n_dist = (all_labels == 0).sum().item()
    print(f"  Node labels — gold (1): {n_gold}  distractor (0): {n_dist}  "
          f"({n_dist / len(all_labels):.2%} distractor)")

    split      = int(len(dataset) * TRAIN_RATIO)
    train_data = dataset[:split]
    val_data   = dataset[split:]
    print(f"Train graphs: {len(train_data)}  Val graphs: {len(val_data)}")

    # Class-balanced graph sampling: each graph's weight = sum of per-node
    # inverse-frequency weights, so graphs with more gold nodes are sampled more.
    train_labels_flat = torch.cat([d.y for d in train_data])
    node_counts = Counter(train_labels_flat.tolist())   # {0: n_dist, 1: n_gold}
    sample_weights = [
        sum(1.0 / node_counts[lbl] for lbl in d.y.tolist())
        for d in train_data
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_data),
        replacement=True,
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(CLASS_WEIGHTS, device=device)
    print(f"Training on {device}  |  class weights: {CLASS_WEIGHTS}  |  patience: {PATIENCE}\n")

    model     = ReasoningGAT(in_channels=397).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_f1      = -1.0
    best_epoch       = 1
    patience_counter = 0

    csv_path = os.path.join(RESULTS_DIR, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "train_f1",
                         "val_loss", "val_acc", "val_f1"])

    hdr = (f"{'Ep':>3}  {'TrLoss':>7} {'TrAcc':>6} {'TrA0':>6} {'TrA1':>6} {'TrF1':>6}"
           f"  {'VaLoss':>7} {'VaAcc':>6} {'VaA0':>6} {'VaA1':>6} {'VaF1':>6}")
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_a0, tr_a1, tr_f1, tr_p, tr_r = train_epoch(
            model, train_loader, optimizer, device, class_weights
        )
        va_loss, va_acc, va_a0, va_a1, va_f1, va_p, va_r = eval_epoch(
            model, val_loader, device, class_weights
        )

        improved = va_f1 > best_val_f1
        if improved:
            best_val_f1      = va_f1
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            tag = " *"
        else:
            patience_counter += 1
            tag = ""

        csv_writer.writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{tr_f1:.6f}",
                             f"{va_loss:.6f}", f"{va_acc:.6f}", f"{va_f1:.6f}"])
        csv_file.flush()

        if epoch % 5 == 0 or improved:
            print(
                f"{epoch:>3}  {tr_loss:>7.4f} {tr_acc:>6.4f} {tr_a0:>6.4f} {tr_a1:>6.4f} {tr_f1:>6.4f}"
                f"  {va_loss:>7.4f} {va_acc:>6.4f} {va_a0:>6.4f} {va_a1:>6.4f} {va_f1:>6.4f}{tag}"
            )

        if patience_counter >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch} — no val F1 gain for {PATIENCE} epochs.")
            break

    csv_file.close()

    # --- results/best_val_f1.txt ---
    with open(os.path.join(RESULTS_DIR, "best_val_f1.txt"), "w") as f:
        f.write(f"Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}\n")

    # Reload best checkpoint for final eval
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            all_preds.append(out.argmax(dim=-1).cpu())
            all_targets.append(batch.y.cpu())
    final_preds   = torch.cat(all_preds)
    final_targets = torch.cat(all_targets)

    # --- results/classification_report.txt ---
    report = compute_full_report(final_preds, final_targets)
    cls_labels = {0: "distractor (0)", 1: "gold (1)"}
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Classification Report — best checkpoint (epoch {best_epoch})\n")
        f.write("=" * 56 + "\n\n")
        f.write(f"{'Class':<18} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}\n")
        f.write("-" * 56 + "\n")
        for cls in [0, 1]:
            m = report[cls]
            f.write(f"{cls_labels[cls]:<18} {m['precision']:>10.4f} {m['recall']:>8.4f}"
                    f" {m['f1']:>8.4f} {m['support']:>9}\n")

    # --- results/confusion_matrix.json ---
    tp = int(((final_preds == 1) & (final_targets == 1)).sum().item())
    fp = int(((final_preds == 1) & (final_targets == 0)).sum().item())
    tn = int(((final_preds == 0) & (final_targets == 0)).sum().item())
    fn = int(((final_preds == 0) & (final_targets == 1)).sum().item())
    confusion = {
        "labels": ["distractor (0)", "gold (1)"],
        "matrix": [[tn, fp], [fn, tp]],
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }
    with open(os.path.join(RESULTS_DIR, "confusion_matrix.json"), "w") as f:
        json.dump(confusion, f, indent=2)

    print(f"\nBest val F1: {best_val_f1:.4f} at epoch {best_epoch}"
          f"  — results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
