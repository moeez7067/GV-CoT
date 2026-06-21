import torch  # MUST be first import — Windows DLL fix

import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"D:\gvcot")
GRAPHS_PKL = BASE_DIR / "graphs"  / "reasoning_graphs.pkl"
MODEL_PT   = BASE_DIR / "models"  / "gat_best.pt"
OUT_DIR    = BASE_DIR / "attribution"
RESULT_DIR = BASE_DIR / "results"

OUT_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(BASE_DIR))
from models.gnn_model import ReasoningGAT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Stage 3] Device: {DEVICE}")

model = ReasoningGAT(in_channels=384, hidden_channels=128, out_channels=2, heads=4)
state = torch.load(MODEL_PT, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
print("[Stage 3] Model loaded ✓")

# ── Load graphs ───────────────────────────────────────────────────────────────
print("[Stage 3] Loading graphs …")
with open(GRAPHS_PKL, "rb") as f:
    graphs = pickle.load(f)

def nx_to_arrays(g):
    nodes = list(g.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    x = torch.tensor(
        np.stack([g.nodes[n]["embedding"] for n in nodes]), dtype=torch.float
    )
    y = torch.tensor([g.nodes[n]["label"] for n in nodes], dtype=torch.long)
    if g.number_of_edges() > 0:
        src, dst = zip(*[(node_to_idx[u], node_to_idx[v]) for u, v in g.edges()])
        edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return x, edge_index, y

total = len(graphs)
split = int(total * 0.80)
val_graphs = graphs[split:]
print(f"[Stage 3] Val set: {len(val_graphs):,} graphs (indices {split}–{total-1})")

# ── Gradient saliency: |x · ∇_x Σ P(gold)| per feature dim ──────────────────
# shap.GradientExplainer cannot be used with GAT models because it batches
# background nodes from different graphs against a fixed edge_index, causing
# out-of-bounds CUDA index errors.  Input×gradient saliency is computed per
# graph independently, which is both correct and architecturally compatible.
print("[Stage 3] Running gradient saliency analysis …")
print("          (This may take several minutes on the full val set)")

MAX_GRAPHS = min(5_000, len(val_graphs))

rng = np.random.default_rng(42)
sampled_idx = rng.choice(len(val_graphs), size=MAX_GRAPHS, replace=False)

saliency_gold_accum = np.zeros(384)
saliency_hall_accum = np.zeros(384)
gold_node_count = 0
hall_node_count = 0

gold_entropies = []
hall_entropies = []

for i, idx in enumerate(sampled_idx):
    if (i + 1) % 500 == 0:
        print(f"  … {i+1}/{MAX_GRAPHS} graphs processed")

    g = val_graphs[idx]
    x_t, ei, y_t = nx_to_arrays(g)
    x_t = x_t.to(DEVICE)
    ei  = ei.to(DEVICE)

    if x_t.shape[0] < 2:
        continue

    # Gradient saliency pass — no torch.no_grad() so autograd runs
    x_t = x_t.detach().requires_grad_(True)
    logits = model(x_t, ei)                       # (N, 2)
    gold_conf_sum = torch.exp(logits[:, 1]).sum()
    gold_conf_sum.backward()

    with torch.no_grad():
        # Input × gradient: captures which features drive gold confidence
        saliency = (x_t * x_t.grad).abs().cpu().numpy()   # (N, 384)
        probs    = torch.softmax(logits.detach(), dim=-1).cpu().numpy()  # (N, 2)

    x_t.grad = None

    labels_np = y_t.numpy()
    for node_i, lbl in enumerate(labels_np):
        if lbl == 1:
            saliency_gold_accum += saliency[node_i]
            gold_node_count += 1
        else:
            saliency_hall_accum += saliency[node_i]
            hall_node_count += 1

    # Prediction entropy per node (lower = more confident)
    eps = 1e-9
    node_entropy = -(probs * np.log(probs + eps)).sum(axis=1)
    for node_i, lbl in enumerate(labels_np):
        ent = float(node_entropy[node_i])
        if lbl == 1:
            gold_entropies.append(ent)
        else:
            hall_entropies.append(ent)

print(f"[Stage 3] SHAP complete. Gold nodes: {gold_node_count:,} | Hall nodes: {hall_node_count:,}")

# ── Summary statistics ────────────────────────────────────────────────────────
mean_shap_gold = saliency_gold_accum / max(gold_node_count, 1)
mean_shap_hall = saliency_hall_accum / max(hall_node_count, 1)

combined_mean = (mean_shap_gold + mean_shap_hall) / 2
top10_idx  = np.argsort(combined_mean)[::-1][:10].tolist()
top10_vals = combined_mean[top10_idx].tolist()

def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

separation_score = 1.0 - cosine_sim(mean_shap_gold, mean_shap_hall)

mean_ent_gold = float(np.mean(gold_entropies)) if gold_entropies else 0.0
mean_ent_hall = float(np.mean(hall_entropies)) if hall_entropies else 0.0
std_ent_gold  = float(np.std(gold_entropies))  if gold_entropies else 0.0
std_ent_hall  = float(np.std(hall_entropies))  if hall_entropies else 0.0

# ── Save shap_scores.json ─────────────────────────────────────────────────────
shap_scores = {
    "meta": {
        "val_graphs_total":      len(val_graphs),
        "graphs_sampled":        int(MAX_GRAPHS),
        "gold_nodes_seen":       int(gold_node_count),
        "hall_nodes_seen":       int(hall_node_count),
        "feature_dim":           384,
        "method":                "gradient_saliency (|x * grad_x sum_P(gold)|)",
    },
    "mean_shap_magnitude_gold":         mean_shap_gold.tolist(),
    "mean_shap_magnitude_hallucinated": mean_shap_hall.tolist(),
    "top10_feature_dims":               top10_idx,
    "top10_mean_magnitudes":            [round(v, 6) for v in top10_vals],
    "separation_score":                 round(separation_score, 6),
    "attention_entropy": {
        "gold_mean": round(mean_ent_gold, 6),
        "gold_std":  round(std_ent_gold,  6),
        "hall_mean": round(mean_ent_hall, 6),
        "hall_std":  round(std_ent_hall,  6),
    },
}

out_json = OUT_DIR / "shap_scores.json"
with open(out_json, "w") as f:
    json.dump(shap_scores, f, indent=2)
print(f"[Stage 3] Saved → {out_json}")

# ── Save shap_report.txt ──────────────────────────────────────────────────────
report_lines = [
    "=" * 65,
    "GV-CoT Stage 3 — Gradient Saliency Attribution Report",
    "=" * 65,
    "",
    f"Val set         : {len(val_graphs):,} graphs",
    f"Graphs sampled  : {MAX_GRAPHS:,}",
    f"Gold nodes seen : {gold_node_count:,}",
    f"Hall nodes seen : {hall_node_count:,}",
    f"Method          : input x gradient saliency  (|x * grad_x sum_P(gold)|)",
    "",
    "-" * 65,
    "Mean saliency magnitude per class (top 10 dims)",
    "-" * 65,
    f"{'Dim':>5}  {'Gold |sal|':>14}  {'Hall |sal|':>14}  {'D (G-H)':>10}",
]
for dim in top10_idx:
    g_val = float(mean_shap_gold[dim])
    h_val = float(mean_shap_hall[dim])
    report_lines.append(
        f"  {dim:>5d}  {g_val:>14.6f}  {h_val:>14.6f}  {g_val - h_val:>+10.6f}"
    )

report_lines += [
    "",
    "-" * 65,
    "Separation score (cosine distance of saliency vectors)",
    "-" * 65,
    f"  {separation_score:.6f}  (0=identical features, 1=orthogonal)",
    "",
    "-" * 65,
    "Prediction entropy per class (lower = more confident)",
    "-" * 65,
    f"  Gold        : {mean_ent_gold:.6f} +/- {std_ent_gold:.6f}",
    f"  Hallucinated: {mean_ent_hall:.6f} +/- {std_ent_hall:.6f}",
    "",
    "-" * 65,
    "Interpretation",
    "-" * 65,
]

if separation_score > 0.10:
    report_lines.append(
        "  GAT uses DIFFERENT feature dimensions to classify gold vs hallucinated."
    )
    report_lines.append(
        "    This confirms the model has learned semantically meaningful distinctions."
    )
else:
    report_lines.append(
        "  Low separation: GAT may rely on the same dims for both classes."
    )
    report_lines.append(
        "    Consider adding node-level features (position, length, connectives)."
    )

if mean_ent_gold < mean_ent_hall:
    report_lines.append(
        "  Model is MORE confident on gold nodes -- correct class ordering."
    )
else:
    report_lines.append(
        "  Model is more confident on hallucinated nodes -- may be over-fitting to majority class."
    )

report_lines += [
    "",
    "=" * 65,
    "Stage 3 Attribution -- COMPLETE",
    "=" * 65,
]

report_txt = RESULT_DIR / "shap_report.txt"
with open(report_txt, "w") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"[Stage 3] Saved → {report_txt}")

print()
print("\n".join(report_lines))
