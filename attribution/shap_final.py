import torch  # MUST be first import — Windows DLL fix

import json
import pickle
import numpy as np
import sys
from pathlib import Path

from torch_geometric.data import Data

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\gvcot")
GRAPHS_PKL  = BASE_DIR / "graphs"  / "reasoning_graphs_augmented.pkl"
MODEL_PT    = BASE_DIR / "models"  / "gat_best.pt"
OUT_DIR     = BASE_DIR / "attribution"
RESULT_DIR  = BASE_DIR / "results"

OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from models.gnn_model import ReasoningGAT  # noqa: E402

# ── Load model ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Stage 5] Device: {DEVICE}")

model = ReasoningGAT(in_channels=397, hidden_channels=128, out_channels=2, heads=4)
state = torch.load(MODEL_PT, map_location=DEVICE, weights_only=False)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
print("[Stage 5] Model loaded OK  (Run 3 -- 397-dim scaled features)")

# ── Load graphs ───────────────────────────────────────────────────────────────
print("[Stage 5] Loading graphs ...")
with open(GRAPHS_PKL, "rb") as f:
    graphs = pickle.load(f)

total      = len(graphs)
split      = int(total * 0.80)
val_graphs = graphs[split:]
print(f"[Stage 5] Val set: {len(val_graphs):,} graphs")

# ── nx_to_pyg ─────────────────────────────────────────────────────────────────
def nx_to_pyg(g):
    node_list   = list(g.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    x = torch.tensor(
        np.array([g.nodes[n]["embedding"] for n in node_list], dtype=np.float32)
    )
    y = torch.tensor(
        [g.nodes[n]["label"] for n in node_list], dtype=torch.long
    )
    if g.number_of_edges() > 0:
        edges      = [(node_to_idx[u], node_to_idx[v]) for u, v in g.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT SALIENCY — per node, per graph
# method: |x * ∂ logit_hallucinated / ∂x|  summed over feature dims
# ─────────────────────────────────────────────────────────────────────────────
print("[Stage 5] Computing gradient saliency ...")

# Global accumulators
saliency_gold_accum  = np.zeros(397)
saliency_hall_accum  = np.zeros(397)
gold_node_count      = 0
hall_node_count      = 0

# Explainability coverage accumulators
coverage_correct     = 0   # graphs where top-saliency node is truly hallucinated
coverage_total       = 0   # graphs with at least one predicted hallucinated node

per_graph_results    = []

for i, g in enumerate(val_graphs):
    if (i + 1) % 3000 == 0:
        print(f"  ... {i+1}/{len(val_graphs)} graphs processed")

    pyg = nx_to_pyg(g)
    x   = pyg.x.to(DEVICE)          # (N, 397)
    ei  = pyg.edge_index.to(DEVICE)
    y   = pyg.y.numpy()             # (N,) true labels

    if x.shape[0] < 2:
        continue

    # ── Gradient saliency ─────────────────────────────────────────────────
    x_inp = x.detach().requires_grad_(True)

    logits = model(x_inp, ei)        # (N, 2)
    # Score toward hallucinated class (class 0) for each node
    hall_score = logits[:, 0].sum()
    hall_score.backward()

    grad        = x_inp.grad.detach().cpu().numpy()  # (N, 397)
    x_np        = x.detach().cpu().numpy()
    saliency    = np.abs(x_np * grad)                # (N, 397)
    node_sal    = saliency.sum(axis=1)               # (N,) scalar per node

    preds = logits.detach().argmax(dim=-1).cpu().numpy()   # (N,)

    # ── Accumulate per-class feature saliency ─────────────────────────────
    for node_i, lbl in enumerate(y):
        if lbl == 1:
            saliency_gold_accum += saliency[node_i]
            gold_node_count     += 1
        else:
            saliency_hall_accum += saliency[node_i]
            hall_node_count     += 1

    # ── Explainability coverage ───────────────────────────────────────────
    # Among nodes predicted as hallucinated, does the highest-saliency one
    # have a true label of hallucinated (0)?
    hall_pred_indices = np.where(preds == 0)[0]

    if len(hall_pred_indices) > 0:
        coverage_total += 1
        # Highest saliency among predicted-hallucinated nodes
        sal_among_hall  = node_sal[hall_pred_indices]
        top_local_idx   = hall_pred_indices[np.argmax(sal_among_hall)]
        true_lbl_of_top = int(y[top_local_idx])

        if true_lbl_of_top == 0:    # truly hallucinated -> correct attribution
            coverage_correct += 1
        correct_attr = (true_lbl_of_top == 0)
    else:
        correct_attr = None         # no hallucinated predictions — skip

    per_graph_results.append({
        "graph_idx"          : split + i,
        "num_nodes"          : int(x.shape[0]),
        "node_saliency_scores": node_sal.tolist(),
        "gat_preds"          : preds.tolist(),
        "true_labels"        : y.tolist(),
        "correct_attribution": correct_attr,
    })

# ── Explainability coverage metric ────────────────────────────────────────────
explainability_coverage = (
    coverage_correct / coverage_total if coverage_total > 0 else 0.0
)

# ── Feature-level saliency ────────────────────────────────────────────────────
mean_sal_gold = saliency_gold_accum / max(gold_node_count, 1)
mean_sal_hall = saliency_hall_accum / max(hall_node_count, 1)

combined_mean = (mean_sal_gold + mean_sal_hall) / 2
top10_idx     = np.argsort(combined_mean)[::-1][:10].tolist()
top10_vals    = combined_mean[top10_idx].tolist()

# Structural vs SBERT feature importance
# dims 0-383 = SBERT, dims 384-396 = structural
sbert_importance      = float(combined_mean[:384].mean())
structural_importance = float(combined_mean[384:].mean())

# Cosine separation between class saliency vectors
def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

separation_score = 1.0 - cosine_sim(mean_sal_gold, mean_sal_hall)

print(f"\n[Stage 5] Explainability coverage : {explainability_coverage:.4f}")
print(f"[Stage 5] Separation score        : {separation_score:.4f}")
print(f"[Stage 5] SBERT importance        : {sbert_importance:.6f}")
print(f"[Stage 5] Structural importance   : {structural_importance:.6f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "meta": {
        "model"            : "Run 3 -- 397-dim scaled (gat_best.pt)",
        "val_graphs"       : len(val_graphs),
        "graphs_processed" : len(per_graph_results),
        "gold_nodes"       : int(gold_node_count),
        "hall_nodes"       : int(hall_node_count),
        "method"           : "gradient saliency |x * d(logit_hall)/dx|",
    },
    "explainability_coverage": {
        "coverage"         : round(explainability_coverage, 4),
        "correct"          : int(coverage_correct),
        "total_with_hall_pred": int(coverage_total),
        "definition"       : (
            "% of graphs where the highest-saliency predicted-hallucinated node "
            "has a true label of hallucinated (0)"
        ),
    },
    "feature_saliency": {
        "separation_score"           : round(separation_score, 4),
        "sbert_dims_mean_importance" : round(sbert_importance, 6),
        "structural_dims_mean_importance": round(structural_importance, 6),
        "top10_feature_dims"         : top10_idx,
        "top10_mean_magnitudes"      : [round(v, 6) for v in top10_vals],
        "mean_saliency_gold"         : mean_sal_gold.tolist(),
        "mean_saliency_hallucinated" : mean_sal_hall.tolist(),
    },
    "per_graph_results": per_graph_results,
}

out_json = OUT_DIR / "shap_final_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"[Stage 5] Saved -> {out_json}")

# ── Report ────────────────────────────────────────────────────────────────────
report_lines = [
    "=" * 65,
    "GV-CoT Stage 5 -- SHAP Attribution Report",
    "=" * 65,
    "",
    f"Model          : Run 3 -- 397-dim scaled features",
    f"Val graphs     : {len(val_graphs):,}",
    f"Gold nodes     : {gold_node_count:,}",
    f"Hall nodes     : {hall_node_count:,}",
    f"Method         : gradient saliency  |x * d(logit_hall)/dx|",
    "",
    "=" * 65,
    "Explainability Coverage  (primary metric)",
    "=" * 65,
    "",
    f"  Coverage     : {explainability_coverage:.4f}",
    f"  Correct      : {coverage_correct:,} / {coverage_total:,} graphs",
    "",
    "  Definition: % of graphs where the highest-saliency",
    "  predicted-hallucinated node is truly hallucinated.",
    "",
    "-" * 65,
    "Feature Saliency Analysis",
    "-" * 65,
    "",
    f"  Separation score (gold vs hall) : {separation_score:.4f}",
    f"  SBERT dims   mean importance    : {sbert_importance:.6f}",
    f"  Structural dims mean importance : {structural_importance:.6f}",
    "",
    f"  {'Dim':>5}  {'Combined |sal|':>16}",
]

for dim, val in zip(top10_idx, top10_vals):
    feat_type = "SBERT" if dim < 384 else f"structural[{dim-384}]"
    report_lines.append(f"  {dim:>5d}  {val:>16.6f}  ({feat_type})")

report_lines += [
    "",
    "-" * 65,
    "SBERT vs Structural Feature Contribution",
    "-" * 65,
    f"  SBERT (dims 0-383)      : {sbert_importance:.6f}",
    f"  Structural (dims 384-396): {structural_importance:.6f}",
]

ratio = structural_importance / sbert_importance if sbert_importance > 0 else 0
report_lines.append(
    f"  Structural / SBERT ratio : {ratio:.4f}"
)

report_lines += [
    "",
    "-" * 65,
    "Interpretation",
    "-" * 65,
]

if explainability_coverage > 0.60:
    report_lines.append(
        f"  [OK] High explainability coverage ({explainability_coverage:.4f}):"
    )
    report_lines.append(
        "    SHAP attribution correctly localises hallucinated steps in most graphs."
    )
elif explainability_coverage > 0.40:
    report_lines.append(
        f"  [~] Moderate explainability coverage ({explainability_coverage:.4f}):"
    )
    report_lines.append(
        "    SHAP attribution partially localises hallucinated steps."
    )
else:
    report_lines.append(
        f"  [!] Low explainability coverage ({explainability_coverage:.4f}):"
    )
    report_lines.append(
        "    SHAP attribution struggles to localise hallucinated steps."
    )

if structural_importance > sbert_importance:
    report_lines.append(
        "  [OK] Structural features have higher mean saliency than SBERT dims."
    )
    report_lines.append(
        "    Position/degree/connective features are driving predictions."
    )
else:
    report_lines.append(
        "  [~] SBERT dims dominate saliency -- semantic content drives predictions"
    )
    report_lines.append(
        "    more than structural position/degree features."
    )

report_lines += [
    "",
    "=" * 65,
    "Stage 5 SHAP Attribution -- COMPLETE",
    "=" * 65,
]

report_txt = RESULT_DIR / "shap_final_report.txt"
with open(report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"[Stage 5] Saved -> {report_txt}")

print()
print("\n".join(report_lines))
