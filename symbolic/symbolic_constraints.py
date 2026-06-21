import torch  # MUST be first import — Windows DLL fix

import json
import pickle
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

from torch_geometric.data import Data
from sklearn.metrics import f1_score, classification_report

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\gvcot")
GRAPHS_PKL  = BASE_DIR / "graphs"  / "reasoning_graphs_augmented.pkl"
MODEL_PT    = BASE_DIR / "models"  / "gat_best.pt"
OUT_DIR     = BASE_DIR / "symbolic"
RESULT_DIR  = BASE_DIR / "results"

OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from models.gnn_model import ReasoningGAT  # noqa: E402

# ── Load model ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Stage 4] Device: {DEVICE}")

model = ReasoningGAT(in_channels=397, hidden_channels=128, out_channels=2, heads=4)
state = torch.load(MODEL_PT, map_location=DEVICE, weights_only=False)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
print("[Stage 4] Model loaded OK")

# ── Load graphs ───────────────────────────────────────────────────────────────
print("[Stage 4] Loading graphs ...")
with open(GRAPHS_PKL, "rb") as f:
    graphs = pickle.load(f)

total = len(graphs)
split = int(total * 0.80)
val_graphs = graphs[split:]
print(f"[Stage 4] Val set: {len(val_graphs):,} graphs")

# ── nx_to_pyg conversion ──────────────────────────────────────────────────────
def nx_to_pyg(g):
    node_list   = list(g.nodes())
    node_to_idx = {n: k for k, n in enumerate(node_list)}

    x = torch.tensor(
        np.array([g.nodes[n]["embedding"] for n in node_list], dtype=np.float32)
    )
    y = torch.tensor(
        [g.nodes[n]["label"] for n in node_list], dtype=torch.long
    )

    if g.number_of_edges() > 0:
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in g.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

# ── Cosine similarity helper ──────────────────────────────────────────────────
def cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — DIAGNOSTIC METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Stage 4 | Part 1] Running diagnostic metrics ...")

CONTRADICTION_THRESHOLD = 0.50   # cosine sim below this -> potential contradiction

# Per-graph diagnostic accumulators
all_true_labels   = []
all_gat_preds     = []

transitivity_violations = 0
total_edges_checked     = 0

contradiction_flags     = 0
total_edges_cosine      = 0

consistency_scores      = []   # per-graph edge-level consistency

graph_diagnostics = []

for i, g in enumerate(val_graphs):
    if (i + 1) % 3000 == 0:
        print(f"  ... {i+1}/{len(val_graphs)} graphs")

    node_list   = list(g.nodes())
    node_to_idx = {n: k for k, n in enumerate(node_list)}
    num_nodes   = len(node_list)

    if num_nodes < 2:
        continue

    # Run GAT
    pyg = nx_to_pyg(g)
    x   = pyg.x.to(DEVICE)
    ei  = pyg.edge_index.to(DEVICE)
    y   = pyg.y.numpy()

    with torch.no_grad():
        logits = model(x, ei)
        preds  = logits.argmax(dim=-1).cpu().numpy()

    all_true_labels.extend(y.tolist())
    all_gat_preds.extend(preds.tolist())

    # Edge-level diagnostics
    edges = list(g.edges())
    if not edges:
        continue

    graph_violations    = 0
    graph_contradictions= 0
    graph_consistent    = 0

    for u, v in edges:
        ui = node_to_idx[u]
        vi = node_to_idx[v]

        pred_u = int(preds[ui])
        pred_v = int(preds[vi])

        # ── Transitivity check ────────────────────────────────────────────
        # If source node is gold but destination is hallucinated -> violation
        if pred_u == 1 and pred_v == 0:
            transitivity_violations += 1
            graph_violations += 1

        total_edges_checked += 1

        # ── Contradiction check ───────────────────────────────────────────
        # Connected nodes with low embedding cosine similarity
        emb_u = g.nodes[u]["embedding"][:384]   # use SBERT dims only
        emb_v = g.nodes[v]["embedding"][:384]
        sim   = cosine_sim(emb_u, emb_v)

        if sim < CONTRADICTION_THRESHOLD:
            contradiction_flags += 1
            graph_contradictions += 1

        total_edges_cosine += 1

        # ── Consistency check ─────────────────────────────────────────────
        # Both nodes agree on their label prediction
        if pred_u == pred_v:
            graph_consistent += 1

    n_edges = len(edges)
    graph_consistency = graph_consistent / n_edges

    consistency_scores.append(graph_consistency)

    graph_diagnostics.append({
        "graph_idx"            : split + i,
        "num_nodes"            : num_nodes,
        "num_edges"            : n_edges,
        "transitivity_violations": graph_violations,
        "contradiction_flags"  : graph_contradictions,
        "consistency_score"    : round(graph_consistency, 4),
    })

# ── Aggregate diagnostic metrics ──────────────────────────────────────────────
transitivity_violation_rate = (
    transitivity_violations / total_edges_checked
    if total_edges_checked > 0 else 0.0
)
contradiction_rate = (
    contradiction_flags / total_edges_cosine
    if total_edges_cosine > 0 else 0.0
)
mean_consistency_score = float(np.mean(consistency_scores)) if consistency_scores else 0.0

# GAT baseline F1
gat_f1_gold = f1_score(all_true_labels, all_gat_preds, pos_label=1)
gat_f1_hall = f1_score(all_true_labels, all_gat_preds, pos_label=0)
gat_f1_macro= f1_score(all_true_labels, all_gat_preds, average="macro")

print(f"\n  Transitivity violation rate : {transitivity_violation_rate:.4f}")
print(f"  Contradiction rate          : {contradiction_rate:.4f}")
print(f"  Mean consistency score      : {mean_consistency_score:.4f}")
print(f"  GAT baseline F1 (gold)      : {gat_f1_gold:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — OPTIONAL CORRECTION EXPERIMENT
# Clearly labeled as heuristic upper-bound — NOT a main claim
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Stage 4 | Part 2] Running heuristic correction experiment ...")
print("  (NOTE: this is an upper-bound heuristic, not a learned method)")

corrected_preds = []
correction_count = 0

for i, g in enumerate(val_graphs):
    node_list   = list(g.nodes())
    node_to_idx = {n: k for k, n in enumerate(node_list)}
    num_nodes   = len(node_list)

    if num_nodes < 2:
        continue

    pyg = nx_to_pyg(g)
    x   = pyg.x.to(DEVICE)
    ei  = pyg.edge_index.to(DEVICE)

    with torch.no_grad():
        logits = model(x, ei)
        preds  = logits.argmax(dim=-1).cpu().numpy().copy()

    # Build predecessor map (for transitivity override)
    predecessors = defaultdict(list)
    for u, v in g.edges():
        predecessors[node_to_idx[v]].append(node_to_idx[u])

    # Transitivity override:
    # If node is predicted hallucinated but ALL predecessors are gold -> override to gold
    corrected = preds.copy()
    for node_i in range(num_nodes):
        if corrected[node_i] == 0:   # predicted hallucinated
            preds_of_preds = [preds[p] for p in predecessors[node_i]]
            if preds_of_preds and all(p == 1 for p in preds_of_preds):
                corrected[node_i] = 1
                correction_count += 1

    corrected_preds.extend(corrected.tolist())

# ── Corrected F1 ─────────────────────────────────────────────────────────────
sym_f1_gold  = f1_score(all_true_labels, corrected_preds, pos_label=1)
sym_f1_hall  = f1_score(all_true_labels, corrected_preds, pos_label=0)
sym_f1_macro = f1_score(all_true_labels, corrected_preds, average="macro")

f1_delta_gold = sym_f1_gold  - gat_f1_gold
f1_delta_macro= sym_f1_macro - gat_f1_macro

print(f"\n  Corrections applied         : {correction_count:,}")
print(f"  GAT F1 (gold)               : {gat_f1_gold:.4f}")
print(f"  Symbolic post-proc F1 (gold): {sym_f1_gold:.4f}  (delta {f1_delta_gold:+.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
symbolic_output = {
    "meta": {
        "val_graphs"            : len(val_graphs),
        "graphs_processed"      : len(graph_diagnostics),
        "total_edges_checked"   : total_edges_checked,
        "contradiction_threshold": CONTRADICTION_THRESHOLD,
    },

    # ── Part 1: Diagnostic metrics ─────────────────────────────────────────
    "diagnostic": {
        "transitivity_violation_rate": round(transitivity_violation_rate, 4),
        "contradiction_rate"         : round(contradiction_rate, 4),
        "mean_consistency_score"     : round(mean_consistency_score, 4),
        "gat_baseline": {
            "f1_gold" : round(gat_f1_gold,  4),
            "f1_hall" : round(gat_f1_hall,  4),
            "f1_macro": round(gat_f1_macro, 4),
        },
    },

    # ── Part 2: Heuristic correction experiment ────────────────────────────
    "heuristic_correction_experiment": {
        "WARNING": "Upper-bound heuristic only. Not a main claim.",
        "corrections_applied": correction_count,
        "symbolic_postproc": {
            "f1_gold" : round(sym_f1_gold,   4),
            "f1_hall" : round(sym_f1_hall,   4),
            "f1_macro": round(sym_f1_macro,  4),
        },
        "delta_vs_gat": {
            "f1_gold_delta" : round(f1_delta_gold,  4),
            "f1_macro_delta": round(f1_delta_macro, 4),
        },
    },

    "per_graph_diagnostics": graph_diagnostics,
}

out_json = OUT_DIR / "symbolic_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(symbolic_output, f, indent=2)
print(f"\n[Stage 4] Saved -> {out_json}")

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
report_lines = [
    "=" * 65,
    "GV-CoT Stage 4 - Symbolic Constraints Report",
    "=" * 65,
    "",
    f"Val graphs processed : {len(graph_diagnostics):,}",
    f"Total edges checked  : {total_edges_checked:,}",
    f"Contradiction threshold (cosine sim): {CONTRADICTION_THRESHOLD}",
    "",
    "=" * 65,
    "PART 1 - DIAGNOSTIC METRICS  (main result)",
    "=" * 65,
    "",
    "-" * 65,
    "Transitivity",
    "-" * 65,
    f"  Violation rate : {transitivity_violation_rate:.4f}",
    f"  ({transitivity_violations:,} edges where gold->hallucinated transition detected)",
    "",
    "-" * 65,
    "Contradiction Detection",
    "-" * 65,
    f"  Contradiction rate : {contradiction_rate:.4f}",
    f"  ({contradiction_flags:,} edges where cosine sim < {CONTRADICTION_THRESHOLD})",
    "",
    "-" * 65,
    "Graph Consistency Score",
    "-" * 65,
    f"  Mean consistency : {mean_consistency_score:.4f}",
    "  (proportion of edges where both endpoints share same predicted label)",
    "",
    "-" * 65,
    "GAT Baseline F1 (reference)",
    "-" * 65,
    f"  Gold (1) F1  : {gat_f1_gold:.4f}",
    f"  Hall (0) F1  : {gat_f1_hall:.4f}",
    f"  Macro F1     : {gat_f1_macro:.4f}",
    "",
    "=" * 65,
    "PART 2 - HEURISTIC CORRECTION EXPERIMENT",
    "  [!] Upper-bound heuristic only. NOT a main claim.",
    "  Rule: if node predicted hallucinated but all predecessors",
    "        are predicted gold -> override to gold.",
    "=" * 65,
    "",
    f"  Corrections applied          : {correction_count:,}",
    f"  Symbolic post-proc F1 (gold) : {sym_f1_gold:.4f}  (delta {f1_delta_gold:+.4f} vs GAT)",
    f"  Symbolic post-proc F1 (hall) : {sym_f1_hall:.4f}",
    f"  Symbolic post-proc Macro F1  : {sym_f1_macro:.4f}  (delta {f1_delta_macro:+.4f} vs GAT)",
    "",
    "-" * 65,
    "Interpretation",
    "-" * 65,
]

# Auto-interpret
if transitivity_violation_rate > 0.10:
    report_lines.append(
        f"  [!] High transitivity violation rate ({transitivity_violation_rate:.4f}):"
    )
    report_lines.append(
        "    GAT predictions frequently break the gold->gold chain assumption."
    )
else:
    report_lines.append(
        f"  [OK] Low transitivity violation rate ({transitivity_violation_rate:.4f}):"
    )
    report_lines.append(
        "    GAT predictions are mostly consistent with reasoning chain flow."
    )

if contradiction_rate > 0.20:
    report_lines.append(
        f"  [!] High contradiction rate ({contradiction_rate:.4f}):"
    )
    report_lines.append(
        "    Many connected node pairs have low semantic similarity."
    )
else:
    report_lines.append(
        f"  [OK] Low contradiction rate ({contradiction_rate:.4f}):"
    )
    report_lines.append(
        "    Connected nodes are semantically coherent."
    )

report_lines += [
    "",
    "=" * 65,
    "Stage 4 Symbolic Constraints - COMPLETE",
    "=" * 65,
]

report_txt = RESULT_DIR / "symbolic_report.txt"
with open(report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"[Stage 4] Saved -> {report_txt}")

print()
print("\n".join(report_lines))
