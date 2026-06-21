import torch  # MUST be first import — Windows DLL fix

import json
import os
import pickle
import sys

import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from torch_geometric.data import Data

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_PKL = os.path.join(ROOT, "graphs",     "musique_cot_graphs.pkl")
MODEL_PT   = os.path.join(ROOT, "models",     "gat_best.pt")
RESULT_DIR = os.path.join(ROOT, "results")
OUT_DIR    = os.path.join(ROOT, "evaluation")

sys.path.insert(0, ROOT)
from models.gnn_model import ReasoningGAT  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MuSiQue CoT Eval] Device: {DEVICE}")

# ── Augment: apply same 13 structural features + 0.1 scaling as Run 3 ─────────
def augment_graph(g):
    nodes  = list(g.nodes())
    n      = len(nodes)
    denom  = max(n - 1, 1)

    for node_idx, node in enumerate(nodes):
        orig_emb = g.nodes[node]["embedding"]
        if len(orig_emb) == 397:
            return  # already augmented
        text    = g.nodes[node]["text"]
        text_lo = text.lower()

        position    = 0.5 if n == 1 else node_idx / denom
        is_first    = 1.0 if node_idx == 0       else 0.0
        is_last     = 1.0 if node_idx == (n - 1) else 0.0
        in_deg      = g.in_degree(node)  / denom
        out_deg     = g.out_degree(node) / denom
        total_deg   = (g.in_degree(node) + g.out_degree(node)) / (2 * denom)
        c_because   = 1.0 if "because"   in text_lo else 0.0
        c_therefore = 1.0 if "therefore" in text_lo else 0.0
        c_since     = 1.0 if "since"     in text_lo else 0.0
        c_hence     = 1.0 if "hence"     in text_lo else 0.0
        c_thus      = 1.0 if "thus"      in text_lo else 0.0
        c_so        = 1.0 if " so "      in text_lo else 0.0
        norm_length = min(len(text.split()) / 50.0, 1.0)

        structural = np.array([
            position, is_first, is_last,
            in_deg, out_deg, total_deg,
            c_because, c_therefore, c_since, c_hence, c_thus, c_so,
            norm_length,
        ], dtype=np.float32) * 0.1

        g.nodes[node]["embedding"] = np.concatenate([orig_emb, structural]).astype(np.float32)

# ── Load model ─────────────────────────────────────────────────────────────────
model = ReasoningGAT(in_channels=397, hidden_channels=128, out_channels=2, heads=4)
state = torch.load(MODEL_PT, map_location=DEVICE, weights_only=False)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
print("[MuSiQue CoT Eval] Model loaded OK  (Run 3 -- trained on HotpotQA)")

# ── Load graphs ────────────────────────────────────────────────────────────────
print(f"[MuSiQue CoT Eval] Loading {GRAPHS_PKL} ...")
with open(GRAPHS_PKL, "rb") as f:
    graphs = pickle.load(f)
print(f"[MuSiQue CoT Eval] {len(graphs):,} graphs loaded")

# ── Augment graphs to 397 dims ─────────────────────────────────────────────────
print("[MuSiQue CoT Eval] Augmenting graphs to 397-dim ...")
for g in graphs:
    augment_graph(g)

# Verify dim
sample_node = list(graphs[0].nodes())[0]
dim = len(graphs[0].nodes[sample_node]["embedding"])
print(f"[MuSiQue CoT Eval] Embedding dim: {dim} (expected 397)")
assert dim == 397, f"Expected 397, got {dim}"

# ── nx_to_pyg ──────────────────────────────────────────────────────────────────
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

# ── Inference ──────────────────────────────────────────────────────────────────
print("[MuSiQue CoT Eval] Running GAT inference ...")

all_preds  = []
all_labels = []

for i, g in enumerate(graphs):
    if (i + 1) % 3000 == 0:
        print(f"  ... {i+1:,}/{len(graphs):,} graphs")

    pyg = nx_to_pyg(g)
    x   = pyg.x.to(DEVICE)
    ei  = pyg.edge_index.to(DEVICE)
    y   = pyg.y.numpy()

    with torch.no_grad():
        logits = model(x, ei)
        preds  = logits.argmax(dim=-1).cpu().numpy()

    all_preds.extend(preds.tolist())
    all_labels.extend(y.tolist())

# ── Metrics ────────────────────────────────────────────────────────────────────
f1_gold  = f1_score(all_labels, all_preds, pos_label=1)
f1_hall  = f1_score(all_labels, all_preds, pos_label=0)
f1_macro = f1_score(all_labels, all_preds, average="macro")
report   = classification_report(
    all_labels, all_preds,
    target_names=["distractor (0)", "gold (1)"]
)

hotpot_f1_gold  = 0.6473
hotpot_f1_hall  = 0.6124
hotpot_f1_macro = 0.6299

delta_gold  = f1_gold  - hotpot_f1_gold
delta_macro = f1_macro - hotpot_f1_macro

print(f"\n[MuSiQue CoT Eval] F1 Gold  : {f1_gold:.4f}  (HotpotQA: {hotpot_f1_gold:.4f}  delta {delta_gold:+.4f})")
print(f"[MuSiQue CoT Eval] F1 Hall  : {f1_hall:.4f}  (HotpotQA: {hotpot_f1_hall:.4f})")
print(f"[MuSiQue CoT Eval] F1 Macro : {f1_macro:.4f}  (HotpotQA: {hotpot_f1_macro:.4f}  delta {delta_macro:+.4f})")

# ── Save JSON ──────────────────────────────────────────────────────────────────
results = {
    "dataset"    : "MuSiQue CoT chains (answerable only)",
    "model"      : "Run 3 -- 397-dim scaled, trained on HotpotQA",
    "graphs"     : len(graphs),
    "total_nodes": len(all_labels),
    "musique_cot": {
        "f1_gold" : round(f1_gold,  4),
        "f1_hall" : round(f1_hall,  4),
        "f1_macro": round(f1_macro, 4),
    },
    "hotpotqa_reference": {
        "f1_gold" : hotpot_f1_gold,
        "f1_hall" : hotpot_f1_hall,
        "f1_macro": hotpot_f1_macro,
    },
    "delta": {
        "f1_gold_delta" : round(delta_gold,  4),
        "f1_macro_delta": round(delta_macro, 4),
    },
    "classification_report": report,
}

out_json = os.path.join(OUT_DIR, "musique_cot_eval_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n[MuSiQue CoT Eval] Saved -> {out_json}")

# ── Report ─────────────────────────────────────────────────────────────────────
report_lines = [
    "=" * 65,
    "GV-CoT -- MuSiQue CoT Cross-Dataset Evaluation",
    "=" * 65,
    "",
    f"Model          : Run 3 -- 397-dim scaled (trained on HotpotQA)",
    f"Test dataset   : MuSiQue CoT chains (answerable questions only)",
    f"Graphs         : {len(graphs):,}",
    f"Total nodes    : {len(all_labels):,}",
    "",
    "-" * 65,
    "Classification Report",
    "-" * 65,
    report,
    "-" * 65,
    "HotpotQA vs MuSiQue CoT Comparison",
    "-" * 65,
    f"{'Metric':<20} {'HotpotQA':>12} {'MuSiQue CoT':>12} {'Delta':>10}",
    f"{'F1 Gold (1)':<20} {hotpot_f1_gold:>12.4f} {f1_gold:>12.4f} {delta_gold:>+10.4f}",
    f"{'F1 Hall (0)':<20} {hotpot_f1_hall:>12.4f} {f1_hall:>12.4f} {f1_hall - hotpot_f1_hall:>+10.4f}",
    f"{'F1 Macro':<20} {hotpot_f1_macro:>12.4f} {f1_macro:>12.4f} {delta_macro:>+10.4f}",
    "",
    "-" * 65,
    "Interpretation",
    "-" * 65,
]

if delta_macro > -0.05:
    report_lines.append(
        f"  [OK] Strong generalisation: Macro F1 drop < 5pts ({delta_macro:+.4f})."
    )
    report_lines.append(
        "    GV-CoT transfers well to MuSiQue CoT chains without retraining."
    )
elif delta_macro > -0.10:
    report_lines.append(
        f"  [~] Moderate generalisation: Macro F1 drop {delta_macro:+.4f}."
    )
    report_lines.append(
        "    Some domain shift from HotpotQA to MuSiQue CoT chains."
    )
else:
    report_lines.append(
        f"  [!] Limited generalisation: Macro F1 drop {delta_macro:+.4f}."
    )
    report_lines.append(
        "    Significant domain shift -- model may be HotpotQA-specific."
    )

report_lines += [
    "",
    "Note: Both HotpotQA and MuSiQue CoT graphs use CoT chain steps as nodes.",
    "      Node structure is comparable; domain shift is the main variable.",
    "",
    "=" * 65,
    "MuSiQue CoT Evaluation -- COMPLETE",
    "=" * 65,
]

report_txt = os.path.join(RESULT_DIR, "musique_cot_eval_report.txt")
with open(report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"[MuSiQue CoT Eval] Saved -> {report_txt}")

print()
print("\n".join(report_lines))
