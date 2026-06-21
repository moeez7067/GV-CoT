import torch  # MUST be first import — Windows DLL fix

import json
import pickle
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

from torch_geometric.data import Data

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\gvcot")
MUSIQUE_PKL = BASE_DIR / "graphs"     / "musique_graphs.pkl"
MODEL_PT    = BASE_DIR / "models"     / "gat_best.pt"
RESULT_DIR  = BASE_DIR / "results"
OUT_DIR     = BASE_DIR / "evaluation"

sys.path.insert(0, str(BASE_DIR))
from models.gnn_model import ReasoningGAT  # noqa: E402

# ── Load model ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MuSiQue Eval] Device: {DEVICE}")

model = ReasoningGAT(in_channels=397, hidden_channels=128, out_channels=2, heads=4)
state = torch.load(MODEL_PT, map_location=DEVICE, weights_only=False)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
print("[MuSiQue Eval] Model loaded OK  (Run 3 -- trained on HotpotQA)")

# ── Load graphs ───────────────────────────────────────────────────────────────
print("[MuSiQue Eval] Loading musique_graphs.pkl ...")
with open(MUSIQUE_PKL, "rb") as f:
    graphs = pickle.load(f)
print(f"[MuSiQue Eval] {len(graphs):,} graphs loaded")

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

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("[MuSiQue Eval] Running GAT inference ...")

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

# ── Metrics ───────────────────────────────────────────────────────────────────
f1_gold  = f1_score(all_labels, all_preds, pos_label=1)
f1_hall  = f1_score(all_labels, all_preds, pos_label=0)
f1_macro = f1_score(all_labels, all_preds, average="macro")
report   = classification_report(
    all_labels, all_preds,
    target_names=["distractor (0)", "gold (1)"]
)

# HotpotQA reference numbers
hotpot_f1_gold  = 0.6473
hotpot_f1_hall  = 0.6124
hotpot_f1_macro = 0.6299

delta_gold  = f1_gold  - hotpot_f1_gold
delta_macro = f1_macro - hotpot_f1_macro

print(f"\n[MuSiQue Eval] F1 Gold  : {f1_gold:.4f}  (HotpotQA: {hotpot_f1_gold:.4f}  delta {delta_gold:+.4f})")
print(f"[MuSiQue Eval] F1 Hall  : {f1_hall:.4f}  (HotpotQA: {hotpot_f1_hall:.4f})")
print(f"[MuSiQue Eval] F1 Macro : {f1_macro:.4f}  (HotpotQA: {hotpot_f1_macro:.4f}  delta {delta_macro:+.4f})")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "dataset"    : "MuSiQue (answerable only)",
    "model"      : "Run 3 -- 397-dim scaled, trained on HotpotQA",
    "graphs"     : len(graphs),
    "total_nodes": len(all_labels),
    "musique": {
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

out_json = OUT_DIR / "musique_eval_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n[MuSiQue Eval] Saved -> {out_json}")

# ── Report ────────────────────────────────────────────────────────────────────
report_lines = [
    "=" * 65,
    "GV-CoT -- MuSiQue Cross-Dataset Evaluation",
    "=" * 65,
    "",
    f"Model          : Run 3 -- 397-dim scaled (trained on HotpotQA)",
    f"Test dataset   : MuSiQue (answerable questions only)",
    f"Graphs         : {len(graphs):,}",
    f"Total nodes    : {len(all_labels):,}",
    "",
    "-" * 65,
    "Classification Report",
    "-" * 65,
    report,
    "-" * 65,
    "HotpotQA vs MuSiQue Comparison",
    "-" * 65,
    f"{'Metric':<20} {'HotpotQA':>12} {'MuSiQue':>12} {'Delta':>10}",
    f"{'F1 Gold (1)':<20} {hotpot_f1_gold:>12.4f} {f1_gold:>12.4f} {delta_gold:>+10.4f}",
    f"{'F1 Hall (0)':<20} {hotpot_f1_hall:>12.4f} {f1_hall:>12.4f} {f1_hall-hotpot_f1_hall:>+10.4f}",
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
        "    GV-CoT transfers well to MuSiQue without retraining."
    )
elif delta_macro > -0.10:
    report_lines.append(
        f"  [~] Moderate generalisation: Macro F1 drop {delta_macro:+.4f}."
    )
    report_lines.append(
        "    Some domain shift from HotpotQA CoT steps to MuSiQue paragraphs."
    )
else:
    report_lines.append(
        f"  [!] Limited generalisation: Macro F1 drop {delta_macro:+.4f}."
    )
    report_lines.append(
        "    Significant domain shift -- model is HotpotQA-specific."
    )

report_lines += [
    "",
    "Note: HotpotQA used CoT chain steps as nodes (avg 7.58 nodes/graph).",
    "      MuSiQue uses context paragraphs as nodes (20 nodes/graph).",
    "      Performance gap partly reflects this structural difference.",
    "",
    "=" * 65,
    "MuSiQue Evaluation -- COMPLETE",
    "=" * 65,
]

report_txt = RESULT_DIR / "musique_eval_report.txt"
with open(report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"[MuSiQue Eval] Saved -> {report_txt}")

print()
print("\n".join(report_lines))
