import torch  # MUST be first import — Windows DLL fix

import json
import pickle
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

from torch_geometric.data import Data

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\gvcot")
GRAPHS_PKL  = BASE_DIR / "graphs"  / "reasoning_graphs_augmented.pkl"
MODEL_PT    = BASE_DIR / "models"  / "gat_best.pt"
HOTPOT_JSON = BASE_DIR / "data"    / "hotpotqa_train.json"
COT_JSON    = BASE_DIR / "data"    / "cot_chains.json"
OUT_DIR     = BASE_DIR / "evaluation"
RESULT_DIR  = BASE_DIR / "results"

OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from models.gnn_model import ReasoningGAT  # noqa: E402

# ── Load ground truth ─────────────────────────────────────────────────────────
print("[Stage 4] Loading HotpotQA ground truth …")
with open(HOTPOT_JSON, encoding="utf-8") as f:
    hotpot = json.load(f)

gt_lookup = {rec["id"]: rec["answer"] for rec in hotpot}
print(f"[Stage 4] Ground truth loaded: {len(gt_lookup):,} records")

print("[Stage 4] Loading CoT chains ...")
with open(COT_JSON, encoding="utf-8") as f:
    cot_chains = json.load(f)
cot_lookup = {rec["id"]: {"cot_chain": rec["cot_chain"], "model_name": rec["model_name"]} for rec in cot_chains}
print(f"[Stage 4] CoT lookup: {len(cot_lookup):,} entries")

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
print("[Stage 4] Loading graphs …")
with open(GRAPHS_PKL, "rb") as f:
    graphs = pickle.load(f)

total = len(graphs)
split = int(total * 0.80)
val_graphs = graphs[split:]
print(f"[Stage 4] Val set: {len(val_graphs):,} graphs")

# ── nx_to_pyg conversion ──────────────────────────────────────────────────────
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
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in g.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

# ── Answer matching ───────────────────────────────────────────────────────────
def extract_llm_answer(cot_chain: str) -> str:
    """Extract text after '# Final Answer' marker."""
    lower = cot_chain.lower()
    marker = "# final answer"
    idx = lower.rfind(marker)
    if idx == -1:
        idx = lower.rfind("final answer")
    if idx == -1:
        return ""
    return cot_chain[idx:].strip()

def answers_match(cot_chain: str, gt: str) -> bool:
    """Check if ground truth answer appears in the LLM's final answer section."""
    final_section = extract_llm_answer(cot_chain)
    if not final_section:
        return False
    return gt.strip().lower() in final_section.lower()

# ── Evaluate val set ──────────────────────────────────────────────────────────
print("[Stage 4] Evaluating val set …")

results = []
skipped_no_gt  = 0
skipped_no_ans = 0

for i, g in enumerate(val_graphs):
    if (i + 1) % 2000 == 0:
        print(f"  … {i+1}/{len(val_graphs)} graphs processed")

    graph_id   = g.graph.get("id", "")
    question   = g.graph.get("question", "")
    chain      = cot_lookup.get(graph_id, {})
    cot_chain  = chain.get("cot_chain", "")
    model_name = chain.get("model_name", "unknown")

    # Skip if no ground truth
    if graph_id not in gt_lookup:
        skipped_no_gt += 1
        continue

    gt_answer = gt_lookup[graph_id]

    if not cot_chain:
        skipped_no_ans += 1
        continue

    # Convert and run GAT
    pyg = nx_to_pyg(g)
    x   = pyg.x.to(DEVICE)
    ei  = pyg.edge_index.to(DEVICE)

    with torch.no_grad():
        logits = model(x, ei)
        preds  = logits.argmax(dim=-1).cpu().numpy()   # 0=hall, 1=gold

    num_nodes  = len(preds)
    num_gold   = int(preds.sum())
    gold_ratio = num_gold / num_nodes if num_nodes > 0 else 0.0
    correct    = answers_match(cot_chain, gt_answer)

    results.append({
        "id"               : graph_id,
        "question"         : question,
        "cot_chain_snippet": cot_chain[-200:] if cot_chain else "",
        "gt_answer"        : gt_answer,
        "correct"          : correct,
        "num_nodes"        : num_nodes,
        "num_gold"         : num_gold,
        "gold_ratio"       : round(gold_ratio, 4),
        "model_name"       : model_name,
    })

print(f"[Stage 4] Evaluated {len(results):,} graphs")
print(f"          Skipped (no GT): {skipped_no_gt} | Skipped (no answer): {skipped_no_ans}")

# ── Compute aggregate statistics ──────────────────────────────────────────────
total_eval   = len(results)
total_correct= sum(r["correct"] for r in results)
overall_acc  = total_correct / total_eval if total_eval > 0 else 0.0

# Gold ratio stats per correctness group
correct_ratios   = [r["gold_ratio"] for r in results if r["correct"]]
incorrect_ratios = [r["gold_ratio"] for r in results if not r["correct"]]

mean_gr_correct   = np.mean(correct_ratios)   if correct_ratios   else 0.0
mean_gr_incorrect = np.mean(incorrect_ratios) if incorrect_ratios else 0.0

# Pearson correlation: gold_ratio vs correctness (binary)
gold_ratios  = np.array([r["gold_ratio"] for r in results])
correctness  = np.array([float(r["correct"]) for r in results])
if gold_ratios.std() > 0:
    pearson_r = float(np.corrcoef(gold_ratios, correctness)[0, 1])
else:
    pearson_r = 0.0

# Accuracy by gold_ratio bucket
def bucket_acc(results, lo, hi):
    sub = [r for r in results if lo <= r["gold_ratio"] < hi]
    if not sub:
        return None, 0
    return sum(r["correct"] for r in sub) / len(sub), len(sub)

buckets = [
    ("0.0-0.2",  0.0, 0.2),
    ("0.2-0.4",  0.2, 0.4),
    ("0.4-0.6",  0.4, 0.6),
    ("0.6-0.8",  0.6, 0.8),
    ("0.8-1.0",  0.8, 1.01),
]
bucket_results = []
for label, lo, hi in buckets:
    acc, n = bucket_acc(results, lo, hi)
    bucket_results.append((label, acc, n))

# Accuracy by model
by_model = defaultdict(list)
for r in results:
    by_model[r["model_name"]].append(r["correct"])

model_accs = {
    m: (sum(v) / len(v), len(v))
    for m, v in by_model.items()
}

# ── Save results JSON ─────────────────────────────────────────────────────────
eval_output = {
    "meta": {
        "val_graphs_total"  : len(val_graphs),
        "graphs_evaluated"  : total_eval,
        "skipped_no_gt"     : skipped_no_gt,
        "skipped_no_answer" : skipped_no_ans,
    },
    "overall": {
        "accuracy"          : round(overall_acc, 4),
        "correct"           : int(total_correct),
        "total"             : total_eval,
    },
    "gold_ratio_vs_correctness": {
        "mean_gold_ratio_correct"  : round(mean_gr_correct,   4),
        "mean_gold_ratio_incorrect": round(mean_gr_incorrect,  4),
        "delta"                    : round(mean_gr_correct - mean_gr_incorrect, 4),
        "pearson_r"                : round(pearson_r, 4),
    },
    "accuracy_by_gold_ratio_bucket": {
        label: {"accuracy": round(acc, 4) if acc is not None else None, "n": n}
        for label, acc, n in bucket_results
    },
    "accuracy_by_model": {
        m: {"accuracy": round(acc, 4), "n": n}
        for m, (acc, n) in model_accs.items()
    },
    "per_graph_results": results,
}

out_json = OUT_DIR / "answer_eval_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(eval_output, f, indent=2, ensure_ascii=False)
print(f"[Stage 4] Saved -> {out_json}")

# ── Save report ───────────────────────────────────────────────────────────────
report_lines = [
    "=" * 65,
    "GV-CoT Stage 4 - Answer Evaluation Report (Option A)",
    "=" * 65,
    "",
    f"Val graphs evaluated : {total_eval:,}",
    f"Overall answer accuracy: {overall_acc:.4f} ({total_correct:,}/{total_eval:,} correct)",
    "",
    "-" * 65,
    "Gold Ratio vs Answer Correctness",
    "-" * 65,
    f"  Mean gold ratio - CORRECT answers  : {mean_gr_correct:.4f}",
    f"  Mean gold ratio - INCORRECT answers: {mean_gr_incorrect:.4f}",
    f"  Delta (correct - incorrect)         : {mean_gr_correct - mean_gr_incorrect:+.4f}",
    f"  Pearson r (gold_ratio ~ correctness): {pearson_r:+.4f}",
    "",
    "-" * 65,
    "Answer Accuracy by Gold Ratio Bucket",
    "-" * 65,
]
for label, acc, n in bucket_results:
    if acc is not None:
        report_lines.append(f"  gold_ratio {label}  ->  accuracy {acc:.4f}  (n={n:,})")
    else:
        report_lines.append(f"  gold_ratio {label}  ->  no samples")

report_lines += [
    "",
    "-" * 65,
    "Answer Accuracy by LLM",
    "-" * 65,
]
for m, (acc, n) in sorted(model_accs.items()):
    report_lines.append(f"  {m:<35} accuracy {acc:.4f}  (n={n:,})")

report_lines += [
    "",
    "-" * 65,
    "Interpretation",
    "-" * 65,
]

# Auto-interpret pearson
if pearson_r > 0.05:
    report_lines.append(
        f"  Positive correlation (r={pearson_r:.4f}): higher gold ratio -> more correct answers."
    )
    report_lines.append(
        "    GV-CoT hallucination detection correlates with answer quality."
    )
elif pearson_r < -0.05:
    report_lines.append(
        f"  Negative correlation (r={pearson_r:.4f}): unexpected inverse relationship."
    )
    report_lines.append(
        "    Investigate labeling or answer matching pipeline."
    )
else:
    report_lines.append(
        f"  Near-zero correlation (r={pearson_r:.4f}): gold ratio does not predict answer correctness."
    )
    report_lines.append(
        "    GAT predictions may not yet be strong enough to gate answer quality."
    )

# Delta interpretation
if mean_gr_correct > mean_gr_incorrect:
    report_lines.append(
        f"  Correct answers have higher mean gold ratio (+{mean_gr_correct - mean_gr_incorrect:.4f})."
    )
else:
    report_lines.append(
        f"  Incorrect answers have higher or equal mean gold ratio."
    )

report_lines += [
    "",
    "=" * 65,
    "Stage 4 Answer Evaluation - COMPLETE",
    "=" * 65,
]

report_txt = RESULT_DIR / "answer_eval_report.txt"
with open(report_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"[Stage 4] Saved -> {report_txt}")

print()
print("\n".join(report_lines))
