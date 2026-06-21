import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # must be before sentence_transformers to fix Windows DLL load order

import csv
import json
import pickle
import random
import numpy as np
from collections import Counter

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH    = os.path.join(ROOT, "graphs", "reasoning_graphs.pkl")
COT_PATH    = os.path.join(ROOT, "data", "cot_chains.json")
RESULTS_DIR = os.path.join(ROOT, "results")

RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────

print(f"Loading {PKL_PATH} ...")
with open(PKL_PATH, "rb") as f:
    graphs = pickle.load(f)
print(f"  {len(graphs):,} graphs loaded.")

print(f"Loading {COT_PATH} ...")
with open(COT_PATH, encoding="utf-8") as f:
    cot_data = json.load(f)
cot_lookup = {r["id"]: r.get("cot_chain", "") for r in cot_data}

rng = random.Random(RANDOM_SEED)

# Report lines accumulate for file output
report_lines = []

def log(s=""):
    print(s)
    report_lines.append(s)


# ── 1. Dataset Statistics ──────────────────────────────────────────────────

log("=" * 70)
log("STAGE 1 VERIFICATION REPORT")
log("=" * 70)

node_counts = [g.number_of_nodes() for g in graphs]
edge_counts = [g.number_of_edges() for g in graphs]
total_nodes = sum(node_counts)
total_edges = sum(edge_counts)

all_labels = [
    d.get("label", -1)
    for g in graphs
    for _, d in g.nodes(data=True)
]
n_gold = sum(1 for l in all_labels if l == 1)
n_dist = sum(1 for l in all_labels if l == 0)

log("\n── 1. Dataset Statistics ─────────────────────────────────────────────")
log(f"  Total graphs     : {len(graphs):,}")
log(f"  Total nodes      : {total_nodes:,}")
log(f"  Total edges      : {total_edges:,}")
log(f"  Avg nodes/graph  : {np.mean(node_counts):.2f}")
log(f"  Median nodes     : {np.median(node_counts):.1f}")
log(f"  Min nodes        : {min(node_counts)}")
log(f"  Max nodes        : {max(node_counts)}")
log(f"  Avg edges/graph  : {np.mean(edge_counts):.2f}")
log(f"  Gold nodes  (1)  : {n_gold:,}  ({n_gold / total_nodes:.2%})")
log(f"  Distractor  (0)  : {n_dist:,}  ({n_dist / total_nodes:.2%})")

stats = {
    "total_graphs":          len(graphs),
    "total_nodes":           total_nodes,
    "total_edges":           total_edges,
    "avg_nodes_per_graph":   round(float(np.mean(node_counts)), 4),
    "median_nodes_per_graph": float(np.median(node_counts)),
    "min_nodes_per_graph":   int(min(node_counts)),
    "max_nodes_per_graph":   int(max(node_counts)),
    "avg_edges_per_graph":   round(float(np.mean(edge_counts)), 4),
    "gold_nodes":            n_gold,
    "distractor_nodes":      n_dist,
    "gold_percent":          round(n_gold / total_nodes * 100, 4),
}


# ── 2. Graph Structure Validation ─────────────────────────────────────────

log("\n── 2. Graph Structure Validation ─────────────────────────────────────")

empty_graphs   = []
no_edge_graphs = []
missing_emb    = []
wrong_dim      = []
missing_label  = []

for i, g in enumerate(graphs):
    gid   = g.graph.get("id", f"<missing:{i}>")
    nodes = list(g.nodes(data=True))

    if len(nodes) == 0:
        empty_graphs.append(gid)
        continue

    if g.number_of_edges() == 0:
        no_edge_graphs.append(gid)

    for nid, attrs in nodes:
        if "embedding" not in attrs:
            missing_emb.append((gid, nid))
        else:
            dim = np.array(attrs["embedding"]).shape
            if dim != (384,):
                wrong_dim.append((gid, nid, dim))
        if "label" not in attrs:
            missing_label.append((gid, nid))


def _report_check(label, issues, limit=5):
    if issues:
        log(f"  [FAIL] {label}: {len(issues):,} (first {min(limit, len(issues))} shown)")
        for item in issues[:limit]:
            log(f"         {item}")
    else:
        log(f"  [OK]   {label}: 0")


_report_check("Empty graphs",                   empty_graphs)
_report_check("Graphs with no edges",           no_edge_graphs)
_report_check("Nodes missing embedding",         missing_emb)
_report_check("Nodes with wrong embedding dim",  wrong_dim)
_report_check("Nodes missing label",             missing_label)

n_structural = len(empty_graphs) + len(no_edge_graphs) + len(missing_emb) + len(wrong_dim) + len(missing_label)
log(f"\n  Total structural issues: {n_structural}")

stats["structural"] = {
    "empty_graphs":              len(empty_graphs),
    "no_edge_graphs":            len(no_edge_graphs),
    "nodes_missing_embedding":   len(missing_emb),
    "nodes_wrong_embedding_dim": len(wrong_dim),
    "nodes_missing_label":       len(missing_label),
}


# ── 3. Random Sample Inspection ───────────────────────────────────────────

log("\n── 3. Random Sample Inspection (20 graphs) ───────────────────────────")

for g in rng.sample(graphs, min(20, len(graphs))):
    n_g   = g.number_of_nodes()
    n_gld = sum(1 for _, d in g.nodes(data=True) if d.get("label") == 1)
    log(f"\n  ID : {g.graph.get('id', '?')}")
    log(f"  Q  : {g.graph.get('question', '')[:100]}")
    log(f"  Ans: {g.graph.get('answer', '')[:80]}")
    log(f"  Nodes: {n_g}  Gold: {n_gld}  Dist: {n_g - n_gld}")
    for nid, attrs in g.nodes(data=True):
        tag = "GOLD" if attrs.get("label") == 1 else "DIST"
        log(f"    [{nid:>2} {tag}] {attrs.get('text', '')[:100]}")


# ── 4. Label Quality Audit → CSV ──────────────────────────────────────────

log("\n── 4. Label Quality Audit ────────────────────────────────────────────")

all_node_refs = [
    (g, nid, attrs)
    for g in graphs
    for nid, attrs in g.nodes(data=True)
]
sample_nodes = rng.sample(all_node_refs, min(100, len(all_node_refs)))

csv_path = os.path.join(RESULTS_DIR, "stage1_sample_nodes.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "node_text", "label", "original_cot", "gold_answer"])
    for g, nid, attrs in sample_nodes:
        qid = g.graph.get("id", "")
        writer.writerow([
            qid,
            attrs.get("text", ""),
            attrs.get("label", ""),
            cot_lookup.get(qid, "")[:500],
            g.graph.get("answer", ""),
        ])

log(f"  {len(sample_nodes)} random nodes saved to {csv_path}")


# ── 5. Data Leakage Checks ────────────────────────────────────────────────

log("\n── 5. Data Leakage Checks ────────────────────────────────────────────")

id_counts = Counter(g.graph.get("id", "") for g in graphs)
q_counts  = Counter(g.graph.get("question", "") for g in graphs)

dup_ids = {k: v for k, v in id_counts.items() if v > 1}
dup_qs  = {k: v for k, v in q_counts.items()  if v > 1}

if dup_ids:
    log(f"  [FAIL] Duplicate graph IDs: {len(dup_ids):,}  (e.g. {list(dup_ids.keys())[:3]})")
else:
    log(f"  [OK]   No duplicate graph IDs  ({len(id_counts):,} unique)")

if dup_qs:
    log(f"  [FAIL] Duplicate questions: {len(dup_qs):,}  (e.g. {list(dup_qs.keys())[0][:60]}...)")
else:
    log(f"  [OK]   No duplicate questions  ({len(q_counts):,} unique)")

stats["leakage"] = {
    "duplicate_ids":       len(dup_ids),
    "duplicate_questions": len(dup_qs),
}


# ── 6. Final Verdict ──────────────────────────────────────────────────────

log("\n── 6. Final Verdict ──────────────────────────────────────────────────")

gold_pct  = n_gold / total_nodes * 100
avg_nodes = float(np.mean(node_counts))

chk_labels    = "HEALTHY" if 5.0  <= gold_pct  <= 60.0 else "WARNING"
chk_size      = "OK"      if 3.0  <= avg_nodes <= 20.0 else "WARNING"
chk_leakage   = "OK"      if not dup_ids and not dup_qs  else "WARNING"
chk_structure = "OK"      if n_structural == 0            else "WARNING"

ready = all(v in ("OK", "HEALTHY") for v in [chk_labels, chk_size, chk_leakage, chk_structure])

log(f"  Label distribution   : {chk_labels:<8}  ({gold_pct:.2f}% gold)")
log(f"  Graph size           : {chk_size:<8}  (avg {avg_nodes:.2f} nodes/graph)")
log(f"  No data leakage      : {chk_leakage}")
log(f"  Structural integrity : {chk_structure}")
log("")
log(f"  {'>>> READY FOR STAGE 2 <<<' if ready else '>>> ISSUES FOUND — review before Stage 2 <<<'}")

stats["verdict"] = {
    "label_health":        chk_labels,
    "graph_size":          chk_size,
    "no_leakage":          chk_leakage,
    "structural_integrity": chk_structure,
    "ready_for_stage2":    ready,
}


# ── Save outputs ──────────────────────────────────────────────────────────

report_path = os.path.join(RESULTS_DIR, "stage1_verification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"\nReport  → {report_path}")

stats_path = os.path.join(RESULTS_DIR, "stage1_statistics.json")
with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print(f"Stats   → {stats_path}")
print(f"CSV     → {csv_path}")
