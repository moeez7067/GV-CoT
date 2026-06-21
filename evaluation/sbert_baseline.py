import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # must be before sentence_transformers to fix Windows DLL load order

import pickle
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH    = os.path.join(ROOT, "graphs", "reasoning_graphs.pkl")
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_PATH    = os.path.join(RESULTS_DIR, "sbert_baseline.txt")

TRAIN_RATIO = 0.8

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load and split ─────────────────────────────────────────────────────────────

print(f"Loading {PKL_PATH} ...")
with open(PKL_PATH, "rb") as f:
    graphs = pickle.load(f)

split = int(len(graphs) * TRAIN_RATIO)
val_graphs = graphs[split:]

# ── Collect val labels ────────────────────────────────────────────────────────

labels = [
    attrs.get("label", 0)
    for g in val_graphs
    for _, attrs in g.nodes(data=True)
]
labels = np.array(labels, dtype=int)

n_val_graphs = len(val_graphs)
n_nodes      = len(labels)
n_gold       = int((labels == 1).sum())
n_hall       = int((labels == 0).sum())

lines = []

def log(s=""):
    print(s)
    lines.append(s)

log("=" * 60)
log("SBERT BASELINE REPORT")
log("=" * 60)

log("\n── Val Set Statistics ────────────────────────────────────")
log(f"  Val graphs   : {n_val_graphs:,}")
log(f"  Val nodes    : {n_nodes:,}")
log(f"  Gold   (1)   : {n_gold:,}  ({n_gold / n_nodes:.2%})")
log(f"  Halluc (0)   : {n_hall:,}  ({n_hall / n_nodes:.2%})")

# ── Baseline: predict all 0 ───────────────────────────────────────────────────

pred_all0 = np.zeros(n_nodes, dtype=int)

log("\n── Baseline: Predict ALL 0 (hallucinated) ────────────────")
log(f"  F1        : {f1_score(labels, pred_all0, zero_division=0):.4f}")
log(f"  Precision : {precision_score(labels, pred_all0, zero_division=0):.4f}")
log(f"  Recall    : {recall_score(labels, pred_all0, zero_division=0):.4f}")

# ── Baseline: predict all 1 ───────────────────────────────────────────────────

pred_all1 = np.ones(n_nodes, dtype=int)

log("\n── Baseline: Predict ALL 1 (gold) ───────────────────────")
log(f"  F1        : {f1_score(labels, pred_all1, zero_division=0):.4f}")
log(f"  Precision : {precision_score(labels, pred_all1, zero_division=0):.4f}")
log(f"  Recall    : {recall_score(labels, pred_all1, zero_division=0):.4f}")

log("")

# ── Save ──────────────────────────────────────────────────────────────────────

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved → {OUT_PATH}")
