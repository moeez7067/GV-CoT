import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # must be before sentence_transformers to fix Windows DLL load order

import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH    = os.path.join(ROOT, "graphs", "reasoning_graphs.pkl")
HOTPOT_PATH = os.path.join(ROOT, "data", "hotpotqa_train.json")

MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def cosine_sim(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# --- Load graphs ---
print(f"Loading {PKL_PATH} ...")
with open(PKL_PATH, "rb") as f:
    graphs = pickle.load(f)
print(f"  {len(graphs):,} graphs loaded.")

# --- Build gold_lookup from HotpotQA ---
print(f"Loading {HOTPOT_PATH} ...")
with open(HOTPOT_PATH, encoding="utf-8") as f:
    hotpot_data = json.load(f)

gold_lookup = {}
for sample in hotpot_data:
    sf_pairs = set(zip(
        sample["supporting_facts"]["title"],
        sample["supporting_facts"]["sent_id"],
    ))
    texts = []
    for title, sents in zip(sample["context"]["title"], sample["context"]["sentences"]):
        for sid, sent_text in enumerate(sents):
            if (title, sid) in sf_pairs:
                t = sent_text.strip()
                if t:
                    texts.append(t)
    gold_lookup[sample["id"]] = texts

# --- Build flat gold text list aligned to graphs ---
all_gold_texts = []
gold_ranges    = []
for g in graphs:
    gts   = gold_lookup.get(g.graph["id"], [])
    start = len(all_gold_texts)
    all_gold_texts.extend(gts)
    gold_ranges.append((start, len(all_gold_texts)))

all_answers = [g.graph["answer"] for g in graphs]

# --- Encode ---
print("Loading SBERT model ...")
model = SentenceTransformer(MODEL_NAME, device="cuda")

print(f"Encoding {len(all_gold_texts):,} gold fact sentences ...")
gold_embs_flat = model.encode(
    all_gold_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True,
) if all_gold_texts else np.zeros((0, 384), dtype="float32")

print(f"Encoding {len(all_answers):,} answer strings ...")
answer_embs = model.encode(
    all_answers, batch_size=256, show_progress_bar=True, convert_to_numpy=True,
)

# --- Precompute per-node (max_gold_sim, ans_sim) in one pass ---
print("Computing per-node similarities ...")
node_sims = []  # [(max_gold_sim, ans_sim), ...]

for i, g in enumerate(graphs):
    gs, ge    = gold_ranges[i]
    gold_embs = gold_embs_flat[gs:ge] if ge > gs else None
    ans_emb   = answer_embs[i]

    for _, attrs in g.nodes(data=True):
        emb = attrs["embedding"]

        if gold_embs is not None and len(gold_embs) > 0:
            max_gold = max(cosine_sim(emb, gf) for gf in gold_embs)
        else:
            max_gold = 0.0

        node_sims.append((max_gold, cosine_sim(emb, ans_emb)))

total = len(node_sims)
print(f"  {total:,} nodes processed.\n")

# --- Threshold table ---
print(f"{'Threshold':>10} | {'Gold':>10} | {'Hallucinated':>14} | {'Gold%':>8}")
print("-" * 52)

for thresh in THRESHOLDS:
    n_gold = sum(1 for mg, ans in node_sims if mg >= thresh or ans >= thresh)
    n_hall = total - n_gold
    pct    = n_gold / total * 100
    print(f"{thresh:>10.2f} | {n_gold:>10,} | {n_hall:>14,} | {pct:>7.2f}%")
