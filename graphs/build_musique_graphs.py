import torch  # MUST be first import — Windows DLL fix

import json
import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\gvcot")
MUSIQUE_JSON= BASE_DIR / "data"   / "musique_train.json"
OUT_PKL     = BASE_DIR / "graphs" / "musique_graphs.pkl"
SIM_THRESH  = 0.60
SCALE       = 0.1     # same structural feature scaling as Run 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[MuSiQue Build] Device: {DEVICE}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("[MuSiQue Build] Loading musique_train.json ...")
with open(MUSIQUE_JSON, encoding="utf-8") as f:
    data = json.load(f)

answerable = [r for r in data if r["answerable"]]
print(f"[MuSiQue Build] Answerable questions: {len(answerable):,} / {len(data):,}")

# ── Load SBERT ────────────────────────────────────────────────────────────────
print("[MuSiQue Build] Loading SBERT model ...")
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
print("[MuSiQue Build] SBERT loaded OK")

# ── Structural feature helper ─────────────────────────────────────────────────
def structural_features(text, node_idx, num_nodes, in_deg, out_deg):
    n  = max(num_nodes - 1, 1)
    tl = text.lower()
    feats = [
        node_idx / n,                                      # 0  position
        1.0 if node_idx == 0 else 0.0,                    # 1  is_first
        1.0 if node_idx == num_nodes - 1 else 0.0,        # 2  is_last
        in_deg  / n,                                       # 3  in_degree
        out_deg / n,                                       # 4  out_degree
        (in_deg + out_deg) / (2 * n),                     # 5  total_degree
        1.0 if "because"   in tl else 0.0,                # 6
        1.0 if "therefore" in tl else 0.0,                # 7
        1.0 if "since"     in tl else 0.0,                # 8
        1.0 if "hence"     in tl else 0.0,                # 9
        1.0 if "thus"      in tl else 0.0,                # 10
        1.0 if " so "      in tl else 0.0,                # 11
        min(len(text.split()) / 50.0, 1.0),               # 12 norm length
    ]
    return np.array(feats, dtype=np.float32) * SCALE

# ── Build graphs ──────────────────────────────────────────────────────────────
print("[MuSiQue Build] Building graphs ...")

graphs    = []
skipped   = 0
gold_total= 0
dist_total= 0

BATCH_SIZE = 500   # encode paragraphs in batches for speed

for qi, record in enumerate(answerable):
    if (qi + 1) % 2000 == 0:
        print(f"  ... {qi+1:,}/{len(answerable):,} questions processed")

    paragraphs = record["paragraphs"]
    num_nodes  = len(paragraphs)

    if num_nodes < 2:
        skipped += 1
        continue

    # Node texts: title + paragraph text
    texts  = [f"{p['title']}. {p['paragraph_text']}" for p in paragraphs]
    labels = [1 if p["is_supporting"] else 0 for p in paragraphs]

    # SBERT encode
    embeddings = sbert.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)   # (num_nodes, 384)

    # Build graph
    G = nx.DiGraph()
    G.graph["id"]       = record["id"]
    G.graph["question"] = record["question"]
    G.graph["answer"]   = record["answer"]

    # Add nodes (temporary — structural features added after edges)
    for i in range(num_nodes):
        G.add_node(i, text=texts[i], label=labels[i], embedding=embeddings[i])

    # Add edges: cosine similarity >= SIM_THRESH
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            sim = float(np.dot(embeddings[i], embeddings[j]))  # normalized -> dot = cosine
            if sim >= SIM_THRESH:
                G.add_edge(i, j)
                G.add_edge(j, i)

    # Update structural features now that edges exist
    for i in range(num_nodes):
        in_d  = G.in_degree(i)
        out_d = G.out_degree(i)
        sf    = structural_features(texts[i], i, num_nodes, in_d, out_d)
        aug   = np.concatenate([embeddings[i], sf]).astype(np.float32)  # (397,)
        G.nodes[i]["embedding"] = aug

    gold_total += sum(labels)
    dist_total += sum(1 - l for l in labels)
    graphs.append(G)

print(f"\n[MuSiQue Build] Graphs built      : {len(graphs):,}")
print(f"[MuSiQue Build] Skipped (<2 nodes): {skipped}")
print(f"[MuSiQue Build] Gold nodes        : {gold_total:,}")
print(f"[MuSiQue Build] Distractor nodes  : {dist_total:,}")
print(f"[MuSiQue Build] Gold ratio        : {gold_total/(gold_total+dist_total):.4f}")
print(f"[MuSiQue Build] Embedding dim     : {list(graphs[0].nodes(data=True))[0][1]['embedding'].shape[0]}")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\n[MuSiQue Build] Saving -> {OUT_PKL} ...")
with open(OUT_PKL, "wb") as f:
    pickle.dump(graphs, f)

size_mb = OUT_PKL.stat().st_size / 1e6
print(f"[MuSiQue Build] Saved. File size: {size_mb:.1f} MB")
print("[MuSiQue Build] COMPLETE")
