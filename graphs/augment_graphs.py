import torch  # MUST be first import — Windows DLL fix

import os
import pickle
import numpy as np

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PKL   = os.path.join(ROOT, "graphs", "reasoning_graphs.pkl")
OUTPUT_PKL  = os.path.join(ROOT, "graphs", "reasoning_graphs_augmented.pkl")

CONNECTIVES = ["because", "therefore", "since", "hence", "thus"]

print(f"Loading {INPUT_PKL} ...")
with open(INPUT_PKL, "rb") as f:
    graphs = pickle.load(f)
print(f"Loaded {len(graphs):,} graphs.")

for graph_num, g in enumerate(graphs):
    if graph_num % 10_000 == 0:
        print(f"  Processing graph {graph_num:,} / {len(graphs):,} ...")

    nodes    = list(g.nodes())
    n        = len(nodes)
    denom    = max(n - 1, 1)   # avoid div-by-zero for single-node graphs

    for node_idx, node in enumerate(nodes):
        orig_emb = g.nodes[node]["embedding"]   # (384,) float32
        text     = g.nodes[node]["text"]
        text_lo  = text.lower()

        # ── 13 structural features ────────────────────────────────────────────
        position      = 0.5 if n == 1 else node_idx / denom
        is_first      = 1.0 if node_idx == 0         else 0.0
        is_last       = 1.0 if node_idx == (n - 1)   else 0.0
        in_deg        = g.in_degree(node)  / denom
        out_deg       = g.out_degree(node) / denom
        total_deg     = (g.in_degree(node) + g.out_degree(node)) / (2 * denom)

        c_because     = 1.0 if "because"   in text_lo else 0.0
        c_therefore   = 1.0 if "therefore" in text_lo else 0.0
        c_since       = 1.0 if "since"     in text_lo else 0.0
        c_hence       = 1.0 if "hence"     in text_lo else 0.0
        c_thus        = 1.0 if "thus"      in text_lo else 0.0
        c_so          = 1.0 if " so "      in text_lo else 0.0
        norm_length   = min(len(text.split()) / 50.0, 1.0)

        structural = np.array([
            position,
            is_first,
            is_last,
            in_deg,
            out_deg,
            total_deg,
            c_because,
            c_therefore,
            c_since,
            c_hence,
            c_thus,
            c_so,
            norm_length,
        ], dtype=np.float32) * 0.1  # scale to match SBERT magnitude (~std 0.05)

        g.nodes[node]["embedding"] = np.concatenate([orig_emb, structural]).astype(np.float32)

print(f"All {len(graphs):,} graphs augmented.")

# Verify dimensionality on the first graph's first node
sample_g    = graphs[0]
sample_node = list(sample_g.nodes())[0]
dim         = len(sample_g.nodes[sample_node]["embedding"])
print(f"Embedding dim check: {dim} (expected 397)")
assert dim == 397, f"Expected 397, got {dim}"

print(f"\nSaving to {OUTPUT_PKL} ...")
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(graphs, f)

size_mb = os.path.getsize(OUTPUT_PKL) / (1024 ** 2)
print(f"Saved. File size: {size_mb:.1f} MB")
print("Done.")
