import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # must be before sentence_transformers to fix Windows DLL load order

import pickle

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH = os.path.join(ROOT, "graphs", "reasoning_graphs.pkl")

print(f"Loading {PKL_PATH} ...")
with open(PKL_PATH, "rb") as f:
    graphs = pickle.load(f)

n_before  = len(graphs)
filtered  = [g for g in graphs if g.number_of_edges() > 0]
n_removed = n_before - len(filtered)

print(f"  Total graphs before : {n_before:,}")
print(f"  Graphs removed      : {n_removed:,}  (0-edge / single-node)")
print(f"  Total graphs after  : {len(filtered):,}")

print(f"\nOverwriting {PKL_PATH} ...")
with open(PKL_PATH, "wb") as f:
    pickle.dump(filtered, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Done.")
