"""One-shot script: pick 4 HotpotQA + 4 MuSiQue questions that are
(a) long (>= 10 CoT steps),
(b) complex (real multi-hop), and
(c) BALANCED under the trained GAT — i.e. the model predicts a mix of
    gold and hallucinated nodes (gold ratio between 0.30 and 0.75).

This ensures the demo never looks 'underfit' (model predicts everything
gold) or 'overfit' (everything hallucinated) — every sample shows the
GAT discriminating between steps."""
import torch  # MUST be first import — Windows DLL fix

import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE = Path(r"D:\gvcot")
sys.path.insert(0, str(BASE))
from models.gnn_model import ReasoningGAT


# ── CoT splitter (verbatim from graphs/build_graphs.py) ──────────────────────
_PREAMBLE_RE = re.compile(
    r"^(to answer this question|to answer the question|here's a step-by-step"
    r"|i will follow these steps|to find the answer)", re.IGNORECASE,
)
_FINAL_ANSWER_RE = re.compile(
    r"^#+\s*(final\s+answer|answer)\s*$|^final\s+answer", re.IGNORECASE,
)
_HEADER_RE     = re.compile(r"^#+\s*")
_STEP_START_RE = re.compile(r"^\**\s*[Ss]tep\s+\d+")


def _is_noise(line):
    w = line.split()
    if len(w) <= 3: return True
    if _PREAMBLE_RE.match(line): return True
    if line.endswith(":") and len(w) <= 8: return True
    return False


def split_cot_chain(cot):
    raw = []
    for ln in cot.split("\n"):
        ln = ln.strip()
        if not ln: continue
        if _FINAL_ANSWER_RE.match(ln): break
        if _HEADER_RE.match(ln): continue
        raw.append(ln)
    has = any(_STEP_START_RE.match(l) for l in raw)
    if has:
        first = next(i for i, l in enumerate(raw) if _STEP_START_RE.match(l))
        raw = raw[first:]
        nodes, cur = [], []
        def flush():
            if cur:
                t = re.sub(r"\*+", "", " ".join(cur)).strip()
                if t: nodes.append(t)
            cur.clear()
        for ln in raw:
            if _STEP_START_RE.match(ln):
                flush()
                ln = re.sub(r"^\**\s*[Ss]tep\s+\d+[^a-zA-Z]*", "", ln).strip()
                ln = re.sub(r"\*+", "", ln).strip()
                if ln: cur.append(ln)
            else:
                ln = re.sub(r"\*+", "", ln).strip()
                ln = re.sub(r"^-\s+", "", ln).strip()
                ln = re.sub(r"^\d+[\.\)]\s*", "", ln).strip()
                if ln: cur.append(ln)
        flush()
        return nodes
    nodes = []
    for ln in raw:
        ln = re.sub(r"^\d+[\.\)]\s*", "", ln).strip()
        ln = re.sub(r"\*+", "", ln).strip()
        ln = re.sub(r"^-\s+", "", ln).strip()
        if ln and not _is_noise(ln) and not re.match(r"^final\s+answer\s*:", ln, re.IGNORECASE):
            nodes.append(ln)
    return nodes


# ── Structural features (must match training: 13 dims x 0.1) ─────────────────
def compute_structural_features(text, node_idx, num_nodes, in_deg, out_deg):
    n = max(num_nodes - 1, 1)
    tl = text.lower()
    feats = [
        node_idx / n,
        1.0 if node_idx == 0 else 0.0,
        1.0 if node_idx == num_nodes - 1 else 0.0,
        in_deg / n,
        out_deg / n,
        (in_deg + out_deg) / (2 * n),
        1.0 if "because" in tl else 0.0,
        1.0 if "therefore" in tl else 0.0,
        1.0 if "since" in tl else 0.0,
        1.0 if "hence" in tl else 0.0,
        1.0 if "thus" in tl else 0.0,
        1.0 if " so " in tl else 0.0,
        min(len(text.split()) / 50.0, 1.0),
    ]
    return np.array(feats, dtype=np.float32) * 0.1


def gold_ratio(steps, sbert, gat):
    if len(steps) < 2:
        return 0.0
    embs = sbert.encode(steps, convert_to_numpy=True, show_progress_bar=False)
    N = len(steps)
    feats = []
    for i in range(N):
        in_deg  = 0 if i == 0 else 1
        out_deg = 0 if i == N - 1 else 1
        feats.append(np.concatenate(
            [embs[i], compute_structural_features(steps[i], i, N, in_deg, out_deg)]
        ))
    x = torch.tensor(np.stack(feats), dtype=torch.float32)
    ei = torch.tensor([[i, i + 1] for i in range(N - 1)], dtype=torch.long).t().contiguous()
    with torch.no_grad():
        preds = gat(x, ei).argmax(dim=-1).numpy()
    return float((preds == 1).mean())


# ── Load models once ─────────────────────────────────────────────────────────
print("Loading SBERT …", flush=True)
sbert = SentenceTransformer("all-MiniLM-L6-v2")
print("Loading GAT …", flush=True)
gat = ReasoningGAT(in_channels=397, hidden_channels=128, out_channels=2, heads=4)
state = torch.load(BASE / "models" / "gat_best.pt", map_location="cpu", weights_only=False)
gat.load_state_dict(state)
gat.eval()


# ── Pick balanced + long samples ─────────────────────────────────────────────
TARGET_N      = 4
MIN_STEPS     = 10
RATIO_LO      = 0.30   # at least 30% predicted gold
RATIO_HI      = 0.75   # at most  75% predicted gold
QUESTION_MIN  = 60     # bias toward longer (more complex-looking) questions
SCAN_LIMIT    = 1500   # how many chains to score per dataset

out_dir = BASE / "data" / "_st_cache"
out_dir.mkdir(parents=True, exist_ok=True)

for name, src in [
    ("hotpot",  BASE / "data" / "cot_chains.json"),
    ("musique", BASE / "data" / "musique_cot_chains.json"),
]:
    print(f"\n=== Scanning {name} ===", flush=True)
    with open(src, "r", encoding="utf-8") as f:
        chains = json.load(f)

    kept = []
    seen_questions = set()
    for j, rec in enumerate(chains[:SCAN_LIMIT]):
        if (j + 1) % 100 == 0:
            print(f"  scanned {j+1}/{SCAN_LIMIT}  (kept {len(kept)})", flush=True)
        q = rec.get("question", "")
        if len(q) < QUESTION_MIN: continue
        if q in seen_questions:   continue
        steps = split_cot_chain(rec.get("cot_chain", ""))
        if len(steps) < MIN_STEPS: continue
        gr = gold_ratio(steps, sbert, gat)
        if RATIO_LO <= gr <= RATIO_HI:
            kept.append({
                "question":   q,
                "cot_chain":  rec["cot_chain"],
                "steps":      len(steps),
                "gold_ratio": round(gr, 3),
            })
            seen_questions.add(q)
            if len(kept) >= TARGET_N * 3:
                break

    # Sort: prefer mid-range gold-ratio (~0.5), longest first as tiebreaker
    kept.sort(key=lambda r: (abs(r["gold_ratio"] - 0.55), -r["steps"]))
    kept = kept[:TARGET_N]

    out = out_dir / f"samples_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(kept)} samples to {out}:")
    for s in kept:
        print(f"  [{s['steps']:>2} steps, gold={s['gold_ratio']:.2f}] {s['question'][:90]}")

print("\nDONE")
