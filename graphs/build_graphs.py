import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import json
import os
import pickle
import re
import sys

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COT_PATH      = os.path.join(ROOT, "data", "cot_chains.json")
HOTPOT_PATH   = os.path.join(ROOT, "data", "hotpotqa_train.json")
OUTPUT_PATH   = os.path.join(ROOT, "graphs", "reasoning_graphs.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"
SIM_THRESH = 0.6


_PREAMBLE_RE = re.compile(
    r"^(to answer this question"
    r"|to answer the question"
    r"|here's a step-by-step"
    r"|i will follow these steps"
    r"|to find the answer)",
    re.IGNORECASE,
)

_FINAL_ANSWER_RE = re.compile(
    r"^#+\s*(final\s+answer|answer)\s*$"
    r"|^final\s+answer",
    re.IGNORECASE,
)
_HEADER_RE     = re.compile(r"^#+\s*")
_STEP_START_RE = re.compile(r"^\**\s*[Ss]tep\s+\d+")


def _is_noise(line: str) -> bool:
    words = line.split()
    if len(words) <= 3:
        return True
    if _PREAMBLE_RE.match(line):
        return True
    if line.endswith(":") and len(words) <= 8:
        return True
    return False


def split_cot_chain(cot_chain: str) -> list:
    # Pass 1: raw line cleanup — drop empties, headers, stop at Final Answer
    raw_lines = []
    for line in cot_chain.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _FINAL_ANSWER_RE.match(line):
            break
        if _HEADER_RE.match(line):
            continue
        raw_lines.append(line)

    has_steps = any(_STEP_START_RE.match(l) for l in raw_lines)

    if has_steps:
        # Drop preamble lines that appear before the first Step marker
        first_step = next(i for i, l in enumerate(raw_lines) if _STEP_START_RE.match(l))
        raw_lines = raw_lines[first_step:]

        nodes = []
        current_parts = []

        def flush():
            if current_parts:
                text = re.sub(r'\*+', '', " ".join(current_parts)).strip()
                if text:
                    nodes.append(text)
            current_parts.clear()

        for line in raw_lines:
            if _STEP_START_RE.match(line):
                flush()
                line = re.sub(r'^\**\s*[Ss]tep\s+\d+[^a-zA-Z]*', '', line).strip()
                line = re.sub(r'\*+', '', line).strip()
                if line:
                    current_parts.append(line)
            else:
                line = re.sub(r'\*+', '', line).strip()
                line = re.sub(r'^-\s+', '', line).strip()
                line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if line:
                    current_parts.append(line)

        flush()
        return nodes

    else:
        # Fallback (chains without Step markers): one cleaned line = one node
        nodes = []
        for line in raw_lines:
            line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            line = re.sub(r'\*+', '', line).strip()
            line = re.sub(r'^-\s+', '', line).strip()
            if line and not _is_noise(line) and not re.match(r'^final\s+answer\s*:', line, re.IGNORECASE):
                nodes.append(line)
        return nodes


def cosine_sim(a, b) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def label_sentence(embedding, gold_embeddings, answer_embedding) -> int:
    """Return 1 if embedding is similar to any gold fact or the answer above SIM_THRESH."""
    if gold_embeddings is not None and len(gold_embeddings) > 0:
        if max(cosine_sim(embedding, g) for g in gold_embeddings) >= SIM_THRESH:
            return 1
    if answer_embedding is not None and cosine_sim(embedding, answer_embedding) >= SIM_THRESH:
        return 1
    return 0


def build_graph(record: dict, gold_embeddings, answer_embedding, sentences: list, node_embs) -> nx.DiGraph:
    """Build a directed graph over CoT chain sentences.

    Each node is one CoT step with:
      text      — sentence text
      embedding — SBERT 384-dim vector
      label     — 1 if cosine similarity to a gold fact or answer >= SIM_THRESH, else 0

    Edges are directed sequential links between consecutive CoT steps.
    """
    g = nx.DiGraph()
    g.graph["id"]       = record["id"]
    g.graph["question"] = record["question"]
    g.graph["answer"]   = record["answer"]

    for idx, (text, emb) in enumerate(zip(sentences, node_embs)):
        g.add_node(idx, text=text, embedding=emb, label=label_sentence(emb, gold_embeddings, answer_embedding))

    for i in range(len(sentences) - 1):
        g.add_edge(i, i + 1)

    return g


def main():
    # --- Load CoT chains (primary input) ---
    if not os.path.exists(COT_PATH):
        print(f"ERROR: Not found: {COT_PATH}")
        sys.exit(1)
    print(f"Loading {COT_PATH} ...")
    with open(COT_PATH, encoding="utf-8") as f:
        cot_data = json.load(f)
    print(f"Loaded {len(cot_data)} CoT chains.")

    # --- Load HotpotQA for gold supporting facts only ---
    if not os.path.exists(HOTPOT_PATH):
        print(f"ERROR: Not found: {HOTPOT_PATH}")
        sys.exit(1)
    print(f"Loading {HOTPOT_PATH} for supporting facts ...")
    with open(HOTPOT_PATH, encoding="utf-8") as f:
        hotpot_data = json.load(f)

    # Build lookup: {sample_id: [gold sentence texts]}
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
    print(f"Built gold lookup for {len(gold_lookup)} samples.")

    # --- Split CoT chains and build flat sentence list for batch embedding ---
    all_texts      = []
    sample_sents   = []
    sample_ranges  = []

    for record in cot_data:
        sents = split_cot_chain(record.get("cot_chain", ""))
        if not sents:
            sents = ["(empty)"]
        sample_sents.append(sents)
        start = len(all_texts)
        all_texts.extend(sents)
        sample_ranges.append((start, len(all_texts)))

    # --- Encode all sentences in one flat SBERT pass ---
    print(f"Encoding {len(all_texts)} CoT sentences ...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    all_embeddings = model.encode(
        all_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # --- Encode gold facts in one flat SBERT pass ---
    all_gold_texts = []
    gold_ranges    = []
    for record in cot_data:
        gts   = gold_lookup.get(record["id"], [])
        start = len(all_gold_texts)
        all_gold_texts.extend(gts)
        gold_ranges.append((start, len(all_gold_texts)))

    print(f"Encoding {len(all_gold_texts)} gold fact sentences ...")
    gold_embeddings_flat = model.encode(
        all_gold_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    ) if all_gold_texts else np.zeros((0, 384), dtype="float32")

    # --- Encode answer strings ---
    all_answers = [record.get("answer", "") for record in cot_data]
    print(f"Encoding {len(all_answers)} answer strings ...")
    answer_embeddings = model.encode(
        all_answers,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # --- Build graphs ---
    print("Building graphs ...")
    graphs = []
    for i, (record, sents, (start, end)) in enumerate(zip(cot_data, sample_sents, sample_ranges)):
        gs, ge    = gold_ranges[i]
        gold_embs = gold_embeddings_flat[gs:ge] if ge > gs else None
        ans_emb   = answer_embeddings[i]
        graphs.append(build_graph(record, gold_embs, ans_emb, sents, all_embeddings[start:end]))

    n_nodes_total = sum(g.number_of_nodes() for g in graphs)
    n_gold_total  = sum(
        sum(1 for _, d in g.nodes(data=True) if d["label"] == 1)
        for g in graphs
    )
    print(f"Built {len(graphs)} graphs — {n_nodes_total} total nodes, "
          f"{n_gold_total} gold ({n_gold_total / n_nodes_total:.2%})")

    # --- Save ---
    print(f"Saving to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved.")

    # Sanity check on first graph
    g0     = graphs[0]
    n_gold = sum(1 for _, d in g0.nodes(data=True) if d["label"] == 1)
    n_dist = g0.number_of_nodes() - n_gold
    print(f"\n--- First graph ---")
    print(f"  ID      : {g0.graph['id']}")
    print(f"  Question: {g0.graph['question']}")
    print(f"  Nodes   : {g0.number_of_nodes()} total  ({n_gold} gold, {n_dist} distractor)")
    print(f"  Edges   : {g0.number_of_edges()}")
    for nid, attrs in g0.nodes(data=True):
        tag = "GOLD" if attrs["label"] == 1 else "DIST"
        print(f"    [{nid:>2} {tag}] {attrs['text'][:72]}")


if __name__ == "__main__":
    main()
