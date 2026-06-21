import torch  # MUST be first import — Windows DLL fix

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import sys
import json
import time
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

# ── Path / project setup ──────────────────────────────────────────────────────
BASE_DIR        = Path(r"D:\gvcot")
MODEL_PT        = BASE_DIR / "models" / "gat_best.pt"
HOTPOT_CHAINS   = BASE_DIR / "data"   / "cot_chains.json"
MUSIQUE_CHAINS  = BASE_DIR / "data"   / "musique_cot_chains.json"
SAMPLES_HOTPOT  = BASE_DIR / "data"   / "_st_cache" / "samples_hotpot.json"
SAMPLES_MUSIQUE = BASE_DIR / "data"   / "_st_cache" / "samples_musique.json"

sys.path.insert(0, str(BASE_DIR))


# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="GV-CoT — Reasoning Verifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Heavy imports deferred until after set_page_config so the page chrome
# renders immediately while the first cache call warms in the background.
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pyvis.network import Network


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    /* Metric cards */
    .gv-metric {
        border: 1px solid #2E3540;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        background-color: #161B22;
    }
    .gv-metric-label {
        font-size: 12px;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .gv-metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #00C851;
    }
    /* Disclaimer box */
    .gv-disclaimer {
        border: 1px solid #FFA500;
        background-color: rgba(255, 165, 0, 0.08);
        border-radius: 8px;
        padding: 12px 14px;
        margin: 12px 0;
        font-size: 13px;
        color: #FFD89A;
    }
    .gv-disclaimer b { color: #FFA500; }

    /* Custom-mode warning box */
    .gv-custom-warn {
        border: 1.5px solid #FF4444;
        background-color: rgba(255, 68, 68, 0.08);
        border-radius: 8px;
        padding: 14px 16px;
        margin: 12px 0;
        color: #FFBABA;
    }
    .gv-custom-warn b { color: #FF4444; }

    /* Result-summary card */
    .gv-result {
        border-radius: 8px;
        padding: 14px 16px;
        background-color: #161B22;
        border: 1px solid #2E3540;
        margin-bottom: 10px;
    }
    .gv-result .v { font-size: 28px; font-weight: 700; }
    .gv-result .l { font-size: 12px; color: #8B949E; text-transform: uppercase; }
    .gv-green  { color: #00C851; }
    .gv-red    { color: #FF4444; }
    .gv-blue   { color: #2E75B6; }

    /* Gold-ratio gauge */
    .gv-gauge-wrap {
        border: 1px solid #2E3540;
        background: #161B22;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 10px 0 18px 0;
    }
    .gv-gauge-head {
        display: flex; justify-content: space-between; align-items: baseline;
        margin-bottom: 8px;
    }
    .gv-gauge-head .lbl {
        font-size: 12px; color: #8B949E;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .gv-gauge-head .val {
        font-size: 22px; font-weight: 700; color: #00C851;
    }
    .gv-gauge-bar {
        position: relative; width: 100%; height: 18px;
        background: #21262D; border-radius: 9px; overflow: hidden;
        border: 1px solid #2E3540;
    }
    .gv-gauge-fill {
        height: 100%;
        background: linear-gradient(90deg, #FF4444 0%, #FFA500 50%, #00C851 100%);
        border-radius: 9px;
    }
    .gv-gauge-foot {
        display: flex; justify-content: space-between;
        font-size: 11px; color: #8B949E; margin-top: 4px;
    }
    .gv-gauge-note {
        margin-top: 10px; font-size: 13px; color: #C9D1D9;
    }
    .gv-gauge-note b { color: #2E75B6; }

    /* Spotlight callouts */
    .gv-spotlight {
        border: 1.5px solid #FF4444;
        background-color: rgba(255, 68, 68, 0.06);
        border-radius: 8px;
        padding: 14px 16px;
        margin: 14px 0 18px 0;
    }
    .gv-spotlight h4 { color: #FF4444; margin: 0 0 6px 0; font-size: 15px; }
    .gv-spotlight .item {
        background: #161B22;
        border-left: 3px solid #FF4444;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 13px;
        color: #C9D1D9;
    }
    .gv-spotlight .item b { color: #FF4444; }
    .gv-spotlight .clean {
        color: #00C851;
        font-size: 14px;
        padding: 6px 0;
    }
    /* Citation box */
    .gv-cite {
        border-left: 4px solid #2E75B6;
        background-color: #161B22;
        padding: 12px 14px;
        font-style: italic;
        font-size: 13px;
        color: #C9D1D9;
        border-radius: 4px;
    }

    /* Pipeline step heading */
    .gv-step {
        font-size: 15px;
        font-weight: 600;
        margin: 14px 0 6px 0;
        color: #58A6FF;
    }

    /* Tables */
    .gv-table { width: 100%; border-collapse: collapse; }
    .gv-table th, .gv-table td {
        border: 1px solid #2E3540;
        padding: 8px 10px;
        text-align: left;
        font-size: 13px;
    }
    .gv-table th {
        background-color: #161B22;
        color: #58A6FF;
        font-weight: 600;
    }
    .gv-table tr.star { background-color: rgba(0, 200, 81, 0.08); }
</style>
""", unsafe_allow_html=True)


# ── CoT splitter (copied verbatim from graphs/build_graphs.py) ────────────────
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
        nodes = []
        for line in raw_lines:
            line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            line = re.sub(r'\*+', '', line).strip()
            line = re.sub(r'^-\s+', '', line).strip()
            if line and not _is_noise(line) and not re.match(r'^final\s+answer\s*:', line, re.IGNORECASE):
                nodes.append(line)
        return nodes


# ── Structural features (13-dim, scaled ×0.1 — Run 3 recipe) ──────────────────
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
        1.0 if "because"  in tl else 0.0,
        1.0 if "therefore" in tl else 0.0,
        1.0 if "since"    in tl else 0.0,
        1.0 if "hence"    in tl else 0.0,
        1.0 if "thus"     in tl else 0.0,
        1.0 if " so "     in tl else 0.0,
        min(len(text.split()) / 50.0, 1.0),
    ]
    return np.array(feats, dtype=np.float32) * 0.1


# ── Pearson-r answer-accuracy buckets (from the cross-eval table) ────────────
_GOLD_RATIO_BUCKETS = [
    (0.0, 0.2, 0.155, "0.0 – 0.2"),
    (0.2, 0.4, 0.185, "0.2 – 0.4"),
    (0.4, 0.6, 0.267, "0.4 – 0.6"),
    (0.6, 0.8, 0.345, "0.6 – 0.8"),
    (0.8, 1.0 + 1e-9, 0.417, "0.8 – 1.0"),
]


def expected_accuracy(gold_ratio: float):
    """Map predicted gold-ratio to the empirical answer-accuracy bucket
    (Pearson r = +0.195 — higher ratio predicts more correct answers)."""
    for lo, hi, acc, label in _GOLD_RATIO_BUCKETS:
        if lo <= gold_ratio < hi:
            return acc, label
    return _GOLD_RATIO_BUCKETS[-1][2], _GOLD_RATIO_BUCKETS[-1][3]


# ── Cached resources (all lazy — only fire on demand) ────────────────────────
@st.cache_resource(show_spinner="Loading SBERT encoder …")
def load_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Loading trained GAT …")
def load_gat():
    from models.gnn_model import ReasoningGAT
    model = ReasoningGAT(in_channels=397, hidden_channels=128, out_channels=2, heads=4)
    state = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data(show_spinner="Loading sample questions …")
def get_sample_questions(dataset: str):
    """Reads a tiny precomputed JSON (~4 questions). Never touches the
    145 MB / 38 MB chain files."""
    src = SAMPLES_HOTPOT if dataset == "hotpot" else SAMPLES_MUSIQUE
    if not src.exists():
        return []
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading HotpotQA chains for search …")
def load_hotpot_chains():
    """Only called when the user actually types in the search box."""
    with open(HOTPOT_CHAINS, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading MuSiQue chains for search …")
def load_musique_chains():
    """Only called when the user actually types in the search box."""
    with open(MUSIQUE_CHAINS, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Pyvis helper ──────────────────────────────────────────────────────────────
def build_net(steps, node_colors=None, node_sizes=None, node_labels=None,
              edges_built=None, n_visible=None):
    n_visible = len(steps) if n_visible is None else n_visible
    net = Network(
        height="500px", width="100%",
        bgcolor="#0E1117", font_color="white",
        directed=True,
    )
    net.set_options("""
    {
        "physics": {"enabled": false},
        "interaction": {"hover": true, "zoomView": true, "dragView": true},
        "nodes": {"borderWidth": 2, "borderWidthSelected": 3,
                  "shape": "dot", "shadow": true}
    }
    """)
    for i in range(n_visible):
        color = (node_colors[i] if node_colors else "#888888")
        size  = (node_sizes[i]  if node_sizes  else 20)
        label = (node_labels[i] if node_labels else f"Step {i+1}")
        # Layout: linear horizontal arrangement
        x = i * 180 - (n_visible - 1) * 90
        y = 0
        net.add_node(
            i,
            label=label,
            title=steps[i][:300] + ("…" if len(steps[i]) > 300 else ""),
            color=color,
            size=size,
            x=x, y=y,
            physics=False,
            font={"size": 14, "color": "white"},
        )
    edge_limit = (edges_built if edges_built is not None else n_visible - 1)
    for i in range(edge_limit):
        if i + 1 < n_visible:
            net.add_edge(i, i + 1, arrows="to", color="#555555")
    return net


def draw_chain_mpl(steps, colors, sizes_norm, edges_built):
    """Fast matplotlib renderer for the live pipeline.
    Renders in ~10 ms per frame vs ~200 ms for a PyVis iframe replacement."""
    N = len(steps)
    width = max(8.5, min(18, 1.4 * N))
    fig, ax = plt.subplots(figsize=(width, 2.4), facecolor="#0E1117")
    ax.set_facecolor("#0E1117")

    xs = list(range(N))
    ys = [0] * N

    for i in range(edges_built):
        if i + 1 < N:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.20, ys[i + 1]),
                xytext=(xs[i] + 0.20, ys[i]),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=2, mutation_scale=18),
            )

    for i in range(N):
        c = colors[i] if i < len(colors) else "#888888"
        size = 600 + (sizes_norm[i] if i < len(sizes_norm) else 0.0) * 1100
        ax.scatter(xs[i], ys[i], s=size, c=c, edgecolors="white",
                   linewidths=2.0, zorder=5)
        ax.text(xs[i], ys[i], str(i + 1), ha="center", va="center",
                color="white", fontsize=11, fontweight="bold", zorder=6)

    ax.set_xlim(-0.7, N - 0.3)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


# ── Helpers for the 3D Neural-Universe graph (Three.js scene) ────────────────
def _find_clusters(preds: np.ndarray, min_run: int = 3):
    """Return list of (start, end_inclusive) ranges of consecutive
    hallucinated nodes of length ≥ min_run."""
    clusters, run = [], 0
    for i, p in enumerate(preds):
        if p == 0:
            run += 1
        else:
            if run >= min_run:
                clusters.append((i - run, i - 1))
            run = 0
    if run >= min_run:
        clusters.append((len(preds) - run, len(preds) - 1))
    return clusters


def _longest_gold_streak(preds: np.ndarray) -> int:
    best = cur = 0
    for p in preds:
        if p == 1:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


_LABEL_STOP = {
    "a", "an", "the", "of", "to", "for", "in", "on", "at", "and", "but", "or",
    "with", "by", "as", "is", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "i", "we", "they",
    "so", "if", "then", "thus", "hence", "therefore",
}


def _step_label(text: str) -> str:
    """Derive a short 1-word label from a CoT step (first meaningful token)."""
    if not text:
        return ""
    for raw in text.replace("\n", " ").split():
        w = raw.strip(":.,;!?\"'()[]{}*")
        if not w:
            continue
        if w.lower() in _LABEL_STOP:
            continue
        return w[:16]
    first = text.strip().split(" ", 1)[0]
    return first[:16] or "Step"


GRAPH_TEMPLATE_PATH = BASE_DIR / "graph_template.html"

# Offline Three.js bundling lives in three_bundle.py (import-safe, unit-testable).
from three_bundle import inline_three_offline   # noqa: E402


def render_3d_graph(steps, preds, confidences, saliency, embeddings,
                    sim_thresh: float = 0.35, max_sim_edges: int = 20):
    """Build the cinematic Three.js Neural-Universe scene.
    Returns (html_string, stats) — caller renders with components.html()
    and shows the stats card below it."""
    N = len(steps)
    preds_np       = np.asarray(preds)
    confidences_np = np.asarray(confidences, dtype=float)
    saliency_np    = np.asarray(saliency, dtype=float)
    emb            = np.asarray(embeddings, dtype=np.float32)

    # Cosine similarity matrix
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb_n = emb / norms
    sim   = emb_n @ emb_n.T
    np.fill_diagonal(sim, 0.0)

    # Semantic edges — top-K by similarity, skipping immediate neighbours
    candidates = []
    for i in range(N):
        for j in range(i + 2, N):
            if sim[i, j] > sim_thresh:
                candidates.append((float(sim[i, j]), i, j))
    candidates.sort(reverse=True)
    semantic_edges = [
        {"i": int(i), "j": int(j), "w": round(float(w), 3)}
        for (w, i, j) in candidates[:max_sim_edges]
    ]

    sequential_edges = [[i, i + 1] for i in range(N - 1)]
    labels = [_step_label(s) or f"Step {i+1}" for i, s in enumerate(steps)]

    data = {
        "steps":       list(steps),
        "labels":      labels,
        "preds":       [int(p) for p in preds_np],
        "confidences": [float(c) for c in confidences_np],
        "saliency":    [float(s) for s in saliency_np],
        "sequential":  sequential_edges,
        "semantic":    semantic_edges,
    }

    if not GRAPH_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"graph_template.html not found at {GRAPH_TEMPLATE_PATH}. "
            "Make sure it lives next to app.py."
        )
    template  = GRAPH_TEMPLATE_PATH.read_text(encoding="utf-8")
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html_str  = template.replace("__DATA_JSON__", data_json)
    html_str  = inline_three_offline(html_str)   # vendor Three.js locally (no CDN)

    clusters = _find_clusters(preds_np, min_run=3)
    stats = {
        "nodes":      N,
        "seq_edges":  N - 1,
        "sim_edges":  len(semantic_edges),
        "avg_sim":    float(np.mean(sim[np.triu_indices(N, k=1)])) if N > 1 else 0.0,
        "density":    float(((N - 1) + len(semantic_edges)) / max(1, N * (N - 1) / 2)),
        "longest_gold_streak": _longest_gold_streak(preds_np),
        "clusters":   clusters,
    }
    return html_str, stats


def render_net(net, height=520):
    try:
        html = net.generate_html(notebook=False)
    except Exception:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as tf:
            net.write_html(tf.name, notebook=False, open_browser=False)
            path = tf.name
        with open(path, "r", encoding="utf-8") as rf:
            html = rf.read()
        os.unlink(path)
    components.html(html, height=height, scrolling=False)


# ── Inference pipeline (returns predictions + confidence + saliency) ──────────
def run_inference(steps, sbert, gat):
    N = len(steps)
    embeddings = sbert.encode(steps, convert_to_numpy=True)  # (N, 384)

    feats = []
    for i in range(N):
        in_deg  = 0 if i == 0 else 1
        out_deg = 0 if i == N - 1 else 1
        s = compute_structural_features(steps[i], i, N, in_deg, out_deg)
        feats.append(np.concatenate([embeddings[i], s]))   # (397,)
    x = torch.tensor(np.stack(feats), dtype=torch.float32)
    edge_index = torch.tensor(
        [[i, i + 1] for i in range(N - 1)], dtype=torch.long
    ).t().contiguous() if N > 1 else torch.zeros((2, 0), dtype=torch.long)

    # Forward (predictions + confidence)
    with torch.no_grad():
        logits = gat(x, edge_index)              # log-softmax outputs
        probs  = logits.exp().numpy()            # (N, 2)
        preds  = probs.argmax(axis=1)            # 1 = gold, 0 = hallucinated
        conf   = probs.max(axis=1)               # per-node confidence

    # Gradient saliency (input × |grad|)
    x_inp  = x.detach().requires_grad_(True)
    out    = gat(x_inp, edge_index)
    out[:, 0].sum().backward()
    saliency = (x_inp.grad.abs() * x_inp.abs()).sum(dim=1).detach().numpy()
    saliency = saliency / (saliency.max() + 1e-9)

    return preds, conf, saliency, embeddings


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔬 GV-CoT")
    st.markdown(
        "<div style='color:#8B949E; font-size:13px; margin-top:-12px;'>"
        "Graph-Based Verification of Chain-of-Thought Reasoning</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#C9D1D9; font-size:12px;'>"
        "Moeez &nbsp;|&nbsp; F2023332094 &nbsp;|&nbsp; UMT &nbsp;|&nbsp; 2026</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        "<div class='gv-metric'>"
        "<div class='gv-metric-label'>🎯 Best F1</div>"
        "<div class='gv-metric-value'>0.6473</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='gv-metric'>"
        "<div class='gv-metric-label'>🔍 Explainability</div>"
        "<div class='gv-metric-value'>82.9%</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='gv-metric'>"
        "<div class='gv-metric-label'>📈 Pearson r</div>"
        "<div class='gv-metric-value' style='color:#2E75B6;'>+0.1951</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### About")
    st.markdown(
        "GV-CoT is a post-hoc verification framework that converts LLM "
        "reasoning chains into graphs and uses a Graph Attention Network "
        "to detect hallucinated steps — without external knowledge bases "
        "or multiple LLM calls."
    )

    st.markdown("### Datasets")
    st.markdown(
        "- 📚 **HotpotQA**: 89,625 chains\n"
        "- 📚 **MuSiQue**: 19,937 chains\n"
        "- 🤖 **3 LLMs**: Claude Haiku · Llama 3.1 8B · GPT-4.1-nano"
    )

    st.markdown(
        "<div class='gv-disclaimer'>"
        "<b>⚠️ Research Prototype</b><br>"
        "This tool is for academic demonstration only."
        "<ul style='margin:6px 0 0 -14px;'>"
        "<li>Best results on multi-hop factual questions</li>"
        "<li>Custom input reliability not guaranteed</li>"
        "<li>Not for production or clinical use</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("📑 Citation"):
        st.markdown(
            "<div class='gv-cite'>"
            "Moeez (2026). <b>GV-CoT: Graph-Based Verification of "
            "Chain-of-Thought Reasoning in LLMs</b>. University of "
            "Management &amp; Technology (UMT). F2023332094."
            "</div>",
            unsafe_allow_html=True,
        )


# ── Session state defaults ────────────────────────────────────────────────────
st.session_state.setdefault("mode", "HotpotQA")
st.session_state.setdefault("ta_question", "")
st.session_state.setdefault("ta_chain", "")
st.session_state.setdefault("last_result", None)


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_verify, tab_perf, tab_about = st.tabs([
    "🔍 Verify Reasoning",
    "📊 Model Performance",
    "ℹ About",
])


# =============================================================================
# TAB 1 — Verify Reasoning
# =============================================================================
with tab_verify:
    st.markdown("## 🔍 Verify a Chain-of-Thought")

    # Dataset mode
    st.markdown("##### Dataset Mode")
    mcol1, mcol2, mcol3 = st.columns(3)
    if mcol1.button("🟦 HotpotQA", use_container_width=True,
                    type=("primary" if st.session_state["mode"] == "HotpotQA" else "secondary")):
        st.session_state["mode"] = "HotpotQA"
        st.session_state["ta_question"] = ""
        st.session_state["ta_chain"] = ""
        st.rerun()
    if mcol2.button("🟩 MuSiQue", use_container_width=True,
                    type=("primary" if st.session_state["mode"] == "MuSiQue" else "secondary")):
        st.session_state["mode"] = "MuSiQue"
        st.session_state["ta_question"] = ""
        st.session_state["ta_chain"] = ""
        st.rerun()
    if mcol3.button("🟧 Custom Input", use_container_width=True,
                    type=("primary" if st.session_state["mode"] == "Custom" else "secondary")):
        st.session_state["mode"] = "Custom"
        st.session_state["ta_question"] = ""
        st.session_state["ta_chain"] = ""
        st.rerun()

    mode = st.session_state["mode"]
    st.markdown("---")

    # Sample-questions for dataset modes
    if mode in ("HotpotQA", "MuSiQue"):
        st.markdown("##### ✨ Try a complex question")
        ds = "hotpot" if mode == "HotpotQA" else "musique"
        samples = get_sample_questions(ds)
        cols = st.columns(4)
        for i, sample in enumerate(samples):
            short = sample["question"][:70] + ("…" if len(sample["question"]) > 70 else "")
            if cols[i].button(f"💡 {short}\n\n_{sample['steps']} steps_",
                              key=f"sample_{mode}_{i}", use_container_width=True):
                st.session_state["ta_question"] = sample["question"]
                st.session_state["ta_chain"]    = sample["cot_chain"]
                st.rerun()

        # Search
        st.markdown("##### 🔍 Search questions")
        q = st.text_input("Search…", label_visibility="collapsed",
                          placeholder=f"Search {mode} questions…", key=f"search_{mode}")
        if q:
            chains = load_hotpot_chains() if mode == "HotpotQA" else load_musique_chains()
            q_lower = q.lower()
            matches = []
            for rec in chains[:8000]:
                if q_lower in rec["question"].lower():
                    matches.append(rec)
                if len(matches) >= 5:
                    break
            if not matches:
                st.info("No matches in first 8 000 records.")
            for j, rec in enumerate(matches):
                if st.button(f"→ {rec['question']}", key=f"hit_{mode}_{j}",
                             use_container_width=True):
                    st.session_state["ta_question"] = rec["question"]
                    st.session_state["ta_chain"]    = rec["cot_chain"]
                    st.rerun()

    # Custom-input warning
    if mode == "Custom":
        st.markdown(
            "<div class='gv-custom-warn'>"
            "<b>⚠️ Custom Input Mode</b><br>"
            "The model was trained on HotpotQA-style multi-hop questions. "
            "Results on custom questions may be unreliable.<br>"
            "<b>For best results:</b> use multi-hop factual questions requiring "
            "2+ reasoning steps. Paste a CoT chain from ChatGPT or Claude."
            "</div>",
            unsafe_allow_html=True,
        )

    # Input fields — read straight from session_state (the buttons above
    # write into ta_question / ta_chain, which the widgets pick up on rerun).
    st.markdown("##### Inputs")
    question = st.text_area(
        "Question",
        height=80,
        key="ta_question",
    )
    cot_chain = st.text_area(
        "Chain-of-Thought",
        height=240,
        key="ta_chain",
        placeholder="Paste a Step-by-step reasoning chain here…",
    )

    # Run button
    rcol1, rcol2, rcol3 = st.columns([1, 1, 2])
    with rcol2:
        run_button = st.button(
            "▶ Run GV-CoT",
            type="primary",
            use_container_width=True,
        )

    # ── Animated pipeline ─────────────────────────────────────────────────────
    if run_button:
        if not cot_chain.strip():
            st.error("Please provide a Chain-of-Thought to verify.")
        else:
            try:
                steps = split_cot_chain(cot_chain)
                if len(steps) < 2:
                    st.warning(
                        "⚠️ Chain too short — please provide at least 2 reasoning steps."
                    )
                else:
                    # Lazy-load heavy models only when actually running inference
                    sbert = load_sbert()
                    gat   = load_gat()
                    st.markdown("### 🔄 Live Pipeline")

                    # ── Persistent graph canvas (updated in place) ────────────
                    graph_box = st.empty()
                    N = len(steps)
                    current_colors = ["#888888"] * N
                    current_sizes  = [0.0] * N

                    # STEP 1 ── Split chain
                    with st.status("✂️ Step 1: Splitting reasoning chain…",
                                   expanded=True) as s1:
                        listing = st.empty()
                        buf = []
                        for s in steps:
                            buf.append(s)
                            listing.markdown(
                                "\n".join(
                                    f"**{i+1}.** {x[:160]}"
                                    + ("…" if len(x) > 160 else "")
                                    for i, x in enumerate(buf)
                                )
                            )
                            time.sleep(0.03)
                        s1.update(
                            label=f"✅ Step 1 — Split into {N} reasoning steps",
                            state="complete", expanded=False,
                        )

                    # STEP 2 ── SBERT encode
                    with st.status("⚙️ Step 2: Encoding steps with SBERT…",
                                   expanded=True) as s2:
                        prog       = st.progress(0)
                        enc_status = st.empty()
                        embeddings = []
                        for i, s in enumerate(steps):
                            enc_status.markdown(
                                f"`SBERT` encoding step **{i+1}/{N}**: "
                                f"_{s[:80]}{'…' if len(s) > 80 else ''}_"
                            )
                            embeddings.append(
                                sbert.encode([s], convert_to_numpy=True)[0]
                            )
                            prog.progress((i + 1) / N)
                            time.sleep(0.02)
                        embeddings = np.stack(embeddings)
                        enc_status.markdown("✓ All 384-dim embeddings ready.")
                        s2.update(
                            label=f"✅ Step 2 — Encoded {N} steps to 384-dim vectors",
                            state="complete", expanded=False,
                        )

                    # STEP 3 ── Build graph node by node (live)
                    with st.status("🕸️ Step 3: Building reasoning graph…",
                                   expanded=True) as s3:
                        st.caption(
                            "Each grey node is a reasoning step; "
                            "directed edges link consecutive steps."
                        )
                        for i in range(N):
                            fig = draw_chain_mpl(
                                steps,
                                colors=current_colors[: i + 1] + ["#0E1117"] * (N - i - 1),
                                sizes_norm=current_sizes,
                                edges_built=i,
                            )
                            graph_box.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            time.sleep(0.06)
                        s3.update(
                            label=f"✅ Step 3 — Built directed graph ({N} nodes, {N-1} edges)",
                            state="complete", expanded=False,
                        )

                    # STEP 4 ── GAT inference + colour flip
                    with st.status("🧠 Step 4: Running Graph Attention Network…",
                                   expanded=True) as s4:
                        st.caption(
                            "GAT classifies each step as ✅ Gold or ❌ Hallucinated."
                        )
                        preds, conf, saliency, sbert_embeddings = run_inference(steps, sbert, gat)
                        target_colors = [
                            "#00C851" if preds[i] == 1 else "#FF4444"
                            for i in range(N)
                        ]
                        flip_status = st.empty()
                        for i in range(N):
                            current_colors[i] = target_colors[i]
                            tag = "✅ Gold" if preds[i] == 1 else "❌ Hallucinated"
                            flip_status.markdown(
                                f"Step **{i+1}** → {tag} "
                                f"(confidence **{conf[i]:.2f}**)"
                            )
                            fig = draw_chain_mpl(
                                steps,
                                colors=current_colors,
                                sizes_norm=current_sizes,
                                edges_built=N - 1,
                            )
                            graph_box.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            time.sleep(0.09)
                        n_gold = int((preds == 1).sum())
                        n_hall = int((preds == 0).sum())
                        s4.update(
                            label=f"✅ Step 4 — GAT predictions: {n_gold} gold · {n_hall} hallucinated",
                            state="complete", expanded=False,
                        )

                    # STEP 5 ── SHAP saliency sizes
                    with st.status("📊 Step 5: Computing SHAP attribution scores…",
                                   expanded=True) as s5:
                        st.caption(
                            "Node size now reflects each step's contribution "
                            "to the model's decision (input × gradient)."
                        )
                        sal_status = st.empty()
                        for i in range(N):
                            current_sizes[i] = float(saliency[i])
                            sal_status.markdown(
                                f"Step **{i+1}** SHAP saliency = **{saliency[i]:.3f}**"
                            )
                            fig = draw_chain_mpl(
                                steps,
                                colors=current_colors,
                                sizes_norm=current_sizes,
                                edges_built=N - 1,
                            )
                            graph_box.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            time.sleep(0.06)
                        top_idx = int(np.argmax(saliency))
                        s5.update(
                            label=f"✅ Step 5 — Top-saliency step: #{top_idx+1} "
                                  f"(score {saliency[top_idx]:.3f})",
                            state="complete", expanded=False,
                        )

                    # ── Result panel ──────────────────────────────────────────
                    st.markdown("---")
                    st.markdown("### 📋 Results")

                    n_gold = int((preds == 1).sum())
                    n_hall = int((preds == 0).sum())

                    # ── #1 Hallucinated-step spotlight ────────────────────────
                    hall_idx = [i for i in range(N) if preds[i] == 0]
                    if hall_idx:
                        spot = [
                            "<div class='gv-spotlight'>",
                            f"<h4>🚨 {len(hall_idx)} step{'s' if len(hall_idx)!=1 else ''} "
                            f"flagged as hallucinated</h4>",
                        ]
                        for i in hall_idx:
                            spot.append(
                                f"<div class='item'><b>Step {i+1}</b> "
                                f"&nbsp;·&nbsp; confidence {conf[i]:.2f} "
                                f"&nbsp;·&nbsp; SHAP {saliency[i]:.2f}<br>"
                                f"{steps[i]}</div>"
                            )
                        spot.append("</div>")
                        st.markdown("".join(spot), unsafe_allow_html=True)
                    else:
                        st.markdown(
                            "<div class='gv-spotlight' style='border-color:#00C851; "
                            "background-color:rgba(0,200,81,0.06);'>"
                            "<div class='clean'>✅ No hallucinated steps detected — "
                            "all reasoning steps were classified as gold.</div>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.markdown(
                        f"<div class='gv-result'>"
                        f"<div class='l'>Total Steps</div>"
                        f"<div class='v gv-blue'>{len(steps)}</div></div>",
                        unsafe_allow_html=True,
                    )
                    rc2.markdown(
                        f"<div class='gv-result'>"
                        f"<div class='l'>✅ Gold Steps</div>"
                        f"<div class='v gv-green'>{n_gold}</div></div>",
                        unsafe_allow_html=True,
                    )
                    rc3.markdown(
                        f"<div class='gv-result'>"
                        f"<div class='l'>❌ Hallucinated</div>"
                        f"<div class='v gv-red'>{n_hall}</div></div>",
                        unsafe_allow_html=True,
                    )

                    # ── #2 Gold-ratio gauge ───────────────────────────────────
                    gold_ratio = n_gold / N
                    exp_acc, bucket_label = expected_accuracy(gold_ratio)
                    pct = gold_ratio * 100
                    st.markdown(
                        f"<div class='gv-gauge-wrap'>"
                        f"<div class='gv-gauge-head'>"
                        f"<span class='lbl'>Gold ratio (predicted)</span>"
                        f"<span class='val'>{pct:.1f}%</span>"
                        f"</div>"
                        f"<div class='gv-gauge-bar'>"
                        f"<div class='gv-gauge-fill' style='width:{pct:.1f}%;'></div>"
                        f"</div>"
                        f"<div class='gv-gauge-foot'>"
                        f"<span>0% (all hallucinated)</span>"
                        f"<span>100% (all gold)</span>"
                        f"</div>"
                        f"<div class='gv-gauge-note'>"
                        f"Empirically (Pearson r = +0.195) a chain in the "
                        f"<b>{bucket_label}</b> gold-ratio bucket has answer "
                        f"accuracy ≈ <b>{exp_acc*100:.1f}%</b>."
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # ── #3 Interactive 3D neural-network-style graph ──────────
                    st.markdown("#### 🌐 Explore the reasoning graph in 3D")
                    st.caption(
                        "Drag to rotate · scroll to zoom · shift-drag to pan · "
                        "hover any node for the full step text, label, confidence, "
                        "and SHAP score. **Green** = valid reasoning · "
                        "**Red** = hallucinated · Size = SHAP influence · "
                        "Blue connections = semantic similarity between steps."
                    )

                    # Chain-health bar (birds-eye view above the graph)
                    seg = []
                    for i in range(N):
                        c = "#00E676" if preds[i] == 1 else "#FF1744"
                        tip = f"Step {i+1}: {'Gold' if preds[i]==1 else 'Hallucinated'} ({conf[i]:.0%})"
                        seg.append(
                            f"<div title='{tip}' style='flex:1; "
                            f"background:{c}; height:100%; "
                            f"border-right:1px solid #0E1117;'></div>"
                        )
                    st.markdown(
                        "<div style='font-size:11px; color:#8B949E; "
                        "letter-spacing:0.5px; text-transform:uppercase; "
                        "margin:6px 0 4px 0;'>Chain health</div>"
                        f"<div style='display:flex; height:18px; width:100%; "
                        f"border-radius:5px; overflow:hidden; "
                        f"border:1px solid #2E3540;'>{''.join(seg)}</div>",
                        unsafe_allow_html=True,
                    )

                    try:
                        html_str, graph_stats = render_3d_graph(
                            steps, preds, conf, saliency, sbert_embeddings
                        )
                        components.html(html_str, height=920, scrolling=False)
                    except FileNotFoundError as e:
                        st.error(f"💥 {e}")
                        html_str, graph_stats = None, {
                            "nodes": len(steps), "seq_edges": max(0, len(steps) - 1),
                            "sim_edges": 0, "avg_sim": 0.0, "density": 0.0,
                            "longest_gold_streak": _longest_gold_streak(preds),
                            "clusters": _find_clusters(np.asarray(preds), min_run=3),
                        }

                    # Graph-stats card
                    st.markdown(
                        f"<div style='font-size:13px; color:#C9D1D9; "
                        f"background:#161B22; border:1px solid #2E3540; "
                        f"border-radius:6px; padding:8px 12px; margin-top:4px;'>"
                        f"<b>Nodes:</b> {graph_stats['nodes']} &nbsp;·&nbsp; "
                        f"<b>Edges:</b> {graph_stats['seq_edges']} sequential "
                        f"+ {graph_stats['sim_edges']} similarity &nbsp;·&nbsp; "
                        f"<b>Avg similarity:</b> {graph_stats['avg_sim']:.2f} &nbsp;·&nbsp; "
                        f"<b>Density:</b> {graph_stats['density']:.2f} &nbsp;·&nbsp; "
                        f"<b>Longest gold streak:</b> {graph_stats['longest_gold_streak']}"
                        + (
                            f" &nbsp;·&nbsp; <span style='color:#FF8A80;'>"
                            f"<b>⚠ {len(graph_stats['clusters'])} hallucination "
                            f"cluster{'s' if len(graph_stats['clusters'])!=1 else ''}</b>"
                            f"</span>"
                            if graph_stats["clusters"] else ""
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                    # Interactive HTML export — open the cinematic scene in any browser
                    if html_str:
                        st.download_button(
                            "🌌 Export Interactive Graph as HTML",
                            data=html_str.encode("utf-8"),
                            file_name="gvcot_reasoning_graph.html",
                            mime="text/html",
                        )

                    # Per-step table
                    rows = [
                        "<table class='gv-table'>",
                        "<tr><th>Step</th><th>Text (first 60 chars)</th>"
                        "<th>Label</th><th>Confidence</th><th>SHAP</th></tr>",
                    ]
                    for i in range(len(steps)):
                        label_html = (
                            "<span class='gv-green'>✅ Gold</span>"
                            if preds[i] == 1 else
                            "<span class='gv-red'>❌ Hallucinated</span>"
                        )
                        rows.append(
                            f"<tr><td>{i+1}</td>"
                            f"<td>{steps[i][:60]}{'…' if len(steps[i])>60 else ''}</td>"
                            f"<td>{label_html}</td>"
                            f"<td>{conf[i]:.3f}</td>"
                            f"<td>{saliency[i]:.3f}</td></tr>"
                        )
                    rows.append("</table>")
                    st.markdown("".join(rows), unsafe_allow_html=True)

                    # Download
                    out_json = {
                        "question":   question,
                        "steps":      steps,
                        "predictions": [int(p) for p in preds],
                        "confidence":  [float(c) for c in conf],
                        "saliency":    [float(s) for s in saliency],
                    }
                    st.download_button(
                        "⬇ Download Results (JSON)",
                        data=json.dumps(out_json, indent=2),
                        file_name="gvcot_results.json",
                        mime="application/json",
                    )
            except Exception as e:
                st.error(f"💥 Inference failed: {e}")
                st.exception(e)


# =============================================================================
# TAB 2 — Model Performance
# =============================================================================
with tab_perf:
    st.markdown("## 📊 Model Performance")

    st.markdown("### Training Ablation")
    st.markdown("""
<table class='gv-table'>
<tr><th>Method</th><th>Features</th><th>F1 Gold</th><th>F1 Hall</th>
    <th>Gold Recall</th><th>Hall Recall</th></tr>
<tr><td>SBERT baseline</td><td>Predict all gold</td><td>0.5860</td>
    <td>—</td><td>1.000</td><td>0.000</td></tr>
<tr><td>GV-CoT Run 1</td><td>384-dim SBERT</td><td>0.6199</td>
    <td>0.6371</td><td>0.7305</td><td>0.5566</td></tr>
<tr class='star'><td><b>GV-CoT Run 3 ★</b></td><td>397-dim scaled</td>
    <td><b>0.6473</b></td><td>0.6124</td><td>0.8179</td><td>0.4982</td></tr>
</table>
""", unsafe_allow_html=True)

    st.markdown("### Cross-Dataset Generalisation")
    st.markdown("""
<table class='gv-table'>
<tr><th>Setting</th><th>F1 Macro</th><th>% HotpotQA retained</th></tr>
<tr><td>HotpotQA (trained)</td><td>0.6299</td><td>100%</td></tr>
<tr><td>MuSiQue paragraphs (unfair)</td><td>0.3157</td><td>50.1%</td></tr>
<tr class='star'><td><b>MuSiQue CoT chains (fair)</b></td>
    <td><b>0.5337</b></td><td><b>84.7%</b></td></tr>
</table>
""", unsafe_allow_html=True)

    st.markdown("### Answer Accuracy vs Gold Ratio")
    st.markdown("""
<table class='gv-table'>
<tr><th>Gold Ratio</th><th>Accuracy</th><th>n</th></tr>
<tr><td>0.0 – 0.2</td><td>15.5%</td><td>1,394</td></tr>
<tr><td>0.2 – 0.4</td><td>18.5%</td><td>1,381</td></tr>
<tr><td>0.4 – 0.6</td><td>26.7%</td><td>1,748</td></tr>
<tr><td>0.6 – 0.8</td><td>34.5%</td><td>2,392</td></tr>
<tr class='star'><td><b>0.8 – 1.0</b></td><td><b>41.7%</b></td><td>10,735</td></tr>
</table>
<div style='color:#8B949E; font-size:13px; margin-top:4px;'>
Pearson <b>r = +0.1951</b> — higher gold ratio predicts answer correctness.
</div>
""", unsafe_allow_html=True)

    st.markdown("### Symbolic & SHAP Summary")
    st.markdown("""
<table class='gv-table'>
<tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
<tr><td>Transitivity violation rate</td><td>0.0773</td>
    <td>Low — GAT respects chain flow</td></tr>
<tr><td>Contradiction rate</td><td>0.4054</td>
    <td>Expected multi-hop topic jumps</td></tr>
<tr><td>Consistency score</td><td>0.8591</td>
    <td>High local coherence</td></tr>
<tr><td>Explainability coverage</td><td><b>0.8290</b></td>
    <td>+25 pts over random (57.9%)</td></tr>
<tr><td>Structural / SBERT ratio</td><td>2.44×</td>
    <td>Structural features dominate</td></tr>
</table>
""", unsafe_allow_html=True)

    st.markdown("### Comparison with Related Work")
    st.markdown("""
<table class='gv-table'>
<tr><th>Method</th><th>Task</th><th>Ext KB</th><th>Multi-LLM</th>
    <th>Step-level</th><th>Metric</th></tr>
<tr><td>SelfCheckGPT (2023)</td><td>WikiBio</td><td>No</td>
    <td>Yes 3–5×</td><td>No</td><td>AUC ~0.76</td></tr>
<tr><td>FActScoring (2023)</td><td>Biography</td><td>Yes</td>
    <td>Yes</td><td>No</td><td>Prec ~63%</td></tr>
<tr><td>CoVe (2023)</td><td>List QA</td><td>No</td>
    <td>Yes</td><td>No</td><td>Acc varies</td></tr>
<tr class='star'><td><b>GV-CoT (HotpotQA)</b></td><td>HotpotQA</td>
    <td>No</td><td>No 1×</td><td><b>Yes</b></td><td>F1=0.6473</td></tr>
<tr class='star'><td><b>GV-CoT (MuSiQue)</b></td><td>MuSiQue</td>
    <td>No</td><td>No 1×</td><td><b>Yes</b></td><td>F1=0.5337</td></tr>
</table>
<div style='color:#8B949E; font-size:13px; margin-top:6px;'>
Direct numerical comparison not possible due to different evaluation
datasets. Comparison is methodological and contextual.
</div>
""", unsafe_allow_html=True)


# =============================================================================
# TAB 3 — About
# =============================================================================
with tab_about:
    st.markdown("## ℹ About GV-CoT")
    st.markdown(
        "**GV-CoT** is a graph-based post-hoc verifier for Large-Language-Model "
        "chain-of-thought reasoning. It converts a CoT chain into a directed graph "
        "of reasoning steps, encodes each step with SBERT, augments with 13 "
        "structural features, then trains a 2-layer Graph Attention Network to "
        "classify each step as **gold** (faithful) or **hallucinated** (distractor) "
        "— without any external knowledge base or extra LLM calls."
    )

    st.markdown("### Pipeline")
    st.markdown(
        "0. **CoT Generation** — 89 625 chains from 3 LLMs\n"
        "1. **Graph Construction** — SBERT embeddings + sequential edges\n"
        "2. **GAT Training** — 2-layer Graph Attention Network\n"
        "3. **Saliency Diagnostic** — feature separation analysis\n"
        "4. **Answer Evaluation** — gold ratio vs correctness (r = +0.195)\n"
        "4b. **Symbolic Constraints** — transitivity + contradiction detection\n"
        "5. **SHAP Attribution** — 82.9% explainability coverage\n"
        "+. **MuSiQue Validation** — 84.7% of HotpotQA performance retained"
    )

    st.markdown("### Key Findings")
    st.markdown(
        "- F1 = **0.6473** on HotpotQA (+10.5% over baseline)\n"
        "- Structural features add **+2.74 F1 points** over SBERT-only\n"
        "- Gold ratio predicts answer correctness (r = +0.195)\n"
        "- Symbolic constraints degrade performance (semantic ≠ logical)\n"
        "- SHAP localises hallucinated steps in **82.9%** of graphs\n"
        "- MuSiQue: **84.7%** of HotpotQA performance retained (zero-shot)"
    )

    st.markdown("### Limitations")
    st.markdown(
        "- F1 ceiling ≈ 0.65 — representation is limited\n"
        "- Semantic edges ≠ logical entailment\n"
        "- Data contamination possible (public benchmarks)\n"
        "- Custom input reliability not guaranteed"
    )

    st.markdown("### Citation")
    st.markdown(
        "<div class='gv-cite'>"
        "Moeez (2026). <b>GV-CoT: Graph-Based Verification of Chain-of-Thought "
        "Reasoning in Large Language Models</b>. University of Management &amp; "
        "Technology (UMT). Student ID: F2023332094."
        "</div>",
        unsafe_allow_html=True,
    )


# ── Footer (all tabs) ─────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='color:#FFA500; font-size:12px; line-height:1.5;'>"
    "⚠️ <b>Disclaimer</b>: GV-CoT is a research prototype developed as part of "
    "a university project at UMT. Results are for academic purposes only and "
    "should not be used for production, clinical, legal, or safety-critical "
    "applications. Performance is best on multi-hop factual questions similar "
    "to HotpotQA. Cross-domain results may vary."
    "</div>",
    unsafe_allow_html=True,
)
st.caption(
    "GV-CoT  |  Moeez F2023332094  |  UMT 2026  |  "
    "Built with Streamlit · PyTorch Geometric · SBERT"
)
