# GV-CoT — Graph-Based Verification of Chain-of-Thought Reasoning

> Post-hoc hallucination detection framework using Graph Attention Networks + SHAP attribution.
> Built on HotpotQA and MuSiQue. F1 = 0.647 | Explainability Coverage = 82.9%

**Authors:** Abdul Moeez (F2023332094), Manahil Ahmad (F2023332060), Malikya Munawar (F2023332059), Abubakar Siddique (F2023332019)
**Course:** Generative AI, Section D1 | **Supervisor:** Dr. Shaista Habib
**University of Management and Technology (UMT), Lahore | 2026**

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/moeez7067/GV-CoT.git
cd GV-CoT
```

### 2. Download large data files (NOT in this repo)
The following files are **too large for GitHub** and are hosted separately on Google Drive. They are **not included** when you clone — you must download them manually.

📁 **[Download GV-CoT Data](https://drive.google.com/drive/folders/19ryd-g8q8X2Xhi-l02alK4o68MUkhE7x?usp=sharing)**

| File | Size | Goes in |
|---|---|---|
| `reasoning_graphs.pkl` | 1.14 GB | `graphs/` |
| `reasoning_graphs_augmented.pkl` | 1.19 GB | `graphs/` |
| `musique_graphs.pkl` | 827 MB | `graphs/` |
| `musique_cot_graphs.pkl` | 282 MB | `graphs/` |
| `hotpotqa_train.json` | 616 MB | `data/` |
| `musique_train.json` | 498 MB | `data/` |
| `cot_chains.json` | 145 MB | `data/` |
| `musique_cot_chains.json` | 39 MB | `data/` |

After downloading, place each file in the folder shown above. The `graphs/` and `data/` folders already exist in the cloned repo — just drop the files in.

> **Note:** `models/gat_best.pt` (the trained model, 1 MB) and `results/**` (all evaluation outputs) ARE already included in this repo — no download needed for those.

### 3. Set up environment (Miniconda)
```bash
conda create -n gvcot python=3.11
conda activate gvcot
```

**If you just want to run the app (most people — no GPU needed):**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install torch-geometric==2.8.0
pip install -r requirements.txt
```

**If you have an NVIDIA GPU and want to retrain the model:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.8.0
pip install -r requirements.txt
```

### 4. Run the frontend
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## What's in This Repo vs. Google Drive

| Included in GitHub repo | Hosted on Google Drive (download separately) |
|---|---|
| All Python source code (`.py` files) | Large `.pkl` graph files (~3.4 GB total) |
| `app.py` + Streamlit frontend | Large `.json` chain/training files (~1.3 GB total) |
| `models/gat_best.pt` (trained model, 1 MB) | |
| `results/**` (all evaluation outputs, tables, metrics) | |
| `graph_template.html`, `static/three/` (vendored JS) | |
| `requirements.txt`, `.gitignore`, `LICENSE` | |

---

## Project Structure

```
GV-CoT/
├── app.py                        # Streamlit frontend
├── graph_template.html           # 3D graph template (required by app.py)
├── static/three/                 # Vendored Three.js (offline-capable)
├── requirements.txt
├── README.md
│
├── data/                         # CoT chain generation scripts
│   ├── stage0_cot_generation.py
│   ├── stage0_musique_openai_batch.py
│   └── _st_cache/samples_*.json  # Small cached samples (included)
│   (large .json chain files → download from Google Drive)
│
├── graphs/                       # Graph building scripts
│   ├── build_graphs.py
│   ├── augment_graphs.py
│   └── build_musique_cot_graphs.py
│   (large .pkl graph files → download from Google Drive)
│
├── models/
│   ├── gnn_model.py
│   ├── train_gnn.py
│   └── gat_best.pt               # Trained model (included, 1 MB)
│
├── attribution/                  # SHAP analysis
│   ├── shap_analysis.py
│   └── shap_final.py
│
├── symbolic/                     # Symbolic constraint experiments
│   └── symbolic_constraints.py
│
├── evaluation/                   # Evaluation scripts
│   ├── answer_eval.py
│   └── musique_cot_eval.py
│
└── results/                      # All saved results (included)
    ├── run1_baseline/
    ├── run2_augmented/
    ├── run3_scaled/
    ├── stage4_answer_eval/
    ├── stage4_symbolic/
    ├── stage5_shap/
    └── musique_cot_eval/
```

---

## Pipeline

| Stage | Script | Status | Key Result |
|-------|--------|--------|------------|
| 0 — HotpotQA | `data/stage0_cot_generation.py` | DONE | 89,625 CoT chains |
| 0 — MuSiQue | `data/stage0_musique_openai_batch.py` | DONE | 19,937 CoT chains |
| 1 | `graphs/build_graphs.py` | DONE | 88,246 graphs, 670K nodes |
| 2 | `models/train_gnn.py` | DONE | F1 = 0.647 (Run 3) |
| 3 | `attribution/shap_analysis.py` | DONE | Separation = 0.004 |
| 4a | `evaluation/answer_eval.py` | DONE | r = +0.195 |
| 4b | `symbolic/symbolic_constraints.py` | DONE | Consistency = 0.859 |
| 5 | `attribution/shap_final.py` | DONE | Coverage = 82.9% |
| + | `evaluation/musique_cot_eval.py` | DONE | F1 = 0.534 (84.7% retained) |

---

## Key Results

| Metric | Value |
|--------|-------|
| HotpotQA F1 (Run 3) | **0.6473** |
| SBERT Baseline F1 | 0.5860 |
| Improvement over baseline | +10.5% |
| Answer accuracy correlation (Pearson r) | +0.195 |
| SHAP Explainability Coverage | **82.9%** |
| MuSiQue F1 (zero-shot) | 0.5337 (84.7% retained) |

---

## Requirements

**To run the app / demo (what most people will do):**
- Python 3.11
- No GPU required — a normal CPU runs the trained model fine
- ~4 GB RAM minimum
- ~8 GB free disk space (for downloaded data + graph files)

**To retrain the model from scratch (optional, not needed to use the app):**
- CUDA-capable GPU recommended (tested on RTX 4050)
- Training will be very slow on CPU

---

## Citation

```
Moeez, A., Ahmad, M., Munawar, M., Siddique, A. (2026).
GV-CoT: Structural Graph Attention for Step-Level Hallucination
Localization in Chain-of-Thought Reasoning. University of
Management and Technology (UMT), Generative AI, Section D1.
Supervisor: Dr. Shaista Habib.
```
