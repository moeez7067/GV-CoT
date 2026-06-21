# GV-CoT — Graph-Based Verification of Chain-of-Thought Reasoning
Student: Moeez | ID: F2023332094 | UMT | 2026

## Pipeline Status
| Stage | Script | Status | Key Result |
|-------|--------|--------|------------|
| Stage 0 HotpotQA | data/stage0_cot_generation.py | DONE | 89,625 CoT chains |
| Stage 0 MuSiQue  | data/stage0_musique_openai_batch.py | DONE | 19,937 CoT chains |
| Stage 1 | graphs/build_graphs.py | DONE | 88,246 graphs, 670K nodes |
| Stage 2 | models/train_gnn.py | DONE | F1=0.6473 (Run 3) |
| Stage 3 | attribution/shap_analysis.py | DONE | Separation=0.004 |
| Stage 4a | evaluation/answer_eval.py | DONE | r=+0.195 |
| Stage 4b | symbolic/symbolic_constraints.py | DONE | Consistency=0.859 |
| Stage 5 | attribution/shap_final.py | DONE | Coverage=82.9% |
| MuSiQue Eval | evaluation/musique_cot_eval.py | DONE | F1=0.5337 (84.7% retained) |

## Key Results
- HotpotQA F1: 0.6473 (+10.5% over baseline)
- MuSiQue F1: 0.5337 (84.7% of HotpotQA retained, zero-shot)
- Answer correlation: r=+0.195
- Explainability coverage: 82.9%
- Symbolic constraints: negative result (semantic != logical)

## Key Files
- models/gat_best.pt — final trained model (Run 3, 397-dim)
- graphs/reasoning_graphs_augmented.pkl — HotpotQA graphs
- graphs/musique_cot_graphs.pkl — MuSiQue CoT graphs
- data/cot_chains.json — 89,625 HotpotQA chains
- data/musique_cot_chains.json — 19,937 MuSiQue chains
