# GV-CoT Experiment Results

This folder stores all experiment outputs so results can be 
reviewed without rerunning code.

## Structure

### Stage 4 — GAT Training Results
- training_log.csv — epoch-by-epoch metrics (loss, accuracy, F1)
- best_val_f1.txt — best validation F1 score achieved
- confusion_matrix.png — confusion matrix on validation set
- classification_report.txt — precision, recall, F1 per class

### Stage 5 — SHAP Attribution Results  
- shap_scores.json — per-node SHAP importance scores
- shap_summary_plot.png — SHAP summary visualization
- top_hallucinated_nodes.json — highest scoring hallucinated nodes

### Comparison Tables
- hallucination_detection_comparison.csv — GV-CoT vs SelfCheckGPT vs FActScorer vs SAFE
- answer_accuracy_comparison.csv — Option A vs CoT vs Self-Consistency vs Chain-of-Verification

### Dataset Statistics
- cot_diversity_stats.json — LLM distribution (30% Claude / 33% Groq / 36% OpenAI)
- label_distribution.txt — gold vs distractor node counts
