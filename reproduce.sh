#!/usr/bin/env bash
# reproduce.sh — Run all ARAT experiments end-to-end.
# Usage: bash reproduce.sh
set -euo pipefail

echo "=== Reproducibility Pipeline ==="
echo ""

echo "[1/10] Downloading data..."
python data/fetch_data.py

echo "[2/10] Main UNSW-NB15 pipeline (Table 1, error dependence §6.2, CIs, θ sweep, hi-sev, Fig 2)..."
python src/run_unsw.py

echo "[3/10] Baseline comparison (Table 1)..."
python src/run_unsw_baselines.py

echo "[4/10] RF-config ablation, Table 2 (UNSW v1 vs v2)..."
python src/run_unsw_ablation.py

echo "[5/10] Diabetes validation (§6.5)..."
python src/run_diabetes.py

echo "[6/10] Diabetes RF-config ablation (Table 2, Diabetes columns)..."
python src/run_diabetes_ablation.py

echo "[7/10] SVM substitution, §6.5 (agent robustness, UNSW + Diabetes)..."
python src/run_svm_substitution.py

echo "[8/10] Base-learner pair sweep, Table 3 (R1 robustness)..."
python src/run_pair_sweep.py

echo "[9/10] Single-model UQ baseline vs ARAT, Table 4 (R3)..."
python src/run_conformal_baseline.py

echo "[10/10] LLM committee table from committed soft-probs (§6.7)..."
python src/llm_committee/fill_table.py \
    --large results/llm_committee/TCGAStage_crossfamily_large.csv

echo ""
echo "=== Done. Results in results/, figures in figures/ ==="
