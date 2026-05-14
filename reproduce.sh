#!/usr/bin/env bash
# reproduce.sh — Run all ARAT experiments end-to-end.
# Usage: bash reproduce.sh
set -euo pipefail

echo "=== Reproducibility Pipeline ==="
echo ""

echo "[1/6] Downloading data..."
python data/fetch_data.py

echo "[2/6] Main UNSW-NB15 pipeline (Tables 1-2, CIs, θ sweep, hi-sev, figure)..."
python src/run_unsw.py

echo "[3/6] Baseline comparison..."
python src/run_unsw_baselines.py

echo "[4/6] Ablation study (v1 vs v2)..."
python src/run_unsw_ablation.py

echo "[5/6] Diabetes validation..."
python src/run_diabetes.py

echo "[6/6] SVM substitution (agent robustness)..."
python src/run_svm_substitution.py

echo ""
echo "=== Done. Results in results/, figures in figures/ ==="
