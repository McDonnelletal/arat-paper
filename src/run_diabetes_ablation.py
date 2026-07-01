#!/usr/bin/env python3
"""ARAT -- Diabetes RF-configuration ablation (Table 2, Diabetes columns).

v1 (lighter): RF n_estimators=100; v2 (heavier): RF n_estimators=500.
Both configs use max_depth=15, min_samples_leaf=5, balanced class weights;
k-NN k=5; SEED=42. Reads the pre-encoded 21-feature split in data/diabetes/.

Outputs: results/diabetes_ablation.json (7 mechanism indicators per config).
Usage:   python src/run_diabetes_ablation.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path

SEED = 42
DATA = Path(__file__).resolve().parents[1] / "data" / "diabetes"
OUT = Path(__file__).resolve().parents[1] / "results"


def phi_coef(a_wrong, b_wrong):
    """Mean-square-contingency phi between the two agents' error indicators."""
    n11 = int((a_wrong & b_wrong).sum()); n00 = int((~a_wrong & ~b_wrong).sum())
    n10 = int((a_wrong & ~b_wrong).sum()); n01 = int((~a_wrong & b_wrong).sum())
    den = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return float((n11 * n00 - n10 * n01) / den) if den else 0.0


def indicators(rf_proba, knn_proba, rf_pred, knn_pred, y):
    ens = (rf_proba + knn_proba) / 2.0
    disagree = rf_pred != knn_pred
    agree = ~disagree
    pred_a = ens.argmax(1).astype(int)                       # ARAT-A: soft vote ...
    pred_a[disagree] = np.maximum(rf_pred[disagree], knn_pred[disagree])  # ... conservative override
    errors = pred_a != y
    under = pred_a < y
    return {
        "rf_exact": round(float((rf_pred == y).mean()), 4),
        "arat_under": round(float(under.mean()), 4),
        "dang_recall": round(float((pred_a[y == 2] == 2).sum() / max((y == 2).sum(), 1)), 4),
        "phi": round(phi_coef(rf_pred != y, knn_pred != y), 4),
        "disagree_rate": round(float(disagree.mean()), 4),
        "err_agree_pct": round(float((errors & agree).sum() / max(errors.sum(), 1)), 4),
        "under_agree_pct": round(float((under & agree).sum() / max(under.sum(), 1)), 4),
    }


def main():
    tr = pd.read_csv(DATA / "diabetes_train.csv")
    te = pd.read_csv(DATA / "diabetes_test.csv")
    fcols = [c for c in tr.columns if c != "target"]
    sc = StandardScaler()
    X_train = sc.fit_transform(tr[fcols].values.astype(np.float32)).astype(np.float32)
    X_test = sc.transform(te[fcols].values.astype(np.float32)).astype(np.float32)
    y_train = tr["target"].values.astype(int)
    y_test = te["target"].values.astype(int)
    print(f"Diabetes ablation: train={len(y_train):,} test={len(y_test):,} feats={len(fcols)}")

    # k-NN is the fixed second agent; only the RF configuration changes v1 -> v2.
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_train, y_train)
    knn_proba = knn.predict_proba(X_test).astype(np.float64)
    knn_pred = knn.predict(X_test).astype(int)

    out = {}
    for tag, n_trees in [("v1", 100), ("v2", 500)]:
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=15, min_samples_leaf=5,
                                    class_weight="balanced", random_state=SEED, n_jobs=-1).fit(X_train, y_train)
        rf_proba = rf.predict_proba(X_test).astype(np.float64)
        rf_pred = rf.predict(X_test).astype(int)
        out[tag] = {"rf_n_estimators": n_trees, **indicators(rf_proba, knn_proba, rf_pred, knn_pred, y_test)}
        print(f"  {tag} (RF={n_trees}): {out[tag]}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "diabetes_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/diabetes_ablation.json")


if __name__ == "__main__":
    main()
