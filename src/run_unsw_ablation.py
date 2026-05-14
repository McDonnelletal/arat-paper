"""
ARAT — Ablation study: v1 (RF=200, 37 features) vs v2 (RF=500, 30 features).

Usage:
    python src/run_unsw_ablation.py

Outputs:
    results/table3_ablation.csv
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import gc, time, warnings

warnings.filterwarnings("ignore")

import inspect as _inspect
try:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _REPO_ROOT = Path(_inspect.getframeinfo(_inspect.currentframe()).filename).resolve().parent.parent

DATA_DIR = _REPO_ROOT / "data" / "unsw_nb15"
OUT_DIR = _REPO_ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

SEVERITY_MAP = {"Normal": 0, "Reconnaissance": 1, "Fuzzers": 1, "Analysis": 1,
                "Backdoor": 2, "DoS": 2, "Exploits": 2, "Generic": 2,
                "Shellcode": 3, "Worms": 3}
CATS = ["proto", "service", "state"]
DROP_COLS = ["id", "label", "attack_cat", "sev"]
SEED = 42


def compute_metrics(pred, y):
    return ((pred == y).mean(), (pred < y).mean(), (pred >= y).mean(),
            (pred[y == 3] == 3).sum() / max((y == 3).sum(), 1))


def run_config(X_train, X_test, y_train, y_test, n_trees):
    """Train RF+kNN, apply Config A routing, return metrics dict."""
    rf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test).astype(int)
    rf_proba = rf.predict_proba(X_test).astype(np.float64)
    rf_acc = (rf_pred == y_test).mean()
    del rf; gc.collect()

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test).astype(int)
    knn_proba = knn.predict_proba(X_test).astype(np.float64)
    del knn; gc.collect()

    ens_proba = (rf_proba + knn_proba) / 2
    disagree = rf_pred != knn_pred; agree = ~disagree
    pred_a = ens_proba.argmax(axis=1).astype(int)
    pred_a[disagree] = np.maximum(rf_pred[disagree], knn_pred[disagree])

    e, u, a, h = compute_metrics(pred_a, y_test)
    rf_w = rf_pred != y_test; knn_w = knn_pred != y_test
    a11, a10 = int((~rf_w & ~knn_w).sum()), int((rf_w & ~knn_w).sum())
    a01, a00 = int((~rf_w & knn_w).sum()), int((rf_w & knn_w).sum())
    phi = (a11*a00 - a10*a01) / np.sqrt(float((a11+a10)*(a01+a00)*(a11+a01)*(a10+a00)))
    total_err = int((pred_a != y_test).sum())
    err_agree = int((agree & (pred_a != y_test)).sum())
    total_under = int((pred_a < y_test).sum())
    under_agree = int((agree & (pred_a < y_test)).sum())

    return {"rf_exact": round(rf_acc, 4), "arat_exact": round(e, 4), "under": round(u, 4),
            "adj": round(a, 4), "hi_recall": round(h, 4), "phi": round(phi, 4),
            "disagree": round(disagree.mean(), 4),
            "err_agree": round(err_agree / total_err, 4) if total_err > 0 else 0,
            "under_agree": round(under_agree / total_under, 4) if total_under > 0 else 0}


def main():
    t0 = time.time()
    tr = pd.read_csv(DATA_DIR / "UNSW_NB15_training-set.csv")
    te = pd.read_csv(DATA_DIR / "UNSW_NB15_testing-set.csv")
    for df in [tr, te]:
        df["attack_cat"] = df["attack_cat"].fillna("Normal").str.strip()
        df["sev"] = df["attack_cat"].map(SEVERITY_MAP)
    feat = [c for c in tr.columns if c not in DROP_COLS]
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr[CATS] = oe.fit_transform(tr[CATS]); te[CATS] = oe.transform(te[CATS])
    X_tr_raw = np.nan_to_num(tr[feat].values.astype(np.float32))
    X_te_raw = np.nan_to_num(te[feat].values.astype(np.float32))
    y_train, y_test = tr["sev"].values.astype(int), te["sev"].values.astype(int)

    cat_mask = np.array([f in CATS for f in feat])
    rng = np.random.RandomState(SEED); ix = rng.choice(len(y_train), 20000, replace=False)
    mi = mutual_info_classif(X_tr_raw[ix], y_train[ix], discrete_features=cat_mask, n_neighbors=5, random_state=SEED)
    mi_ranked = sorted(zip(feat, mi), key=lambda x: -x[1])
    all_above = [f for f, s in mi_ranked if s > 0.01]

    # v1: 37 features, RF=200
    sel_v1 = [feat.index(f) for f in all_above[:37]]
    sc1 = StandardScaler(); X_tr1 = sc1.fit_transform(X_tr_raw[:, sel_v1]).astype(np.float32); X_te1 = sc1.transform(X_te_raw[:, sel_v1]).astype(np.float32)
    # v2: 30 features, RF=500
    sel_v2 = [feat.index(f) for f in all_above[:30]]
    sc2 = StandardScaler(); X_tr2 = sc2.fit_transform(X_tr_raw[:, sel_v2]).astype(np.float32); X_te2 = sc2.transform(X_te_raw[:, sel_v2]).astype(np.float32)

    print("Running v1 (RF=200, 37 features)...")
    v1 = run_config(X_tr1, X_te1, y_train, y_test, 200)
    print("Running v2 (RF=500, 30 features)...")
    v2 = run_config(X_tr2, X_te2, y_train, y_test, 500)

    df = pd.DataFrame({"v1": v1, "v2": v2})
    df.index.name = "metric"
    print(f"\n{df.to_string()}")
    df.to_csv(OUT_DIR / "table3_ablation.csv")
    print(f"\nSaved: {OUT_DIR / 'table3_ablation.csv'}")
    print(f"Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
