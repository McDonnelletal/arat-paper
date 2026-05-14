"""
ARAT — Baseline comparison on UNSW-NB15.

Trains LightGBM, XGBoost, CatBoost, and cost-sensitive variants.
Compares against RF, kNN, soft vote, conservative majority, ARAT-A.

Usage:
    python src/run_unsw_baselines.py

Outputs:
    results/baselines_full_comparison.csv
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import beta as sp_beta
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
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


def cp95(k, n):
    return sp_beta.ppf(0.025, k, n - k + 1), sp_beta.ppf(0.975, k + 1, n - k)


def main():
    t0 = time.time()

    # --- Load & preprocess (same as run_unsw.py) ---
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
    N = len(y_test)

    cat_mask = np.array([f in CATS for f in feat])
    rng = np.random.RandomState(SEED)
    ix = rng.choice(len(y_train), 20000, replace=False)
    mi = mutual_info_classif(X_tr_raw[ix], y_train[ix], discrete_features=cat_mask, n_neighbors=5, random_state=SEED)
    selected = [f for f, s in sorted(zip(feat, mi), key=lambda x: -x[1]) if s > 0.01][:30]
    sel_idx = [feat.index(f) for f in selected]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_tr_raw[:, sel_idx]).astype(np.float32)
    X_test = sc.transform(X_te_raw[:, sel_idx]).astype(np.float32)
    print(f"Data loaded: train={len(y_train):,}, test={N:,}, features={len(selected)}")

    # --- Train all models ---
    models = {}

    print("Training RF(500)..."); rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1); rf.fit(X_train, y_train); models["RF only"] = rf.predict(X_test).astype(int); rf_proba = rf.predict_proba(X_test).astype(np.float64); del rf; gc.collect()
    print("Training kNN(5)..."); knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1); knn.fit(X_train, y_train); models["kNN only"] = knn.predict(X_test).astype(int); knn_proba = knn.predict_proba(X_test).astype(np.float64); del knn; gc.collect()

    # Ensemble strategies
    ens_proba = (rf_proba + knn_proba) / 2
    disagree = models["RF only"] != models["kNN only"]
    models["Soft vote"] = ens_proba.argmax(axis=1).astype(int)
    models["Cons. majority"] = np.where(~disagree, models["RF only"], np.maximum(models["RF only"], models["kNN only"]))
    pred_a = ens_proba.argmax(axis=1).astype(int); pred_a[disagree] = np.maximum(models["RF only"][disagree], models["kNN only"][disagree]); models["ARAT-A"] = pred_a

    print("Training LightGBM (balanced)..."); m = lgb.LGBMClassifier(num_leaves=127, n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1); m.fit(X_train, y_train); models["LightGBM"] = m.predict(X_test).astype(int); del m; gc.collect()
    for alpha in [2, 3, 5, 50]:
        print(f"Training LGB cost-sens α={alpha}..."); sw = np.ones(len(y_train)); sw[y_train == 3] = alpha; m = lgb.LGBMClassifier(num_leaves=127, n_estimators=500, random_state=SEED, n_jobs=-1, verbose=-1); m.fit(X_train, y_train, sample_weight=sw); models[f"LGB α={alpha}"] = m.predict(X_test).astype(int); del m; gc.collect()
    print("Training XGBoost..."); xw = compute_sample_weight("balanced", y_train); m = xgb.XGBClassifier(n_estimators=500, max_depth=6, use_label_encoder=False, eval_metric="mlogloss", random_state=SEED, n_jobs=-1, verbosity=0); m.fit(X_train, y_train, sample_weight=xw); models["XGBoost"] = m.predict(X_test).astype(int); del m; gc.collect()
    print("Training CatBoost..."); m = CatBoostClassifier(iterations=500, depth=6, random_seed=SEED, verbose=0, auto_class_weights="Balanced"); m.fit(X_train, y_train); models["CatBoost"] = m.predict(X_test).astype(int).ravel(); del m; gc.collect()

    # --- Report ---
    print(f"\n{'='*72}\n  BASELINE COMPARISON (n={N:,})\n{'='*72}")
    print(f"  {'Model':<20s} {'Exact':>7s} {'Under':>7s} {'Adj':>7s} {'Hi-R':>7s} {'CP95(adj)':>16s}")
    rows = []
    for name, pred in models.items():
        e, u, a, h = compute_metrics(pred, y_test)
        lo, hi = cp95(int((pred >= y_test).sum()), N)
        print(f"  {name:<20s} {e:>7.4f} {u:>7.4f} {a:>7.4f} {h:>7.4f} [{lo:.4f},{hi:.4f}]")
        rows.append({"model": name, "exact": round(e, 4), "under": round(u, 4), "adj": round(a, 4), "hi_recall": round(h, 4), "cp95_lo": round(lo, 4), "cp95_hi": round(hi, 4)})

    pd.DataFrame(rows).to_csv(OUT_DIR / "baselines_full_comparison.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'baselines_full_comparison.csv'}")
    print(f"Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
