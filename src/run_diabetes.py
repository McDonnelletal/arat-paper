"""
ARAT — Diabetes 130-US Hospitals readmission validation experiment.

Reproduces Table 4 (Diabetes) from the paper.
Uses 3-class ordinal target:
    0 = No readmission
    1 = Readmitted >30 days
    2 = Readmitted <30 days

Dataset: UCI ID 296 — Diabetes 130-US Hospitals for years 1999-2008
    https://archive.ics.uci.edu/dataset/296

Pre-split CSVs: data/diabetes/diabetes_train.csv (n=81,412)
                data/diabetes/diabetes_test.csv  (n=20,354)

Pipeline:
    1. RF (500 trees, max_depth=15, min_samples_leaf=5, balanced) + k-NN (k=5)
    2. Conservative override routing: on disagreement, emit more severe class
    3. Unanimous-Normal safety flag: both predict 0 AND entropy > theta => escalate
    4. Logistic regression meta-model for escalation scoring

Usage:
    python src/run_diabetes.py

Outputs:
    results/diabetes_results.json
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy as sp_entropy, beta as sp_beta, norm as sp_norm
from pathlib import Path
import json, time, gc, warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
import inspect as _inspect
try:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _REPO_ROOT = Path(
        _inspect.getframeinfo(_inspect.currentframe()).filename
    ).resolve().parent.parent

DATA_DIR = _REPO_ROOT / "data" / "diabetes"
OUT_DIR = _REPO_ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
THETA = 0.10  # Safety flag entropy threshold
CLASS_NAMES = ["No readmission", ">30 days", "<30 days"]
N_CLASSES = 3
DANGEROUS_CLASS = 2  # "<30 days" readmission is the most severe


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(pred, y):
    """Exact accuracy, under-prediction rate, adjusted accuracy, dangerous-class recall."""
    return (
        (pred == y).mean(),
        (pred < y).mean(),
        (pred >= y).mean(),
        (pred[y == DANGEROUS_CLASS] == DANGEROUS_CLASS).sum()
        / max((y == DANGEROUS_CLASS).sum(), 1),
    )


def cp95(k, n):
    """Clopper-Pearson 95% confidence interval."""
    return (
        sp_beta.ppf(0.025, k, n - k + 1),
        sp_beta.ppf(0.975, k + 1, n - k),
    )


def bca_ci(data_func, n, n_boot=2000, seed=42):
    """BCa bootstrap 95% CI."""
    rng_b = np.random.RandomState(seed)
    theta_hat = data_func(np.arange(n))
    boot_vals = np.array(
        [data_func(rng_b.choice(n, n, replace=True)) for _ in range(n_boot)]
    )
    prop_below = np.clip((boot_vals < theta_hat).mean(), 0.001, 0.999)
    z0 = sp_norm.ppf(prop_below)
    sub_n = min(2000, n)
    jack_idx = rng_b.choice(n, sub_n, replace=False)
    full_idx = np.arange(n)
    jack_vals = np.array(
        [data_func(np.concatenate([full_idx[:j], full_idx[j + 1 :]])) for j in jack_idx]
    )
    jm = jack_vals.mean()
    num = ((jm - jack_vals) ** 3).sum()
    den = 6 * (((jm - jack_vals) ** 2).sum()) ** 1.5
    a_hat = num / den if den != 0 else 0
    z_lo, z_hi = sp_norm.ppf(0.025), sp_norm.ppf(0.975)
    a1 = sp_norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
    a2 = sp_norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))
    return np.percentile(boot_vals, 100 * a1), np.percentile(boot_vals, 100 * a2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    # --- Load pre-split data ---
    train_path = DATA_DIR / "diabetes_train.csv"
    test_path = DATA_DIR / "diabetes_test.csv"
    if not train_path.exists() or not test_path.exists():
        print("Diabetes data not found. Expected:")
        print(f"  {train_path}")
        print(f"  {test_path}")
        print("Run: python data/fetch_data.py")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["target"].values.astype(int)
    X_test_raw = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["target"].values.astype(int)
    N = len(y_test)

    print(f"Diabetes 130-US Hospitals (UCI 296)")
    print(f"  Train: {len(y_train):,}  Test: {N:,}  Features: {len(feature_cols)}")
    print(f"  Classes: {N_CLASSES} — {CLASS_NAMES}")
    print(f"  Dangerous class: {DANGEROUS_CLASS} ({CLASS_NAMES[DANGEROUS_CLASS]})")
    print(f"  Class distribution (test): {np.bincount(y_test, minlength=N_CLASSES).tolist()}")

    # --- Scale features ---
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train_raw).astype(np.float32)
    X_test = sc.transform(X_test_raw).astype(np.float32)

    # --- Train agents ---
    print("\nTraining RF (500, depth=15, leaf=5, balanced)...")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test).astype(int)
    rf_proba = rf.predict_proba(X_test).astype(np.float64)
    del rf; gc.collect()

    print("Training k-NN (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test).astype(int)
    knn_proba = knn.predict_proba(X_test).astype(np.float64)
    del knn; gc.collect()

    # --- Routing strategies ---
    ens_proba = (rf_proba + knn_proba) / 2
    ens_entropy = sp_entropy(ens_proba.T, base=2)
    disagree = rf_pred != knn_pred
    agree = ~disagree

    # Soft vote
    pred_soft = ens_proba.argmax(axis=1).astype(int)

    # Conservative majority (same as ARAT-A for 2 agents)
    pred_cons = np.where(agree, rf_pred, np.maximum(rf_pred, knn_pred))

    # ARAT-A: ensemble argmax on agree, conservative override on disagree
    pred_a = ens_proba.argmax(axis=1).astype(int)
    pred_a[disagree] = np.maximum(rf_pred[disagree], knn_pred[disagree])

    # ARAT-B: ARAT-A + unanimous-Normal safety flag
    pred_b = pred_a.copy()
    unan_normal = agree & (pred_b == 0)
    flag = unan_normal & (ens_entropy > THETA)
    pred_b[flag] = 1  # Escalate to next severity level

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 1: Routing strategies
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("TABLE 1: ROUTING STRATEGIES")
    print(f"{'='*70}")
    t1 = {}
    for name, pred in [("RF only", rf_pred), ("kNN only", knn_pred),
                       ("Soft vote", pred_soft), ("Cons. majority", pred_cons),
                       ("ARAT-A", pred_a), ("ARAT-B", pred_b)]:
        e, u, a, h = compute_metrics(pred, y_test)
        t1[name] = {"exact": round(e, 4), "under": round(u, 4),
                    "adj": round(a, 4), "hi_recall": round(h, 4)}
        print(f"  {name:<16s}  exact={e:.4f}  under={u:.4f}  adj={a:.4f}  dang_R={h:.4f}")

    # --- ARAT-B auto-decided subset ---
    non_esc = ~flag
    n_auto = int(non_esc.sum())
    coverage = non_esc.mean()
    p_auto, yt_auto = pred_a[non_esc], y_test[non_esc]
    auto_adj = (p_auto >= yt_auto).mean()
    k_auto = int((p_auto >= yt_auto).sum())
    auto_cp95_lo, auto_cp95_hi = cp95(k_auto, n_auto)
    print(f"\n  ARAT-B auto: n={n_auto:,} ({coverage*100:.1f}% coverage), "
          f"adj={auto_adj:.4f}, CP95=[{auto_cp95_lo:.4f}, {auto_cp95_hi:.4f}]")
    t1["ARAT-B (auto)"] = {"n": n_auto, "coverage": round(coverage, 4),
                           "adj": round(auto_adj, 4),
                           "cp95_adj_lo": round(auto_cp95_lo, 4),
                           "cp95_adj_hi": round(auto_cp95_hi, 4)}

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 2: Error dependence (phi correlation)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("TABLE 2: ERROR DEPENDENCE")
    print(f"{'='*70}")
    rf_wrong = rf_pred != y_test
    knn_wrong = knn_pred != y_test
    p_rf, p_knn = rf_wrong.mean(), knn_wrong.mean()
    p_both = (rf_wrong & knn_wrong).mean()
    p_indep = p_rf * p_knn
    inflation = p_both / p_indep if p_indep > 0 else 0

    a11 = int((~rf_wrong & ~knn_wrong).sum())
    a10 = int((rf_wrong & ~knn_wrong).sum())
    a01 = int((~rf_wrong & knn_wrong).sum())
    a00 = int((rf_wrong & knn_wrong).sum())
    denom = np.sqrt(float((a11 + a10) * (a01 + a00) * (a11 + a01) * (a10 + a00)))
    phi = (a11 * a00 - a10 * a01) / denom if denom > 0 else 0

    total_errors = int((pred_a != y_test).sum())
    errors_agree = int((agree & (pred_a != y_test)).sum())
    total_under = int((pred_a < y_test).sum())
    under_agree = int((agree & (pred_a < y_test)).sum())
    disagree_rate = disagree.mean()

    t2 = {
        "p_rf_wrong": round(p_rf, 4), "p_knn_wrong": round(p_knn, 4),
        "p_both_wrong": round(p_both, 4), "expected_indep": round(p_indep, 4),
        "inflation": round(inflation, 2), "phi": round(phi, 4),
        "errors_agree": errors_agree, "total_errors": total_errors,
        "err_agree_pct": round(errors_agree / max(total_errors, 1), 4),
        "under_agree": under_agree, "total_under": total_under,
        "under_agree_pct": round(under_agree / max(total_under, 1), 4),
        "disagree_rate": round(disagree_rate, 4),
    }
    for k, v in t2.items():
        print(f"  {k}: {v}")

    # ═══════════════════════════════════════════════════════════════════════════
    # ESCALATION META-MODEL (Logistic Regression)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("ESCALATION META-MODEL")
    print(f"{'='*70}")
    rf_conf = rf_proba.max(axis=1)
    knn_conf = knn_proba.max(axis=1)
    ens_margin = ens_proba.max(axis=1) - np.sort(ens_proba, axis=1)[:, -2]

    meta_X = np.column_stack([
        ens_margin * ens_entropy, ens_entropy, disagree.astype(float),
        rf_conf, knn_conf, rf_pred.astype(float), knn_pred.astype(float),
        np.abs(rf_conf - knn_conf), unan_normal.astype(float),
    ])
    meta_y = (pred_a < y_test).astype(int)  # under-prediction label

    mms = MinMaxScaler()
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(mms.fit_transform(meta_X), meta_y)
    scores = lr.predict_proba(mms.transform(meta_X))[:, 1]
    auroc = roc_auc_score(meta_y, scores)
    print(f"  AUROC (under-prediction): {auroc:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIDENCE INTERVALS
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("CONFIDENCE INTERVALS")
    print(f"{'='*70}")
    ci = {}
    for name, pred in [("arat_a", pred_a), ("arat_b", pred_b)]:
        k = int((pred >= y_test).sum())
        lo, hi = cp95(k, N)
        ci[f"{name}_adj_cp95"] = (round(lo, 4), round(hi, 4))
        print(f"  {name} adj: [{lo:.4f}, {hi:.4f}]")

    n_dang = int((y_test == DANGEROUS_CLASS).sum())
    for name, pred in [("arat_a", pred_a), ("arat_b", pred_b)]:
        k_d = int((pred[y_test == DANGEROUS_CLASS] == DANGEROUS_CLASS).sum())
        lo, hi = cp95(k_d, n_dang)
        ci[f"{name}_dang_cp95"] = (round(lo, 4), round(hi, 4))
        print(f"  {name} dang_R: [{lo:.4f}, {hi:.4f}]")

    # Phi BCa CI
    rf_wrong_ci = rf_pred != y_test
    knn_wrong_ci = knn_pred != y_test

    def phi_func(idx):
        rw, kw = rf_wrong_ci[idx], knn_wrong_ci[idx]
        _a11 = ((~rw) & (~kw)).sum(); _a10 = (rw & (~kw)).sum()
        _a01 = ((~rw) & kw).sum(); _a00 = (rw & kw).sum()
        d = np.sqrt(float((_a11+_a10)*(_a01+_a00)*(_a11+_a01)*(_a10+_a00)))
        return (_a11*_a00 - _a10*_a01) / d if d > 0 else 0

    phi_lo, phi_hi = bca_ci(phi_func, N)
    ci["phi_bca"] = (round(phi_lo, 4), round(phi_hi, 4))
    print(f"  phi BCa: [{phi_lo:.4f}, {phi_hi:.4f}]")

    ci["arat_b_auto_adj_cp95"] = (round(auto_cp95_lo, 4), round(auto_cp95_hi, 4))

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY COMPARISON BLOCK
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY — PAPER COMPARISON")
    print(f"{'='*70}")
    e_a, u_a, adj_a, h_a = compute_metrics(pred_a, y_test)
    print(f"  Exact accuracy (ARAT-A):     {e_a:.4f}")
    print(f"  Under-prediction (ARAT-A):   {u_a:.4f}")
    print(f"  Adjusted accuracy (ARAT-A):  {adj_a:.4f}")
    print(f"  Dangerous-class recall:      {h_a:.4f}")
    print(f"  Phi (error correlation):     {phi:.4f}")
    print(f"  Disagreement rate:           {disagree_rate:.4f}")
    print(f"  Meta-model AUROC:            {auroc:.4f}")
    print(f"  Safety flag rate (theta={THETA}): {flag.mean():.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    all_results = {
        "dataset": {
            "name": "Diabetes 130-US Hospitals",
            "uci_id": 296,
            "train_n": len(y_train),
            "test_n": N,
            "features": len(feature_cols),
            "n_classes": N_CLASSES,
            "class_names": CLASS_NAMES,
            "dangerous_class": DANGEROUS_CLASS,
        },
        "models": {
            "rf": {"n_estimators": 500, "max_depth": 15, "min_samples_leaf": 5,
                   "class_weight": "balanced"},
            "knn": {"n_neighbors": 5},
        },
        "table1": t1,
        "table2": t2,
        "escalation_meta": {"auroc": round(auroc, 4)},
        "confidence_intervals": {k: list(v) for k, v in ci.items()},
    }

    out_path = OUT_DIR / "diabetes_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved: {out_path}")
    print(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
