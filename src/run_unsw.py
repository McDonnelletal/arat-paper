"""
ARAT — Main UNSW-NB15 experiment pipeline.

Reproduces: Table 1 (routing strategies), Table 2 (error dependence),
Sections 6.3-6.5, all confidence intervals, θ sweep, high-severity
analysis, and confusion matrix figure.

Usage:
    python src/run_unsw.py              # run everything (default)
    python src/run_unsw.py --mode all   # same as above
    python src/run_unsw.py --mode sweep # θ sweep only
    python src/run_unsw.py --mode hisev # high-severity analysis only
    python src/run_unsw.py --mode figure # confusion matrix only

Outputs:
    results/table1_routing_strategies.csv
    results/table2_error_dependence.csv
    results/confidence_intervals.csv
    results/all_results.json
    results/theta_sweep_results.csv
    results/high_severity_unan_normal_analysis.json
    figures/confusion_matrix.pdf
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
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
    _REPO_ROOT = Path(_inspect.getframeinfo(_inspect.currentframe()).filename).resolve().parent.parent

DATA_DIR = _REPO_ROOT / "data" / "unsw_nb15"
OUT_DIR = _REPO_ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = _REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEVERITY_MAP = {
    "Normal": 0,
    "Reconnaissance": 1, "Fuzzers": 1, "Analysis": 1,
    "Backdoor": 2, "DoS": 2, "Exploits": 2, "Generic": 2,
    "Shellcode": 3, "Worms": 3,
}
CATS = ["proto", "service", "state"]
DROP_COLS = ["id", "label", "attack_cat", "sev"]
THETA = 0.10  # Safety flag entropy threshold
SEED = 42
THETAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
CLASS_NAMES = ["Normal", "Low", "Medium", "High"]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(pred, y):
    """Exact accuracy, under-prediction rate, adjusted accuracy, high-class recall."""
    return (
        (pred == y).mean(),
        (pred < y).mean(),
        (pred >= y).mean(),
        (pred[y == 3] == 3).sum() / max((y == 3).sum(), 1),
    )


def cp95(k, n):
    """Clopper-Pearson 95% confidence interval."""
    return sp_beta.ppf(0.025, k, n - k + 1), sp_beta.ppf(0.975, k + 1, n - k)


def bca_ci(data_func, n, n_boot=2000, seed=42):
    """BCa bootstrap 95% CI."""
    rng_b = np.random.RandomState(seed)
    theta_hat = data_func(np.arange(n))
    boot_vals = np.array([data_func(rng_b.choice(n, n, replace=True)) for _ in range(n_boot)])
    prop_below = np.clip((boot_vals < theta_hat).mean(), 0.001, 0.999)
    z0 = sp_norm.ppf(prop_below)
    sub_n = min(2000, n)
    jack_idx = rng_b.choice(n, sub_n, replace=False)
    full_idx = np.arange(n)
    jack_vals = np.array([data_func(np.concatenate([full_idx[:j], full_idx[j+1:]])) for j in jack_idx])
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
def main(mode="all"):
    t0 = time.time()

    # --- Load data ---
    print("Loading UNSW-NB15...")
    tr = pd.read_csv(DATA_DIR / "UNSW_NB15_training-set.csv")
    te = pd.read_csv(DATA_DIR / "UNSW_NB15_testing-set.csv")
    for df in [tr, te]:
        df["attack_cat"] = df["attack_cat"].fillna("Normal").str.strip()
        df["sev"] = df["attack_cat"].map(SEVERITY_MAP)

    feat = [c for c in tr.columns if c not in DROP_COLS]
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr[CATS] = oe.fit_transform(tr[CATS])
    te[CATS] = oe.transform(te[CATS])

    X_tr_raw = np.nan_to_num(tr[feat].values.astype(np.float32))
    X_te_raw = np.nan_to_num(te[feat].values.astype(np.float32))
    y_train = tr["sev"].values.astype(int)
    y_test = te["sev"].values.astype(int)
    N = len(y_test)
    print(f"  train={len(y_train):,}, test={N:,}")

    # --- Feature selection (MI, top-30) ---
    print("Feature selection...")
    cat_mask = np.array([f in CATS for f in feat])
    rng = np.random.RandomState(SEED)
    ix = rng.choice(len(y_train), 20000, replace=False)
    mi = mutual_info_classif(X_tr_raw[ix], y_train[ix], discrete_features=cat_mask, n_neighbors=5, random_state=SEED)
    mi_scores = sorted(zip(feat, mi), key=lambda x: -x[1])
    selected = [f for f, s in mi_scores if s > 0.01][:30]
    sel_idx = [feat.index(f) for f in selected]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_tr_raw[:, sel_idx]).astype(np.float32)
    X_test = sc.transform(X_te_raw[:, sel_idx]).astype(np.float32)
    print(f"  {len(selected)} features selected")

    # --- Train agents ---
    print("Training RF(500)...")
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test).astype(int)
    rf_proba = rf.predict_proba(X_test).astype(np.float64)
    del rf; gc.collect()

    print("Training kNN(5)...")
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

    pred_soft = ens_proba.argmax(axis=1).astype(int)
    pred_cons = np.where(agree, rf_pred, np.maximum(rf_pred, knn_pred))
    pred_a = ens_proba.argmax(axis=1).astype(int)
    pred_a[disagree] = np.maximum(rf_pred[disagree], knn_pred[disagree])
    pred_b = pred_a.copy()
    unan_normal = agree & (pred_b == 0)
    flag = unan_normal & (ens_entropy > THETA)
    pred_b[flag] = 1

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE: TABLES (default / all)
    # ═══════════════════════════════════════════════════════════════════════════
    if mode in ("all", "tables"):
        # --- Table 1 ---
        print("\n=== TABLE 1 ===")
        t1 = {}
        for name, pred in [("RF only", rf_pred), ("kNN only", knn_pred), ("Soft vote", pred_soft),
                           ("Cons. majority", pred_cons), ("ARAT-A", pred_a), ("ARAT-B", pred_b)]:
            e, u, a, h = compute_metrics(pred, y_test)
            t1[name] = {"exact": round(e, 4), "under": round(u, 4), "adj": round(a, 4), "hi_recall": round(h, 4)}
            print(f"  {name:<16s}  exact={e:.4f}  under={u:.4f}  adj={a:.4f}  hi_R={h:.4f}")

        pd.DataFrame(t1).T.to_csv(OUT_DIR / "table1_routing_strategies.csv")

        # --- ARAT v2 auto-decided subset (non-escalated) ---
        print("\n=== ARAT v2 AUTO-DECIDED (non-escalated) ===")
        non_esc = ~flag
        n_auto = int(non_esc.sum())
        coverage = non_esc.mean()
        p_auto, yt_auto = pred_a[non_esc], y_test[non_esc]
        auto_exact = (p_auto == yt_auto).mean()
        auto_under = (p_auto < yt_auto).mean()
        auto_adj = (p_auto >= yt_auto).mean()
        hi_auto = (yt_auto == 3)
        auto_hi_r = (p_auto[hi_auto] == 3).sum() / max(hi_auto.sum(), 1)
        k_auto = int((p_auto >= yt_auto).sum())
        auto_cp95_lo, auto_cp95_hi = cp95(k_auto, n_auto)
        print(f"  n_auto={n_auto:,} ({coverage*100:.1f}% coverage)")
        print(f"  exact={auto_exact:.4f}  under={auto_under:.4f}  adj={auto_adj:.4f}  hi_R={auto_hi_r:.4f}")
        print(f"  CP95(adj): [{auto_cp95_lo:.4f}, {auto_cp95_hi:.4f}]")
        t1["ARAT-B (auto)"] = {"exact": round(auto_exact, 4), "under": round(auto_under, 4),
                               "adj": round(auto_adj, 4), "hi_recall": round(auto_hi_r, 4),
                               "n": n_auto, "coverage": round(coverage, 4),
                               "cp95_adj_lo": round(auto_cp95_lo, 4), "cp95_adj_hi": round(auto_cp95_hi, 4)}

        # --- Table 2: Error dependence ---
        print("\n=== TABLE 2 ===")
        rf_wrong = rf_pred != y_test
        knn_wrong = knn_pred != y_test
        p_rf, p_knn = rf_wrong.mean(), knn_wrong.mean()
        p_both = (rf_wrong & knn_wrong).mean()
        p_indep = p_rf * p_knn
        inflation = p_both / p_indep
        a11 = int((~rf_wrong & ~knn_wrong).sum())
        a10 = int((rf_wrong & ~knn_wrong).sum())
        a01 = int((~rf_wrong & knn_wrong).sum())
        a00 = int((rf_wrong & knn_wrong).sum())
        phi = (a11 * a00 - a10 * a01) / np.sqrt(float((a11+a10)*(a01+a00)*(a11+a01)*(a10+a00)))
        total_errors = int((pred_a != y_test).sum())
        errors_agree = int((agree & (pred_a != y_test)).sum())
        total_under = int((pred_a < y_test).sum())
        under_agree = int((agree & (pred_a < y_test)).sum())

        t2 = {"p_rf_wrong": round(p_rf, 4), "p_knn_wrong": round(p_knn, 4),
               "p_both_wrong": round(p_both, 4), "expected_indep": round(p_indep, 4),
               "inflation": round(inflation, 2), "phi": round(phi, 4),
               "errors_agree": errors_agree, "total_errors": total_errors,
               "err_agree_pct": round(errors_agree / total_errors, 4),
               "under_agree": under_agree, "total_under": total_under,
               "under_agree_pct": round(under_agree / total_under, 4) if total_under > 0 else 0}
        for k, v in t2.items():
            print(f"  {k}: {v}")

        pd.DataFrame([t2]).to_csv(OUT_DIR / "table2_error_dependence.csv", index=False)

        # --- Section 6.4: Escalation meta-model ---
        print("\n=== ESCALATION MODEL ===")
        rf_conf = rf_proba.max(axis=1)
        knn_conf = knn_proba.max(axis=1)
        ens_margin = ens_proba.max(axis=1) - np.sort(ens_proba, axis=1)[:, -2]
        meta_X = np.column_stack([ens_margin * ens_entropy, ens_entropy, disagree.astype(float),
                                  rf_conf, knn_conf, rf_pred.astype(float), knn_pred.astype(float),
                                  np.abs(rf_conf - knn_conf), unan_normal.astype(float)])
        meta_y = (pred_a < y_test).astype(int)
        mms = MinMaxScaler()
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        lr.fit(mms.fit_transform(meta_X), meta_y)
        scores = lr.predict_proba(mms.transform(meta_X))[:, 1]
        print(f"  Global AUROC: {roc_auc_score(meta_y, scores):.4f}")

        # --- Confidence intervals ---
        print("\n=== CONFIDENCE INTERVALS ===")
        ci = {}
        for name, pred in [("arat_a", pred_a), ("arat_b", pred_b)]:
            k = int((pred >= y_test).sum())
            lo, hi = cp95(k, N)
            ci[f"{name}_adj_cp95"] = (round(lo, 4), round(hi, 4))
            print(f"  {name} adj: [{lo:.4f}, {hi:.4f}]")

        n_hi = int((y_test == 3).sum())
        for name, pred in [("arat_a", pred_a), ("arat_b", pred_b)]:
            k_hi = int((pred[y_test == 3] == 3).sum())
            lo, hi = cp95(k_hi, n_hi)
            ci[f"{name}_hi_cp95"] = (round(lo, 4), round(hi, 4))
            print(f"  {name} hi_R: [{lo:.4f}, {hi:.4f}]")

        rf_wrong_ci = rf_pred != y_test
        knn_wrong_ci = knn_pred != y_test

        def phi_func(idx):
            rw, kw = rf_wrong_ci[idx], knn_wrong_ci[idx]
            a11 = ((~rw) & (~kw)).sum(); a10 = (rw & (~kw)).sum()
            a01 = ((~rw) & kw).sum(); a00 = (rw & kw).sum()
            d = np.sqrt(float((a11+a10)*(a01+a00)*(a11+a01)*(a10+a00)))
            return (a11*a00 - a10*a01) / d if d > 0 else 0

        phi_lo, phi_hi = bca_ci(phi_func, N)
        ci["phi_bca"] = (round(phi_lo, 4), round(phi_hi, 4))
        print(f"  phi BCa: [{phi_lo:.4f}, {phi_hi:.4f}]")

        ci["arat_b_auto_adj_cp95"] = (round(auto_cp95_lo, 4), round(auto_cp95_hi, 4))
        print(f"  arat_b_auto adj: [{auto_cp95_lo:.4f}, {auto_cp95_hi:.4f}]")

        ci_df = pd.DataFrame([{"metric": k, "lower": v[0], "upper": v[1]} for k, v in ci.items()])
        ci_df.to_csv(OUT_DIR / "confidence_intervals.csv", index=False)

        # --- Full JSON dump ---
        all_results = {"dataset": {"train_n": len(y_train), "test_n": N, "features": len(selected)},
                       "table1": t1, "table2": t2, "confidence_intervals": {k: list(v) for k, v in ci.items()}}
        with open(OUT_DIR / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE: THETA SWEEP
    # ═══════════════════════════════════════════════════════════════════════════
    if mode in ("all", "sweep"):
        a_e, a_u, a_adj, a_hi = compute_metrics(pred_a, y_test)
        print(f"\n{'='*70}\n  θ SWEEP (n={N:,})\n{'='*70}")
        print(f"  {'θ':<7s} {'flag%':>7s} {'exact':>7s} {'under':>7s} {'adj':>7s} {'hi_R':>7s} {'Δpp':>7s}")
        rows = []
        for theta in THETAS:
            flag_t = unan_normal & (ens_entropy > theta)
            pred_bt = pred_a.copy(); pred_bt[flag_t] = 1
            e, u, a, h = compute_metrics(pred_bt, y_test)
            lo, hi = cp95(int((pred_bt >= y_test).sum()), N)
            rows.append({"theta": theta, "flagged_pct": round(flag_t.mean(), 4), "exact": round(e, 4),
                         "under": round(u, 4), "adj": round(a, 4), "hi_recall": round(h, 4),
                         "delta_pp": round((a - a_adj) * 100, 2), "cp95_lo": round(lo, 4), "cp95_hi": round(hi, 4)})
            print(f"  {theta:<7.3f} {flag_t.mean()*100:>6.2f}% {e:>7.4f} {u:>7.4f} {a:>7.4f} {h:>7.4f} {(a-a_adj)*100:>+6.2f}")

        pd.DataFrame(rows).to_csv(OUT_DIR / "theta_sweep_results.csv", index=False)
        print(f"\nSaved: {OUT_DIR / 'theta_sweep_results.csv'}")

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE: HIGH-SEVERITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    if mode in ("all", "hisev"):
        print("\n=== HIGH-SEVERITY ANALYSIS ===")
        hi_mask = y_test == 3
        hi_in_unan = hi_mask & unan_normal
        hi_total = int(hi_mask.sum())
        hi_in_count = int(hi_in_unan.sum())

        flag_010 = unan_normal & (ens_entropy > 0.10)
        flag_005 = unan_normal & (ens_entropy > 0.005)

        results_hisev = {
            "hi_total": hi_total,
            "unan_normal_total": int(unan_normal.sum()),
            "hi_in_unan_normal": hi_in_count,
            "hi_in_unan_pct": round(hi_in_count / hi_total, 4) if hi_total > 0 else 0,
            "flag_theta_0.10_catches": int((hi_in_unan & flag_010).sum()),
            "flag_theta_0.10_catch_pct": round((hi_in_unan & flag_010).sum() / hi_in_count, 4) if hi_in_count > 0 else 0,
            "flag_theta_0.005_catches": int((hi_in_unan & flag_005).sum()),
            "flag_theta_0.005_catch_pct": round((hi_in_unan & flag_005).sum() / hi_in_count, 4) if hi_in_count > 0 else 0,
        }
        print(json.dumps(results_hisev, indent=2))
        with open(OUT_DIR / "high_severity_unan_normal_analysis.json", "w") as f:
            json.dump(results_hisev, f, indent=2)
        print(f"\nSaved: {OUT_DIR / 'high_severity_unan_normal_analysis.json'}")

    # ═══════════════════════════════════════════════════════════════════════════
    # MODE: CONFUSION MATRIX FIGURE
    # ═══════════════════════════════════════════════════════════════════════════
    if mode in ("all", "figure"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("\n=== CONFUSION MATRIX ===")
        n_classes = 4
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_test, pred_a):
            cm[t, p] += 1
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(n_classes):
            for j in range(n_classes):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]:,})", ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
        ax.set_xticklabels(CLASS_NAMES, fontsize=9); ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted Severity", fontsize=10); ax.set_ylabel("True Severity", fontsize=10)
        ax.set_title("ARAT Confusion Matrix", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "confusion_matrix.pdf", bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR / 'confusion_matrix.pdf'}")

    print(f"\nDone. Elapsed: {time.time()-t0:.0f}s")
    print(f"Results saved to: {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARAT UNSW-NB15 pipeline")
    parser.add_argument("--mode", choices=["all", "tables", "sweep", "hisev", "figure"],
                        default="all", help="Which analysis to run (default: all)")
    args = parser.parse_args()
    main(mode=args.mode)
