"""
ARAT — SVM Substitution Experiment.

Replaces kNN with SVM-RBF in the ARAT routing pipeline.
Reports phi, disagreement rate, under-prediction rate, and
compares against the baseline RF+kNN ARAT system.

Datasets: UNSW-NB15, Diabetes (if available).

Usage:
    python src/run_svm_substitution.py

Outputs:
    results/svm_substitution_results.json
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from pathlib import Path
import inspect as _inspect
import gc, time, json, warnings

warnings.filterwarnings("ignore")

try:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    _REPO_ROOT = Path(_inspect.getframeinfo(_inspect.currentframe()).filename).resolve().parent.parent

DATA_DIR = _REPO_ROOT / "data" / "unsw_nb15"
OUT_DIR = _REPO_ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

SEVERITY_MAP = {
    "Normal": 0,
    "Reconnaissance": 1, "Fuzzers": 1, "Analysis": 1,
    "Backdoor": 2, "DoS": 2, "Exploits": 2, "Generic": 2,
    "Shellcode": 3, "Worms": 3,
}
CATS = ["proto", "service", "state"]
DROP_COLS = ["id", "label", "attack_cat", "sev"]
SEED = 42


def compute_metrics(pred, y):
    """Exact accuracy, under-prediction rate, adjusted accuracy, high-class recall."""
    return {
        "exact": round(float((pred == y).mean()), 4),
        "under": round(float((pred < y).mean()), 4),
        "adj": round(float((pred >= y).mean()), 4),
        "hi_recall": round(float((pred[y == 3] == 3).sum() / max((y == 3).sum(), 1)), 4),
    }


def compute_phi(pred_a, pred_b, y):
    """Phi coefficient between error patterns of two agents."""
    a_wrong = (pred_a != y).astype(int)
    b_wrong = (pred_b != y).astype(int)
    n = len(y)
    n11 = int(((1 - a_wrong) & (1 - b_wrong)).sum())  # both correct
    n10 = int((a_wrong & (1 - b_wrong)).sum())          # a wrong, b correct
    n01 = int(((1 - a_wrong) & b_wrong).sum())          # a correct, b wrong
    n00 = int((a_wrong & b_wrong).sum())                 # both wrong
    denom = np.sqrt(float((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00)))
    if denom == 0:
        return 0.0
    return round(float(n11*n00 - n10*n01) / denom, 4)


def run_unsw():
    """Run SVM substitution on UNSW-NB15."""
    print("=" * 70)
    print("  UNSW-NB15: SVM SUBSTITUTION EXPERIMENT")
    print("=" * 70)
    t0 = time.time()

    # --- Load & preprocess ---
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
    print(f"  Data: train={len(y_train):,}, test={N:,}")

    # --- Feature selection (MI, top-30) ---
    cat_mask = np.array([f in CATS for f in feat])
    rng = np.random.RandomState(SEED)
    ix = rng.choice(len(y_train), 20000, replace=False)
    mi = mutual_info_classif(X_tr_raw[ix], y_train[ix], discrete_features=cat_mask,
                             n_neighbors=5, random_state=SEED)
    selected = [f for f, s in sorted(zip(feat, mi), key=lambda x: -x[1]) if s > 0.01][:30]
    sel_idx = [feat.index(f) for f in selected]

    sc = StandardScaler()
    X_train = sc.fit_transform(X_tr_raw[:, sel_idx]).astype(np.float32)
    X_test = sc.transform(X_te_raw[:, sel_idx]).astype(np.float32)
    print(f"  Features: {len(selected)}")

    # --- Train RF (same as baseline) ---
    print("  Training RF(500)...")
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test).astype(int)
    rf_proba = rf.predict_proba(X_test).astype(np.float64)
    del rf; gc.collect()

    # --- Train kNN (baseline second agent) ---
    print("  Training kNN(5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test).astype(int)
    knn_proba = knn.predict_proba(X_test).astype(np.float64)
    del knn; gc.collect()

    # --- Train SVM-RBF (substitution agent) ---
    # SVM-RBF is O(n^2-n^3); subsample training to 30k stratified for tractability.
    # Predictions and routing are evaluated on the FULL test set.
    from sklearn.model_selection import StratifiedShuffleSplit
    N_SVM_TRAIN = 30000
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N_SVM_TRAIN, random_state=SEED)
    svm_idx = next(sss.split(X_train, y_train))[0]
    print(f"  Training SVM-RBF (C=10, balanced, n={N_SVM_TRAIN:,} stratified subsample)...")
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
              random_state=SEED, probability=True)
    svm.fit(X_train[svm_idx], y_train[svm_idx])
    svm_pred = svm.predict(X_test).astype(int)
    svm_proba = svm.predict_proba(X_test).astype(np.float64)
    del svm; gc.collect()

    # --- ARAT routing: RF + kNN (baseline) ---
    ens_proba_knn = (rf_proba + knn_proba) / 2
    disagree_knn = rf_pred != knn_pred
    pred_a_knn = ens_proba_knn.argmax(axis=1).astype(int)
    pred_a_knn[disagree_knn] = np.maximum(rf_pred[disagree_knn], knn_pred[disagree_knn])

    # --- ARAT routing: RF + SVM (substitution) ---
    ens_proba_svm = (rf_proba + svm_proba) / 2
    disagree_svm = rf_pred != svm_pred
    pred_a_svm = ens_proba_svm.argmax(axis=1).astype(int)
    pred_a_svm[disagree_svm] = np.maximum(rf_pred[disagree_svm], svm_pred[disagree_svm])

    # --- Compute metrics ---
    baseline = compute_metrics(pred_a_knn, y_test)
    substitution = compute_metrics(pred_a_svm, y_test)

    phi_baseline = compute_phi(rf_pred, knn_pred, y_test)
    phi_svm = compute_phi(rf_pred, svm_pred, y_test)

    disagree_rate_knn = round(float(disagree_knn.mean()), 4)
    disagree_rate_svm = round(float(disagree_svm.mean()), 4)

    # --- Report ---
    print(f"\n  {'Metric':<20s} {'RF+kNN':>10s} {'RF+SVM':>10s} {'Delta':>10s}")
    print(f"  {'-'*50}")
    for k in ["exact", "under", "adj", "hi_recall"]:
        delta = substitution[k] - baseline[k]
        print(f"  {k:<20s} {baseline[k]:>10.4f} {substitution[k]:>10.4f} {delta:>+10.4f}")
    print(f"  {'phi':<20s} {phi_baseline:>10.4f} {phi_svm:>10.4f} {phi_svm - phi_baseline:>+10.4f}")
    print(f"  {'disagree_rate':<20s} {disagree_rate_knn:>10.4f} {disagree_rate_svm:>10.4f} {disagree_rate_svm - disagree_rate_knn:>+10.4f}")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")

    return {
        "dataset": "UNSW-NB15",
        "n_train": len(y_train), "n_test": N, "svm_train_subsample": 30000,
        "baseline_rf_knn": {**baseline, "phi": phi_baseline, "disagree_rate": disagree_rate_knn},
        "substitution_rf_svm": {**substitution, "phi": phi_svm, "disagree_rate": disagree_rate_svm},
        "delta": {
            k: round(substitution[k] - baseline[k], 4) for k in baseline
        } | {
            "phi": round(phi_svm - phi_baseline, 4),
            "disagree_rate": round(disagree_rate_svm - disagree_rate_knn, 4),
        },
    }


def run_diabetes():
    """Run SVM substitution on Diabetes 130-US Hospitals (UCI 296)."""
    diab_dir = _REPO_ROOT / "data" / "diabetes"
    train_path = diab_dir / "diabetes_train.csv"
    test_path = diab_dir / "diabetes_test.csv"
    if not train_path.exists() or not test_path.exists():
        print("\n  Diabetes data not found — skipping.")
        return None

    print("\n" + "=" * 70)
    print("  DIABETES 130-US HOSPITALS: SVM SUBSTITUTION EXPERIMENT")
    print("=" * 70)
    t0 = time.time()

    # Load pre-split data (same splits as run_diabetes.py)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["target"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["target"].values.astype(int)
    N = len(y_test)
    print(f"  Data: train={len(y_train):,}, test={N:,}, classes={np.unique(y_train).tolist()}")

    # Scale
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train).astype(np.float32)
    X_test = sc.transform(X_test).astype(np.float32)

    # Train RF (same config as run_diabetes.py)
    print("  Training RF(500, depth=15, leaf=5, balanced)...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test).astype(int)
    rf_proba = rf.predict_proba(X_test).astype(np.float64)
    del rf; gc.collect()

    # Train kNN
    print("  Training kNN(5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test).astype(int)
    knn_proba = knn.predict_proba(X_test).astype(np.float64)
    del knn; gc.collect()

    # Train SVM-RBF
    # Use subsample for SVM (full dataset too large)
    n_svm_train = min(50000, len(y_train))
    rng = np.random.RandomState(SEED)
    svm_idx = rng.choice(len(y_train), n_svm_train, replace=False)
    print(f"  Training SVM-RBF (C=10, subsample={n_svm_train:,})...")
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
              random_state=SEED, probability=True)
    svm.fit(X_train[svm_idx], y_train[svm_idx])
    svm_pred = svm.predict(X_test).astype(int)
    svm_proba = svm.predict_proba(X_test).astype(np.float64)
    del svm; gc.collect()

    # ARAT routing: RF + kNN
    ens_proba_knn = (rf_proba + knn_proba) / 2
    disagree_knn = rf_pred != knn_pred
    pred_a_knn = ens_proba_knn.argmax(axis=1).astype(int)
    pred_a_knn[disagree_knn] = np.maximum(rf_pred[disagree_knn], knn_pred[disagree_knn])

    # ARAT routing: RF + SVM
    ens_proba_svm = (rf_proba + svm_proba) / 2
    disagree_svm = rf_pred != svm_pred
    pred_a_svm = ens_proba_svm.argmax(axis=1).astype(int)
    pred_a_svm[disagree_svm] = np.maximum(rf_pred[disagree_svm], svm_pred[disagree_svm])

    # Metrics (diabetes is 3-class: 0,1,2 — hi_recall uses class 2)
    def diab_metrics(pred, y):
        hi = (y == 2)
        return {
            "exact": round(float((pred == y).mean()), 4),
            "under": round(float((pred < y).mean()), 4),
            "adj": round(float((pred >= y).mean()), 4),
            "hi_recall": round(float((pred[hi] == 2).sum() / max(hi.sum(), 1)), 4),
        }

    baseline = diab_metrics(pred_a_knn, y_test)
    substitution = diab_metrics(pred_a_svm, y_test)
    phi_baseline = compute_phi(rf_pred, knn_pred, y_test)
    phi_svm = compute_phi(rf_pred, svm_pred, y_test)
    disagree_rate_knn = round(float(disagree_knn.mean()), 4)
    disagree_rate_svm = round(float(disagree_svm.mean()), 4)

    print(f"\n  {'Metric':<20s} {'RF+kNN':>10s} {'RF+SVM':>10s} {'Delta':>10s}")
    print(f"  {'-'*50}")
    for k in ["exact", "under", "adj", "hi_recall"]:
        delta = substitution[k] - baseline[k]
        print(f"  {k:<20s} {baseline[k]:>10.4f} {substitution[k]:>10.4f} {delta:>+10.4f}")
    print(f"  {'phi':<20s} {phi_baseline:>10.4f} {phi_svm:>10.4f} {phi_svm - phi_baseline:>+10.4f}")
    print(f"  {'disagree_rate':<20s} {disagree_rate_knn:>10.4f} {disagree_rate_svm:>10.4f} {disagree_rate_svm - disagree_rate_knn:>+10.4f}")

    print(f"\n  Elapsed: {time.time()-t0:.0f}s")

    return {
        "dataset": "Diabetes",
        "n_train": len(y_train), "n_test": N,
        "baseline_rf_knn": {**baseline, "phi": phi_baseline, "disagree_rate": disagree_rate_knn},
        "substitution_rf_svm": {**substitution, "phi": phi_svm, "disagree_rate": disagree_rate_svm},
        "delta": {
            k: round(substitution[k] - baseline[k], 4) for k in baseline
        } | {
            "phi": round(phi_svm - phi_baseline, 4),
            "disagree_rate": round(disagree_rate_svm - disagree_rate_knn, 4),
        },
    }


def main():
    t0 = time.time()
    results = {}

    # UNSW-NB15
    results["unsw_nb15"] = run_unsw()

    # Diabetes
    diab = run_diabetes()
    if diab:
        results["diabetes"] = diab

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: PRODUCTIVE DISAGREEMENT ASSESSMENT")
    print("=" * 70)
    for ds_key, ds in results.items():
        name = ds["dataset"]
        phi_b = ds["baseline_rf_knn"]["phi"]
        phi_s = ds["substitution_rf_svm"]["phi"]
        dis_b = ds["baseline_rf_knn"]["disagree_rate"]
        dis_s = ds["substitution_rf_svm"]["disagree_rate"]
        und_b = ds["baseline_rf_knn"]["under"]
        und_s = ds["substitution_rf_svm"]["under"]
        print(f"\n  {name}:")
        print(f"    phi:       {phi_b:.4f} -> {phi_s:.4f} (delta={phi_s-phi_b:+.4f})")
        print(f"    disagree:  {dis_b:.4f} -> {dis_s:.4f} (delta={dis_s-dis_b:+.4f})")
        print(f"    under:     {und_b:.4f} -> {und_s:.4f} (delta={und_s-und_b:+.4f})")
        if phi_s < phi_b:
            print(f"    -> LOWER phi: errors are LESS correlated (productive disagreement)")
        else:
            print(f"    -> HIGHER phi: errors are MORE correlated (convergent)")

    # Save
    out_path = OUT_DIR / "svm_substitution_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
