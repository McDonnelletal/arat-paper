"""
ARAT — Base-learner pair sweep on UNSW-NB15 (reviewer R1 robustness response).

Pushes multiple base-learner pairs through the IDENTICAL ARAT routing
(soft vote -> conservative override on disagreement) and reports, per pair:
  phi (error correlation), disagreement rate, soft-vote under-prediction,
  ARAT (override) under-prediction, and the override-vs-soft reduction.

Generalises run_svm_substitution.py: same preprocessing (MI top-30,
StandardScaler), same routing, same SEED. Re-runs reproduce the saved
substitution / main-run numbers to within third-decimal environment drift
(library/BLAS versions shift phi by ~0.002 and the under-prediction rates by
<0.05pp between otherwise identical runs).

NOTE ON THE PAPER'S TABLE 3: the RF + k-NN (main) row in the paper cites the
MAIN-RUN artifacts for that pair (phi 0.612 / disagree 13.8% / soft-vote under
4.80% / override under 2.18% / delta +2.62pp; see
results/table2_error_dependence.csv, results/baselines_full_comparison.csv and
the unsw baseline block of results/svm_substitution_results.json) so that
Tables 1-3 of the paper are mutually consistent. This sweep's own committed
values for that pair differ from those in the third decimal (see above); the
other four pairs exist only in this sweep.

ARAT under = override-only (routing-layer output, before the unanimous-Normal
safety flag), so every pair is compared on equal footing. The full RF+kNN system
with the safety flag reaches 1.70% (Table 1).

Usage:
    python src/run_pair_sweep.py

Outputs:
    results/pair_sweep_results.json
    results/pair_sweep_table.tex   (paste-ready LaTeX)
"""

import gc
import json
import time
import warnings
import inspect as _inspect
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

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
N_SVM_TRAIN = 30000
PAIRS = [("RF", "kNN"), ("RF", "XGB"), ("RF", "LGB"), ("RF", "SVM"), ("kNN", "SVM")]


def compute_phi(pred_a, pred_b, y):
    """Phi coefficient between the two agents' error patterns (matches run_svm_substitution.py)."""
    a_wrong = (pred_a != y).astype(int)
    b_wrong = (pred_b != y).astype(int)
    n11 = int(((1 - a_wrong) & (1 - b_wrong)).sum())   # both correct
    n10 = int((a_wrong & (1 - b_wrong)).sum())          # a wrong, b correct
    n01 = int(((1 - a_wrong) & b_wrong).sum())          # a correct, b wrong
    n00 = int((a_wrong & b_wrong).sum())                 # both wrong
    denom = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return 0.0 if denom == 0 else round(float(n11 * n00 - n10 * n01) / denom, 4)


def route(proba_a, pred_a, proba_b, pred_b, y):
    """Identical ARAT routing: soft vote, then conservative override on disagreement."""
    soft = ((proba_a + proba_b) / 2).argmax(axis=1).astype(int)
    disagree = pred_a != pred_b
    override = soft.copy()
    override[disagree] = np.maximum(pred_a[disagree], pred_b[disagree])
    hi = (y == 3)
    return {
        "phi": compute_phi(pred_a, pred_b, y),
        "disagree_rate": round(float(disagree.mean()), 4),
        "under_soft": round(float((soft < y).mean()), 4),
        "under_override": round(float((override < y).mean()), 4),
        "exact_override": round(float((override == y).mean()), 4),
        "hi_recall_override": round(float((override[hi] == 3).sum() / max(hi.sum(), 1)), 4),
    }


def main():
    t0 = time.time()

    # --- Load & preprocess (identical to run_svm_substitution.py / run_unsw_baselines.py) ---
    if not (DATA_DIR / "UNSW_NB15_training-set.csv").exists():
        raise SystemExit(
            "UNSW-NB15 data not found in data/unsw_nb15/.\n"
            "Run 'python data/fetch_data.py' (manual download may be required; "
            "see data/README.md).")
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
    print(f"Data: train={len(y_train):,} test={N:,} features={len(selected)}", flush=True)

    # --- Train each base learner ONCE; keep (pred, proba) on the full test set ---
    P = {}

    print("Training RF(500)...", flush=True)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    P["RF"] = (rf.predict(X_test).astype(int), rf.predict_proba(X_test).astype(np.float64))
    del rf; gc.collect()

    print("Training kNN(5)...", flush=True)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    P["kNN"] = (knn.predict(X_test).astype(int), knn.predict_proba(X_test).astype(np.float64))
    del knn; gc.collect()

    print(f"Training SVM-RBF (C=10, balanced, {N_SVM_TRAIN:,} stratified subsample)...", flush=True)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N_SVM_TRAIN, random_state=SEED)
    svm_idx = next(sss.split(X_train, y_train))[0]
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
              random_state=SEED, probability=True)
    svm.fit(X_train[svm_idx], y_train[svm_idx])
    P["SVM"] = (svm.predict(X_test).astype(int), svm.predict_proba(X_test).astype(np.float64))
    del svm; gc.collect()

    print("Training XGBoost(500, depth=6, balanced)...", flush=True)
    xw = compute_sample_weight("balanced", y_train)
    xgm = xgb.XGBClassifier(n_estimators=500, max_depth=6, eval_metric="mlogloss",
                            random_state=SEED, n_jobs=-1, verbosity=0)
    xgm.fit(X_train, y_train, sample_weight=xw)
    P["XGB"] = (xgm.predict(X_test).astype(int), xgm.predict_proba(X_test).astype(np.float64))
    del xgm; gc.collect()

    print("Training LightGBM(500, num_leaves=127, balanced)...", flush=True)
    lgm = lgb.LGBMClassifier(num_leaves=127, n_estimators=500, class_weight="balanced",
                             random_state=SEED, n_jobs=-1, verbose=-1)
    lgm.fit(X_train, y_train)
    P["LGB"] = (lgm.predict(X_test).astype(int), lgm.predict_proba(X_test).astype(np.float64))
    del lgm; gc.collect()

    acc = {k: round(float((v[0] == y_test).mean()), 4) for k, v in P.items()}
    print("Per-agent exact accuracy:", acc, flush=True)

    # --- Sweep pairs through the identical routing ---
    results = {}
    for a, b in PAIRS:
        pred_a, proba_a = P[a]
        pred_b, proba_b = P[b]
        m = route(proba_a, pred_a, proba_b, pred_b, y_test)
        m["reduction_pp"] = round((m["under_soft"] - m["under_override"]) * 100, 2)
        m["productive"] = bool(m["under_override"] < m["under_soft"])
        results[f"{a}+{b}"] = m

    # --- Console table ---
    hdr = (f"{'Pair':<10s}{'phi':>8s}{'disagree':>10s}{'soft-under':>12s}"
           f"{'ARAT-under':>12s}{'reduce':>9s}{'prod?':>7s}")
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for pair, m in results.items():
        print(f"{pair:<10s}{m['phi']:>8.4f}{m['disagree_rate'] * 100:>9.2f}%"
              f"{m['under_soft'] * 100:>11.2f}%{m['under_override'] * 100:>11.2f}%"
              f"{m['reduction_pp']:>+8.2f} {'yes' if m['productive'] else 'NO':>6s}")

    # --- Paste-ready LaTeX ---
    name_tex = {"RF+kNN": r"RF + $k$-NN (main)", "RF+XGB": "RF + XGBoost",
                "RF+LGB": "RF + LightGBM", "RF+SVM": "RF + SVM", "kNN+SVM": r"$k$-NN + SVM"}
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Base-learner pair sweep on UNSW-NB15 ($n=82{,}332$). Every pair is"
        r" pushed through the identical ARAT routing (soft vote, then conservative"
        r" override on disagreement); under-prediction is the dangerous error. The"
        r" override reduces under-prediction whenever the agents produce productive"
        r" disagreement, and the size of the gain tracks how complementary their errors"
        r" are (lower $\phi$). ARAT under is the override-only routing layer; the full"
        r" RF\,+\,$k$-NN system with the unanimous-Normal flag reaches 1.70\%"
        r" (Table~\ref{tab:baseline}).}",
        r"\label{tab:pairsweep}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Agent pair & $\phi$ & Disagree & Soft-vote under & ARAT under & $\Delta$ \\",
        r"\midrule",
    ]
    # Rows ordered by ascending phi, matching the paper's Table 3. The paper's
    # RF + $k$-NN (main) row cites the main-run artifacts (phi 0.612 / 4.80% /
    # 2.18% / +2.62pp) for cross-table consistency; this sweep's re-run of that
    # pair differs in the third decimal (see module docstring).
    for pair, m in sorted(results.items(), key=lambda kv: kv[1]["phi"]):
        lines.append(
            f"{name_tex.get(pair, pair)} & {m['phi']:.3f} & {m['disagree_rate'] * 100:.1f}\\% "
            f"& {m['under_soft'] * 100:.2f}\\% & {m['under_override'] * 100:.2f}\\% "
            f"& ${m['reduction_pp']:+.2f}$\\,pp \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines)

    (OUT_DIR / "pair_sweep_results.json").write_text(json.dumps(
        {"dataset": "UNSW-NB15", "n_test": N, "seed": SEED,
         "svm_train_subsample": N_SVM_TRAIN, "agent_accuracy": acc,
         "note": "ARAT under = override-only (routing-layer, pre safety-flag); "
                 "full RF+kNN with unanimous-Normal flag reaches 1.70% (Table 1). "
                 "The paper's Table 3 RF+k-NN (main) row cites the MAIN-RUN artifacts "
                 "(phi 0.612 / 13.8% / 4.80% / 2.18% / +2.62pp) for cross-table "
                 "consistency; this sweep's re-run of that pair differs in the third "
                 "decimal (environment drift between otherwise identical runs).",
         "pairs": results}, indent=2))
    (OUT_DIR / "pair_sweep_table.tex").write_text(tex)
    print("\n" + tex)
    print(f"\nSaved: results/pair_sweep_results.json + results/pair_sweep_table.tex")
    print(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
