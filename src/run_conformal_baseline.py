"""
ARAT — Single-model uncertainty-quantification baseline (reviewer R3).

Benchmarks ARAT's multi-agent routing against standard single-model UQ at MATCHED
human-review budgets, to show single-model UQ cannot recover the cross-agent
correlated failures ARAT targets:

  (1) Maximum-softmax-probability selective prediction (MSP; Hendrycks & Gimpel 2017,
      Geifman & El-Yaniv 2017) on RF and LightGBM (fit on full train, so RF reproduces
      Table 1's 3.76% base under-prediction as a correctness check).
  (2) Split-conformal Adaptive Prediction Sets (APS; Romano et al. 2020) on RF
      (fit on a proper-train split, calibrated on a held-out split).

For each method we escalate the most-uncertain B% of test cases to human review
(B in {6.79%, 15%}, matching ARAT's flag rate and the Section 6.3 budget) and report:
  - residual under-prediction over the FULL test set (escalated cases assumed resolved,
    directly comparable to ARAT's 1.70%);
  - dangerous-recall@budget: the fraction of the model's under-predictions the
    escalation captures (compare ARAT's 94.3% at the 15% budget).

The ARAT reference cells come from run_unsw.py's held-out escalation-router
block (3-seed means; see the "escalation" section of results/all_results.json):
recall at both budgets uses the router-ordered queue, and the 15%-budget
residual under-prediction applies the FULL routing (override + mandatory
safety flag + router top-up to 15%, escalated cases assumed resolved - the
same semantics as the UQ baselines' residual-under column).

Usage:
    python src/run_conformal_baseline.py
Outputs:
    results/uq_baseline_results.json
    results/uq_baseline_table.tex   (paste-ready LaTeX)
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
from sklearn.model_selection import StratifiedShuffleSplit
import lightgbm as lgb

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
BUDGETS = [0.0679, 0.15]
ALPHA = 0.10  # conformal target miscoverage (90% sets) for the sanity check
# ARAT reference points from the paper (full system / Section 6.3):
# From run_unsw.py's held-out escalation block (3-seed means): recall at the
# 6.79% and 15% budgets, and full-routing residual under at 15% (paper Table 4).
ARAT_REF = {"softvote_under": 0.0480, "under_at_6.79pct": 0.0170,
            "recall_at_6.79pct": 0.7450, "under_at_15pct": 0.0021,
            "recall_at_15pct": 0.9432}


def load_unsw():
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
    return X_train, y_train, X_test, y_test


def selective_metrics(pred, y, u):
    """Escalate the top-B% most-uncertain (largest u) cases; report residual under
    (over all test, escalated assumed resolved) and dangerous-recall@budget."""
    N = len(y)
    under = pred < y
    tot_under = int(under.sum())
    order = np.argsort(-u)  # most uncertain first
    base_under = round(float(under.mean()), 4)
    out = {"base_under": base_under, "total_under": tot_under, "budgets": {}}
    for B in BUDGETS:
        n_esc = int(round(B * N))
        esc = np.zeros(N, bool)
        esc[order[:n_esc]] = True
        out["budgets"][f"{B:.4f}"] = {
            "budget": B,
            "n_escalated": n_esc,
            "residual_under": round(float((under & ~esc).sum()) / N, 4),
            "dangerous_recall": round(float((under & esc).sum()) / max(tot_under, 1), 4),
        }
    return out


def aps_calibration_q(proba_cal, y_cal, alpha):
    """Non-randomised APS calibration quantile."""
    order = np.argsort(-proba_cal, axis=1)
    sorted_p = np.take_along_axis(proba_cal, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    rank_true = (proba_cal > proba_cal[np.arange(len(y_cal)), y_cal][:, None]).sum(axis=1)
    E = csum[np.arange(len(y_cal)), rank_true]
    n = len(y_cal)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(E, level, method="higher"))


def aps_apply(proba_test, y_test, q):
    """Return per-test APS set size, coverage, and an uncertainty score for ranking."""
    order = np.argsort(-proba_test, axis=1)
    sorted_p = np.take_along_axis(proba_test, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    K = proba_test.shape[1]
    set_size = np.minimum((csum < q).sum(axis=1) + 1, K)
    rank_true = (proba_test > proba_test[np.arange(len(y_test)), y_test][:, None]).sum(axis=1)
    covered = rank_true < set_size
    maxprob = proba_test.max(axis=1)
    # uncertainty for selective prediction: larger set first, ties broken by lower confidence
    u = set_size.astype(float) + (1.0 - maxprob)
    return set_size, float(covered.mean()), u


def main():
    t0 = time.time()
    X_train, y_train, X_test, y_test = load_unsw()
    N = len(y_test)
    print(f"Data: train={len(y_train):,} test={N:,}", flush=True)

    results = {"dataset": "UNSW-NB15", "n_test": N, "seed": SEED, "arat_reference": ARAT_REF, "methods": {}}

    # --- MSP selective prediction: RF and LightGBM, fit on FULL train ---
    print("Training RF(500) on full train...", flush=True)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test).astype(np.float64)
    rf_pred = rf_proba.argmax(1).astype(int)
    results["methods"]["RF + MSP selective"] = selective_metrics(
        rf_pred, y_test, 1.0 - rf_proba.max(1))
    del rf; gc.collect()

    print("Training LightGBM(500) on full train...", flush=True)
    lgm = lgb.LGBMClassifier(num_leaves=127, n_estimators=500, class_weight="balanced",
                             random_state=SEED, n_jobs=-1, verbose=-1)
    lgm.fit(X_train, y_train)
    lgb_proba = lgm.predict_proba(X_test).astype(np.float64)
    lgb_pred = lgb_proba.argmax(1).astype(int)
    results["methods"]["LightGBM + MSP selective"] = selective_metrics(
        lgb_pred, y_test, 1.0 - lgb_proba.max(1))
    del lgm; gc.collect()

    # --- Split-conformal APS on RF (proper-train / calibration split) ---
    print("Split-conformal APS on RF (80/20 train/calibration split)...", flush=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, cal_idx = next(sss.split(X_train, y_train))
    rf_c = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_c.fit(X_train[tr_idx], y_train[tr_idx])
    proba_cal = rf_c.predict_proba(X_train[cal_idx]).astype(np.float64)
    proba_te = rf_c.predict_proba(X_test).astype(np.float64)
    pred_te = proba_te.argmax(1).astype(int)
    q = aps_calibration_q(proba_cal, y_train[cal_idx], ALPHA)
    set_size, coverage, u_conf = aps_apply(proba_te, y_test, q)
    conf = selective_metrics(pred_te, y_test, u_conf)
    conf["conformal_coverage"] = round(coverage, 4)
    conf["mean_set_size"] = round(float(set_size.mean()), 3)
    conf["frac_ambiguous_set_ge_2"] = round(float((set_size >= 2).mean()), 4)
    conf["alpha"] = ALPHA
    results["methods"]["RF + split-conformal (APS)"] = conf
    del rf_c; gc.collect()

    # --- Console report ---
    print("\n" + "=" * 88)
    print(f"{'Method':<30s}{'base-under':>11s}{'budget':>9s}{'resid-under':>13s}{'danger-recall':>15s}")
    print("-" * 88)
    for name, m in results["methods"].items():
        for bkey, b in m["budgets"].items():
            print(f"{name:<30s}{m['base_under']*100:>10.2f}%{b['budget']*100:>8.2f}%"
                  f"{b['residual_under']*100:>12.2f}%{b['dangerous_recall']*100:>14.1f}%")
    print("-" * 88)
    print(f"{'ARAT (paper)':<30s}{ARAT_REF['softvote_under']*100:>10.2f}%{6.79:>8.2f}%"
          f"{ARAT_REF['under_at_6.79pct']*100:>12.2f}%{ARAT_REF['recall_at_6.79pct']*100:>14.1f}%")
    print(f"{'ARAT (paper)':<30s}{'':>11s}{15.00:>8.2f}%"
          f"{ARAT_REF['under_at_15pct']*100:>12.2f}%{ARAT_REF['recall_at_15pct']*100:>14.1f}%")
    cm = results["methods"]["RF + split-conformal (APS)"]
    print(f"\nConformal sanity: coverage={cm['conformal_coverage']:.3f} (target {1-ALPHA:.2f}), "
          f"mean set size={cm['mean_set_size']}, ambiguous (>=2)={cm['frac_ambiguous_set_ge_2']*100:.1f}%")

    # --- Paste-ready LaTeX ---
    def row(name, m):
        b1 = m["budgets"][f"{BUDGETS[0]:.4f}"]
        b2 = m["budgets"][f"{BUDGETS[1]:.4f}"]
        return (f"{name} & {m['base_under']*100:.2f}\\% & {b1['residual_under']*100:.2f}\\% "
                f"& {b1['dangerous_recall']*100:.1f}\\% & {b2['residual_under']*100:.2f}\\% "
                f"& {b2['dangerous_recall']*100:.1f}\\% \\\\")
    L = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
        r"\caption{Single-model uncertainty quantification versus ARAT at matched review"
        r" budgets on UNSW-NB15. Each UQ method escalates its most-uncertain cases to human"
        r" review; residual under-prediction is over the full test set (escalated cases"
        r" assumed resolved, comparable to ARAT's 1.70\%) and recall is the fraction of the"
        r" model's under-predictions the escalation captures. At the 6.79\% budget no"
        r" single-model UQ matches ARAT's 1.70\% residual under-prediction, because"
        r" confident correlated errors carry low uncertainty and are escalated last;"
        r" ARAT's deterministic override removes them without spending review budget.}",
        r"\label{tab:uq}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"& & \multicolumn{2}{c}{Budget $6.79\%$} & \multicolumn{2}{c}{Budget $15\%$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Method & Base under & Resid.\ under & Recall & Resid.\ under & Recall \\",
        r"\midrule",
        row("RF + MSP selective", results["methods"]["RF + MSP selective"]),
        row("LightGBM + MSP selective", results["methods"]["LightGBM + MSP selective"]),
        row("RF + split-conformal (APS)", results["methods"]["RF + split-conformal (APS)"]),
        r"\midrule",
        f"ARAT (full routing) & {ARAT_REF['softvote_under']*100:.2f}\\% "
        f"& \\textbf{{{ARAT_REF['under_at_6.79pct']*100:.2f}\\%}} "
        f"& \\textbf{{{ARAT_REF['recall_at_6.79pct']*100:.1f}\\%}} "
        f"& \\textbf{{{ARAT_REF['under_at_15pct']*100:.2f}\\%}} "
        f"& \\textbf{{{ARAT_REF['recall_at_15pct']*100:.1f}\\%}} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    tex = "\n".join(L)
    (OUT_DIR / "uq_baseline_results.json").write_text(json.dumps(results, indent=2))
    (OUT_DIR / "uq_baseline_table.tex").write_text(tex)
    print("\n" + tex)
    print(f"\nSaved: results/uq_baseline_results.json + results/uq_baseline_table.tex")
    print(f"Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
