#!/usr/bin/env python3
"""Fill the LLM-committee table from the saved soft-probability CSV (CPU only).

Reads the per-instance soft-prob CSV and fills every row of the committee table:
soft vote, conservative override, override + safety flag, plus the phi /
disagreement / under-prediction figures and the soft-vote -> override -> flag
reconciliation.

    python fill_table.py --large results/llm_committee/TCGAStage_crossfamily_large.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route  # noqa: E402


def load_softprob_csv(path):
    rows = list(csv.DictReader(open(path)))
    K = sum(1 for c in rows[0] if c.startswith("A_p") and c[3:].isdigit())
    y = np.array([int(r["target"]) for r in rows])
    predA = np.array([int(r["A_pred"]) for r in rows])
    predB = np.array([int(r["B_pred"]) for r in rows])
    pA = np.array([[float(r[f"A_p{c}"]) for c in range(K)] for r in rows])
    pB = np.array([[float(r[f"B_p{c}"]) for c in range(K)] for r in rows])
    return y, predA, predB, pA, pB


def pct(x):
    return "N/A" if x is None else f"{x*100:.2f}\\%"


def summarise(m):
    ok = route.reconciliation_ok(m)
    print(f"\n### committee  n={m['n']}  acc_A={m['acc_A']:.3f} acc_B={m['acc_B']:.3f}")
    print(f"  soft-vote under = {m['under_soft_vote']*100:.2f}%")
    print(f"  override  under = {m['under_override']*100:.2f}%   "
          f"(-{m['reduction_override']*100:.2f}pp vs soft-vote)")
    print(f"  +flag     under = {m['under_override_flag']*100:.2f}%   "
          f"(-{m['reduction_flag']*100:.2f}pp; flag rate {m['flag_rate']*100:.1f}%, "
          f"catches {m['flag_catches']})")
    print(f"  reconciliation (monotone: soft-vote >= override >= +flag): "
          f"{'OK' if ok else 'MISMATCH'}")
    print(f"  phi={m['phi']:.4f}  disagreement={m['disagreement']*100:.2f}%")


def emit_table(m):
    """Filled tabular for the committee table."""
    print("\n" + "=" * 70)
    print("FILLED committee table:")
    print("=" * 70)
    print(rf"Soft vote (averaged probabilities) & {pct(m['under_soft_vote'])} \\")
    print(rf"ARAT (conservative override)       & {pct(m['under_override'])} \\")
    print(rf"ARAT (override + safety flag)      & \textbf{{{pct(m['under_override_flag'])}}} \\")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--large", required=True,
                    help="soft-prob CSV for the committee")
    ap.add_argument("--theta", type=float, default=0.10,
                    help="safety-flag entropy threshold (bits)")
    args = ap.parse_args()

    y, pa, pb, PA, PB = load_softprob_csv(args.large)
    metrics = route.route_metrics(y, pa, pb, PA, PB, theta=args.theta)
    summarise(metrics)
    emit_table(metrics)


if __name__ == "__main__":
    main()
