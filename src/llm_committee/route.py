#!/usr/bin/env python3
"""Classical ARAT routing functions, applied to LLM agents.

These mirror Section 4 (ARAT Architecture) of the paper EXACTLY; the two
LLM agents simply replace RF + k-NN as the belief-state producers. Pure NumPy,
no GPU -- operates on the per-instance soft-prob CSVs.

Severity is ordinal 0..K-1 (higher = more severe). Under-prediction = pred < y
(predicted a lower severity than the truth) -- the primary safety metric.

Routing layers (paper Section 4, RA_3):
  * Soft vote (naive baseline): argmax of the element-wise average of the two
    agents' probability vectors.
  * Conservative override: on disagreement emit the MORE SEVERE prediction
    (max of the two argmax classes). Eliminates disagreement-based
    under-predictions by construction.
  * Unanimous-Normal safety flag: when BOTH agents predict Normal (class 0) AND
    the merged-distribution entropy exceeds theta, the case is surfaced for
    mandatory analyst review -- so it is no longer a *silent* under-prediction.

Entropy units: scipy Shannon entropy in BITS (base 2), matching the classical
flag in run_unsw.py. theta defaults to 0.10 as in the paper.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import entropy as sp_entropy


def merged_probs(pA: np.ndarray, pB: np.ndarray) -> np.ndarray:
    """Element-wise average of the two agents' probability distributions."""
    return 0.5 * (pA + pB)


def soft_vote_pred(pA: np.ndarray, pB: np.ndarray) -> np.ndarray:
    """Naive baseline: argmax of the averaged probability vectors."""
    return merged_probs(pA, pB).argmax(axis=1)


def conservative_override_pred(predA: np.ndarray, predB: np.ndarray) -> np.ndarray:
    """On disagreement take the more severe class; identity on agreement.
    For ordinal severity (higher index = more severe) this is the element-wise
    max of the two agents' argmax predictions."""
    return np.maximum(predA, predB)


def unanimous_normal_flag(predA: np.ndarray, predB: np.ndarray,
                          pA: np.ndarray, pB: np.ndarray,
                          theta: float = 0.10) -> np.ndarray:
    """Both agents predict Normal (class 0) AND merged entropy > theta -> flag
    for mandatory review (paper Section 4)."""
    H = sp_entropy(merged_probs(pA, pB) + 1e-12, base=2, axis=1)   # bits, matching run_unsw.py
    return (predA == 0) & (predB == 0) & (H > theta)


def under_pred_rate(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred < y).mean())


def phi_coefficient(eA: np.ndarray, eB: np.ndarray):
    """Phi (mean-square-contingency) between two binary error vectors."""
    eA = eA.astype(bool); eB = eB.astype(bool)
    n11 = int((eA & eB).sum()); n10 = int((eA & ~eB).sum())
    n01 = int((~eA & eB).sum()); n00 = int((~eA & ~eB).sum())
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    phi = (n11 * n00 - n10 * n01) / denom if denom else float("nan")
    return phi, (n11, n10, n01, n00)


def route_metrics(y, predA, predB, pA, pB, theta: float = 0.10) -> dict:
    """All routing quantities for one committee at one scale."""
    n = len(y)
    sv = soft_vote_pred(pA, pB)
    ovr = conservative_override_pred(predA, predB)
    flag = unanimous_normal_flag(predA, predB, pA, pB, theta=theta)

    under_sv = under_pred_rate(sv, y)
    under_ovr = under_pred_rate(ovr, y)
    # override + flag: an override under-prediction that the flag surfaces for
    # review is no longer a silent miss.
    ovr_under_mask = ovr < y
    flag_catches = int((ovr_under_mask & flag).sum())
    under_ovr_flag = float((ovr_under_mask & ~flag).mean())

    eA = predA != y; eB = predB != y
    phi, cells = phi_coefficient(eA, eB)
    disagreement = float((predA != predB).mean())
    agreed_under = float(((predA == predB) & (predA < y)).mean())

    return {
        "n": n,
        "under_soft_vote": under_sv,
        "under_override": under_ovr,
        "under_override_flag": under_ovr_flag,
        "reduction_override": under_sv - under_ovr,          # soft-vote -> override
        "reduction_flag": under_ovr - under_ovr_flag,        # override  -> +flag
        "flag_rate": float(flag.mean()),
        "flag_catches": flag_catches,
        "phi": phi, "phi_cells": cells,
        "disagreement": disagreement,
        "agreed_under": agreed_under,
        "acc_A": float((predA == y).mean()),
        "acc_B": float((predB == y).mean()),
    }


def reconciliation_ok(m: dict, tol: float = 1e-9) -> bool:
    """Classical-headline reconciliation. The decomposition
        under(soft-vote) - reduction(override) - reduction(flag) = under(override+flag)
    is an arithmetic identity; the MEANINGFUL property it encodes is that each
    safeguard only *removes* under-predictions, i.e. the layers are monotone
    non-increasing (soft-vote >= override >= override+flag). A negative reduction
    would mean a 'safeguard' made safety worse, which must not happen."""
    return m["reduction_override"] >= -tol and m["reduction_flag"] >= -tol
