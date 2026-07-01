#!/usr/bin/env python3
"""Prep TCGA pathology reports -> cancer-stage severity task.

Builds data/tcga/tcga_stage_ordinal.csv for the LLM committee, where:
  text   = a real pathology report (gross + microscopic findings), with the
           EXPLICIT stage / TNM codes redacted so the model must SYNTHESISE the
           stage from the findings rather than read it off the page.
  target = AJCC pathologic stage, I->1 II->2 III->3 IV->4.

So under-prediction (pred < target) == UNDER-STAGING: judging a more advanced
cancer as earlier-stage -> under-treatment. That is exactly ARAT's "dangerous
under-prediction" safety metric, and Stage IV is the rare advanced-disease tail.

Two real, citable sources (no credentialing):
  * Reports : Kefeli & Tatonetti, "TCGA-Reports", Patterns (Cell Press) 2024.
              github.com/tatonetti-lab/tcga-path-reports  (TCGA_Reports.csv.zip)
  * Stage   : Liu et al., "An Integrated TCGA Pan-Cancer Clinical Data Resource"
              (TCGA-CDR), Cell 2018.  TCGA-CDR-SupplementalTableS1.xlsx, sheet
              "TCGA-CDR", columns bcr_patient_barcode + ajcc_pathologic_tumor_stage.

The stage LABEL comes from TCGA-CDR (clean ground truth); the report text is the
INPUT with stage codes removed -- so the label is never simply copied from the text.

Usage (needs openpyxl for the .xlsx):
    python3 src/llm_committee/prep_tcga_stage.py \
        --cdr /path/to/TCGA-CDR-SupplementalTableS1.xlsx
(reports are auto-cloned from GitHub; pass --reports to use a local copy.)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPORTS_REPO = "https://github.com/tatonetti-lab/tcga-path-reports.git"

# --- pure helpers (unit-testable, no I/O) ----------------------------------

_BARCODE = re.compile(r"(TCGA-[0-9A-Za-z]{2}-[0-9A-Za-z]{4})")
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}
# NB: no trailing \b -- sub-stages like "Stage IIIA"/"IVB" must map to III/IV
# (alternation is longest-first, so it never mis-splits IV/III/II/I).
_STAGE_LABEL = re.compile(r"stage\s*(IV|III|II|I)", re.IGNORECASE)

# Explicit stage / TNM codes to redact from the INPUT text. We keep the
# descriptive findings (tumour size, node counts, "metastasis", grade) -- those
# are the legitimate basis for staging; we strip only the answer-giving codes.
_REDACT = [
    re.compile(r"\b[ycrp]{0,2}t(?:is|x|[0-4][a-d]?)\s*[ycrp]?n[x0-3][a-c]?\s*"
               r"[ycrp]?m[x0-1][a-c]?\b", re.IGNORECASE),                # pT3N1M0, ypT2aN0M0
    re.compile(r"\b(?:ajcc\s+)?(?:pathologic(?:al)?\s+|clinical\s+|overall\s+|"
               r"final\s+|tumou?r\s+)?stage\s*[:\-]?\s*(?:0|IV|III|II|I|[0-4])"
               r"\s*[A-C]?\d?\b", re.IGNORECASE),                        # (AJCC) Stage IIIA
    re.compile(r"\b[ycr]?p[TNM](?:is|x|[0-4][a-d]?)\b", re.IGNORECASE),  # pT3, ypN1, pM1
]


def extract_barcode(value) -> str | None:
    """First 12-char TCGA patient barcode found in a filename/id string."""
    if not isinstance(value, str):
        return None
    m = _BARCODE.search(value)
    return m.group(1).upper() if m else None


def stage_to_ordinal(raw) -> int | None:
    """AJCC stage string -> 1..4 (I..IV). None for stage 0/X, ambiguous, or missing."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper()
    if not s or "NOS" in s or "/" in s:          # ambiguous combined stages -> drop
        return None
    m = _STAGE_LABEL.search(s) or re.match(r"(IV|III|II|I)", s)
    return _ROMAN.get(m.group(1).upper()) if m else None


def deleak_text(text: str) -> str:
    """Remove explicit stage / TNM codes; collapse whitespace. Findings are kept."""
    if not isinstance(text, str):
        return ""
    for pat in _REDACT:
        text = pat.sub(" [stage redacted] ", text)
    return re.sub(r"\s+", " ", text).strip()


def _find(cols, *needles, exact=None):
    low = {c.lower(): c for c in cols}
    if exact and exact.lower() in low:
        return low[exact.lower()]
    for c in cols:
        if needles and all(n in c.lower() for n in needles):
            return c
    return None


# --- I/O -------------------------------------------------------------------

def fetch_reports(cache: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    repo = cache / "tcga-path-reports"
    if not repo.exists():
        print(f"git clone {REPORTS_REPO} -> {repo}")
        subprocess.run(["git", "clone", "--depth", "1", REPORTS_REPO, str(repo)], check=True)
    for name in ("TCGA_Reports.csv.zip", "TCGA_Reports.csv"):
        if (repo / name).exists():
            return repo / name
    hits = sorted(repo.glob("**/*TCGA_Reports*.csv*"))
    if hits:
        return hits[0]
    sys.exit(f"FATAL: TCGA_Reports.csv(.zip) not found under {repo}")


def load_reports(path: Path) -> pd.DataFrame:
    if str(path).endswith(".zip"):
        # the published zip carries macOS junk (__MACOSX/._*), so pandas' own
        # zip handling refuses it ("multiple files"); pick the real CSV ourselves.
        import zipfile
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist()
                       if n.lower().endswith(".csv")
                       and not n.startswith("__MACOSX")
                       and not Path(n).name.startswith("._")]
            if not members:
                sys.exit(f"FATAL: no usable .csv inside {path} (members: {z.namelist()})")
            with z.open(members[0]) as fh:
                df = pd.read_csv(fh)
    else:
        df = pd.read_csv(path)
    print(f"reports: {len(df)} rows; columns: {list(df.columns)}")
    text_col = _find(df.columns, exact="text") or _find(df.columns, "report") or _find(df.columns, "note")
    id_col = (_find(df.columns, exact="patient_filename") or _find(df.columns, "patient")
              or _find(df.columns, "filename") or _find(df.columns, "barcode") or _find(df.columns, "id"))
    if text_col is None or id_col is None:
        sys.exit(f"FATAL: report text/id columns not found (text={text_col} id={id_col}); cols={list(df.columns)}")
    out = pd.DataFrame({"barcode": df[id_col].map(extract_barcode),
                        "text": df[text_col].astype(str)})
    return out.dropna(subset=["barcode"])


def load_cdr(path: Path) -> pd.DataFrame:
    try:
        cdr = pd.read_excel(path, sheet_name="TCGA-CDR")
    except ImportError:
        sys.exit("FATAL: reading the .xlsx needs openpyxl -> `pip install openpyxl`")
    bc = _find(cdr.columns, exact="bcr_patient_barcode") or _find(cdr.columns, "barcode")
    st = _find(cdr.columns, exact="ajcc_pathologic_tumor_stage") or _find(cdr.columns, "stage")
    if bc is None or st is None:
        sys.exit(f"FATAL: CDR barcode/stage columns not found (barcode={bc} stage={st}); cols={list(cdr.columns)}")
    out = pd.DataFrame({"barcode": cdr[bc].astype(str).str.upper(),
                        "stage_ord": cdr[st].map(stage_to_ordinal)})
    return out.dropna(subset=["stage_ord"]).astype({"stage_ord": int})


# --- orchestration ---------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdr", required=True,
                    help="path to TCGA-CDR-SupplementalTableS1.xlsx (Liu et al. 2018)")
    ap.add_argument("--reports", default=None, help="local TCGA_Reports.csv(.zip); else auto-cloned")
    ap.add_argument("--cache", default="/tmp/tcga_cache", help="clone cache (off /home quota)")
    ap.add_argument("--n", type=int, default=12000, help="max rows to write (shuffled)")
    ap.add_argument("--out", default=str(REPO / "data/tcga/tcga_stage_ordinal.csv"))
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="truncate de-leaked report text to fit the 4096-token context "
                         "(front-keep; truncating rather than dropping avoids biasing away "
                         "from the longer, more-advanced Stage IV reports)")
    args = ap.parse_args()

    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    reports = load_reports(fetch_reports(cache, args.reports))
    cdr = load_cdr(Path(args.cdr))
    print(f"CDR: {len(cdr)} patients with a clean AJCC stage (I-IV)")

    df = reports.merge(cdr, on="barcode", how="inner")
    df["text"] = df["text"].map(deleak_text).str.slice(0, args.max_chars)
    df = df[df["text"].str.len() > 30].drop_duplicates(subset=["barcode"]).copy()
    df["target"] = df["stage_ord"]

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)   # shuffle (iloc[:N] = random)
    if len(df) > args.n:
        df = df.iloc[: args.n].copy()
    out = pd.DataFrame({"text": df["text"], "target": df["target"]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nwrote {len(out)} reports -> {args.out}")
    print("stage distribution (4 = Stage IV metastatic = most severe; under-pred = under-staging):")
    for k, v in (out["target"].value_counts(normalize=True).sort_index() * 100).round(1).items():
        print(f"  target {k} (Stage {'I'*k if k < 4 else 'IV'}): {v}%")
    ex = out["text"].iloc[0]
    print(f"\nde-leak check -- example input the model sees (stage codes redacted):\n  {ex[:300]}...")
    print("\nnext: run the two-LLM committee over these reports through your "
          "vLLM inference pipeline to produce per-instance soft-probs, then fill "
          "the table with fill_table.py (see src/llm_committee/README.md).")


if __name__ == "__main__":
    main()
