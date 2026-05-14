"""
Download and prepare datasets for ARAT experiments.

UNSW-NB15: Downloads from the official UNSW Research repository.
Diabetes:  Downloads from UCI ML Repository and applies preprocessing.

Usage:
    python data/fetch_data.py
"""

import os
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
UNSW_DIR = ROOT / "unsw_nb15"
DIABETES_DIR = ROOT / "diabetes"


def fetch_unsw_nb15():
    """Download UNSW-NB15 official train/test CSVs."""
    UNSW_DIR.mkdir(exist_ok=True)

    train_file = UNSW_DIR / "UNSW_NB15_training-set.csv"
    test_file = UNSW_DIR / "UNSW_NB15_testing-set.csv"

    if train_file.exists() and test_file.exists():
        print(f"  UNSW-NB15 already present: {train_file.stat().st_size:,} bytes")
        return

    # NOTE: The official UNSW-NB15 dataset requires manual download from:
    # https://research.unsw.edu.au/projects/unsw-nb15-dataset
    #
    # If automated download fails, manually place these files in data/unsw_nb15/:
    #   - UNSW_NB15_training-set.csv (175,341 rows)
    #   - UNSW_NB15_testing-set.csv  (82,332 rows)

    print("  UNSW-NB15: Attempting download...")
    base_url = "https://research.unsw.edu.au/sites/default/files/documents"
    for fname in ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]:
        url = f"{base_url}/{fname}"
        dest = UNSW_DIR / fname
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"    Downloaded: {fname}")
        except Exception as e:
            print(f"    FAILED: {fname} ({e})")
            print(f"    Please download manually from the UNSW website.")
            print(f"    See data/README.md for instructions.")
            return

    # Verify row counts
    tr = pd.read_csv(train_file)
    te = pd.read_csv(test_file)
    assert len(tr) == 175341, f"Training set: expected 175,341 rows, got {len(tr)}"
    assert len(te) == 82332, f"Testing set: expected 82,332 rows, got {len(te)}"
    print(f"  UNSW-NB15 verified: train={len(tr):,}, test={len(te):,}")


def fetch_diabetes():
    """Download Diabetes 130-US Hospitals dataset from UCI and preprocess."""
    DIABETES_DIR.mkdir(exist_ok=True)

    train_file = DIABETES_DIR / "diabetes_train.csv"
    test_file = DIABETES_DIR / "diabetes_test.csv"

    if train_file.exists() and test_file.exists():
        print(f"  Diabetes already present: {train_file.stat().st_size:,} bytes")
        return

    print("  Diabetes: Downloading from UCI...")
    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    zip_path = DIABETES_DIR / "diabetes_raw.zip"

    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DIABETES_DIR)
        zip_path.unlink()
        print("    Downloaded and extracted.")
    except Exception as e:
        print(f"    FAILED: {e}")
        print(f"    Please download manually. See data/README.md.")
        return

    # Preprocessing will be done by run_diabetes.py if raw files exist
    print("  Diabetes: Raw data ready. Preprocessing happens at runtime.")


if __name__ == "__main__":
    print("Fetching datasets...")
    print()
    fetch_unsw_nb15()
    print()
    fetch_diabetes()
    print()
    print("Done.")
