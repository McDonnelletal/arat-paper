# Data

Raw datasets are not included in this repository. Run `fetch_data.py` to
download them automatically, or follow the manual instructions below.

## UNSW-NB15

Official network intrusion detection dataset by Moustafa & Slay (2015).

- **Source**: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Files needed**: `UNSW_NB15_training-set.csv` (175,341 rows) and
  `UNSW_NB15_testing-set.csv` (82,332 rows)
- **Place in**: `data/unsw_nb15/`

The official split uses the training CSV (175,341) for training and the
testing CSV (82,332) as the holdout evaluation set.

## Diabetes Readmission

UCI Machine Learning Repository dataset (Strack et al., 2014).

- **Source**: https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008
- **Place in**: `data/diabetes/`
- **Reproducibility**: the pre-encoded 21-feature split
  (`diabetes_train.csv` / `diabetes_test.csv`, each with a `target` column:
  0 = No readmission, 1 = >30 days, 2 = <30 days) is **bundled in this repo**
  (UCI 296 is CC BY, so redistribution is permitted), so `run_diabetes.py`,
  `run_diabetes_ablation.py` (Table 2 Diabetes columns), and the Diabetes arm of
  `run_svm_substitution.py` run directly. `fetch_data.py` also fetches the raw
  `diabetic_data.csv` for reference. The train/test split is a fixed 80/20 split
  (not stratified): the test set is class-balance-shifted (60.4/28.9/10.6% vs
  train 52.3/36.4/11.3%), which additionally exercises robustness to
  distribution shift.

## Automated Download

```bash
python data/fetch_data.py
```

## TCGA pathology reports (LLM committee)

The cross-family LLM committee (`src/llm_committee/`) stages TCGA pathology
reports. Build the dataset with:

```bash
python src/llm_committee/prep_tcga_stage.py \
    --cdr /path/to/TCGA-CDR-SupplementalTableS1.xlsx
```

- **Reports**: Kefeli & Tatonetti, *TCGA-Reports*, Patterns (2024) — auto-cloned by the script.
- **Stage labels**: Liu et al., *TCGA-CDR*, Cell (2018) — provide `TCGA-CDR-SupplementalTableS1.xlsx`.
- **Output**: `data/tcga/tcga_stage_ordinal.csv` (de-leaked report text + ordinal AJCC stage).
