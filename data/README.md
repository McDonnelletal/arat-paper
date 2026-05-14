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
- **Preprocessing**: The fetch script applies the same preprocessing as the
  paper (feature selection, target encoding) and creates an 80/20 split
  with random_state=42.
- **Place in**: `data/diabetes/`

## Automated Download

```bash
python data/fetch_data.py
```
