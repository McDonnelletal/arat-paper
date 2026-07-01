# LLM committee under ARAT routing (TCGA cancer staging)

Reproducibility code for the paper's frontier-LLM transfer result: the same ARAT
routing that arbitrates the RF + k-NN committee is applied **unchanged** to a
cross-family pair of large language models, to test whether the under-prediction
reduction carries over from classical learners to frontier models.

**Task.** Assign the overall AJCC cancer stage (I-IV) to 6,112 TCGA pathology
reports, with the explicit stage / TNM codes redacted from the text so each model
must synthesise the stage from the findings. Here under-prediction is
**under-staging** (predicting an earlier stage than the truth), which maps to
under-treatment, the dangerous error.

**Committee.** Two instruction-tuned LLMs from different families (Qwen2.5-72B and
Llama-3.1-70B). Each emits an ordinal stage distribution; the two distributions go
through the identical routing: soft vote, conservative override, unanimous
lowest-stage safety flag.

## Reproduce the committee table (CPU, no GPU)

The per-instance soft-probabilities are committed, so the table regenerates offline
in seconds:

```bash
python src/llm_committee/fill_table.py \
    --large results/llm_committee/TCGAStage_crossfamily_large.csv
```

This prints the soft-vote, override, and override-plus-flag under-staging rates
(16.52% to 13.87% to 12.12%), the phi / disagreement statistics, and the
monotonicity check.

## Files

| File | Role |
|------|------|
| `route.py` | The classical ARAT routing (soft vote, conservative override, unanimous lowest-stage flag), applied to the LLM soft-probs. Pure NumPy, CPU. |
| `fill_table.py` | Fill the committee table from the saved soft-probs. CPU. |
| `prep_tcga_stage.py` | Build the de-leaked report-to-stage dataset from TCGA-Reports + TCGA-CDR. Standalone, CPU. |

## Data

`results/llm_committee/TCGAStage_crossfamily_large.csv` holds one row per
report: the true stage (`target`, ordinal 0-3), each agent's hard prediction
(`A_pred`, `B_pred`), and each agent's 4-class probability vector (`A_p0..A_p3`,
`B_p0..B_p3`). No report text or identifiers are included.

## How the soft-probs were produced

The two LLMs (Qwen2.5-72B and Llama-3.1-70B, each tensor-parallel across 4 A100s)
were run over the de-leaked reports with vLLM through the project's standard
inference path, emitting a per-instance ordinal stage distribution per agent. Those
vectors are the committed CSV above; the routing and table are then reproduced
offline by `route.py` / `fill_table.py`, so no GPU is needed to regenerate the
paper's numbers. To rebuild the dataset from source, run `prep_tcga_stage.py` with
the TCGA-CDR supplement (see `../../data/README.md`).

## Method notes

- **Severity** is ordinal `0..K-1` (higher = more severe); **under-prediction** = `pred < y`.
- **Soft vote** = argmax of the averaged probability vectors.
- **Conservative override** = element-wise max of the two argmax predictions (more
  severe on disagreement; identity on agreement).
- **Safety flag** = both agents predict the lowest class AND the merged-distribution
  Shannon entropy (bits) exceeds theta (default 0.10), surfacing the case for review.
  This matches the entropy base of the classical `run_unsw.py` flag, so the routing
  is identical to the classical committee.
