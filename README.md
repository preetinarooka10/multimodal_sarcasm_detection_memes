# Sarcasm Detection Repository (Notebook-Aligned Package)

This package is aligned to the uploaded final executed notebook:

`notebooks/01_memotion_train_in_domain.ipynb`


## Why this package was revised

The uploaded final notebook uses the text model string:

`ai4bharat/indic-bert`

The scripts in this folder are aligned to that executed notebook implementation. For consistency, the manuscript, README, BACKBONE_NOTE, and reviewer-facing notes should all use the same backbone description:
- IndicBERT (`ai4bharat/indic-bert`)

## Frozen-project policy

This package is designed for the current reviewer-response stage, where the safest policy is:

- do not rerun long end-to-end training unless absolutely necessary
- keep the final executed notebook as the primary evidence file
- treat split artifacts as frozen once the manuscript is fixed
- do not claim outputs from supplementary scripts unless they were actually executed
- do not silently mix notebook versions in the repo narrative

## Repo layout

```text
repo/
├── README.md
├── BACKBONE_NOTE.txt
├── notebooks/
│   └── 01_memotion_train_in_domain.ipynb
├── scripts/
│   ├── 01_audit_notebook_alignment.py
│   ├── 02_export_repo_metadata.py
│   ├── 03_build_memotion_subset_and_splits.py
│   ├── 04_bootstrap_ci_from_predictions.py
│   ├── 05_fusion_baselines_from_branch_probs.py
│   ├── 06_seed_summary_from_run_dirs.py
│   └── 07_mustard_paths_and_keyframe_audit.py
└── artifacts/
    ├── splits/
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   ├── full_binary_subset.csv
    │   └── subset_summary.json
    └── repo_metadata/
        ├── notebook_alignment_summary.json
        ├── repo_metadata.json
        └── BACKBONE_NOTE.txt
```

## What each script does

### 01_audit_notebook_alignment.py
Read-only notebook audit.

Use this first if you want a machine-readable summary of:
- actual text backbone in code
- visual backbone names
- split style and seed
- output artifact filenames detected in the notebook

This script does not train or evaluate anything.

### 02_export_repo_metadata.py
Generate clean metadata files for release preparation:
- `repo_metadata.json`
- `BACKBONE_NOTE.txt`

Use this after Script 01, and optionally pass the frozen `subset_summary.json`.

### 03_build_memotion_subset_and_splits.py
Notebook-aligned split builder.

This version mirrors the final notebook more closely than the older lightweight reviewer script:
- robust image-column inference
- robust sarcasm-label inference
- text column preference `text_corrected -> text_ocr`
- 70/15/15 stratified split with seed 42 by default

Important:
if your manuscript is already frozen with approved split files, do not regenerate them.

### 04_bootstrap_ci_from_predictions.py
Bootstrap confidence intervals from saved notebook predictions.

Supports notebook-style prediction files such as:
- `predictions.csv`
- `predictions_final_eval_fixed.csv`

Typical use:
- `--prob_col p_multi` for the main multimodal output
- `--prob_col p_ens` for the ensemble output

### 05_fusion_baselines_from_branch_probs.py
Evaluate simple non-GWO fusion baselines from branch probabilities.

Required columns:
- `y_true`
- `p_text`
- `p_image`
- `p_emoji`
- `p_multi` or `p_multimodal_branch`

Important:
the notebook's default `predictions.csv` does not include all branch probabilities.  
So this script needs a richer export file.

### 06_seed_summary_from_run_dirs.py
Aggregate notebook-style metrics across multiple run folders.

Supported JSON targets:
- `metrics.json`
- `final_eval_fixed.json`
- `best_metrics.json`
- `mustard_metrics.json`

Use this only if you genuinely have multiple valid runs.

### 07_mustard_paths_and_keyframe_audit.py
Audit MUStARD paths and optionally build a notebook-aligned evaluation CSV.

This script matches the final notebook's MUStARD assumptions:
- utterance clips under `raw_videos/mmsd_raw_data/utterances_final`
- annotations from `data/sarcasm_data.json`
- keyframes under `_keyframes_utt`
- evaluation CSV under `_eval/mustard_eval.csv`

By default it performs only an audit.  
Add `--build_eval_csv` if you want the CSV generated.

## Suggested usage order

### A. Release-alignment only
Use this when you are preparing the repo but not rerunning experiments.

```bash
python scripts/01_audit_notebook_alignment.py \
  --notebook notebooks/01_memotion_train_in_domain.ipynb \
  --out_json artifacts/repo_metadata/notebook_alignment_summary.json

python scripts/02_export_repo_metadata.py \
  --notebook_alignment_json artifacts/repo_metadata/notebook_alignment_summary.json \
  --subset_summary artifacts/splits/subset_summary.json \
  --out_dir artifacts/repo_metadata
```

### B. Split transparency only
Use only if you must rebuild splits from the raw Memotion CSV.

```bash
python scripts/03_build_memotion_subset_and_splits.py \
  --input_csv /path/to/labels.csv \
  --output_dir artifacts/splits \
  --notebook_name "sarcasm_9_localgpu_updated_v4 - Copy (2).ipynb"
```

### C. Supplementary statistical reporting
Use only on already-saved prediction files.

```bash
python scripts/04_bootstrap_ci_from_predictions.py \
  --pred_csv runs/best_run/predictions.csv \
  --prob_col p_multi \
  --out_json runs/best_run/ci_multi.json
```

### D. Simple fusion baselines
Use only if you have a richer branch-probability CSV.

```bash
python scripts/05_fusion_baselines_from_branch_probs.py \
  --pred_csv runs/best_run/predictions_final_eval_fixed.csv \
  --out_csv runs/best_run/fusion_baselines.csv \
  --out_json runs/best_run/fusion_baselines.json
```

### E. Multi-run summary
```bash
python scripts/06_seed_summary_from_run_dirs.py \
  --pattern "runs_sarcasm/*" \
  --which final_eval_fixed \
  --out_csv artifacts/seed_summary.csv \
  --out_json artifacts/seed_summary.json
```

### F. MUStARD audit
```bash
python scripts/07_mustard_paths_and_keyframe_audit.py \
  --mustard_root /path/to/MUStARD-master \
  --out_json artifacts/mustard_audit.json
```

## Expected notebook-aligned artifacts

### In-domain main run
The notebook writes or references these artifacts in the primary run directory:
- `best_model.pt`
- `best_metrics.json`
- `final_eval_fixed.json`
- `metrics.json`
- `predictions.csv`
- `classification_report.txt`
- `confusion_matrix.png`
- `roc.png`
- `pr.png`

### MUStARD external evaluation
The notebook writes:
- `mustard_eval.csv`
- `mustard_metrics.json`
- `mustard_predictions.csv`
- `mustard_classification_report.txt`
- `mustard_confusion_matrix.png`
- `mustard_roc.png`
- `mustard_pr.png`

### Ablation study
The notebook writes:
- `ablation_summary.csv`
- per-model `history.csv`
- per-model `predictions.csv`
- per-model `summary.json`

## Important caution about frozen split provenance

If your existing `subset_summary.json` still says:
- `source_notebook = "sarcasm (10).ipynb"`

then do not falsely claim that those split files were exported from the final notebook.  
Use one of these two honest approaches:

1. keep the split files and explicitly say they were exported from an earlier notebook revision within the same finalized pipeline, or
2. rebuild the split files once using the final notebook-aligned script and then freeze them

For the current revision stage, option 1 is usually safer if the manuscript is already fixed around the current split counts.

## Manuscript-facing consistency checklist

Before final public release, check these items:

- text backbone name matches between paper and repo
- split counts in paper match `subset_summary.json`
- Data Availability no longer says “upon reasonable request”
- README does not imply that supplementary scripts were executed if they were not
- reviewer-facing claims about CI, fusion baselines, or seed summaries are backed by actual output files

## Minimal release package

If you want the safest public package, include only:
- final executed notebook
- README
- BACKBONE_NOTE
- split artifacts
- scripts 01–07

Avoid uploading extra notebooks unless they clearly support the same final narrative.
