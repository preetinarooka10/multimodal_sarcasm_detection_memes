# A Multimodal Attention-Based Framework for Sarcasm Detection in Memes

This repository contains code and reproducibility resources for multimodal sarcasm detection in memes using textual, visual, and emoji-related signals.

## Overview

The project studies sarcasm detection in a code-mixed meme setting using a multimodal pipeline that combines:

-  **dual-branch visual representation**
- **emoji-aware features**
- **cross-modal interaction** and **incongruity-aware modeling**
- a supplementary **Grey Wolf Optimization (GWO)** based post-hoc fusion analysis

The work is intended as an empirical study of a task-specific multimodal configuration for sarcasm detection in memes.

## Repository Contents

```text
sarcasm_repo_aligned/
├── README.md
├── notebooks/
│   └── memotion_train_in_domain.ipynb
└── scripts/
    ├── audit_notebook_alignment.py
    ├── bootstrap_ci_from_predictions.py
    ├── build_memotion_subset_and_splits.py
    ├── export_repo_metadata.py
    ├── fusion_baselines_from_branch_probs.py
    ├── mustard_paths_and_keyframe_audit.py
    └── seed_summary_from_run_dirs.py
```

## Main Notebook

The primary notebook in this repository is:

- `notebooks/memotion_train_in_domain.ipynb`

This notebook contains the main in-domain training and evaluation workflow used for the project.

## Scripts

The `scripts/` folder includes utilities for data preparation, audit, evaluation support, and analysis:

- `build_memotion_subset_and_splits.py` — prepares the Memotion-derived subset and train/validation/test splits
- `bootstrap_ci_from_predictions.py` — computes bootstrap-based summary statistics from saved predictions
- `fusion_baselines_from_branch_probs.py` — analyzes fusion behavior from branch probabilities
- `seed_summary_from_run_dirs.py` — summarizes seed-wise run information
- `mustard_paths_and_keyframe_audit.py` — checks MUStARD paths and keyframe availability
- `audit_notebook_alignment.py` — checks repository consistency with the main notebook
- `export_repo_metadata.py` — exports repository metadata for documentation and tracking

## Datasets

This project uses public benchmark resources:

- **Memotion** for the in-domain meme setting
- **MUStARD** for external validation under domain shift

Please obtain these datasets from their original public sources and use them according to their respective licenses and terms.

Useful source links:

- Memotion: `https://github.com/rockangator/memotion-analysis`
- MUStARD: `https://github.com/soujanyaporia/MUStARD`



## Reproducibility

The repository is organized to support transparency and reproducibility of the implemented pipeline. It includes the main notebook and supporting scripts used for data preparation, alignment checks, and analysis.

If you use this repository, make sure that dataset paths, environment dependencies, and output directories are configured correctly for your system.

## Citation

If you use this repository in academic work, please cite the corresponding paper.

## Contact

For questions related to the code or manuscript, please contact the corresponding author listed in the paper.
