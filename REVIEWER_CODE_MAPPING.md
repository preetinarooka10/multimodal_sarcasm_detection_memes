# Reviewer Code Mapping (Notebook-Aligned)

## Primary evidence
- `notebooks/01_memotion_train_in_domain.ipynb`  
  Final executed notebook aligned to the current repo narrative.

## Reproducibility and transparency helpers
- `scripts/01_audit_notebook_alignment.py`  
  Extract implementation facts directly from the notebook.
- `scripts/02_export_repo_metadata.py`  
  Build release metadata and `BACKBONE_NOTE.txt`.
- `scripts/03_build_memotion_subset_and_splits.py`  
  Notebook-aligned Memotion split export.
- `scripts/04_bootstrap_ci_from_predictions.py`  
  Bootstrap CIs from saved prediction files.
- `scripts/05_fusion_baselines_from_branch_probs.py`  
  Deterministic non-GWO fusion baselines from branch probabilities.
- `scripts/06_seed_summary_from_run_dirs.py`  
  Summary over multiple saved run directories.
- `scripts/07_mustard_paths_and_keyframe_audit.py`  
  MUStARD path audit and optional eval CSV generation.

## Important usage note
These helper scripts are supplementary. They should not be cited as executed evidence unless
their outputs are actually generated and included in the repository.
