# A Multimodal Attention-Based Framework for Sarcasm Detection in Memes

This repository contains the reproducibility package for the manuscript:

**A Multimodal Attention-Based Framework for Sarcasm Detection in Memes**

The repository is organized around the final synchronized artifacts used for:
- in-domain evaluation on a curated sarcasm-oriented meme dataset derived from the publicly distributed Memotion benchmark
- external validation on MUStARD
- ablation analysis
- five-seed stability analysis of post-hoc GWO fusion on fixed branch predictions
- generation of final tables and figures reported in the manuscript

---

## 1. Repository overview

```text
multimodal_sarcasm_detection_memes/
├── README.md
├── configs/
├── notebooks/
├── splits/
├── artifacts/
│   ├── in_domain/
│   ├── mustard/
│   ├── ablation/
│   └── tables/
├── figures/
└── docs/   (optional, if added later)
```

### Main folders

- **configs/**  
  Training/evaluation configuration files used in the final runs.

- **notebooks/**  
  Final synchronized notebooks used to produce the reported results.

- **splits/**  
  Train/validation/test split definitions and summary files for the curated in-domain setup.

- **artifacts/in_domain/**  
  Final in-domain evaluation outputs for the primary multimodal operating point.

- **artifacts/mustard/**  
  Final external evaluation outputs on MUStARD.

- **artifacts/ablation/**  
  Final ablation-study outputs.

- **artifacts/tables/**  
  Final exported table files corresponding to the manuscript.

- **figures/**  
  Final publication figures used in the manuscript.

---

## 2. What is the primary reported result?

The manuscript reports the following as the **primary operating point**:

- **In-domain primary result**: multimodal head with validation-tuned threshold
- **External primary result**: multimodal head evaluated on MUStARD using the validation-tuned threshold

### Important note on GWO
Grey Wolf Optimization (GWO) is included in this repository as a **supplementary post-hoc fusion analysis**, not as the main reported model. The five-seed GWO stability study is based on **fixed branch predictions** and should not be interpreted as a full multi-seed end-to-end retraining analysis of the underlying multimodal model.

---

## 3. Data access

This repository does **not** redistribute the raw benchmark datasets unless redistribution is explicitly permitted by the original source.

### 3.1 In-domain meme dataset
The in-domain experiments use a curated binary sarcasm-oriented setup derived from the publicly distributed **Memotion** benchmark.

Please obtain the dataset from the original/publicly distributed source referenced in the paper and place the files locally according to the folder structure expected by the notebooks.

### 3.2 MUStARD
External validation is performed on **MUStARD**.

Official/public project source:
- MUStARD: `https://github.com/soujanyaporia/MUStARD`

Please obtain MUStARD from the original source and place it locally according to the folder structure expected by the external-evaluation notebook.

### 3.3 Expected local folder structure
Adapt paths as needed for your own machine. A typical layout is:

```text
D:\AI\
├── projects\
│   └── sarcasm_project\
├── outputs\
│   ├── runs_sarcasm_improve\
│   ├── runs_sarcasm_ablation\
│   └── final_submission_package\
└── datasets\
    └── MUStARD\
```

If you use different local paths, update the path variables inside the notebooks before execution.

---

## 4. Split definitions

The curated in-domain split definitions used in the manuscript are provided under:

- `splits/train.csv`
- `splits/val.csv`
- `splits/test.csv`
- `splits/full_binary_subset.csv`
- `splits/subset_summary.json`

These files document the final synchronized in-domain split organization used in the manuscript.

---

## 5. Final notebooks

The reproducibility package is organized primarily around the following final notebooks:

- `notebooks/01_in_domain_final.ipynb`  
  Final in-domain evaluation notebook corresponding to the primary multimodal in-domain results

- `notebooks/02_mustard_external_final.ipynb`  
  Final external-validation notebook for MUStARD

- `notebooks/03_ablation_final.ipynb`  
  Final ablation notebook used to generate the cleaned ablation outputs

- `notebooks/04_gwo_stability_table3.ipynb`  
  Five-seed fixed-branch GWO stability notebook used for Table 3

If an additional packaging or consolidation notebook is included, it should be treated as an artifact-generation helper rather than the main scientific training/evaluation pipeline.

---

## 6. Configurations

The final configuration files are stored under `configs/`.

Recommended primary config file:
- `configs/IMPROVE_V4_CONFIG.json`

If present, `configs/BEST_CONFIG.json` is retained as a reference configuration from earlier synchronized runs.

---

## 7. Artifact provenance: mapping between manuscript items and repository files

### Table 2 — In-domain primary results
Source file:
- `artifacts/tables/table2_final.csv`

Primary values reported in the manuscript:
- Multimodal (threshold 0.50): Accuracy = 0.678, Macro-F1 = 0.488, ROC-AUC = 0.503
- Multimodal (tuned threshold = 0.525): Accuracy = 0.647, Macro-F1 = 0.506, ROC-AUC = 0.503

### Table 3 — Five-seed GWO stability analysis
Source files:
- `artifacts/tables/table3_final.csv`
- `artifacts/tables/table3_seedwise_gwo_stability.csv`

This table reflects the stochasticity of post-hoc GWO fusion on fixed branch predictions.

### Table 4 — Ablation study
Source files:
- `artifacts/ablation/table4_final.csv`
- `artifacts/ablation/ablation_summary.csv`

### Figure 5 — In-domain supplementary GWO/ensemble diagnostic
Source files:
- `figures/fig5_confusion_matrix_ensemble.png`
- `figures/fig5_roc_ensemble.png`
- `figures/fig5_pr_ensemble.png`

### Figure 6 — External MUStARD validation
Source files:
- `figures/fig6_confusion_matrix_multimodal.png`
- `figures/fig6_roc_multimodal.png`
- `figures/fig6_pr_multimodal.png`

### Main in-domain artifact files
Located in:
- `artifacts/in_domain/`

Typical contents:
- `best_metrics.json`
- `final_eval_fixed.json`
- `predictions.csv`
- `classification_report.txt`
- `confusion_matrix.png`
- `roc.png`
- `pr.png`

### Main MUStARD artifact files
Located in:
- `artifacts/mustard/`

Typical contents:
- `external_eval.json`
- `classification_report.txt`
- `fig6_confusion_matrix_multimodal.png`
- `fig6_roc_multimodal.png`
- `fig6_pr_multimodal.png`

---

## 8. Reproducing the manuscript outputs

### 8.1 In-domain primary results
Open and run:

- `notebooks/01_in_domain_final.ipynb`

This notebook produces the primary in-domain multimodal outputs and related artifacts.

### 8.2 External validation on MUStARD
Open and run:

- `notebooks/02_mustard_external_final.ipynb`

This notebook produces the final external evaluation outputs.

### 8.3 Ablation study
Open and run:

- `notebooks/03_ablation_final.ipynb`

This notebook produces the cleaned ablation outputs used in the paper.

### 8.4 GWO stability analysis
Open and run:

- `notebooks/04_gwo_stability_table3.ipynb`

This notebook generates the five-seed fixed-branch GWO stability results used in Table 3.

---

## 9. Environment setup

At minimum, create a Python environment with the dependencies required by the notebooks.

Recommended files to provide in the repository root:
- `requirements.txt`
or
- `environment.yml`

If these are not yet present, add them before final submission so that the manuscript’s Data Availability statement remains accurate.

---

## 10. Notes on interpretation

This repository is intended to support reproducibility of the **final synchronized manuscript results**.

Please note:

1. The primary scientific interpretation in the manuscript is based on the **multimodal head** rather than the post-hoc GWO fusion output.
2. GWO is included mainly as a **diagnostic supplementary analysis**.
3. External validation on MUStARD is intentionally challenging and should be interpreted as a domain-shift assessment rather than a matched-distribution evaluation.
4. The repository reflects the final synchronized artifacts used to align the manuscript with the reported results.

---

## 11. Recommended citation

If you use this repository, please cite the associated manuscript.

---

## 12. Contact

For correspondence regarding the manuscript and reproducibility package, please contact the authors listed in the paper.
