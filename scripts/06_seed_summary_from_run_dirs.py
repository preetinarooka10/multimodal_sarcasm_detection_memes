#!/usr/bin/env python3
"""
06_seed_summary_from_run_dirs.py

Aggregate notebook-style metrics across multiple run directories.

Aligned with these notebook outputs:
- metrics.json
- final_eval_fixed.json
- best_metrics.json
- mustard_metrics.json

Example:
python scripts/06_seed_summary_from_run_dirs.py \
  --pattern "runs_sarcasm/*" \
  --which final_eval_fixed \
  --out_csv artifacts/seed_summary.csv \
  --out_json artifacts/seed_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


WHICH_TO_FILE = {
    "metrics": "metrics.json",
    "final_eval_fixed": "final_eval_fixed.json",
    "best_metrics": "best_metrics.json",
    "mustard_metrics": "mustard_metrics.json",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="Glob pattern for run directories or JSON files.")
    ap.add_argument("--which", choices=sorted(WHICH_TO_FILE.keys()), default="metrics")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    return ap.parse_args()


def extract_row(which: str, source: str, data: dict) -> dict:
    if which == "metrics":
        blk_multi = data.get("test_multi", {})
        blk_ens = data.get("test_ens", {})
        return {
            "source": source,
            "multi_accuracy": blk_multi.get("acc"),
            "multi_macro_f1": blk_multi.get("macro_f1"),
            "multi_roc_auc": blk_multi.get("auc"),
            "ens_accuracy": blk_ens.get("acc"),
            "ens_macro_f1": blk_ens.get("macro_f1"),
            "ens_roc_auc": blk_ens.get("auc"),
            "thr_multi": data.get("thr_multi"),
            "thr_ens": data.get("thr_ens"),
        }

    if which == "final_eval_fixed":
        return {
            "source": source,
            "multi_accuracy": data.get("test_multi_acc"),
            "multi_macro_f1": data.get("test_multi_f1"),
            "multi_roc_auc": data.get("test_multi_auc"),
            "ens_accuracy": data.get("test_ens_acc"),
            "ens_macro_f1": data.get("test_ens_f1"),
            "ens_roc_auc": data.get("test_ens_auc"),
            "thr_multi": data.get("val_multi_thr"),
            "thr_ens": data.get("val_ens_thr"),
        }

    if which == "best_metrics":
        val_multi = data.get("val_multi", {})
        val_ens = data.get("val_ens", {})
        return {
            "source": source,
            "multi_accuracy": val_multi.get("acc"),
            "multi_macro_f1": val_multi.get("macro_f1"),
            "multi_roc_auc": val_multi.get("auc"),
            "ens_accuracy": val_ens.get("acc"),
            "ens_macro_f1": val_ens.get("macro_f1"),
            "ens_roc_auc": val_ens.get("auc"),
            "thr_multi": data.get("val_multi_thr"),
            "thr_ens": data.get("val_ens_thr"),
            "best_epoch": data.get("best_epoch"),
        }

    if which == "mustard_metrics":
        return {
            "source": source,
            "multi_accuracy": data.get("mustard_acc"),
            "multi_macro_f1": data.get("mustard_macro_f1"),
            "multi_roc_auc": data.get("mustard_roc_auc"),
            "thr_multi": data.get("thr_multi"),
            "n_samples": data.get("mustard_size"),
            "pos_frac": data.get("mustard_pos_frac"),
        }

    raise ValueError(f"Unsupported mode: {which}")


def mean_std(series: pd.Series) -> dict | None:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return None
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "n": int(len(vals)),
    }


def main() -> None:
    args = parse_args()
    hits = sorted(glob.glob(args.pattern))
    if not hits:
        raise FileNotFoundError(f"No matches for pattern: {args.pattern}")

    rows: list[dict] = []
    target_name = WHICH_TO_FILE[args.which]

    for hit in hits:
        path = hit
        if os.path.isdir(hit):
            path = os.path.join(hit, target_name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows.append(extract_row(args.which, hit, data))
        except Exception as exc:
            rows.append({"source": hit, "error": str(exc)})

    if not rows:
        raise RuntimeError("No readable JSON files found.")

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = {
        "which": args.which,
        "file_expected": target_name,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "summary": {
            col: mean_std(df[col])
            for col in df.columns
            if col not in {"source", "error"}
        },
        "notes": [
            "This script only summarizes existing run artifacts.",
            "Do not report seed statistics unless the underlying runs are genuine independent runs.",
        ],
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
