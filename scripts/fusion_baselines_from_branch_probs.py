#!/usr/bin/env python3
"""
fusion_baselines_from_branch_probs.py

Evaluate simple non-GWO fusion baselines from branch probability CSVs.


"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


REQUIRED_MIN = {"y_true", "p_text", "p_image", "p_emoji"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    return ap.parse_args()


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr, float(best_f1)


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    thr, _ = tune_threshold(y_true, y_prob)
    pred = (y_prob >= thr).astype(int)
    return {
        "fusion_name": name,
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "roc_auc": safe_auc(y_true, y_prob),
    }


def main() -> None:
    args = parse_args()
    pred_csv = Path(args.pred_csv)
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    df = pd.read_csv(pred_csv)

    missing = REQUIRED_MIN - set(df.columns)
    if missing:
        raise KeyError(
            f"CSV is missing required columns: {sorted(missing)}. "
            "The notebook's default predictions.csv does not contain branch probabilities. "
            "Use a richer export such as predictions_final_eval_fixed.csv or another CSV "
            "that includes p_text, p_image, p_emoji, and p_multi/p_multimodal_branch."
        )

    multi_col = "p_multi" if "p_multi" in df.columns else "p_multimodal_branch"
    if multi_col not in df.columns:
        raise KeyError("Could not find p_multi or p_multimodal_branch in the CSV.")

    y_true = df["y_true"].astype(int).to_numpy()
    p_text = df["p_text"].astype(float).to_numpy()
    p_image = df["p_image"].astype(float).to_numpy()
    p_emoji = df["p_emoji"].astype(float).to_numpy()
    p_multi = df[multi_col].astype(float).to_numpy()

    baselines = {
        "text_only": p_text,
        "image_only": p_image,
        "emoji_only": p_emoji,
        "multimodal_branch_only": p_multi,
        "equal_mean_all": np.mean(np.stack([p_text, p_image, p_emoji, p_multi], axis=1), axis=1),
        "equal_mean_text_image": np.mean(np.stack([p_text, p_image], axis=1), axis=1),
        "equal_mean_text_multi": np.mean(np.stack([p_text, p_multi], axis=1), axis=1),
        "equal_mean_image_multi": np.mean(np.stack([p_image, p_multi], axis=1), axis=1),
        "equal_mean_text_image_multi": np.mean(np.stack([p_text, p_image, p_multi], axis=1), axis=1),
        "median_all": np.median(np.stack([p_text, p_image, p_emoji, p_multi], axis=1), axis=1),
    }

    rows = [evaluate(name, y_true, y_prob) for name, y_prob in baselines.items()]
    rows = sorted(rows, key=lambda d: (d["macro_f1"], d["roc_auc"] if d["roc_auc"] is not None else -1.0), reverse=True)

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    out = {
        "pred_csv": str(pred_csv),
        "multi_col_used": multi_col,
        "n_samples": int(len(df)),
        "results": rows,
        "notes": [
            "These are simple deterministic fusion baselines for reviewer-facing analysis.",
            "Do not claim any baseline result in the manuscript unless this script was actually run on a valid branch-probability CSV.",
            "If your only available file is predictions.csv from the default notebook final-eval cell, you still need a richer export with per-branch probabilities.",
        ],
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(out_df.to_string(index=False))
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
