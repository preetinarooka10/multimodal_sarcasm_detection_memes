#!/usr/bin/env python3
"""
bootstrap_ci_from_predictions.py

Compute bootstrap confidence intervals from a notebook-style predictions CSV.

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--prob_col", default="p_multi", help="Probability column to evaluate.")
    ap.add_argument("--thr", type=float, default=None, help="Threshold for binary metrics. If omitted, tries to infer from prediction column or defaults to 0.5.")
    ap.add_argument("--pred_col", default=None, help="Optional hard-prediction column. If provided, threshold inference is skipped.")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", required=True)
    return ap.parse_args()


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def infer_threshold(df: pd.DataFrame, prob_col: str, pred_col: str | None, explicit_thr: float | None) -> tuple[float, str]:
    if explicit_thr is not None:
        return float(explicit_thr), "explicit"

    if pred_col and pred_col in df.columns:
        return 0.5, f"using explicit pred_col={pred_col}"

    auto_map = {
        "p_multi": "yhat_multi",
        "p_ens": "yhat_ens",
        "p_multimodal_branch": "pred_multi_tuned",
    }
    guess = auto_map.get(prob_col)
    if guess and guess in df.columns:
        return 0.5, f"inferred from available hard predictions via {guess}"

    return 0.5, "default_0.5"


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, thr: float, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc_vals: list[float] = []
    f1_vals: list[float] = []
    auc_vals: list[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        yh = (yp >= thr).astype(int)

        acc_vals.append(float(accuracy_score(yt, yh)))
        f1_vals.append(float(f1_score(yt, yh, average="macro", zero_division=0)))
        auc = safe_auc(yt, yp)
        if auc is not None:
            auc_vals.append(auc)

    def pack(vals: list[float]) -> dict | None:
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(arr.mean()),
            "lower_95": float(np.percentile(arr, 2.5)),
            "upper_95": float(np.percentile(arr, 97.5)),
            "n_boot_valid": int(len(arr)),
        }

    point_yhat = (y_prob >= thr).astype(int)
    return {
        "point_estimate": {
            "accuracy": float(accuracy_score(y_true, point_yhat)),
            "macro_f1": float(f1_score(y_true, point_yhat, average="macro", zero_division=0)),
            "roc_auc": safe_auc(y_true, y_prob),
        },
        "bootstrap_ci": {
            "accuracy": pack(acc_vals),
            "macro_f1": pack(f1_vals),
            "roc_auc": pack(auc_vals),
        },
    }


def main() -> None:
    args = parse_args()
    pred_csv = Path(args.pred_csv)
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    df = pd.read_csv(pred_csv)
    if "y_true" not in df.columns:
        raise KeyError("Prediction CSV must contain y_true.")
    if args.prob_col not in df.columns:
        raise KeyError(f"Probability column '{args.prob_col}' not found in CSV. Available: {list(df.columns)}")

    thr, thr_source = infer_threshold(df, args.prob_col, args.pred_col, args.thr)
    y_true = df["y_true"].astype(int).to_numpy()
    y_prob = df[args.prob_col].astype(float).to_numpy()

    out = {
        "pred_csv": str(pred_csv),
        "prob_col": args.prob_col,
        "threshold_used": float(thr),
        "threshold_source": thr_source,
        "n_samples": int(len(df)),
    }
    out.update(bootstrap_ci(y_true, y_prob, thr=thr, n_boot=args.n_boot, seed=args.seed))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
