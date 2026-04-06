#!/usr/bin/env python3
"""
03_build_memotion_subset_and_splits.py

Notebook-aligned reproducibility script for the final sarcasm notebook.

This version mirrors the notebook logic more closely than the older reviewer script:
- robust image column inference
- robust sarcasm label mapping
- text column preference: text_corrected -> text_ocr
- 70/15/15 stratified split with seed 42 by default

It is intended for transparency and frozen-split export.
If you already have the manuscript-approved split files, do not regenerate them.

Example:
python scripts/03_build_memotion_subset_and_splits.py \
  --input_csv path/to/labels.csv \
  --output_dir artifacts/splits \
  --notebook_name "sarcasm_9_localgpu_updated_v4 - Copy (2).ipynb"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NEG_TOKENS = {
    "0", "none", "not_sarcastic", "not sarcastic", "no", "na", "nan", "null",
    "non-sarcastic", "notsarcastic", "not-sarcastic"
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--notebook_name", default="sarcasm_9_localgpu_updated_v4 - Copy (2).ipynb")
    ap.add_argument("--image_col", default=None)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--sarcasm_col", default=None)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    return ap.parse_args()


def choose_image_col(df: pd.DataFrame, manual: str | None) -> str:
    if manual:
        if manual not in df.columns:
            raise KeyError(f"image_col '{manual}' not found.")
        return manual
    if "image_name" in df.columns:
        return "image_name"
    img_cands = [c for c in df.columns if re.search(r"(image|file|meme)", c, re.I)]
    if not img_cands:
        raise KeyError("Could not infer image column.")
    return img_cands[0]


def choose_text_col(df: pd.DataFrame, manual: str | None) -> str:
    if manual:
        if manual not in df.columns:
            raise KeyError(f"text_col '{manual}' not found.")
        return manual
    if "text_corrected" in df.columns:
        return "text_corrected"
    if "text_ocr" in df.columns:
        return "text_ocr"
    raise KeyError("Could not infer text column. Expected text_corrected or text_ocr.")


def choose_sarcasm_col(df: pd.DataFrame, manual: str | None) -> str:
    if manual:
        if manual not in df.columns:
            raise KeyError(f"sarcasm_col '{manual}' not found.")
        return manual
    if "sarcasm" in df.columns:
        return "sarcasm"
    sarc_cands = [c for c in df.columns if re.search(r"sarc", c, re.I)]
    if not sarc_cands:
        raise KeyError("Could not infer sarcasm label column.")
    return sarc_cands[0]


def robust_binary_map(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    is_num = s.str.fullmatch(r"-?\d+(\.\d+)?").fillna(False)
    binary = np.zeros(len(s), dtype=int)

    if is_num.any():
        nums = pd.to_numeric(s[is_num], errors="coerce").fillna(0).to_numpy()
        binary[is_num.to_numpy()] = (nums != 0).astype(int)

    mask_str = ~is_num.to_numpy()
    if mask_str.any():
        binary[mask_str] = (~s[mask_str].isin(NEG_TOKENS)).astype(int)

    return binary


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    image_col = choose_image_col(df, args.image_col)
    text_col = choose_text_col(df, args.text_col)
    sarcasm_col = choose_sarcasm_col(df, args.sarcasm_col)

    work = df.copy()
    work["image_name"] = work[image_col].astype(str)
    work["binary_label"] = robust_binary_map(work[sarcasm_col])
    work = work.dropna(subset=[image_col, text_col, sarcasm_col]).reset_index(drop=True)
    work["sample_id"] = np.arange(len(work), dtype=int)

    train_df, temp_df = train_test_split(
        work,
        test_size=(args.val_frac + args.test_frac),
        random_state=args.seed,
        stratify=work["binary_label"],
    )
    rel_test = args.test_frac / (args.val_frac + args.test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        random_state=args.seed,
        stratify=temp_df["binary_label"],
    )

    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    work.to_csv(out / "full_binary_subset.csv", index=False)

    summary = {
        "source_notebook": args.notebook_name,
        "input_csv": str(input_csv),
        "seed": args.seed,
        "text_col_used": text_col,
        "raw_label_col": sarcasm_col,
        "image_col_used": image_col,
        "split_rule": "train/val/test = 70/15/15 stratified on binary_label",
        "counts": {
            "full": int(len(work)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "pos_frac": {
            "full": float(work["binary_label"].mean()) if len(work) else None,
            "train": float(train_df["binary_label"].mean()) if len(train_df) else None,
            "val": float(val_df["binary_label"].mean()) if len(val_df) else None,
            "test": float(test_df["binary_label"].mean()) if len(test_df) else None,
        },
        "notes": [
            "This script mirrors the uploaded notebook's label-mapping and split logic more closely than the earlier lightweight script.",
            "If the manuscript is already frozen with existing split files, keep those files and do not regenerate them.",
            "sample_id is added only for reproducibility bookkeeping and does not change model behavior.",
        ],
    }

    with open(out / "subset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved split files in: {out}")


if __name__ == "__main__":
    main()
