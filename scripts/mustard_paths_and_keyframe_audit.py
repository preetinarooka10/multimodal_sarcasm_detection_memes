#!/usr/bin/env python3
"""
mustard_paths_and_keyframe_audit.py

Audit MUStARD paths and optionally build a notebook-aligned evaluation CSV.

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mustard_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--build_eval_csv", action="store_true")
    ap.add_argument("--out_csv", default=None)
    return ap.parse_args()


def find_paths(root: Path) -> dict[str, Path]:
    return {
        "annotation_json": root / "data" / "sarcasm_data.json",
        "utterance_dir": root / "raw_videos" / "mmsd_raw_data" / "utterances_final",
        "keyframes_dir": root / "_keyframes_utt",
        "default_eval_dir": root / "_eval",
        "default_eval_csv": root / "_eval" / "mustard_eval.csv",
    }


def build_eval_dataframe(annotation_json: Path, keyframes_dir: Path) -> pd.DataFrame:
    with open(annotation_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for key, item in data.items():
        if not isinstance(item, dict):
            continue

        # Notebook-aligned assumptions:
        # - utterance key is the identifier
        # - labels are mapped to binary sarcasm
        # - keyframe is <utterance_id>.jpg
        text_parts = []
        context = item.get("context", [])
        if isinstance(context, list):
            text_parts.extend([str(x) for x in context if x is not None])
        if item.get("utterance") is not None:
            text_parts.append(str(item.get("utterance")))

        keyframe_name = f"{key}.jpg"
        keyframe_path = keyframes_dir / keyframe_name
        label_val = item.get("sarcasm", item.get("label", 0))
        binary = int(label_val) if str(label_val).strip().isdigit() else int(bool(label_val))

        rows.append({
            "utterance_id": str(key),
            "image_name": keyframe_name,
            "text_corrected": " ".join(text_parts).strip(),
            "binary_label": binary,
            "keyframe_exists": keyframe_path.exists(),
        })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.mustard_root)
    if not root.exists():
        raise FileNotFoundError(f"MUStARD root not found: {root}")

    paths = find_paths(root)
    report = {
        "mustard_root": str(root),
        "paths": {k: str(v) for k, v in paths.items()},
        "exists": {k: v.exists() for k, v in paths.items()},
    }

    if paths["utterance_dir"].exists():
        video_files = sorted([p for p in paths["utterance_dir"].glob("*") if p.is_file()])
        report["utterance_file_count"] = len(video_files)
    else:
        report["utterance_file_count"] = None

    if paths["keyframes_dir"].exists():
        jpgs = sorted(paths["keyframes_dir"].glob("*.jpg"))
        report["keyframe_count"] = len(jpgs)
    else:
        report["keyframe_count"] = None

    if args.build_eval_csv:
        if not paths["annotation_json"].exists():
            raise FileNotFoundError(f"Missing annotation JSON: {paths['annotation_json']}")
        if not paths["keyframes_dir"].exists():
            raise FileNotFoundError(f"Missing keyframes directory: {paths['keyframes_dir']}")

        df = build_eval_dataframe(paths["annotation_json"], paths["keyframes_dir"])
        report["eval_rows_total"] = int(len(df))
        report["eval_rows_with_keyframe"] = int(df["keyframe_exists"].sum())
        report["eval_rows_missing_keyframe"] = int((~df["keyframe_exists"]).sum())

        out_csv = Path(args.out_csv) if args.out_csv else paths["default_eval_csv"]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        report["eval_csv_written"] = str(out_csv)
    else:
        report["eval_csv_written"] = None

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
