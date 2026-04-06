#!/usr/bin/env python3
"""
audit_notebook_alignment.py



This script is intentionally read-only. It does not run any training code.
It inspects the notebook JSON and writes a compact implementation summary.


"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True, help="Path to the executed notebook (.ipynb).")
    ap.add_argument("--out_json", required=True, help="Path to save the extracted summary JSON.")
    return ap.parse_args()


def read_notebook_text(notebook_path: Path) -> str:
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    parts = []
    for cell in cells:
        src = cell.get("source", [])
        if isinstance(src, list):
            parts.append("".join(src))
        else:
            parts.append(str(src))
    return "\n\n".join(parts)


def first_group(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1) if m else None


def find_all_strings(pattern: str, text: str) -> list[str]:
    return sorted(set(re.findall(pattern, text, flags=re.MULTILINE)))


def extract_filenames(text: str) -> list[str]:
    patterns = [
        r'os\.path\.join\([^,\n]+,\s*"([^"]+\.(?:json|csv|png|pt|txt|zip))"\)',
        r"os\.path\.join\([^,\n]+,\s*'([^']+\.(?:json|csv|png|pt|txt|zip))'\)",
        r'Path\([^)]*\)\s*/\s*"([^"]+\.(?:json|csv|png|pt|txt|zip))"',
        r"Path\([^)]*\)\s*/\s*'([^']+\.(?:json|csv|png|pt|txt|zip))'",
    ]
    out: set[str] = set()
    for pat in patterns:
        out.update(re.findall(pat, text))
    return sorted(out)


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    text = read_notebook_text(notebook_path)

    text_model = first_group(r'TEXT_MODEL\s*=\s*"([^"]+)"', text)
    clip_model = first_group(r'CLIPVisionModel\.from_pretrained\("([^"]+)"', text)
    data_root = first_group(r'DATA_ROOT\s*=\s*r?"([^"]+)"', text)
    output_root = first_group(r'OUTPUT_ROOT\s*=\s*r?"([^"]+)"', text)
    best_cfg_path = first_group(r'BEST_CFG_PATH\s*=\s*r?"([^"]+)"', text)
    mustard_root = first_group(r'MUSTARD_ROOT\s*=\s*r?"([^"]+)"', text)

    uses_resnet50 = "resnet50(" in text or "ResNet50_Weights" in text
    uses_clip = "CLIPVisionModel" in text
    uses_cross_attn = "MultiheadAttention(embed_dim=768, num_heads=8" in text
    uses_emoji_branch = "emoji_emb" in text and "emoji_proj" in text
    uses_gwo = "def gwo_optimize" in text
    uses_focal = "focal_bce_with_logits" in text

    split_seed = first_group(r"SEED\s*=\s*(\d+)", text)
    max_len = first_group(r"MAX_LEN\s*=\s*(\d+)", text)

    ratio_match = re.search(
        r"train_test_split\(memotion_df,\s*test_size=0\.30.*?train_test_split\(temp_df,\s*test_size=0\.50",
        text,
        flags=re.DOTALL,
    )
    split_description = "70/15/15 stratified split from memotion_df" if ratio_match else None

    filenames = extract_filenames(text)

    summary: dict[str, Any] = {
        "notebook_path": str(notebook_path),
        "detected_backbone": {
            "text_model": text_model,
            "visual_branch_a": "torchvision.models.resnet50" if uses_resnet50 else None,
            "visual_branch_b": clip_model,
            "emoji_branch": uses_emoji_branch,
            "cross_modal_attention": uses_cross_attn,
            "gwo_fusion": uses_gwo,
            "focal_loss": uses_focal,
        },
        "data_and_paths": {
            "data_root": data_root,
            "output_root": output_root,
            "best_cfg_path": best_cfg_path,
            "mustard_root": mustard_root,
        },
        "preprocessing_and_split": {
            "seed": int(split_seed) if split_seed is not None else None,
            "max_len": int(max_len) if max_len is not None else None,
            "split_description": split_description,
            "label_mapping_mode": "robust binary mapping from sarcasm-like column; numeric nonzero => positive; selected negative tokens => negative",
            "text_column_preference": ["text_corrected", "text_ocr"],
        },
        "detected_output_files": filenames,
        "notes": [
            "This summary is extracted from the uploaded notebook, not from the manuscript PDF.",
            "Use this file to keep README and reviewer-facing scripts aligned with the executed code.",
            "Use one consistent backbone name across the notebook, manuscript, README, and reviewer-facing notes.",
        ],
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
