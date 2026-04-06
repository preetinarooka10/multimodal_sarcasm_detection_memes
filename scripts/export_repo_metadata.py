#!/usr/bin/env python3
"""
export_repo_metadata.py


"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook_alignment_json", required=True)
    ap.add_argument("--subset_summary", default=None)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def load_json(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    align = load_json(args.notebook_alignment_json)
    subset = load_json(args.subset_summary) if args.subset_summary else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text_model = align.get("detected_backbone", {}).get("text_model")
    visual_a = align.get("detected_backbone", {}).get("visual_branch_a")
    visual_b = align.get("detected_backbone", {}).get("visual_branch_b")

    counts = subset.get("counts", {})
    pos_frac = subset.get("pos_frac", {})

    metadata = {
        "implementation_source": align.get("notebook_path"),
        "text_backbone_in_code": text_model,
        "vision_backbone_in_code": {
            "branch_a": visual_a,
            "branch_b": visual_b,
        },
        "split_summary": {
            "seed": subset.get("seed"),
            "counts": counts,
            "pos_frac": pos_frac,
            "text_col_used": subset.get("text_col_used"),
            "raw_label_col": subset.get("raw_label_col"),
            "source_notebook_in_subset_summary": subset.get("source_notebook"),
        },
        "release_notes": [
            "This metadata is for repository alignment and reviewer-facing transparency.",
            "If source_notebook in subset_summary.json points to an older notebook revision, describe that honestly in the README.",
            "Do not claim any script-generated results unless those scripts were actually executed on the frozen artifacts.",
        ],
    }

    backbone_note = f"""BACKBONE NOTE
=============
Repository alignment should follow the executed notebook, not the older manuscript wording.

Detected text backbone in code:
- {text_model}

Detected visual backbones in code:
- Branch A: {visual_a}
- Branch B: {visual_b}

Important:
Use the executed notebook as the source of truth for the text backbone and keep the manuscript, README,
and reviewer-facing notes consistent with:
- {text_model}

Frozen split summary:
- Seed: {subset.get("seed")}
- Counts: {counts if counts else 'not provided'}
- Positive fraction: {pos_frac if pos_frac else 'not provided'}

Provenance caution:
- subset_summary source_notebook: {subset.get("source_notebook")}
"""

    (out_dir / "repo_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "BACKBONE_NOTE.txt").write_text(backbone_note, encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    print(f"Saved: {out_dir / 'repo_metadata.json'}")
    print(f"Saved: {out_dir / 'BACKBONE_NOTE.txt'}")


if __name__ == "__main__":
    main()
