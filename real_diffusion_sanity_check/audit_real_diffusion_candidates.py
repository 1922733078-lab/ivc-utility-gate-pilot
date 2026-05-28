#!/usr/bin/env python3
"""Audit the small real-diffusion sanity-check set.

The audit is intentionally simple and transparent. It checks basic image
quality, center occupancy, and target-color dominance, then writes a manifest
compatible with the existing procedural utility-gate script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXP_DIR = Path(__file__).resolve().parent
TARGET_COLORS = {
    "red_square": np.array([220, 45, 45], dtype=np.float64),
    "green_circle": np.array([35, 190, 75], dtype=np.float64),
    "blue_x": np.array([55, 95, 225], dtype=np.float64),
    "yellow_triangle": np.array([230, 195, 45], dtype=np.float64),
    "purple_diamond": np.array([165, 70, 215], dtype=np.float64),
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def foreground_mask(arr: np.ndarray) -> np.ndarray:
    pixels = arr.reshape(-1, 3).astype(np.float64)
    bg = np.percentile(pixels, 20, axis=0)
    dist = np.sqrt(((arr.astype(np.float64) - bg[None, None, :]) ** 2).sum(axis=2))
    threshold = max(28.0, float(np.percentile(dist, 72)))
    return dist > threshold


def target_color_score(arr: np.ndarray, mask: np.ndarray, label: str) -> Tuple[float, float]:
    target = TARGET_COLORS[label]
    if mask.sum() == 0:
        return 0.0, 0.0
    fg = arr[mask].astype(np.float64)
    target_dist = np.sqrt(((fg - target[None, :]) ** 2).sum(axis=1))
    other = [color for key, color in TARGET_COLORS.items() if key != label]
    other_dist = np.min(
        np.stack([np.sqrt(((fg - color[None, :]) ** 2).sum(axis=1)) for color in other], axis=1),
        axis=1,
    )
    target_score = float(np.mean(np.exp(-target_dist / 95.0)))
    margin = float(np.mean(other_dist - target_dist) / 255.0)
    return target_score, margin


def audit_image(path: Path, label: str, config: Dict[str, Any]) -> Dict[str, Any]:
    audit_cfg = config["audit"]
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    scaled = arr.astype(np.float64) / 255.0
    quality_mean = float(scaled.mean())
    quality_std = float(scaled.std())
    mask = foreground_mask(arr)
    foreground_fraction = float(mask.mean())
    h, w = mask.shape
    crop_fraction = float(audit_cfg["center_crop_fraction"])
    y0 = int((1.0 - crop_fraction) * h / 2)
    y1 = int((1.0 + crop_fraction) * h / 2)
    x0 = int((1.0 - crop_fraction) * w / 2)
    x1 = int((1.0 + crop_fraction) * w / 2)
    center_share = float(mask[y0:y1, x0:x1].sum() / max(1, mask.sum()))
    target_score, target_margin = target_color_score(arr, mask, label)

    failures = []
    if quality_std < float(audit_cfg["min_quality_std"]):
        failures.append("low_image_variation")
    if foreground_fraction < float(audit_cfg["min_foreground_fraction"]):
        failures.append("missing_or_tiny_foreground")
    if foreground_fraction > float(audit_cfg["max_foreground_fraction"]):
        failures.append("excessive_foreground_or_clutter")
    if center_share < float(audit_cfg["min_center_foreground_share"]):
        failures.append("off_center_or_fragmented_foreground")
    if target_margin < float(audit_cfg["target_color_margin"]):
        failures.append("weak_target_color_evidence")

    return {
        "quality_mean": quality_mean,
        "quality_std": quality_std,
        "foreground_fraction": foreground_fraction,
        "center_foreground_share": center_share,
        "target_color_score": target_score,
        "target_color_margin": target_margin,
        "gate_decision": "retain" if not failures else "reject",
        "filter_reason": "manual_audit_needed" if not failures else "+".join(failures),
    }


def make_contact_sheet(rows: List[Dict[str, Any]], out_path: Path, root: Path) -> None:
    thumb = 128
    label_h = 42
    cols = 5
    rows_n = int(np.ceil(len(rows) / cols))
    sheet = Image.new("RGB", (cols * thumb, rows_n * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("Arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    for idx, row in enumerate(rows):
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + label_h)
        with Image.open(root / row["path"]) as image:
            sheet.paste(image.convert("RGB").resize((thumb, thumb)), (x, y))
        status = row["gate_decision"]
        text = f"{row['label']} | {status}\n{row['seed']}"
        draw.rectangle([x, y + thumb, x + thumb, y + thumb + label_h], fill=(245, 245, 245))
        draw.text((x + 3, y + thumb + 3), text, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.json")
    parser.add_argument("--out", type=Path, default=EXP_DIR / "outputs")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    out_dir = args.out.resolve()
    rows = read_csv(out_dir / "generation_metadata.csv")
    audited: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    for row in rows:
        image_path = out_dir / row["path"]
        audit = audit_image(image_path, row["label"], config)
        merged = dict(row)
        merged.update(audit)
        audited.append(merged)
        manifest_rows.append(
            {
                "path": row["path"],
                "label": row["label"],
                "label_source": row["label_source"],
                "generator_id": row["generator_id"],
                "prompt": row["prompt"],
                "gate_decision": audit["gate_decision"],
                "filter_reason": audit["filter_reason"],
                "metadata_json": json.dumps(
                    {
                        "seed": int(row["seed"]),
                        "sampler_name": row["sampler_name"],
                        "scheduler": row["scheduler"],
                        "steps": int(row["steps"]),
                        "cfg": float(row["cfg"]),
                        "negative_prompt": row["negative_prompt"],
                        "quality_mean": audit["quality_mean"],
                        "quality_std": audit["quality_std"],
                        "foreground_fraction": audit["foreground_fraction"],
                        "center_foreground_share": audit["center_foreground_share"],
                        "target_color_margin": audit["target_color_margin"],
                    },
                    separators=(",", ":"),
                ),
            }
        )

    audit_fields = list(audited[0].keys()) if audited else []
    manifest_fields = [
        "path",
        "label",
        "label_source",
        "generator_id",
        "prompt",
        "gate_decision",
        "filter_reason",
        "metadata_json",
    ]
    write_csv(out_dir / "real_diffusion_audit.csv", audited, audit_fields)
    write_csv(out_dir / "real_diffusion_manifest.csv", manifest_rows, manifest_fields)
    make_contact_sheet(audited, out_dir / "contact_sheet.png", out_dir)

    total = len(audited)
    retained = sum(1 for row in audited if row["gate_decision"] == "retain")
    reasons: Dict[str, int] = {}
    for row in audited:
        if row["gate_decision"] == "reject":
            reasons[row["filter_reason"]] = reasons.get(row["filter_reason"], 0) + 1
    reason_text = ", ".join(f"{key}: {value}" for key, value in sorted(reasons.items())) or "none"
    note = [
        "# Real-diffusion sanity-check result note",
        "",
        "This small check uses a local real diffusion checkpoint through ComfyUI.",
        "It is an illustrative auditability check, not a diffusion-model benchmark.",
        "",
        f"- Candidate images: {total}",
        f"- Retained by lightweight audit: {retained}",
        f"- Rejected by lightweight audit: {total - retained}",
        f"- Rejection reasons: {reason_text}",
        "",
        "Interpretation:",
        "",
        "- The key output is the candidate-level audit trail, not a claim of improved recognition accuracy.",
        "- Prompt-direct labels from a real diffusion model still require disclosed filtering and retained/rejected counts.",
        "- These outputs can be cited in Supplementary File S1 as a small sanity check supporting the utility-gate reporting logic.",
        "",
    ]
    (out_dir / "real_diffusion_result_note.md").write_text("\n".join(note), encoding="utf-8")
    print(f"Wrote audit for {total} images to {out_dir}")
    print(f"retained={retained}, rejected={total - retained}")
    print(f"rejection reasons: {reason_text}")


if __name__ == "__main__":
    main()

