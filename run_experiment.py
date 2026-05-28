#!/usr/bin/env python3
"""Self-contained low-shot utility-gate experiment.

No torch, torchvision, or sklearn are required. The default procedural dataset
is intentionally small but complete: it creates real low-shot training images,
real held-out test images, and synthetic/proxy candidates with controlled label
drift. A simple prototype classifier supplies both the downstream classifier and
the gate's label-consistency signal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


EXP_DIR = Path(__file__).resolve().parent


@dataclass
class DatasetBundle:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    class_names: List[str]


@dataclass
class PrototypeModel:
    prototypes: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    class_names: List[str]
    feature_config: Dict[str, Any]
    temperature: float


def ensure_inside_experiment(path: Path) -> None:
    resolved = path.resolve()
    root = EXP_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Refusing to write outside {root}: {resolved}")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_inside_experiment(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def draw_shape(
    class_idx: int,
    rng: np.random.Generator,
    size: int,
    domain: str,
    force_color_shift: bool = False,
    artifact: bool = False,
) -> np.ndarray:
    """Draw one small procedural image.

    `domain` controls style. The real train/test domains are similar but not
    identical; the synthetic candidate domain can be useful, off-target, or
    artifact-heavy depending on the caller.
    """
    if artifact:
        base = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        if rng.random() < 0.5:
            stripe_color = rng.integers(0, 255, size=3)
            base[:, ::4, :] = stripe_color
        return base

    palette = [
        np.array([220, 45, 45], dtype=np.int16),
        np.array([35, 190, 75], dtype=np.int16),
        np.array([55, 95, 225], dtype=np.int16),
        np.array([230, 195, 45], dtype=np.int16),
        np.array([165, 70, 215], dtype=np.int16),
    ]
    color = palette[class_idx % len(palette)].copy()
    if force_color_shift:
        color = palette[(class_idx + 1 + int(rng.integers(0, len(palette) - 1))) % len(palette)].copy()

    bg_low, bg_high = (18, 58) if domain != "candidate" else (10, 72)
    bg = tuple(int(x) for x in rng.integers(bg_low, bg_high, size=3))
    image = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(image)

    shift = int(rng.integers(-4, 5))
    scale = int(rng.integers(-2, 4))
    x0 = 8 + shift
    y0 = 8 + int(rng.integers(-4, 5))
    x1 = x0 + 15 + scale
    y1 = y0 + 15 + scale
    color = np.clip(color + rng.integers(-18, 19, size=3), 0, 255)
    fill = tuple(int(v) for v in color)

    shape = class_idx % 5
    if shape == 0:
        draw.rectangle([x0, y0, x1, y1], fill=fill)
    elif shape == 1:
        draw.ellipse([x0, y0, x1 + 1, y1 + 1], fill=fill)
    elif shape == 2:
        width = int(rng.integers(3, 6))
        draw.line([x0, y0, x1, y1], fill=fill, width=width)
        draw.line([x1, y0, x0, y1], fill=fill, width=width)
    elif shape == 3:
        draw.polygon([(x0 + 8, y0 - 1), (x1 + 2, y1 + 2), (x0 - 2, y1 + 2)], fill=fill)
    else:
        draw.polygon([(x0 + 8, y0 - 2), (x1 + 2, y0 + 8), (x0 + 8, y1 + 2), (x0 - 2, y0 + 8)], fill=fill)

    arr = np.asarray(image).astype(np.float32)
    noise_sigma = 7.0 if domain == "train" else 10.0
    if domain == "candidate":
        noise_sigma = 13.0
        contrast = float(rng.uniform(0.7, 1.35))
        center = arr.mean(axis=(0, 1), keepdims=True)
        arr = (arr - center) * contrast + center
    arr += rng.normal(0, noise_sigma, size=arr.shape)

    if domain == "candidate" and rng.random() < 0.25:
        cut = int(rng.integers(5, 13))
        cy = int(rng.integers(0, size - cut + 1))
        cx = int(rng.integers(0, size - cut + 1))
        arr[cy : cy + cut, cx : cx + cut, :] = arr.mean(axis=(0, 1), keepdims=True)

    return np.clip(arr, 0, 255).astype(np.uint8)


def load_procedural_dataset(config: Dict[str, Any]) -> DatasetBundle:
    class_names = list(config["classes"])
    size = int(config["image_size"])
    train_images: List[np.ndarray] = []
    train_labels: List[int] = []
    test_images: List[np.ndarray] = []
    test_labels: List[int] = []
    train_pool_per_class = max(80, int(config["shots_per_class"]) * 10)
    test_per_class = int(config["test_per_class"])
    for class_idx, _name in enumerate(class_names):
        train_rng = np.random.default_rng(10000 + class_idx)
        test_rng = np.random.default_rng(20000 + class_idx)
        for _ in range(train_pool_per_class):
            train_images.append(draw_shape(class_idx, train_rng, size, "train"))
            train_labels.append(class_idx)
        for _ in range(test_per_class):
            test_images.append(draw_shape(class_idx, test_rng, size, "test"))
            test_labels.append(class_idx)
    return DatasetBundle(
        train_images=np.stack(train_images).astype(np.uint8),
        train_labels=np.array(train_labels, dtype=np.int64),
        test_images=np.stack(test_images).astype(np.uint8),
        test_labels=np.array(test_labels, dtype=np.int64),
        class_names=class_names,
    )


def sample_low_shot(
    labels: np.ndarray,
    shots_per_class: int,
    class_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    chosen: List[np.ndarray] = []
    for label in range(class_count):
        idx = np.flatnonzero(labels == label)
        if len(idx) < shots_per_class:
            raise ValueError(f"Class {label} has {len(idx)} samples, need {shots_per_class}.")
        chosen.append(rng.choice(idx, size=shots_per_class, replace=False))
    return np.concatenate(chosen)


def resize_batch(images: np.ndarray, size: int) -> np.ndarray:
    if images.shape[1] == size and images.shape[2] == size:
        return images.astype(np.float32)
    resized = []
    for image in images:
        resized.append(np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR), dtype=np.float32))
    return np.stack(resized)


def extract_features(images: np.ndarray, feature_config: Dict[str, Any]) -> np.ndarray:
    downsample = int(feature_config.get("downsample", 16))
    small = resize_batch(images, downsample) / 255.0
    parts = [small.reshape(small.shape[0], -1)]
    if feature_config.get("include_color_stats", True):
        parts.append(small.mean(axis=(1, 2)))
        parts.append(small.std(axis=(1, 2)))
    if feature_config.get("include_edge_stats", True):
        gray = small.mean(axis=3)
        dx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))[:, None]
        dy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))[:, None]
        parts.extend([dx, dy])
    return np.concatenate(parts, axis=1).astype(np.float64)


def standardize_train(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-4, 1.0, std)
    x = np.clip((features - mean) / std, -8.0, 8.0)
    return x, mean, std


def standardize_apply(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((features - mean) / std, -8.0, 8.0)


def fit_prototype_model(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    feature_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> PrototypeModel:
    features = extract_features(images, feature_config)
    x, mean, std = standardize_train(features)
    prototypes = []
    for class_idx in range(len(class_names)):
        class_features = x[labels == class_idx]
        if len(class_features) == 0:
            raise ValueError(f"No training samples for class {class_idx}.")
        prototypes.append(class_features.mean(axis=0))
    return PrototypeModel(
        prototypes=np.stack(prototypes),
        feature_mean=mean,
        feature_std=std,
        class_names=list(class_names),
        feature_config=feature_config,
        temperature=float(model_config.get("temperature", 1.0)),
    )


def distances_to_prototypes(model: PrototypeModel, images: np.ndarray) -> np.ndarray:
    features = extract_features(images, model.feature_config)
    x = standardize_apply(features, model.feature_mean, model.feature_std)
    diff = x[:, None, :] - model.prototypes[None, :, :]
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def predict(model: PrototypeModel, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = distances_to_prototypes(model, images)
    logits = -distances / max(model.temperature, 1e-6)
    probabilities = softmax(logits)
    pred = probabilities.argmax(axis=1)
    conf = probabilities.max(axis=1)
    min_distance = distances[np.arange(len(images)), pred]
    return pred, conf, min_distance


def evaluate(model: PrototypeModel, images: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    pred, _conf, _dist = predict(model, images)
    class_count = len(model.class_names)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for true, guessed in zip(labels, pred):
        confusion[int(true), int(guessed)] += 1
    per_class = []
    for class_idx in range(class_count):
        tp = int(confusion[class_idx, class_idx])
        fp = int(confusion[:, class_idx].sum() - tp)
        fn = int(confusion[class_idx, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class.append(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(confusion[class_idx, :].sum()),
            }
        )
    return {
        "accuracy": float((pred == labels).mean()),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "per_class": per_class,
        "confusion": confusion,
    }


def candidate_identifier(image: np.ndarray, seed: int, declared_label: int, local_idx: int) -> str:
    payload = image.tobytes() + f"{seed}-{declared_label}-{local_idx}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:14]


def make_candidates(
    real_images: np.ndarray,
    real_labels: np.ndarray,
    class_names: Sequence[str],
    config: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    rng = np.random.default_rng(50000 + seed)
    size = int(config["image_size"])
    per_real = int(config["candidate_per_real"])
    mix = dict(config["candidate_mix"])
    kinds = list(mix.keys())
    probs = np.array([float(mix[key]) for key in kinds], dtype=np.float64)
    probs = probs / probs.sum()
    images: List[np.ndarray] = []
    labels: List[int] = []
    rows: List[Dict[str, Any]] = []
    class_count = len(class_names)
    local_idx = 0
    if per_real <= 0:
        return np.empty((0, size, size, 3), dtype=np.uint8), np.empty((0,), dtype=np.int64), []
    for source_idx, declared_label in enumerate(real_labels):
        declared_label = int(declared_label)
        for _ in range(per_real):
            kind = str(rng.choice(kinds, p=probs))
            if kind == "useful":
                visual_label = declared_label
                image = draw_shape(visual_label, rng, size, "candidate")
                known_drift = False
            elif kind == "label_drift":
                visual_label = int((declared_label + 1 + rng.integers(0, class_count - 1)) % class_count)
                image = draw_shape(visual_label, rng, size, "candidate", force_color_shift=rng.random() < 0.25)
                known_drift = True
            elif kind == "artifact":
                visual_label = declared_label
                image = draw_shape(visual_label, rng, size, "candidate", artifact=True)
                known_drift = True
            else:
                raise ValueError(f"Unsupported candidate kind: {kind}")
            cid = candidate_identifier(image, seed, declared_label, local_idx)
            images.append(image)
            labels.append(declared_label)
            rows.append(
                {
                    "seed": seed,
                    "candidate_id": cid,
                    "source": "procedural_proxy",
                    "candidate_kind": kind,
                    "path": "",
                    "declared_label_idx": declared_label,
                    "declared_label_name": class_names[declared_label],
                    "visual_label_idx": visual_label,
                    "visual_label_name": class_names[visual_label],
                    "known_drift": int(known_drift),
                    "label_source": "direct_label_preservation",
                    "generator_id": "procedural_proxy_generator_v1",
                    "prompt": f"proxy candidate for {class_names[declared_label]}",
                    "external_gate_decision": "",
                    "external_filter_reason": "",
                    "metadata_json": json.dumps({"source_low_shot_index": int(source_idx)}),
                }
            )
            local_idx += 1
    return np.stack(images).astype(np.uint8), np.array(labels, dtype=np.int64), rows


def parse_manifest_label(value: str, class_names: Sequence[str]) -> int:
    text = str(value).strip()
    if text in class_names:
        return list(class_names).index(text)
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(class_names):
            return idx
    raise ValueError(f"Manifest label {value!r} not found in selected classes {class_names}.")


def parse_external_decision(value: str) -> Optional[bool]:
    text = str(value or "").strip().lower()
    if text in {"retain", "accept", "accepted", "pass", "true", "1", "yes"}:
        return True
    if text in {"reject", "rejected", "fail", "false", "0", "no"}:
        return False
    return None


def load_manifest(
    manifest_path: Optional[Path],
    manifest_root: Optional[Path],
    class_names: Sequence[str],
    image_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if manifest_path is None:
        return np.empty((0, image_size, image_size, 3), dtype=np.uint8), np.empty((0,), dtype=np.int64), []
    root = manifest_root or manifest_path.parent
    images: List[np.ndarray] = []
    labels: List[int] = []
    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=2):
            raw_path = row.get("path", "")
            if not raw_path:
                raise ValueError(f"Manifest row {row_idx} has no path.")
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            label = parse_manifest_label(row.get("label", ""), class_names)
            with Image.open(path) as image:
                arr = np.asarray(image.convert("RGB").resize((image_size, image_size), Image.BILINEAR), dtype=np.uint8)
            cid = row.get("candidate_id") or hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:14]
            images.append(arr)
            labels.append(label)
            rows.append(
                {
                    "seed": seed,
                    "candidate_id": cid,
                    "source": "manifest_synthetic",
                    "candidate_kind": "manifest",
                    "path": str(path),
                    "declared_label_idx": label,
                    "declared_label_name": class_names[label],
                    "visual_label_idx": "",
                    "visual_label_name": "",
                    "known_drift": "",
                    "label_source": row.get("label_source", ""),
                    "generator_id": row.get("generator_id", ""),
                    "prompt": row.get("prompt", ""),
                    "external_gate_decision": row.get("gate_decision", ""),
                    "external_filter_reason": row.get("filter_reason", ""),
                    "metadata_json": row.get("metadata_json", ""),
                }
            )
    if not images:
        return np.empty((0, image_size, image_size, 3), dtype=np.uint8), np.empty((0,), dtype=np.int64), []
    return np.stack(images).astype(np.uint8), np.array(labels, dtype=np.int64), rows


def image_quality(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaled = images.astype(np.float64) / 255.0
    return scaled.mean(axis=(1, 2, 3)), scaled.std(axis=(1, 2, 3))


def apply_gate(
    teacher: PrototypeModel,
    images: np.ndarray,
    labels: np.ndarray,
    rows: List[Dict[str, Any]],
    gate_config: Dict[str, Any],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if len(images) == 0:
        return np.zeros((0,), dtype=bool), []
    pred, conf, dist = predict(teacher, images)
    means, stds = image_quality(images)
    retained: List[bool] = []
    audit: List[Dict[str, Any]] = []
    for idx, base_row in enumerate(rows):
        external = parse_external_decision(str(base_row.get("external_gate_decision", "")))
        quality_ok = (
            means[idx] >= float(gate_config["min_quality_mean"])
            and means[idx] <= float(gate_config["max_quality_mean"])
            and stds[idx] >= float(gate_config["min_quality_std"])
        )
        confidence_ok = conf[idx] >= float(gate_config["min_confidence"])
        distance_ok = dist[idx] <= float(gate_config["max_distance"])
        label_match = int(pred[idx]) == int(labels[idx])
        if external is not None:
            keep = bool(external)
            reason = base_row.get("external_filter_reason") or ("external_retain" if keep else "external_reject")
            gate_mode = "external_manifest_decision"
        else:
            keep = bool(quality_ok and confidence_ok and distance_ok and (label_match or not gate_config["require_label_match"]))
            failed = []
            if not quality_ok:
                failed.append("quality")
            if not confidence_ok:
                failed.append("low_confidence")
            if not distance_ok:
                failed.append("far_from_real_prototype")
            if gate_config["require_label_match"] and not label_match:
                failed.append("prototype_label_mismatch")
            reason = "retain" if keep else "+".join(failed)
            gate_mode = "prototype_gate"
        retained.append(keep)
        row = dict(base_row)
        row.update(
            {
                "gate_mode": gate_mode,
                "gate_decision": "retain" if keep else "reject",
                "gate_reason": reason,
                "prototype_pred_idx": int(pred[idx]),
                "prototype_pred_name": teacher.class_names[int(pred[idx])],
                "prototype_confidence": float(conf[idx]),
                "prototype_distance": float(dist[idx]),
                "quality_mean": float(means[idx]),
                "quality_std": float(stds[idx]),
            }
        )
        audit.append(row)
    return np.array(retained, dtype=bool), audit


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    ensure_inside_experiment(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(metrics_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    regimes = ["real_only", "real_plus_ungated", "real_plus_gated"]
    out = []
    for regime in regimes:
        rows = [row for row in metrics_rows if row["regime"] == regime]
        acc = np.array([float(row["accuracy"]) for row in rows], dtype=np.float64)
        f1 = np.array([float(row["macro_f1"]) for row in rows], dtype=np.float64)
        out.append(
            {
                "regime": regime,
                "runs": len(rows),
                "accuracy_mean": float(acc.mean()),
                "accuracy_std": float(acc.std(ddof=0)),
                "macro_f1_mean": float(f1.mean()),
                "macro_f1_std": float(f1.std(ddof=0)),
            }
        )
    return out


def write_result_note(
    path: Path,
    config: Dict[str, Any],
    summary_rows: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
) -> None:
    ensure_inside_experiment(path)
    by_regime = {row["regime"]: row for row in summary_rows}
    total = len(audit_rows)
    retained = sum(1 for row in audit_rows if row["gate_decision"] == "retain")
    drift_total = sum(1 for row in audit_rows if str(row.get("known_drift", "")) == "1")
    drift_retained = sum(
        1 for row in audit_rows if str(row.get("known_drift", "")) == "1" and row["gate_decision"] == "retain"
    )
    rejection_reasons: Dict[str, int] = {}
    for row in audit_rows:
        if row["gate_decision"] == "reject":
            reason = row["gate_reason"]
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    reason_text = ", ".join(f"{key}: {value}" for key, value in sorted(rejection_reasons.items())) or "none"

    lines = [
        "# Volta low-shot utility-gate result note",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Configuration:",
        "",
        f"- Dataset: {config['dataset']}",
        f"- Classes: {', '.join(config['classes'])}",
        f"- Seeds: {config['seeds']}",
        f"- Shots per class: {config['shots_per_class']}",
        f"- Test per class: {config['test_per_class']}",
        f"- Candidate per real image: {config['candidate_per_real']}",
        "",
        "Main metrics:",
        "",
        "| Regime | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for regime in ["real_only", "real_plus_ungated", "real_plus_gated"]:
        row = by_regime[regime]
        lines.append(
            f"| {regime} | {row['accuracy_mean']:.4f} | {row['accuracy_std']:.4f} | "
            f"{row['macro_f1_mean']:.4f} | {row['macro_f1_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Gate audit:",
            "",
            f"- Candidate rows: {total}",
            f"- Retained candidates: {retained}",
            f"- Retention rate: {0.0 if total == 0 else retained / total:.3f}",
            f"- Known drift candidates: {drift_total}",
            f"- Known drift retained: {drift_retained}",
            f"- Rejection reasons: {reason_text}",
            "",
            "Interpretation:",
            "",
            "- The procedural setting is intentionally cheap and controlled; it does not estimate real generator performance.",
            "- The useful comparison is whether ungated candidates distort real held-out utility and whether the gate exposes which candidates were retained or rejected.",
            "- This supports the IVC Opinions Column as internal evidence for an auditable utility gate, not as a reason to convert the manuscript into a regular empirical article.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config: Dict[str, Any], out_dir: Path, manifest: Optional[Path], manifest_root: Optional[Path]) -> None:
    ensure_inside_experiment(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_procedural_dataset(config)
    class_count = len(bundle.class_names)

    metrics_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    audit_rows_all: List[Dict[str, Any]] = []

    for seed in [int(seed) for seed in config["seeds"]]:
        rng = np.random.default_rng(seed)
        train_idx = sample_low_shot(bundle.train_labels, int(config["shots_per_class"]), class_count, rng)
        real_images = bundle.train_images[train_idx]
        real_labels = bundle.train_labels[train_idx]
        teacher = fit_prototype_model(
            real_images,
            real_labels,
            bundle.class_names,
            config["features"],
            config["prototype_classifier"],
        )

        proc_images, proc_labels, proc_rows = make_candidates(real_images, real_labels, bundle.class_names, config, seed)
        man_images, man_labels, man_rows = load_manifest(
            manifest,
            manifest_root,
            bundle.class_names,
            int(config["image_size"]),
            seed,
        )
        candidate_images = np.concatenate([proc_images, man_images], axis=0)
        candidate_labels = np.concatenate([proc_labels, man_labels], axis=0)
        candidate_rows = proc_rows + man_rows
        retained_mask, audit_rows = apply_gate(teacher, candidate_images, candidate_labels, candidate_rows, config["gate"])
        audit_rows_all.extend(audit_rows)

        regimes = {
            "real_only": (real_images, real_labels, 0),
            "real_plus_ungated": (
                np.concatenate([real_images, candidate_images], axis=0),
                np.concatenate([real_labels, candidate_labels], axis=0),
                int(len(candidate_labels)),
            ),
            "real_plus_gated": (
                np.concatenate([real_images, candidate_images[retained_mask]], axis=0),
                np.concatenate([real_labels, candidate_labels[retained_mask]], axis=0),
                int(retained_mask.sum()),
            ),
        }

        for regime, (train_images, train_labels, candidate_train_count) in regimes.items():
            model = fit_prototype_model(
                train_images,
                train_labels,
                bundle.class_names,
                config["features"],
                config["prototype_classifier"],
            )
            result = evaluate(model, bundle.test_images, bundle.test_labels)
            metrics_rows.append(
                {
                    "experiment_name": config["experiment_name"],
                    "dataset": config["dataset"],
                    "regime": regime,
                    "seed": seed,
                    "class_count": class_count,
                    "classes": "|".join(bundle.class_names),
                    "shots_per_class": int(config["shots_per_class"]),
                    "test_per_class": int(config["test_per_class"]),
                    "real_train_count": int(len(real_labels)),
                    "candidate_train_count": candidate_train_count,
                    "candidate_total": int(len(candidate_labels)),
                    "candidate_retained": int(retained_mask.sum()),
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                }
            )
            for class_idx, row in enumerate(result["per_class"]):
                per_class_rows.append(
                    {
                        "experiment_name": config["experiment_name"],
                        "dataset": config["dataset"],
                        "regime": regime,
                        "seed": seed,
                        "class_idx": class_idx,
                        "class_name": bundle.class_names[class_idx],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                        "support": row["support"],
                    }
                )

    summary_rows = summarize(metrics_rows)
    write_csv(
        out_dir / "metrics.csv",
        metrics_rows,
        [
            "experiment_name",
            "dataset",
            "regime",
            "seed",
            "class_count",
            "classes",
            "shots_per_class",
            "test_per_class",
            "real_train_count",
            "candidate_train_count",
            "candidate_total",
            "candidate_retained",
            "accuracy",
            "macro_f1",
        ],
    )
    write_csv(
        out_dir / "metrics_summary.csv",
        summary_rows,
        ["regime", "runs", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"],
    )
    write_csv(
        out_dir / "per_class_metrics.csv",
        per_class_rows,
        [
            "experiment_name",
            "dataset",
            "regime",
            "seed",
            "class_idx",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
        ],
    )
    write_csv(
        out_dir / "gate_audit.csv",
        audit_rows_all,
        [
            "seed",
            "candidate_id",
            "source",
            "candidate_kind",
            "path",
            "declared_label_idx",
            "declared_label_name",
            "visual_label_idx",
            "visual_label_name",
            "known_drift",
            "label_source",
            "generator_id",
            "prompt",
            "gate_mode",
            "gate_decision",
            "gate_reason",
            "prototype_pred_idx",
            "prototype_pred_name",
            "prototype_confidence",
            "prototype_distance",
            "quality_mean",
            "quality_std",
            "external_gate_decision",
            "external_filter_reason",
            "metadata_json",
        ],
    )
    resolved = dict(config)
    resolved["manifest"] = str(manifest) if manifest else None
    resolved["manifest_root"] = str(manifest_root) if manifest_root else None
    resolved["output_dir"] = str(out_dir.resolve())
    write_json(out_dir / "run_config_resolved.json", resolved)
    write_result_note(out_dir / "result_note.md", resolved, summary_rows, audit_rows_all)

    print(f"Wrote results to {out_dir.resolve()}")
    for row in summary_rows:
        print(
            f"{row['regime']}: accuracy={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}, "
            f"macro_f1={row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.json")
    parser.add_argument("--out", type=Path, default=EXP_DIR / "results")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Optional comma-separated seed override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if args.seeds:
        config["seeds"] = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = args.out.resolve()
    manifest = args.manifest.resolve() if args.manifest else None
    manifest_root = args.manifest_root.resolve() if args.manifest_root else None
    if manifest is not None and not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    run(config, out_dir, manifest, manifest_root)


if __name__ == "__main__":
    main()
