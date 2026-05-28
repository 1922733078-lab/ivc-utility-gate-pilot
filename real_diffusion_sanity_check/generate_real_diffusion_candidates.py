#!/usr/bin/env python3
"""Generate a small real-diffusion prompt-label sanity-check set via ComfyUI.

The script assumes that a local ComfyUI server is already running. It queues a
minimal SD1.5 text-to-image workflow and saves returned images plus generation
metadata. This is an illustrative supplement check, not a benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List


EXP_DIR = Path(__file__).resolve().parent


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def request_json(url: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read())


def build_prompt(config: Dict[str, Any], positive_prompt: str, seed: int, filename_prefix: str) -> Dict[str, Any]:
    comfy = config["comfyui"]
    generation = config["generation"]
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": float(comfy["cfg"]),
                "denoise": 1.0,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": comfy["sampler_name"],
                "scheduler": comfy["scheduler"],
                "seed": int(seed),
                "steps": int(comfy["steps"]),
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": comfy["checkpoint"]},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": int(comfy.get("batch_size", 1)),
                "height": int(comfy["height"]),
                "width": int(comfy["width"]),
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": positive_prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": generation["negative_prompt"]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }


def queue_prompt(server: str, prompt: Dict[str, Any], client_id: str) -> str:
    payload = {"prompt": prompt, "client_id": client_id}
    response = request_json(f"http://{server}/prompt", payload)
    return str(response["prompt_id"])


def wait_for_outputs(server: str, prompt_id: str, timeout_s: int = 600) -> List[Dict[str, str]]:
    start = time.time()
    while time.time() - start < timeout_s:
        history = request_json(f"http://{server}/history/{prompt_id}")
        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            images: List[Dict[str, str]] = []
            for node_out in outputs.values():
                for image in node_out.get("images", []):
                    images.append(
                        {
                            "filename": image["filename"],
                            "subfolder": image.get("subfolder", ""),
                            "type": image.get("type", "output"),
                        }
                    )
            if images:
                return images
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}.")


def download_image(server: str, image_ref: Dict[str, str], out_path: Path) -> None:
    query = urllib.parse.urlencode(image_ref)
    url = f"http://{server}/view?{query}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        out_path.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP_DIR / "config.json")
    parser.add_argument("--out", type=Path, default=EXP_DIR / "outputs")
    args = parser.parse_args()

    config = load_json(args.config)
    server = config["comfyui"]["server_address"]
    out_dir = args.out.resolve()
    image_dir = out_dir / "images"
    meta_path = out_dir / "generation_metadata.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    request_json(f"http://{server}/system_stats")
    client_id = str(uuid.uuid4())
    rows: List[Dict[str, Any]] = []
    per_class = int(config["generation"]["images_per_class"])
    base_seed = int(config["generation"]["base_seed"])

    for class_idx, class_info in enumerate(config["classes"]):
        label = class_info["label"]
        prompt = class_info["prompt"]
        for local_idx in range(per_class):
            seed = base_seed + class_idx * 1000 + local_idx
            prefix = f"ivc_real_diffusion/{label}_{local_idx:03d}_seed{seed}"
            workflow = build_prompt(config, prompt, seed, prefix)
            prompt_id = queue_prompt(server, workflow, client_id)
            image_refs = wait_for_outputs(server, prompt_id)
            if len(image_refs) != 1:
                raise RuntimeError(f"Expected one image for {label} seed {seed}, got {len(image_refs)}")
            out_name = f"{label}_{local_idx:03d}_seed{seed}.png"
            out_path = image_dir / out_name
            download_image(server, image_refs[0], out_path)
            rows.append(
                {
                    "path": str(out_path.relative_to(out_dir)),
                    "label": label,
                    "label_source": "prompt_direct",
                    "generator_id": config["comfyui"]["checkpoint"],
                    "prompt": prompt,
                    "negative_prompt": config["generation"]["negative_prompt"],
                    "seed": seed,
                    "sampler_name": config["comfyui"]["sampler_name"],
                    "scheduler": config["comfyui"]["scheduler"],
                    "steps": config["comfyui"]["steps"],
                    "cfg": config["comfyui"]["cfg"],
                    "width": config["comfyui"]["width"],
                    "height": config["comfyui"]["height"],
                    "comfy_prompt_id": prompt_id,
                    "comfy_filename": image_refs[0]["filename"],
                    "comfy_subfolder": image_refs[0]["subfolder"],
                    "comfy_type": image_refs[0]["type"],
                }
            )
            print(f"generated {out_path}")

    with meta_path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "workflow_config_resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} metadata rows to {meta_path}")


if __name__ == "__main__":
    main()

