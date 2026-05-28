# Real-diffusion sanity check for the utility-gate supplement

This directory contains a small ComfyUI-based sanity check for the IVC Opinion
manuscript. It is deliberately scoped as an illustrative validation of the
reporting logic, not as a benchmark of diffusion models.

The check generates a small set of prompt-labelled images from a local SD1.5
checkpoint, records the exact generation metadata, runs a lightweight
image-level audit, and writes a manifest that can be passed into the existing
utility-gate pilot.

## Scope

- Purpose: show that prompt-direct labels from a real diffusion model still
  need visible generation, filtering, and retention records.
- Non-purpose: compare diffusion models, estimate general generator quality, or
  claim that diffusion-generated candidates improve recognition performance.
- Default size: five simple geometric/iconic classes with a small number of
  images per class.

## Files

- `generate_real_diffusion_candidates.py`: starts from a running ComfyUI server
  and queues a minimal text-to-image workflow.
- `audit_real_diffusion_candidates.py`: computes simple image diagnostics,
  creates thumbnails, writes an auditable manifest, and summarizes retention.
- `config.json`: experiment settings.

Outputs are written under `outputs/`.

## Observed run

The fixed run used `DreamShaper8_modelscope_Yntec_dreamshaper_8.safetensors`
through ComfyUI on 2026-05-28. It generated 30 prompt-direct candidates:
five classes, six seeds per class.

The lightweight audit retained 12 candidates and rejected 18:

- `off_center_or_fragmented_foreground`: 4
- `weak_target_color_evidence`: 7
- `off_center_or_fragmented_foreground+weak_target_color_evidence`: 7

When the manifest was passed into the existing low-shot utility-gate script
without adding procedural candidates, the mean macro-F1 values across three
seeds were:

- `real_only`: 0.7327
- `real_plus_ungated`: 0.3969
- `real_plus_gated`: 0.6839

These numbers should be read conservatively. The sanity check supports the
paper's reporting claim that real generated candidates need visible filtering
and retained/rejected counts; it is not evidence that this checkpoint improves
the task.
