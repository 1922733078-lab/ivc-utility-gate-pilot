# Volta full low-shot utility-gate experiment

This directory is a self-contained, no-PyTorch pilot experiment for the IVC
Opinions Column manuscript on synthetic-image utility gates. It writes all
results inside this directory and does not modify the manuscript or submission
package.

Public archive:

- GitHub: https://github.com/1922733078-lab/ivc-utility-gate-pilot
- Version DOI used in the manuscript: https://doi.org/10.5281/zenodo.20506364
- Zenodo concept DOI: https://doi.org/10.5281/zenodo.20424622
- GitHub release: https://github.com/1922733078-lab/ivc-utility-gate-pilot/releases/tag/v1.1.0
- Release ZIP SHA256: `f666fe056fcd5fa87b78e9419a076711bfe0ab7a2ff34c86cfc1c5fab16c769f`

## Experiment question

In a low-shot classification setting, what changes when synthetic/proxy images
are passed through an explicit utility gate before being treated as training
evidence?

The experiment runs three regimes over multiple seeds:

- `real_only`: train a classifier from a few real images per class.
- `real_plus_ungated`: add all candidate synthetic/proxy images, including
  candidates with off-target label drift.
- `real_plus_gated`: add only candidates retained by a disclosed gate.

The default dataset is procedural. It creates clean real training and held-out
test images, then creates candidate synthetic/proxy images with a controlled
mixture of useful variants, off-target label drift, and low-quality artifacts.
This gives a cheap stress test for the manuscript's core claim: realism or
availability is not enough; the data path needs a utility gate.

## Utility gate operationalization

The IVC Opinion defines the gate as: explicit target gap, defensible label
pathway, disclosed filtering, and real held-out task utility. This pilot maps
those checks as follows:

- Target gap: few real images per class, controlled by `shots_per_class`.
- Label pathway: direct label preservation for generated/proxy candidates;
  manifest images must declare their label source.
- Disclosed filtering: every candidate receives a row in `gate_audit.csv` with
  source, declared label, candidate kind, prototype prediction, confidence,
  quality score, retain/reject decision, and reason.
- Real held-out utility: all regimes are evaluated on independently generated
  real held-out procedural images, never on candidate images.

The gate is deliberately simple: a prototype classifier is fitted on the
low-shot real set. A candidate is retained only if it has acceptable image
quality, its nearest prototype agrees with the declared label, and the
prototype-confidence score exceeds the configured threshold.

## Why this does not change the IVC Opinions Column route

This is internal evidence and a reproducibility aid. It is not a benchmark,
not a generator comparison, and not a claim that procedural images stand in for
CIFAR or ImageNet. Its role is to demonstrate the reporting logic: ungated
synthetic data can be misleading, while a gate makes label drift, filtering, and
real held-out utility auditable. The manuscript should remain an Opinions Column
piece proposing a standard, not a regular empirical article.

## Run

From the repository root:

```bash
python3 run_experiment.py --config config.json --out results_reproduce
```

The script requires only Python, numpy, and Pillow:

```bash
python3 -m pip install -r requirements.txt
```

## Outputs

The default run writes:

- `results/metrics.csv`: one row per seed and regime.
- `results/metrics_summary.csv`: mean and standard deviation by regime.
- `results/per_class_metrics.csv`: per-class precision, recall, and F1.
- `results/gate_audit.csv`: candidate-level gate audit.
- `results/result_note.md`: compact result interpretation.
- `results/run_config_resolved.json`: exact configuration used.

## Real-diffusion sanity-check manifest

The companion directory `real_diffusion_sanity_check/` contains a small
ComfyUI run using a local SD1.5-style checkpoint. It is not a generator
benchmark. Its purpose is to show that prompt-direct labels from real generated
images still need disclosed filtering and candidate-level audit records.

To run the manifest-only utility-gate check after the ComfyUI candidates have
been generated and audited:

```bash
python3 run_experiment.py \
  --config config_real_diffusion_manifest_only.json \
  --manifest real_diffusion_sanity_check/outputs/real_diffusion_manifest.csv \
  --manifest-root real_diffusion_sanity_check/outputs \
  --out results_real_diffusion_manifest_only
```

The observed run generated 30 candidates, retained 12 after lightweight audit,
and rejected 18. With the manifest only, the mean macro-F1 values were 0.7327
for real-only, 0.3969 for real-plus-ungated, and 0.6839 for real-plus-gated.
This supports the reporting logic without claiming that the checkpoint improves
the task.

## Manifest interface for real synthetic images

The script can read an optional CSV manifest. Start from
`synthetic_manifest_template.csv`.

Required columns:

- `path`: path to an image file, absolute or relative to `manifest_root`.
- `label`: class name or local class index.

Recommended columns:

- `label_source`: `prompt_direct`, `real_label_transfer`,
  `foundation_model_relabel`, or `human_verified`.
- `generator_id`: generator and version.
- `prompt`: prompt or condition summary.
- `gate_decision`: optional external `retain` or `reject` decision.
- `filter_reason`: short reason when an external decision exists.
- `metadata_json`: compact JSON metadata.

Example:

```bash
python3 run_experiment.py \
  --config config.json \
  --manifest my_manifest.csv \
  --manifest-root synthetic_images \
  --out results_with_manifest
```

Manifest candidates are included in `real_plus_ungated`. For
`real_plus_gated`, explicit external decisions are honored; rows without an
external decision are evaluated by the same prototype gate as procedural
candidates.
