# Volta low-shot utility-gate result note

Generated: 2026-05-28 16:32:56

Configuration:

- Dataset: procedural_real_plus_real_diffusion_manifest
- Classes: red_square, green_circle, blue_x, yellow_triangle, purple_diamond
- Seeds: [0, 1, 2]
- Shots per class: 4
- Test per class: 160
- Candidate per real image: 0

Main metrics:

| Regime | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
| --- | ---: | ---: | ---: | ---: |
| real_only | 0.7333 | 0.0583 | 0.7327 | 0.0554 |
| real_plus_ungated | 0.4671 | 0.0139 | 0.3969 | 0.0167 |
| real_plus_gated | 0.7533 | 0.0016 | 0.6839 | 0.0033 |

Gate audit:

- Candidate rows: 90
- Retained candidates: 36
- Retention rate: 0.400
- Known drift candidates: 0
- Known drift retained: 0
- Rejection reasons: off_center_or_fragmented_foreground: 12, off_center_or_fragmented_foreground+weak_target_color_evidence: 21, weak_target_color_evidence: 21

Interpretation:

- The procedural setting is intentionally cheap and controlled; it does not estimate real generator performance.
- The useful comparison is whether ungated candidates distort real held-out utility and whether the gate exposes which candidates were retained or rejected.
- This supports the IVC Opinions Column as internal evidence for an auditable utility gate, not as a reason to convert the manuscript into a regular empirical article.
