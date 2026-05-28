# Volta low-shot utility-gate result note

Generated: 2026-05-28 16:33:15

Configuration:

- Dataset: procedural
- Classes: red_square, green_circle, blue_x, yellow_triangle, purple_diamond
- Seeds: [0, 1, 2]
- Shots per class: 4
- Test per class: 160
- Candidate per real image: 20

Main metrics:

| Regime | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std |
| --- | ---: | ---: | ---: | ---: |
| real_only | 0.7333 | 0.0583 | 0.7327 | 0.0554 |
| real_plus_ungated | 0.6121 | 0.0577 | 0.5681 | 0.0619 |
| real_plus_gated | 0.6925 | 0.0163 | 0.6896 | 0.0151 |

Gate audit:

- Candidate rows: 1200
- Retained candidates: 219
- Retention rate: 0.182
- Known drift candidates: 931
- Known drift retained: 81
- Rejection reasons: far_from_real_prototype: 19, far_from_real_prototype+prototype_label_mismatch: 72, low_confidence: 22, low_confidence+far_from_real_prototype: 12, low_confidence+far_from_real_prototype+prototype_label_mismatch: 38, low_confidence+prototype_label_mismatch: 113, prototype_label_mismatch: 705

Interpretation:

- The procedural setting is intentionally cheap and controlled; it does not estimate real generator performance.
- The useful comparison is whether ungated candidates distort real held-out utility and whether the gate exposes which candidates were retained or rejected.
- This supports the IVC Opinions Column as internal evidence for an auditable utility gate, not as a reason to convert the manuscript into a regular empirical article.
