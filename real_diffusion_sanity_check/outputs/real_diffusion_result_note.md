# Real-diffusion sanity-check result note

This small check uses a local real diffusion checkpoint through ComfyUI.
It is an illustrative auditability check, not a diffusion-model benchmark.

- Candidate images: 30
- Retained by lightweight audit: 12
- Rejected by lightweight audit: 18
- Rejection reasons: off_center_or_fragmented_foreground: 4, off_center_or_fragmented_foreground+weak_target_color_evidence: 7, weak_target_color_evidence: 7

Interpretation:

- The key output is the candidate-level audit trail, not a claim of improved recognition accuracy.
- Prompt-direct labels from a real diffusion model still require disclosed filtering and retained/rejected counts.
- These outputs can be cited in Supplementary File S1 as a small sanity check supporting the utility-gate reporting logic.
