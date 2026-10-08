# Antigravity continuation prompt — Hackcessible ISL V9

Continue the English-text-to-Indian Sign Language (ISL) research project in:

`C:\Users\manan\Downloads\Hackcessible`

Read `ANTIGRAVITY_ISL_HANDOFF.md` first, inspect the current Git and Kaggle state, and preserve the existing work. Work directly; do not use Codex subagents unless the user asks. Do not infer that a model works from successful training or lower position error alone.

## Current verified state

- No usable or signer-validated ISL text-to-pose model exists.
- V5 remains the prior reference. On the paired 498-row development cohort, position MSE was 0.679197 for true text and 0.689686 for the training-frame-mean baseline. Hand movement was only about 4.5% of reference motion.
- V8 completed privately on Kaggle T4×2 but generated about 3% of reference motion for both true and shuffled text. No checkpoint was exported.
- A V9 run already exists. Kaggle Version 2, script version `356069698`, ran successfully in 685.2 seconds on T4×2. CPU and synchronized two-GPU preflights passed. It trained six epochs, selected epoch 1, and early-stopped at epoch 6. True-text position/velocity MSE was 0.682464/0.327668; V5 was 0.679197/0.327692; frame mean was 0.689686/0.327527. The run failed the V5 position and frame-mean velocity comparisons, true-text non-collapse floor, shuffled velocity/phase checks, and multiple geometry/support checks. It exported aggregate metrics only and no checkpoint.
- Kaggle Version 3, script version `356087946`, is a caption-to-motion retrieval probe accidentally left under the V9 notebook title. It failed after 975.4 seconds on a confidence-support assertion and did not train a model. Preserve its history, but do not treat it as a V9 result.
- The private pairing audit, Version 3 `356248566`, completed in 541.5 seconds. It verified 10,037/10,037 fit/inner-stop cache rows and V5 normalization parity; found five fit and one inner-stop row with unsupported source identity; and found one MiniLM cosine>=0.92 near-caption edge crossing fit and inner-stop. Status: `integrity_clean_with_metadata_gaps`.
- The audit found no FPS, timestamps, sequence windows, or handedness/mirroring metadata. Raw frame counts are present for 7,992/10,037 fit/inner-stop rows. Source provenance partly uses inferred UID aliases. It did not access raw clips, export row-level data, or train a model.
- The corrected local V9 notebook removes the six unsupported-source rows, removes any remaining inner-stop rows participating in a fit/stop near-caption edge, asserts zero such edges remain, and recalculates the split and target-cache hashes. This source-filtered rerun is pending. The 498 development rows remain previously examined by V7/V8/V9 and cannot be called an independent final test.
- Local Python has no PyTorch. The current V9 module passed `py_compile`, and all seven generated notebook code cells parse as Python AST. The same module hash passed Kaggle's CPU smoke and T4×2 forward/backward probe in V9 Version 2; the updated filter itself still needs Kaggle runtime verification.

Current corrected local SHA-256 hashes:

- `scripts/isl_v9_temporal_ranked_transformer.py`: `02F823E0B93DFBE77CE05B6CF59568AB06DC25562A52C4C7369FA5D5D4CEB38A`
- `scripts/build_isl_v9_temporal_notebook.py`: `006E075EC444D3AA202A528574BEA5BF3926A2E5B17577F275A53CBB4C1D2E80`
- `hackcessible-isl-text-to-pose-v9-ranked.ipynb`: `E357AF31F60FB6A20E77C0663848162AB98FD4D9BA13B1E4194D82BCA6D4AF60`

If any byte changes, rebuild and recompute the hashes before importing or training. Never copy gated clip-level data or secrets into GitHub or a public artifact.

## V9 design and acceptance

V9 keeps the V7 non-autoregressive 48-frame Transformer decoder. Its loss combines group-balanced confidence-weighted XYZ pose MSE, signed XYZ framewise velocity MSE, a 0.25 independently normalized aligned speed-profile auxiliary, and a 0.20 same-target true-caption versus filtered-negative temporal hinge. Pose error is excluded from the ranking hinge. Training and inner-stop selection use the same objective. CPU fixtures cover decoding, padding, signed velocity, and hinge direction; the two-GPU probe records component gradient norms.

The old roles are 9,048 fit, 989 inner-stop, and 498 development rows. Before training, filter the five unsupported fit and one unsupported stop identities, detect fit/stop MiniLM cosine>=0.92 overlap on the V5-normalized captions, remove affected stop rows, and assert zero remaining overlap. Recheck exact captions and source/alias disjointness; report aggregate counts and hashes only. Stop and rerun the audit if the expected unsupported-source counts change.

Export remains gated. Require the inner-stop frame-mean pose guardrail, paired improvements versus V5 and frame mean for position and velocity, positive true-versus-shuffled position/velocity/phase intervals, true-text motion of at least 10% of reference in all groups, confidence-filtered bone support, bone-length ratios, and coordinate/endpoint bounds. Any successful run is only an experimental candidate; it cannot establish sign correctness.

## Next steps

1. Regenerate the local notebook and confirm all three hashes above. Inspect the current Kaggle versions before submitting. The latest Kaggle Version 3 is the failed retrieval probe; Version 2 is the older, unfiltered V9 run. Import the corrected local V9 source as a new version while preserving history.
2. Confirm the V5 artifact and iSign pose-cache inputs remain attached, the notebook remains private, Internet is enabled for the pinned public MiniLM encoder, and accelerator is T4×2. Do not reveal or print any Hugging Face token or Kaggle secret.
3. Verify the filter outputs: six unsupported-source rows removed; any cross-role near-caption stop row excluded; zero remaining cross-role cosine>=0.92 edges; exact-caption and source/alias disjointness; updated hashes. The source has assertions for these gates.
4. Run the CPU contract and synchronized T4×2 probe before any epoch. Require a valid filtered ranking pair, finite losses, nonzero pose/velocity/phase gradients, both GPUs exercised, and successful synchronization. If a gate fails, capture the error and stop before training.
5. If preflight passes, let the corrected private run train and evaluate. Record exact version ID, run signature, split and stop-pair hashes, GPU identity, losses/gradient norms, epoch history, development metrics, and checkpoint gate. Export aggregate metrics only.
6. Treat the 498-row cohort as development evidence only. A usable-model claim still requires fresh untouched source/caption-disjoint data, target-free text-to-pose inference, and blinded fluent-ISL-signer review for meaning, handshape, spatial grammar, and naturalness.

## Data, license, and security

The dataset is gated Exploration-Lab/iSign v1.1 and is stated as CC-BY-NC-SA-4.0 for research/non-commercial use. Keep all gated iSign data and clip-level derivatives private. Never redistribute clips, poses, captions, UIDs, source aliases, per-row outputs, contact sheets, or private checkpoints. Never reveal credentials. Do not integrate the model into the React/FastAPI app until the acceptance requirements are met.
