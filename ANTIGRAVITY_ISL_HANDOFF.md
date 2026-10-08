# Hackcessible ISL — Antigravity Handoff

- **Updated:** 2026-10-08
- **Workspace:** C:\Users\manan\Downloads\Hackcessible
- **Purpose:** Evidence-based status and continuation guide for the English-text-to-Indian Sign Language (ISL) model experiments.

## Current verdict

**No usable ISL text-to-pose model has been verified.** V9 did complete a private Kaggle run on T4×2, but its development quality and geometry gates blocked checkpoint export. It trained for six epochs, selected epoch 1, then early-stopped. Its true-text position error was worse than V5, and its velocity error was slightly worse than the training-frame-mean baseline. The true-text motion-collapse gate, paired V5/velocity checks, and several geometry checks failed. No checkpoint was exported, and no fluent signer reviewed any generated signs.

The corrected V9 source, builder, and notebook are local and have passed Python compilation, notebook-cell AST parsing, builder fixture, and Git whitespace checks. The current Kaggle notebook title is misleading: Version 2 is the actual V9 training run; Version 3 contains a caption-to-motion retrieval probe and failed before model training. The pairing audit found unsupported source identities and one cross-role near-caption edge in the old V9 fit/stop roles. The corrected notebook filters these rows, recalculates role hashes, and fails if the overlap remains. It has not yet been run. V9 still can be evaluated only on the 498-row development cohort already examined by V7/V8, so no independent final accuracy claim is available.

## Objective and intended use

The goal is to assist Deaf students, including psychology and medical students, in accessing spoken-language education through high-fidelity Indian Sign Language (ISL) generation. This project branch implements an **English text → ISL pose-sequence model** (75 MediaPipe landmarks: 33 body pose, 21 left hand, 21 right hand across 48 frames).

Full linguistic coverage requires nonmanual facial expressions and sign-linguistic review by fluent ISL signers. Current V5/V7 experiments do not establish a usable English-to-ISL text-to-pose baseline.

## User constraints and working rules

- Preserve and extend this checkout; inspect before replacing or restarting anything.
- The user previously authorized the Kaggle training workflow and had configured access to the gated dataset. Recheck current account, notebook, accelerator, inputs, secret availability, and run state; prior browser/session state may be stale.
- Never print, paste into source, commit, log, export, or include the Hugging Face token. If the Kaggle HF_TOKEN secret is still present, use it only through Kaggle Secrets and verify presence without revealing its value.
- Do not upload gated iSign files or iSign-derived clip-level pose arrays, captions, predictions, UID/alias manifests, contact sheets, or checkpoints to public GitHub. Keep permitted research artifacts private and review license terms before sharing.
- Do not stage the whole checkout. It contains useful untracked local artifacts and scripts; preserve them and stage only explicitly approved, sanitized files if a future commit is requested.
- Give evidence-based status: source review, notebook import, CPU checks, cache preflight, and GPU training are different milestones. Never call V5/V6 a working translator without the success evidence below.

## Repository state and important paths

At the 2026-10-08 continuation, local branch `codex/isl-pairing-target-audit` matched its origin. HEAD was `96986af` (Fix audit caption normalizer shadowing); it includes the corrected pairing-audit notebook. The earlier sanitized archive remains on `codex/isl-v3` at `407044f`. The V9 module, builder, notebook, and handoff files are local untracked changes until explicitly staged.

- 23adf1c — initial V6 progressive decoder notebook.
- fb44188 — CUDA attention-mask repair.
- 407044f — sanitized project archive.

No project AGENTS.md was found. Local git status showed artifacts/ and multiple scripts/*isl* files as **untracked**. These contain useful local research material; inspect carefully, preserve, and do not use git add . or copy raw/clip-level derivatives into a public package.

Key files:

- hackcessible-isl-text-to-pose-v3.ipynb — earlier pose model.
- hackcessible-isl-text-to-pose-v5.ipynb — completed V5 experiment source.
- hackcessible-isl-text-to-pose-v6-progressive.ipynb — V6 notebook configured for Kaggle GPU T4 x2. At local inspection all 8 cells had no execution count and no outputs.
- hackcessible-isl-text-to-pose-v9-ranked.ipynb — locally corrected V9 development experiment notebook; next run is pending.
- scripts/isl_v6_progressive_decoder.py — causal decoder and CPU contract checks.
- scripts/build_isl_v6_progressive_notebook.py — notebook builder.
- scripts/isl_v9_temporal_ranked_transformer.py and scripts/build_isl_v9_temporal_notebook.py — current V9 source and notebook builder; exact hashes are in the V9 section below.
- scripts/build_isl_v5_notebook.py, scripts/isl_v5_data_cell.py, and evaluation/render helpers — local V5/research tooling; several are untracked.
- artifacts/V4/ and artifacts/V5/ — local evaluation artifacts. Some files include per-example identifiers or predictions; do not redistribute them.

Verified SHA-256 of the current local V6 sources:

- Decoder module: 5854613B9A6C1EC72B1D0F0C0E24E3D38258BF7CE0045409D6E175FCF817092E
- Notebook builder: 68C2255A7EA18AB55D3464EDD82814E17AD0F0866B4603FD94225139B125A0DB
- V6 notebook: AA1525331AE2CE985FFC52FC3F74B046CA3D5DCC5C89EB00722C0447EDD2AEA8

The earlier restricted-data-safe archive is ISL_Project_Handoff_2026-10-02.zip. Its SHA-256 is D6C8B932E1E8F96401A6C91C9D172AE692FF8ECC6D1CE7F3C1791D6EC8572231; it was pushed to Brubeee/Hackcessible, branch codex/isl-v3, in commit 407044f6540a130eba44fcc083ba5afa11011324. It contains sanitized source and aggregate handoff material, not the gated dataset, checkpoints, or clip-level outputs. See [the archive on GitHub](https://github.com/Brubeee/Hackcessible/blob/407044f6540a130eba44fcc083ba5afa11011324/ISL_Project_Handoff_2026-10-02.zip).

## Dataset and provenance limits

The experiment uses gated Exploration-Lab/iSign v1.1 from Hugging Face, with pose files/metadata cached in Kaggle inputs. The stated license is **CC-BY-NC-SA-4.0, research/non-commercial use only**. Follow the official gate and license terms; Kaggle availability does not grant permission to redistribute files or derived sample-level data.

V5 used the verified V2 pose cache plus freshly decoded selected V5 clips. Its aggregate manifest reports 2,045 loaded V2 rows and 7,995 effective fresh training rows (10,040 effective training total), with 9,048 fit rows, 989 inner-stop rows, and 3 caption-held-out rows; the paired development test contains 498 effective rows out of 500 selected. V6 previously reported 8,500/8,500 selected fresh clips present in its cache preflight, with no fresh decode failures and two exclusions elsewhere in the cohort. Verify the current mounted cache contents and manifests before reuse.

The local V5 manifest and V9 preflight assertions show zero source-alias and normalized-caption intersection between fit, inner-stop, and the 498-row development cohort; the development cohort has 498 distinct source clusters. Source-cluster provenance is still based on the conservative metadata/UID-derived alias system, not an authoritative signer/source-video audit. V7 and V8 already evaluated these 498 rows, so they are strictly development data now. The attached artifact/cache does not contain a broader unexamined caption-and-pose pool for a new final test. A fresh gated data selection/cache is needed before independent final evaluation. Compute normalization only from fit data.

## Model history and evidence

### V4

V4 established a text-conditioned MiniLM pose-fit signal on its development set but produced only about 6–8% of reference movement. The independent review found similar true-caption/shuffled-caption skeletons and likely source-video leakage risk from UID suffix handling. Its old development-set numbers are not directly comparable to V5 because the evaluation cohorts differ.

### V5 — best completed candidate, not usable

Kaggle run: [V5 notebook version 354226633](https://www.kaggle.com/code/brubee/hackcessible-isl-text-to-pose?scriptVersionId=354226633). It completed and exported artifacts. The compact checkpoint's previously recorded SHA-256 is CFE5CA02DD9287EE74C5C12798BC0D65F6AF87F4071754DDB40E358E0D6C98A1; confirm the mounted checkpoint against this digest before using it.

On the paired 498-row development test, group-balanced normalized **position** MSE was:

| Condition | Position MSE |
|---|---:|
| V5, true text | 0.679199 |
| V5, shuffled text | 0.704455 |
| Frozen V4, true text | 0.700804 |
| Training-frame-mean baseline | 0.689688 |

V5 is modestly better than paired V4 on this metric and has a small true-vs-shuffled gap, but is slightly worse than the frame-mean position baseline. Its true-text body/left-hand/right-hand motion ratios are approximately **5.57% / 4.55% / 4.63%** of reference motion; shuffled text is nearly the same (**5.55% / 4.62% / 4.71%**). Independent visual inspection found broad true/shuffled poses similar and no convincing signs. Lower coordinate error is not sign intelligibility. V5 is not suitable to present as a translator.

### V6 — historical experiment, failed before training

V6 computes frozen **token-level** MiniLM hidden states; the V5 pooled sentence vectors are not interchangeable with those token states. The model is a two-layer causal Transformer decoder with text cross-attention, predicts 75×3 body/hand coordinates over 48 frames, and separates:

- Teacher-forced training (forward_teacher) with the target shifted right, so frame t sees ground truth only through t−1.
- Target-free autoregressive generation (generate) that accepts no pose target and feeds back only prior predictions.

The notebook is configured for GPU T4 x2, batch 16 (free-running evaluation batch 8), up to 30 epochs, patience 5, and a 0.05 velocity term. It intends to choose checkpoints using free-running inner-stop loss, not teacher-forced training loss.

The first Kaggle attempt, scriptVersionId 354254718 (V1/1), passed key preflight checks: V5 checkpoint digest, full fresh cache coverage, V5 role reconstruction, reference-pose parity, normalizer parity, and CPU model/teacher-forward/backward contracts. It then failed in **CUDA target-free generation**, before any training epochs, with CUDA error: misaligned address. The failure traced to PyTorch 2.10 CUDA MultiheadAttention converting a boolean padding mask via _canonical_mask/masked_fill_.

Commit fb44188ffae7d09f96b3fb9fd1de419262c0ec52 repairs that path with contiguous additive FP32 attention masks, explicit FP32 math-only SDPA (flash/memory-efficient/cuDNN attention disabled and TF32 off), plus synchronized per-frame diagnostics. The local CPU contract/source checks passed and an independent review cleared it for a **fresh GPU probe only**. Evidence that this exact repaired source was uploaded and successfully ran on Kaggle is absent. Inspect live Kaggle history and source digest; do not infer success from the commit or notebook metadata.

The failed V6 run was superseded by V7. V7 used the same broad data roles and a non-autoregressive decoder; its successful run does not validate V6's autoregressive model.

### V7 — completed experiment, not a working translator

Kaggle run `355920242` completed in **747.3 seconds** on **T4 x2**. The run reconstructed the frozen roles (9,048 fit / 989 inner-stop / 498 paired development-test rows), passed the 2-GPU forward/backward probe, selected epoch 2, and early-stopped after epoch 7. Its run log recorded the exact V7 module SHA-256 listed below.

| Evaluation | Position MSE | Velocity MSE | Selection score | Body / left / right motion ratio |
|---|---:|---:|---:|---:|
| V7, true text | 0.689558 | 0.333321 | 0.706224 | 19.75% / 20.16% / 18.73% |
| V7, shuffled text | 0.712648 | 0.333418 | 0.729319 | 19.96% / 20.25% / 18.93% |
| V5, true text | 0.679199 | — | — | — |
| V5 training-frame-mean baseline | 0.689688 | — | — | — |

True text beat shuffled text in **66.1%** of paired rows (reported t-statistic **6.925**), but V7's true-text position MSE is about tied with the frame-mean baseline and worse than V5. True and shuffled motion ratios are nearly the same. This shows limited text sensitivity, not sign intelligibility. The test cohort is a V5 development cohort; source-video and signer independence are not established, and there is no fluent-signer review.

An aggregate coordinate audit found 13/498 true-text samples with at least one confidence-valid XY coordinate outside ±5; the frame-47 p99 was 5.867 and maximum 6.938. The shuffled condition also had six frame-47 samples over ±5. Therefore the earlier “zero boundary explosion” claim is false. This does not by itself prove that those outliers are visually catastrophic, but it warrants endpoint and geometry diagnostics in follow-up runs.

Current local artifacts and source hashes:

- Module: `scripts/isl_v7_motion_transformer.py` — `3B666F9B3612725DC7CC3FFD4DDC5976272D201B5EE7D2A0634709677CCC60B7`
- Builder: `scripts/build_isl_v7_motion_notebook.py` — `026945BBD8F6D29D6A55C31385BFD8233C929B6F27BD8066D23726DE75618378`
- Notebook: `hackcessible-isl-text-to-pose-v7-motion.ipynb` — `E124A41D3F8C35B4733C8799896220D6EAFE48E28743D62CDE7CEF2ACBF4D506`
- Checkpoint: `artifacts/V7/isl_v7_motion_export/checkpoints/motion_transformer_minilm_v7.pt`, **18,095,167 bytes**, SHA-256 `767048D56E24D2BAD5309C8F3E810E8E715398A51762AD2117D4DF93136B5ACC`
- Notebook run link: https://www.kaggle.com/code/brubee/hackcessible-isl-text-to-pose-progressive-v6/edit/run/355920242

V7's primary training loss used Smooth L1 coordinate, velocity, acceleration, speed, and endpoint terms. Review found the coordinate-weighted loss denominator omitted the XYZ expansion, the endpoint anchor duplicated the final transition at roughly 25× a regular transition, and generic speed/acceleration objectives could produce text-independent motion. Its checkpoint criterion was position MSE plus 0.05 velocity MSE; variance was diagnostic only, not a gradient. These issues motivated V8.

### V8 — completed experiment, not a working model

Private Kaggle kernel `brubee/hackcessible-isl-text-to-pose-v8-ranked`, script version `356045347`, completed successfully in 1,325.1 seconds on T4 x2. It passed the CPU preflight and synchronized two-GPU forward/backward gate, trained 8 epochs, early-stopped, and selected epoch 3 (inner-stop score 0.68821). Run signature: `d39e4a750adce6794b337a81ed75a18e82c66fc37e1ddac0cfe66673685081ff`.

On the 498-row cohort, V8 reduced position error relative to shuffled text and the frame mean, but did not show meaningful motion conditioning:

| Condition | Position MSE | Velocity MSE | Body / left / right motion ratio |
|---|---:|---:|---:|
| V8, true text | 0.677434 | 0.327699 | 3.02% / 2.88% / 3.34% |
| V8, shuffled text | 0.705478 | 0.327731 | 3.05% / 2.87% / 3.39% |
| V5, true text | 0.679197 | 0.327692 | — |
| V5 training-frame-mean baseline | 0.689686 | 0.327527 | — |

The true-versus-shuffled position gain was 0.03033 with paired bootstrap 95% CI [0.02324, 0.03783]. V8's paired position comparison against V5 crossed zero, and V8 velocity was essentially tied with both V5 and frame mean. Similar true/shuffled motion ratios around 3% confirm motion collapse despite the position gain. The old V8 bone metric did not confidence-filter both endpoints, so its per-clip extreme should not be interpreted as a validated anatomy diagnosis. No checkpoint was exported; quantitative acceptance and geometry gates failed. No signer reviewed V8.

V8 exact local source hashes:

- Module: `scripts/isl_v8_ranked_transformer.py` — `5A18A61DFD0F2B1C395EE99A84FBE2710995E3901A675F53965FCD3D9361F278`
- Builder: `scripts/build_isl_v8_ranked_notebook.py` — `8B71E9595D03D405BA4CA7767F1ABC7C8B8FC6E99F7FD3C73139078F65C2C9E5`
- Notebook: `hackcessible-isl-text-to-pose-v8-ranked.ipynb` — `96029A312C2AE87BD4D209BD91C259876D98EFF8B65E3ED36FBD8988A7A57014`
- Private Kaggle run: [V8 version 1](https://www.kaggle.com/code/brubee/hackcessible-isl-text-to-pose-v8-ranked?scriptVersionId=356045347)

### V9 — completed exploratory run; corrected filtered rerun pending

V9 is a controlled follow-up to V8. It retains the V7 non-autoregressive 48-frame Transformer decoder and uses group-balanced confidence-weighted XYZ coordinate-mean pose error plus signed XYZ framewise velocity MSE and a scale-normalized aligned speed-profile auxiliary. The speed profiles are normalized separately for prediction and reference. A paired hinge compares true-caption and filtered-negative-caption temporal error on the same target; pose error is excluded from the hinge. Fit loss and inner-stop selection use pose + velocity + phase profile + the same hinge. The inner-stop negative mapping is deterministic and hashed. Same-source cluster/alias, exact normalized caption, and MiniLM embedding cosine >=0.92 negatives are excluded.

The local V5 manifest defines 9,048 fit, 989 inner-stop, and 498 development rows. Kaggle Version 2 (`356069698`) ran the earlier V9 code successfully on T4×2 in 685.2 seconds. CPU and synchronized two-GPU preflights passed; training ran six epochs, selected epoch 1, and early-stopped at epoch 6. It produced an aggregate report but exported no checkpoint because quality and geometry gates failed. Its paired 498-row development results were: V9 true-text position/velocity 0.682464/0.327668, V5 0.679197/0.327692, and training-frame mean 0.689686/0.327527. V9 therefore failed to beat V5 on position and failed to beat frame mean on velocity. It also failed the >=10% reference-motion floor, true-versus-shuffled velocity/phase checks, and several bone geometry/support checks. This was an exploratory result because it predates the pairing audit filters below.

The private pairing audit Version 3 (`356248566`) completed in 541.5 seconds. It found 10,037/10,037 fit and inner-stop pose-cache rows with V5 normalization parity, 5 fit and 1 inner-stop rows with unsupported source identity, and one MiniLM cosine>=0.92 near-caption edge crossing fit and inner-stop. The audit status is `integrity_clean_with_metadata_gaps`; source provenance relies partly on inferred UID aliases. The cache has no FPS, timestamps, sequence windows, or handedness/mirroring metadata; raw frame counts are available for 7,992/10,037 rows. It did not access raw clips, emit row-level data, or train a model.

The corrected local notebook excludes the six unsupported-source rows, removes any inner-stop row participating in a remaining fit/stop near-caption edge, asserts zero remaining cross-role near-caption edges, recomputes role/cache hashes, and keeps the 498 development rows unchanged. It is an in-split integrity correction, not a new evaluation cohort or model objective. The prior 498 development rows were already reviewed by V7, V8, and V9 Version 2; they remain development-only. No untouched source/caption/cache pool is available.

Bone diagnostics now require both endpoint confidences >=0.15, minimum support per edge and clip-group, and a minimum reference p95 length before a ratio is scorable. Aggregate per-edge and per-group reference/prediction distributions and unscorable counts are reported. Export remains blocked if any edge/group is unscorable, any per-edge or per-group p95 ratio falls outside [0.50, 2.00], any clip-group p95 ratio exceeds 3.00, or the coordinate/endpoint bounds fail. Quantitative gates also require V9 to beat V5 and frame mean on paired position and velocity intervals, show true-versus-shuffled gains in position, velocity, and phase profile, and keep all true-text motion ratios above a stated 10% floor. Passing these development checks would create only an experimental candidate, never a working-model claim.

Current corrected local V9 artifacts and SHA-256:

- Module: `scripts/isl_v9_temporal_ranked_transformer.py` — `02F823E0B93DFBE77CE05B6CF59568AB06DC25562A52C4C7369FA5D5D4CEB38A`
- Builder: `scripts/build_isl_v9_temporal_notebook.py` — `006E075EC444D3AA202A528574BEA5BF3926A2E5B17577F275A53CBB4C1D2E80`
- Notebook: `hackcessible-isl-text-to-pose-v9-ranked.ipynb` — `E357AF31F60FB6A20E77C0663848162AB98FD4D9BA13B1E4194D82BCA6D4AF60`

The builder's canonicalization fixture, Python compilation, AST parsing of all seven notebook code cells, accelerator/cell-order checks, embedded hashes, and `git diff --check` passed locally. The local Python runtime lacks PyTorch, so the current source could not run its CPU smoke test locally; the same module hash passed the CPU contract and two-GPU preflight in Kaggle Version 2. The corrected notebook has not yet been imported or run on Kaggle. Kaggle Version 3 (`356087946`) is a separate retrieval probe under the V9 title; it failed after 975.4 seconds on a donor-support assertion and did not train a model. Do not confuse it with a V9 run.

## Separate app status

The React/FastAPI prototype serves local English captions, session-level speaker labels, and acoustic/prosody cues. It uses local VAD/Whisper/SpeechBrain paths when model assets are available. Demo speaker-direction values are scripted/simulated; live direction from its mono microphone path is unavailable, simultaneous speech is not separated, and diarization/prosody accuracy is not benchmarked. The text-to-pose ISL model is **not integrated** into that app.

docs/VALIDATION.md documents a Windows validation run from 2026-09-27: 30 backend tests, 13 frontend tests, lint/build/demo checks, a prerecorded PCM WebSocket run, and five controlled synthetic-noise conditions. Those are app-path checks, not ISL model evidence, real-room/microphone accuracy, or formal accessibility validation.

## Required continuation and gates

1. **Check exact source and version state.** Regenerate the notebook and confirm the three hashes above; check Kaggle versions before submitting. The latest local V9 notebook is filtered and its hash differs from the older successful Kaggle V9 version.
2. **Verify the filter contract.** Confirm the V5 manifest has exactly 5 unsupported fit and 1 unsupported stop row, then require the notebook to remove those rows, exclude any stop row connected by a fit/stop cosine>=0.92 edge, assert zero remaining near-caption overlap, and recalculate role/cache hashes. Stop and rerun the pairing audit if its counts change.
3. **Use the existing role split only.** The unfiltered roles are 9,048 fit, 989 inner-stop, and 498 previously examined development rows. Source aliases and exact normalized captions must be disjoint across active roles after filtering. The cache has no untouched final cohort; do not describe a holdout built from V5 training rows as independent.
4. **Run the corrected notebook privately on T4×2.** Confirm V5 artifact and pose-cache inputs, exact checkpoint/normalizer/encoder/cache hashes, and full target/reference parity. The notebook must pass its CPU contract and synchronized two-GPU probe with finite, nonzero pose/velocity/phase gradients and at least one valid filtered pair before the first epoch. Capture any failure and stop before training.
5. **Evaluate only as development evidence.** Compare true-caption with shuffled and no-text controls, and with the V5 true-text and frame-mean predictions on the same 498 rows. Report paired position/velocity/phase intervals, motion ratios, confidence-filtered bone diagnostics, coordinate/endpoint bounds, and every gate. Keep outputs aggregate-only and do not export a checkpoint unless all gates pass.
7. **Keep all gated material private.** Export only aggregate metrics and a gated private experimental checkpoint. Never export clip-level poses/videos/captions/UIDs/aliases, contact sheets, or secrets. Do not modify the React/FastAPI app as part of this model experiment.
8. **Update both handoff documents with evidence.** Distinguish source/build checks, Kaggle import, CPU/GPU runtime, training, development metrics, geometry, export, and signer review. Record any failure. Do not describe a checkpoint or trained experiment as a working ISL translator.

## Definition of success

A corrected V9 runtime milestone requires the pairing filters to pass, input/cache/checkpoint parity, CPU and synchronized T4×2 preflights, and a completed training/evaluation run with aggregate evidence. An experimental checkpoint can be retained only when the inner-stop pose guardrail, paired quantitative checks, text-conditioning comparisons, non-collapse floor, confidence-filtered anatomy coverage, and geometry gates all pass. The 498-row V7/V8/V9 cohort remains development-only; there is no independent final test. A **working-model claim** additionally requires a fresh untouched source/caption-disjoint cohort, target-free inference that accepts text without target poses, and blinded fluent-ISL-signer review for meaning, handshape, spatial grammar, and naturalness. Training completion, a lower coordinate loss, or checkpoint export alone does not establish that signing works.
