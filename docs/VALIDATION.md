# Hackcessible — Validation Report

## Verification scope

Checks were run on the Windows development machine on 2026-09-27. The combined runner covers unit/component tests, lint, a frontend production build, scripted demo WebSocket events, a prerecorded PCM audio stream through the live WebSocket input path, and controlled synthetic-noise replays through that same path. It does not grant microphone permission or test a physical microphone.

Run the current checks with:

```cmd
scripts\test_all.bat
```

This report records a local run with `samples/test_video_60s.wav` available. The audio recording, its downloaded source video, and cached model weights are excluded from the public repository; the runner skips the two sample-dependent replay stages when the local WAV is absent.

## Automated results

| Check | Result | What it demonstrates |
|---|---|---|
| Backend pytest | 30 passed | Event models, prosody signal analysis, bounded live prosody events, pipeline logic, privacy/storage behavior, REST and WebSocket integrations, local WebSocket-origin allowlist |
| Frontend Vitest | 13 passed | Caption components, live prosody display including missing-data states, mic-capture protocol logic, neutral cue labels, simulated-metric disclosure, and privacy-setting gating |
| Frontend lint | Passed, no warnings | Oxlint reported no findings |
| Production frontend build | Passed | TypeScript compilation and Vite production bundling completed |
| Scripted demo WebSocket run | Passed | Health endpoint, connection, scripted three-speaker events, fixture overlap/cues, metric simulation flag, rename broadcast, and no SQLite write in default Privacy Mode |
| Prerecorded PCM WebSocket run | Passed | Actual 16 kHz PCM16 input traversed the live VAD/ASR/diarization/prosody pipeline and emitted captions, 317 live prosody updates, and processing metrics |
| Controlled-noise WebSocket runs | Passed in all 5 conditions | Clean baseline plus pink noise at 20/10 dB SNR, synthetic fan/hum at 10 dB, and a delayed-copy competing-speech proxy at 5 dB; results below |

Pytest emits one environment deprecation warning from Starlette's `TestClient` integration with the installed `httpx`; it does not fail the tests.

## Prerecorded audio output

The WebSocket verifier streamed `samples/test_video_60s.wav` plus one second of trailing silence as PCM16 frames. The captured run produced:

| Measurement | Result |
|---|---:|
| Audio supplied | 61.0 seconds at 16 kHz |
| Wall time for offline replay | 31.34 seconds |
| Final captions | 17 |
| Captions with recognized text | 16 |
| Interim captions | 20 |
| Session speaker clusters | 3 |
| VAD updates | 1,907 |
| Live prosody updates | 317 |
| Pipeline-processing samples | 16 |
| Median pipeline processing | 905.11 ms |
| Maximum pipeline processing | 2,015.46 ms |

`pipeline_processing_ms` times backend work after VAD finalizes a segment. The figures are not end-to-end caption latency and do not show how a physical microphone performs. The stream included one final placeholder event without recognized text. The run also explicitly requests spatial direction and verifies that the live mono path returns no direction.

## Controlled-noise caption replay

The clean 61-second replay also produced 317 audio-timestamped live prosody updates. Messages include voiced pitch and pitch trend, measured RMS level, recognized speaking pace when available, speech and pause timing, and a possible-emphasis heuristic. The UI displays a rolling 20-second chart. These values are estimates; they are not emotion or intent classifications.

`scripts/verify_noisy_audio.py` takes a speech-rich 20-second crop (20–40 seconds) from the bundled recording, mixes repeatable synthetic noise, and sends each 21-second stream (including trailing silence) through the live 16 kHz PCM WebSocket. Every condition emitted 109 live prosody updates.

| Condition | Final captions / recognized | Token edit rate vs clean model output | Observed result |
|---|---:|---:|---|
| Clean baseline | 8 / 8 | 0.000 | Reference output for this model run |
| Pink noise, 20 dB SNR | 8 / 8 | 0.030 | Output stayed close to the clean model output |
| Pink noise, 10 dB SNR | 9 / 9 | 0.030 | Output stayed close; one utterance was split into two captions |
| Synthetic fan/hum, 10 dB SNR | 9 / 9 | 0.030 | Output stayed close; one utterance was split into two captions |
| Delayed-copy speech-babble proxy, 5 dB SNR | 13 / 13 | 0.576 | More fragmented, duplicated, and degraded text appeared |

The edit rate compares noisy output with the recognizer's own clean output, **not WER or accuracy**; no human transcript was used. The speech-babble condition is a delayed, shifted copy of the same recording with added low-level noise, not independent talkers. These deterministic software stress checks are not real room recordings. The competing-speech proxy degrades captions and does not establish reliable overlap handling.

Separately, `scripts/test_subsegment_split.py` passed its selected-clip smoke check: different-speaker clip similarities were -0.002 and 0.045 (below 0.25), while same-speaker clip similarities were 0.483 and 0.597. This checks the examples and threshold in that script; it is not a diarization-accuracy benchmark.

## Actual browser output

The built app was opened at `http://127.0.0.1:8000` and the **Run 3-Speaker Demo** control was used. The rendered page showed the scripted speaker labels, interim and final caption cards, overlap alerts, and acoustic-cue badges. The diagnostics drawer displayed a simulated-metrics notice and `--` in place of the demo's fabricated timings. Browser console error and warning logs were empty.

This is evidence that the interface and scripted WebSocket flow render. The demo reads preset events from JSON; it does not exercise microphone capture, VAD, ASR, or diarization. Separately, a local client streamed bundled PCM through the live WebSocket while the app remained open. The browser displayed the **Live vocal cues** panel, its populated history chart, and live caption cards, verifying event delivery to the React view without granting microphone permission.

## Startup and privacy observations

- The observed startup log made a GET request to Hugging Face's Faster-Whisper model metadata endpoint. The ASR and SpeechBrain weights were already cached and loaded locally during this run. Therefore, inference is local, but startup is not guaranteed to be offline.
- The source path processes microphone PCM in the local service and does not call a third-party speech-inference API. No packet-level network audit was performed.
- Browser WebSocket connections accept the local served app and documented localhost Vite origins; an untrusted website origin is rejected. Non-browser clients without an `Origin` header remain supported for local tools.
- Default Privacy Mode prevented the demo WebSocket run from creating a SQLite transcript file. Separate storage tests verify that explicit opt-in writes and reads transcript rows and that Privacy Mode blocks disk writes.
- Audio buffers are released after segment processing; secure erasure from process memory was not tested or claimed.

## Still unverified

- Physical browser microphone, operating-system device selection, permission prompts, and actual room audio.
- Word error rate, speaker diarization accuracy, live overlap precision/recall, or acoustic-cue accuracy on labeled human speech.
- Performance on other machines, real room noise, reverberant recordings, and independent overlapping talkers.
- Prosody-estimation accuracy on labeled human speech.
- Formal WCAG conformance and evaluation with assistive technologies or Deaf/hard-of-hearing participants.
- Live microphone-array direction estimation and independent transcription of simultaneous speakers.
