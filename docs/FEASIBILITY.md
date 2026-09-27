# Hackcessible — Feasibility and Readiness

## Summary

Hackcessible is a working software prototype for a local English captioning flow. Its UI, WebSocket protocol, voice-activity detection, ASR, speaker clustering, and acoustic cue code can be exercised on this development machine. That evidence does not establish caption accuracy, reliable speaker identification, accessibility conformance, or suitability for real clinical or classroom use.

Inference runs locally after the required model files are available. First-time startup may contact Hugging Face to check or fetch the Faster-Whisper and SpeechBrain models. Privacy Mode only blocks transcript writes to disk; it does not block network traffic.

## Capability and evidence

| Capability | Current implementation | Evidence and limit |
|---|---|---|
| Browser microphone stream | Web Audio API sends mono PCM16 at the requested 16 kHz rate | Code path and sample-audio WebSocket replay checked; physical microphone and browser permission flow not exercised |
| Voice activity detection | Silero VAD processes 512-sample frames | Backend tests, prerecorded WebSocket replay, and controlled synthetic-noise replays; no real-room or microphone evaluation |
| English speech recognition | Faster-Whisper `tiny.en`, CPU INT8 by default | Produces captions from bundled sample audio; synthetic noise replay stayed near the clean model output at 10–20 dB SNR and degraded under the 5 dB competing-speech proxy; no ground-truth word-error benchmark and no multilingual path |
| Speaker clustering | SpeechBrain ECAPA-TDNN embeddings with a handcrafted MFCC fallback | Synthetic regression and sample replay; no labeled diarization benchmark or real classroom validation |
| Acoustic cues | Final-utterance pitch mean/range/contour, RMS level, speaker-relative level, pause-before, ASR words per minute, and a heuristic emphasis cue | Unit tests and live WebSocket replay; acoustic estimates only, with no labeled accuracy or interpretation validation |
| Live prosody display | A rolling 20-second plot and current estimates for voiced F0/trend, dBFS, recognized pace, speech/pause timing, and possible level-plus-pitch emphasis | The live pipeline emits about 5–6 measurements per second; actual PCM replay reached the browser view. The emphasis cue is heuristic and none of these features is an emotion label |
| Overlap indicator | Acoustic heuristic in live pipeline; canned flags in demo data | UI and scripted demo event checked; live overlap detection is unbenchmarked, and voices are not separated |
| Spatial direction | Simulated values in demo fixtures; null provider in the live mono path | Demo labels are marked simulated; no microphone-array estimator is implemented |
| Transcript privacy | In-memory by default; local SQLite requires Privacy Mode off plus save-transcript opt-in | Storage tests check both blocking and opted-in writing/readback; this does not audit operating-system memory or disk behavior |
| Accessibility controls | Font scaling, contrast theme, reduced motion, labels, and toggles | Component checks and build; no formal WCAG audit or assistive-technology evaluation |
| Demo mode | Scripted WebSocket events from JSON | Confirms the display/event protocol only; it does not run VAD, ASR, or diarization |

## Models and runtime requirements

- **Speech recognition:** Faster-Whisper `tiny.en` with CTranslate2. The English model may be downloaded on first use and cached locally.
- **Speaker embeddings:** SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`). Model loading can require a download. If loading fails, the implementation logs a warning and uses a handcrafted MFCC fallback, which has not been quality-benchmarked.
- **Voice activity detection:** Silero VAD ONNX.
- **Deployment target:** The supplied launcher binds to `127.0.0.1`; this repository has not established minimum CPU, RAM, disk, or startup-time requirements. Performance depends on the machine and cached models.

No reliable hardware minimum or cross-machine latency claim is available yet. `pipeline_processing_ms` records backend work after VAD closes a speech segment; it excludes the speech duration and should not be described as user-perceived end-to-end caption latency.

## Data handling

1. Microphone samples remain in process memory while the speech segment is analyzed. The code releases its buffer references after finalization; Python does not guarantee secure memory erasure.
2. Transcript events are held in session memory by default.
3. When Privacy Mode is off and transcript saving is enabled, the backend persists transcript fields to `backend/session_storage.db`. The REST export endpoint can read persisted rows when the session ID is known.
4. Application startup may make model metadata or model-file requests to Hugging Face. The checked-in pipeline does not send audio or transcript text to a third-party inference API.

## Scope that remains future work

- Benchmark transcription and diarization on representative, consented classroom audio.
- Test with physical microphones, browser permissions, noisy rooms, simultaneous speech, and lower-powered hardware.
- Implement and validate live multichannel direction estimation; the existing future-array class is only a placeholder.
- Add source separation if simultaneous speakers need independent captions.
- Run keyboard, screen-reader, color-contrast, and WCAG conformance checks with disabled users.
- Review consent, retention, and safeguards before any use with sensitive clinical discussions.
