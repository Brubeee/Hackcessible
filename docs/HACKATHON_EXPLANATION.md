# Hackcessible — Project Overview for Judges

## The problem

In a group discussion, a plain caption stream can make it difficult to follow which turn belongs to which participant. Deaf and hard-of-hearing students may also benefit from clear visual cues when speech is fast, a pause occurs, or the system estimates a pitch change.

## What this prototype does

Hackcessible displays English captions with session-level speaker labels and heuristic acoustic cues. A label such as “Speaker A” is a software cluster, not a verified identity. The interface avoids emotion labels and leaves interpretation to the user.

- **Who:** ECAPA-TDNN speaker embeddings are clustered into labels for the current session. If the neural model does not load, the app falls back to handcrafted MFCC features. Speaker accuracy has not been benchmarked.
- **What:** Faster-Whisper `tiny.en` transcribes English speech locally once model files are available. It is not a multilingual model, and accuracy has not been measured on a representative dataset.
- **When:** The live pipeline has a heuristic overlap check. The demo contains preset overlap events. Neither demonstrates reliable live overlap detection, and simultaneous voices are not separated.
- **How:** Pitch, pace, pause, emphasis, and relative-volume cues are computed from acoustic features. They are estimates, not claims about emotion or mental state.
- **Where:** Live direction is unavailable in the mono microphone path. Demo direction values are simulated and labeled as such.

## How the live path works

The browser requests microphone access, creates a 16 kHz mono audio context, converts frames to PCM16, and streams them to the local FastAPI WebSocket. The backend applies Silero voice-activity detection and sends low-rate live prosody measurements while audio arrives. The interface plots the most recent 20 seconds of voiced pitch, microphone level, recognized pace when available, speech activity, and possible emphasis. A pause and a short-term pitch trend also appear in the current reading. When VAD finalizes a segment, the backend runs speech recognition, speaker clustering, and utterance-level acoustic analysis; the resulting caption card can show pitch range, measured pace, level, and heuristic cues. These are measurable acoustic estimates, not emotion or intent labels. Prerecorded PCM and the rendered live panel have been checked; a physical microphone has not been tested as part of automated verification.

Timing shown in diagnostics is backend pipeline processing time after VAD finalizes a segment. It is not the full wait from the start or end of a person's speech to the caption appearing. The separate validation report lists the captured replay results.

## Demo mode

Demo mode reads preset events from `samples/demo_conversation.json` and broadcasts their text, labels, overlap flags, acoustic cues, and simulated directions. This is useful for reviewing the interface and WebSocket event format without microphone access. It bypasses microphone streaming, VAD, speech recognition, and diarization; it is not a live-pipeline fallback or a reliability guarantee.

## Privacy and model setup

Speech inference runs locally when the models are present. Startup may contact Hugging Face to check or download Faster-Whisper and SpeechBrain model files. The checked-in inference path does not send audio or transcript text to a cloud speech API. Privacy Mode blocks transcript writes to disk; users must turn it off and separately opt in to local SQLite transcript saving. Audio segments are held in memory during processing, and the program releases its buffer references afterward without promising secure erasure.

## Readiness limits

This is a software prototype, not a validated accessibility or clinical tool. It still needs physical-microphone testing, representative accuracy benchmarks, real noisy-room and independent-overlap tests, assistive-technology review, and formal accessibility evaluation. Controlled synthetic-noise replay is documented in [Validation](VALIDATION.md); it is not field evidence. See [Feasibility](FEASIBILITY.md) for readiness limits.
