# Hackcessible India 2026 — Conversational Visual Layer

Hackcessible is a local web-app prototype for showing English captions with session-level speaker labels and measurable acoustic cues. It is aimed at Deaf and hard-of-hearing students in classes, seminars, and group discussions. Its accuracy and accessibility have not been clinically or formally evaluated.

## What the prototype does

- **Captions:** Browser microphone audio is sent as 16 kHz mono PCM16 to the local FastAPI service. Silero VAD finds speech segments and Faster-Whisper transcribes English locally.
- **Speaker labels:** SpeechBrain ECAPA-TDNN embeddings are clustered into session speaker labels. If the neural model cannot load, the app falls back to handcrafted MFCC features. These labels are not verified identities, and diarization accuracy has not been benchmarked.
- **Acoustic cues:** During live audio, a 20-second chart shows estimated voiced pitch, mic level (dBFS), recognized pace, active speech, and possible emphasis; current readings also show pitch trend and speech/pause timing. Final captions include utterance-level pitch range, pace, level, and heuristic cues. These are acoustic estimates, not emotion or intent labels.
- **Overlap:** The prototype has an experimental acoustic overlap heuristic. The demo's overlap labels are scripted; it does not separate simultaneous voices into independent transcripts.
- **Direction:** Live direction is unavailable with the current mono microphone path. The demo includes explicitly simulated direction values.

## Architecture

```text
Browser microphone
  → 16 kHz mono PCM16 over localhost WebSocket
  → Silero VAD (512-sample / 32 ms frames)
  → Faster-Whisper English transcription
  → ECAPA speaker embedding / MFCC fallback
  → live and utterance-level acoustic cue extraction
  → React caption interface
```

Models may be downloaded or checked through Hugging Face during startup when they are not cached. Once available, speech inference runs locally. Pipeline timing in diagnostics measures backend processing after a speech segment is finalized; it does not measure the whole time a user waits from speaking to seeing a caption.

## Run on Windows

### One-click launcher

Run:

```cmd
scripts\run_app.bat
```

The launcher starts the local service and opens `http://127.0.0.1:8000`.

### Manual setup

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\pip.exe install -r backend\requirements.txt
cd frontend
npm.cmd install
npm.cmd run build
cd ..
$env:PYTHONPATH='backend'
backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The first model setup can require internet access and may take time; later starts can use the local model cache.

## Use the app

- **Live microphone:** Select **Start Mic** and grant the browser permission. Audio is streamed to the local backend. The browser checks that its audio context uses the required 16 kHz sample rate; unsupported rates are reported instead of silently being mislabeled.
- **Demo:** Select **Run 3-Speaker Demo**. This replays preset caption, speaker, overlap, prosody, and simulated-direction events from `samples/demo_conversation.json`. It checks the scripted display flow, not microphone capture, VAD, transcription, or diarization.
- **Privacy and saving:** Privacy Mode blocks transcript writes to disk. To save transcript text in local SQLite, turn Privacy Mode off and separately enable **Opt-In: Persist Transcript to Local SQLite**. Do not enable this for sensitive conversations unless local retention is appropriate.
- **Display controls:** Font size, a high-contrast theme, reduced motion, and feature toggles are available. The interface has not had a formal WCAG conformance audit.
- **Export:** Export the current in-memory session as plain text, WebVTT, or JSON.

## Privacy notes

- The checked-in inference path does not send microphone audio or transcript text to a third-party speech API. The app binds to localhost in the supplied launcher.
- First startup may contact Hugging Face for model metadata or uncached model files. The Privacy Mode toggle controls transcript disk persistence; it is not a network firewall.
- Audio segments are held in process memory while analyzed and their buffer references are released after finalization. This is not a promise of secure memory erasure.
- Transcript text stays in session memory by default. If Privacy Mode is off and transcript saving is enabled, text is written to `backend/session_storage.db`; the REST export endpoint can read persisted sessions when their session IDs are known.
- Accessibility settings are stored in the browser's local storage.

## Validation

Run the repository checks with:

```cmd
scripts\test_all.bat
```

This runs backend and frontend tests and a scripted-demo WebSocket check. If you supply an authorized local clip at `samples/test_video_60s.wav`, it also replays PCM and controlled synthetic-noise cases through the real WebSocket path. That recording, downloaded video, and cached model weights are intentionally not committed. Without the local WAV, the two sample-dependent replay stages are skipped. Synthetic noise does not establish real-room accuracy, and no physical microphone test is claimed. See [the validation report](docs/VALIDATION.md) for results and limits.

## Known limitations

1. The shipped Whisper model is `tiny.en`, so the supported recognition path is English-only. Other languages and accent accuracy have not been validated.
2. Speaker clustering and overlap heuristics have not been evaluated against a labeled benchmark; the app may merge or split speakers incorrectly.
3. Simultaneous speech is not source-separated. Scripted demo overlap is not evidence of live overlap accuracy.
4. Live spatial direction and microphone-array hardware support are not implemented.
5. Physical microphone behavior, performance on other computers, and WCAG conformance still need direct testing.
