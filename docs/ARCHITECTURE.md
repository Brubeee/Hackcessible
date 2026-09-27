# Hackcessible India 2026 — System Architecture

## Overview
Hackcessible is a local web-app prototype for displaying English captions with session-level speaker labels and acoustic cues. Its accuracy and accessibility have not been formally evaluated.

Standard live captions communicate only **what** was said. Hackcessible transforms spoken dialogue into an accessible visual conversation interface by simultaneously communicating:
1. **WHO** is speaking (session-level speaker clusters and color-text badges; not verified identities)
2. **WHAT** they said (English speech-to-text with rolling interim updates)
3. **WHEN** speech may overlap (live heuristic and scripted demo events; not benchmarked)
4. **HOW** it was said (estimated acoustic cues: pitch, energy, speaking rate, pauses)
5. **WHERE** a speaker is located (simulated demo values only; live mono input reports no direction)

---

## High-Level Pipeline Flow

```mermaid
flowchart TD
    subgraph Browser ["Client: React + TypeScript Frontend"]
        MIC["Microphone Audio Capture\n(Web Audio API / 16kHz mono context)"]
        UI["Accessible Visual Interface\n(Captions, Overlaps, Prosody, Speakers)"]
        WS_CLIENT["Client WebSocket\n(Binary PCM + Event Handlers)"]
    end

    subgraph Backend ["Server: FastAPI + Python 3.10 Backend"]
        WS_SERVER["WebSocket Server\n(/ws/conversation)"]
        BUFFER["Speech Segment Accumulator\n(PCM16 input converted to Float32)"]
        VAD["Silero VAD ONNX\n(32ms frame chunking)"]
        
        subgraph SegmentAnalysis ["Sequential Segment Processing"]
            ASR["Faster-Whisper ASR\n(tiny.en INT8 Quantized)"]
            DIAR["Speaker Clusterer\n(SpeechBrain ECAPA; MFCC fallback)"]
            PROS["Prosody Signal Analyzer\n(F0 Trajectory, RMS vs Baseline, Rate)"]
            DOA["Direction Provider\n(Null for live mono input)"]
        end

        FUSION["Event Fusion Pipeline\n(Overlap Detection & Schema Mapping)"]
        REGISTRY["Session Speaker Registry\n(Persistent Color & Rename Store)"]
        STORAGE["Privacy Storage Engine\n(Ephemeral Memory / Opt-in SQLite)"]
    end

    MIC -->|Binary PCM Chunks| WS_CLIENT
    WS_CLIENT <-->|ws://127.0.0.1:8000/ws/conversation| WS_SERVER
    WS_SERVER --> BUFFER
    BUFFER --> VAD
    VAD -->|Speech Segments| ASR
    VAD -->|Speech Segments| DIAR
    VAD -->|Speech Segments| PROS
    DIAR <--> REGISTRY
    PROS <--> REGISTRY
    ASR --> FUSION
    DIAR --> FUSION
    PROS --> FUSION
    DOA --> FUSION
    FUSION --> STORAGE
    FUSION -->|UtteranceEvents| WS_SERVER
    WS_SERVER -->|JSON Stream| WS_CLIENT
    WS_CLIENT --> UI
    HF[("Hugging Face model files / metadata")]
    ASR -.->|Startup may check/download if uncached| HF
    DIAR -.->|Startup may check/download if uncached| HF
```

---

## Sequence Diagram: Live Utterance Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor Speaker as Participant
    participant Mic as Browser Audio
    participant WS as WebSocket
    participant VAD as Silero VAD ONNX
    participant Pipeline as Fusion Pipeline
    participant ASR as Faster-Whisper
    participant Diar as Acoustic Diarizer
    participant Pros as Prosody Analyzer
    participant UI as React Transcript

    Speaker->>Mic: "Could this be linked to social anxiety?"
    Mic->>WS: Stream 16kHz PCM chunks
    WS->>VAD: Ingest 512-sample frame (32ms)
    VAD-->>Pipeline: Speech probability = 0.94 (Speaking=True)
    Pipeline-->>WS: VAD State update (speaking indicator)
    WS-->>UI: Audio visualizer turns green

    opt Rolling Interim Updates (Every 1.2s of continuous speech)
        Pipeline->>ASR: Transcribe partial buffer
        ASR-->>Pipeline: "Could this be linked..."
        Pipeline-->>WS: UtteranceEvent (is_final=False)
        WS-->>UI: Display live italics caption with cursor
    end

    Speaker->>Mic: [Pauses / Finishes sentence]
    Mic->>WS: Stream silence chunks
    VAD-->>Pipeline: Silence threshold reached (Segment Ended)
    
    Pipeline->>ASR: Transcribe finalized audio segment
    ASR-->>Pipeline: English text and word timings
    Pipeline->>Pipeline: Check candidate speaker-turn partitions
    Pipeline->>Diar: Extract ECAPA embedding (or MFCC fallback) and cluster
    Diar-->>Pipeline: Session speaker label
    Pipeline->>Pros: Calculate acoustic cues from this turn
    Pros-->>Pipeline: Estimated pitch, rate, energy, and pause fields

    Pipeline->>Pipeline: Check overlap against preceding utterances
    Pipeline-->>WS: UtteranceEvent (is_final=True)
    WS-->>UI: Render finalized UtteranceCard with [Speaker B] badge & ↗ Rising intonation
    Pipeline->>Pipeline: Clear segment accumulator references after processing
```

---

## Event Data Model

The pipeline communicates through a strongly-typed, JSON-serializable event model:

```json
{
  "id": "utt-8f4b-4a31-92b1",
  "session_id": "session_1774431800",
  "start_time": 14.2,
  "end_time": 17.6,
  "speaker_id": "speaker_2",
  "speaker_label": "Speaker B (Priya)",
  "speaker_color_index": 1,
  "text": "Could this pattern be explained primarily by social anxiety disorder?",
  "is_final": true,
  "overlap": false,
  "overlapping_speakers": [],
  "prosody": {
    "rising_intonation": true,
    "falling_intonation": false,
    "emphasis": false,
    "long_pause_before": false,
    "speech_rate": "normal",
    "relative_volume": "normal",
    "mean_pitch_hz": 218.6,
    "rms_db": -20.8,
    "words_per_minute": 156.0
  },
  "direction": {
    "angle_degrees": 0.0,
    "label": "front",
    "simulated": true,
    "confidence": 0.88
  },
  "confidence": {
    "transcription": 0.94,
    "speaker": 0.91
  }
}
```

---

## Models and external startup requests

Speech inference uses local models once they are available. Faster-Whisper and SpeechBrain can contact Hugging Face during model setup to check or fetch files; the live inference path does not send audio or transcripts to a cloud speech API. Pipeline processing time measures backend work after a segment closes and is not full user-perceived caption latency.

## Loose Coupling and Modularity

Every major subsystem is decoupled behind clean abstract interfaces:

1. **`BaseASR`** (`backend/app/asr/base.py`):
   Allows swapping Faster-Whisper (`tiny.en`, `base.en`, `small.en`) with Whisper.cpp, Vosk, or multilingual models.
2. **`BaseDiarizer`** (`backend/app/diarization/base.py`):
   Separates session-level speaker clustering from speech recognition; the current clusterer uses ECAPA-TDNN with an MFCC fallback.
3. **`DirectionProvider`** (`backend/app/direction/base.py`):
   Returns no direction for live mono audio. Demo records contain simulated direction values; the future array class is only a placeholder.
4. **`SessionStorage`** (`backend/app/storage/database.py`):
   Keeps transcripts in memory by default; persistence requires Privacy Mode off and explicit transcript-save opt-in.
