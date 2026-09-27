"""Unit tests for Event and Utterance data models."""
import pytest
from app.models.events import (
    ConfidenceInfo,
    DebugMetrics,
    DirectionInfo,
    ProsodyFeatures,
    SpeakerInfo,
    UtteranceEvent,
)


def test_prosody_features_defaults():
    prosody = ProsodyFeatures()
    assert prosody.rising_intonation is False
    assert prosody.falling_intonation is False
    assert prosody.emphasis is False
    assert prosody.long_pause_before is False
    assert prosody.speech_rate == "normal"
    assert prosody.relative_volume == "normal"
    assert prosody.mean_pitch_hz is None


def test_utterance_event_serialization():
    event = UtteranceEvent(
        id="utt-101",
        session_id="test_session",
        start_time=1.2,
        end_time=3.8,
        speaker_id="speaker_1",
        speaker_label="Speaker A",
        speaker_color_index=0,
        text="Could anxiety be involved here?",
        overlap=False,
        prosody=ProsodyFeatures(rising_intonation=True, relative_volume="loud"),
        direction=DirectionInfo(angle_degrees=-45.0, label="left", simulated=True)
    )
    d = event.model_dump()
    assert d["id"] == "utt-101"
    assert d["text"] == "Could anxiety be involved here?"
    assert d["prosody"]["rising_intonation"] is True
    assert d["prosody"]["relative_volume"] == "loud"
    assert d["direction"]["simulated"] is True
    assert d["direction"]["angle_degrees"] == -45.0


def test_overlap_utterance():
    event = UtteranceEvent(
        start_time=5.0,
        end_time=7.0,
        speaker_id="speaker_2",
        speaker_label="Speaker B",
        text="Interrupting comment",
        overlap=True,
        overlapping_speakers=["Speaker A"]
    )
    assert event.overlap is True
    assert "Speaker A" in event.overlapping_speakers


def test_debug_metrics_are_real_by_default_and_can_mark_demo_values():
    values = {
        "chunk_duration_ms": 1600.0,
        "vad_latency_ms": 4.2,
        "asr_latency_ms": 78.5,
        "diarization_latency_ms": 14.3,
        "prosody_latency_ms": 6.1,
        "pipeline_processing_ms": 103.1,
        "active_speakers_count": 3,
        "queue_backlog": 0,
    }

    assert DebugMetrics(**values).simulated is False
    assert DebugMetrics(**values, simulated=True).model_dump()["simulated"] is True
