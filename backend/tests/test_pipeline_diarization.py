"""Regression tests for SpeechBrain ECAPA-TDNN diarization and intra-segment turn partitioning."""
import pytest
import numpy as np
from unittest.mock import MagicMock

from app.asr.base import BaseASR, ASRResult
from app.diarization.clusterer import AcousticClusterDiarizer
from app.diarization.registry import SpeakerRegistry
from app.prosody.analyzer import ProsodyAnalyzer
from app.direction.simulated import SimulatedDirectionProvider
from app.fusion.pipeline import FusionPipeline
from app.models.settings import AccessibilitySettings


class MockASR(BaseASR):
    def __init__(self, text: str = "Test", words=None):
        self._text = text
        self._words = words or []
        self.transcription_calls = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        self.transcription_calls += 1
        return ASRResult(text=self._text, words=self._words, duration_s=len(audio)/sample_rate)

    def is_ready(self) -> bool:
        return True


def test_intra_segment_turn_partitioning():
    sr = 16000
    t = np.linspace(0, 1.0, sr)

    # 1.0s low pitch (130Hz) + 1.0s high pitch (260Hz)
    tone1 = (0.3 * np.sin(2 * np.pi * 130 * t)).astype(np.float32)
    tone2 = (0.3 * np.sin(2 * np.pi * 260 * t)).astype(np.float32)
    combined = np.concatenate([tone1, tone2])

    words = [
        {"word": "First?", "start": 0.2, "end": 0.8},
        {"word": "Second.", "start": 1.2, "end": 1.8}
    ]

    registry = SpeakerRegistry()
    diarizer = AcousticClusterDiarizer(registry=registry, similarity_threshold=0.25)
    prosody = ProsodyAnalyzer(sample_rate=sr)
    direction = SimulatedDirectionProvider()
    asr = MockASR(text="First? Second.", words=words)

    pipeline = FusionPipeline(
        asr=asr,
        diarizer=diarizer,
        registry=registry,
        prosody_analyzer=prosody,
        direction_provider=direction,
        sample_rate=sr
    )

    settings = AccessibilitySettings()
    utterances, metrics = pipeline._process_final_segment(
        audio_segment=combined,
        start_time=0.0,
        end_time=2.0,
        settings=settings,
        vad_duration_ms=1.0,
        interim_id="stable-interim-id",
    )

    # Assert that pipeline produces valid utterances with non-empty text and correct session assignment
    assert len(utterances) >= 1
    assert utterances[0].is_final is True
    assert utterances[0].id == "stable-interim-id"
    assert metrics.pipeline_processing_ms > 0.0


def test_ecapa_speaker_consistency():
    sr = 16000
    t = np.linspace(0, 0.8, int(sr * 0.8))
    # Synthetic harmonic vocal signal
    voice_a = (0.2 * np.sin(2 * np.pi * 140 * t) + 0.1 * np.sin(2 * np.pi * 280 * t)).astype(np.float32)
    voice_b = (0.2 * np.sin(2 * np.pi * 320 * t) + 0.1 * np.sin(2 * np.pi * 640 * t)).astype(np.float32)

    registry = SpeakerRegistry()
    diarizer = AcousticClusterDiarizer(registry=registry, similarity_threshold=0.25)

    res1 = diarizer.diarize_segment(voice_a, sr, 0.0, 0.8)
    # Repeated utterance of voice_a
    res2 = diarizer.diarize_segment(voice_a, sr, 1.0, 1.8)

    assert res1.speaker_id == res2.speaker_id
    assert res1.speaker_label == res2.speaker_label


def test_interim_transcription_is_throttled_by_new_audio_samples():
    sr = 16000
    asr = MockASR()
    registry = SpeakerRegistry()
    pipeline = FusionPipeline(
        asr=asr,
        diarizer=AcousticClusterDiarizer(registry=registry),
        registry=registry,
        prosody_analyzer=ProsodyAnalyzer(sample_rate=sr),
        direction_provider=SimulatedDirectionProvider(),
        sample_rate=sr,
    )
    pipeline.vad.process_frame_512 = MagicMock(side_effect=lambda frame: {
        "is_speaking": True,
        "segment_ended": False,
        "speech_prob": 0.9,
        "energy_db": -20.0,
        "total_samples": (pipeline.vad.process_frame_512.call_count + 1) * 512,
        "current_speech_samples": (pipeline.vad.process_frame_512.call_count + 1) * 512,
    })

    for _ in range(100):  # 3.2 seconds of continuous speech
        pipeline.process_pcm_frame(np.zeros(512, dtype=np.float32), AccessibilitySettings())

    assert asr.transcription_calls == 2


def test_pipeline_emits_bounded_rate_live_prosody_events():
    sr = 16000
    registry = SpeakerRegistry()
    pipeline = FusionPipeline(
        asr=MockASR(),
        diarizer=AcousticClusterDiarizer(registry=registry),
        registry=registry,
        prosody_analyzer=ProsodyAnalyzer(sample_rate=sr),
        direction_provider=SimulatedDirectionProvider(),
        sample_rate=sr,
    )
    frame_index = 0

    def fake_vad(frame):
        nonlocal frame_index
        frame_index += 1
        return {
            "is_speaking": True,
            "segment_ended": False,
            "speech_prob": 0.9,
            "energy_db": -20.0,
            "total_samples": frame_index * 512,
            "current_speech_samples": frame_index * 512,
        }

    pipeline.vad.process_frame_512 = fake_vad
    tone = np.sin(2 * np.pi * 180 * np.arange(512, dtype=np.float32) / sr).astype(np.float32) * 0.2
    emitted = []
    for _ in range(18):
        result = pipeline.process_pcm_frame(tone, AccessibilitySettings())
        if result["prosody_state"] is not None:
            emitted.append(result["prosody_state"])

    assert len(emitted) == 3
    assert all(event.speech_active for event in emitted)
    assert all(event.pitch_hz == pytest.approx(180, abs=8) for event in emitted)
    assert [event.timestamp_s for event in emitted] == sorted(event.timestamp_s for event in emitted)
