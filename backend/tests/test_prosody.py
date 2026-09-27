"""Unit tests for acoustic prosody extraction."""
import pytest
import numpy as np
from app.prosody.analyzer import ProsodyAnalyzer
from app.diarization.registry import SpeakerProfile
from app.models.events import LiveProsodyEvent


def test_rising_intonation_detection():
    analyzer = ProsodyAnalyzer(sample_rate=16000)
    sr = 16000
    t = np.linspace(0, 0.8, int(sr * 0.8))
    # Chirp rising from 140Hz to 280Hz
    audio = (0.25 * np.sin(2 * np.pi * (140 + 85 * t) * t)).astype(np.float32)

    features = analyzer.analyze(
        audio=audio,
        words=[{"word": "Really?"}],
        duration_s=0.8,
        time_since_last_utterance=0.5
    )

    assert features.rising_intonation is True
    assert features.falling_intonation is False
    assert features.min_pitch_hz is not None
    assert features.max_pitch_hz is not None
    assert features.pitch_range_semitones is not None
    assert features.max_pitch_hz > features.min_pitch_hz


def test_falling_intonation_detection():
    analyzer = ProsodyAnalyzer(sample_rate=16000)
    sr = 16000
    t = np.linspace(0, 0.8, int(sr * 0.8))
    # Chirp falling from 280Hz to 140Hz
    audio = (0.25 * np.sin(2 * np.pi * (280 - 85 * t) * t)).astype(np.float32)

    features = analyzer.analyze(
        audio=audio,
        words=[{"word": "Done."}],
        duration_s=0.8,
        time_since_last_utterance=0.5
    )

    assert features.falling_intonation is True
    assert features.rising_intonation is False


def test_relative_volume_vs_speaker_baseline():
    analyzer = ProsodyAnalyzer(sample_rate=16000)
    profile = SpeakerProfile("spk1", "Speaker A", 0)
    # Establish baseline around -25 dB
    profile.update_baseline(-25.0, 150.0)

    # Louder utterance (-16 dB)
    sr = 16000
    loud_audio = (0.5 * np.sin(2 * np.pi * 150 * np.linspace(0, 0.5, int(sr * 0.5)))).astype(np.float32)
    res_loud = analyzer.analyze(
        audio=loud_audio,
        words=[{"word": "loud"}],
        duration_s=0.5,
        time_since_last_utterance=0.2,
        speaker_profile=profile
    )
    assert res_loud.relative_volume == "loud"


def test_speech_rate_calculation():
    analyzer = ProsodyAnalyzer(sample_rate=16000)
    audio = np.zeros(16000, dtype=np.float32)
    # 5 words in 1 second = 5 words/sec -> fast
    words = [{"word": w} for w in ["this", "is", "a", "fast", "sentence"]]
    res = analyzer.analyze(
        audio=audio,
        words=words,
        duration_s=1.0,
        time_since_last_utterance=0.2
    )
    assert res.speech_rate == "fast"
    assert res.words_per_minute == 300.0


def test_long_pause_detection():
    analyzer = ProsodyAnalyzer(sample_rate=16000)
    audio = np.zeros(8000, dtype=np.float32)
    res_pause = analyzer.analyze(
        audio=audio,
        words=[{"word": "yes"}],
        duration_s=0.5,
        time_since_last_utterance=1.8  # > 1.2s threshold
    )
    assert res_pause.long_pause_before is True


def test_question_mark_does_not_override_measured_falling_contour():
    analyzer = ProsodyAnalyzer(sample_rate=16000)
    sr = 16000
    t = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False)
    audio = (0.25 * np.sin(2 * np.pi * (280 - 85 * t) * t)).astype(np.float32)
    result = analyzer.analyze(audio, [{"word": "Really?"}], 0.8, 0.2)
    assert result.falling_intonation is True
    assert result.rising_intonation is False


def test_live_frame_reports_pitch_level_direction_rate_and_pause_without_emotion_labels():
    analyzer = ProsodyAnalyzer(sample_rate=16000)

    def tone(frequency: float, amplitude: float = 0.16) -> np.ndarray:
        t = np.arange(3200, dtype=np.float32) / 16000
        return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    for index in range(5):
        baseline = analyzer.analyze_live_frame(
            tone(160), timestamp_s=index * 0.2, speech_active=True,
            speech_duration_s=index * 0.2, pause_duration_s=0.0, speech_rate_wpm=120.0,
        )
    rising = analyzer.analyze_live_frame(
        tone(190, 0.4), timestamp_s=1.0, speech_active=True,
        speech_duration_s=1.0, pause_duration_s=0.0, speech_rate_wpm=180.0,
    )

    assert baseline.pitch_hz == pytest.approx(160, abs=6)
    assert rising.pitch_hz == pytest.approx(190, abs=6)
    assert rising.pitch_direction == "rising"
    assert rising.level_dbfs is not None
    assert rising.relative_level_db is not None and rising.relative_level_db > 6
    assert rising.speech_rate_wpm == 180
    assert rising.emphasis_candidate is True
    assert not any("emotion" in name.lower() for name in LiveProsodyEvent.model_fields)


def test_live_silence_has_no_pitch_or_rate_and_reports_pause():
    analyzer = ProsodyAnalyzer(sample_rate=16000)
    result = analyzer.analyze_live_frame(
        np.zeros(3200, dtype=np.float32), timestamp_s=2.4, speech_active=False,
        speech_duration_s=0.0, pause_duration_s=0.8, speech_rate_wpm=130.0,
    )
    assert result.pitch_hz is None
    assert result.pitch_direction == "unavailable"
    assert result.speech_rate_wpm is None
    assert result.pause_duration_s == pytest.approx(0.8)


def test_live_prosody_serializes_only_bounded_real_measurements():
    event = LiveProsodyEvent(
        timestamp_s=0.8, speech_active=True, pitch_hz=180.0,
        pitch_direction="steady", level_dbfs=-24.0, relative_level_db=1.5,
        speech_rate_wpm=142.0, speech_duration_s=0.8,
        pause_duration_s=0.0, emphasis_candidate=False,
    )
    payload = event.model_dump()
    assert payload["timestamp_s"] == 0.8
    assert payload["pitch_hz"] == 180.0
    assert not any(word in payload for word in ("emotion", "mood", "intent"))
    with pytest.raises(Exception):
        LiveProsodyEvent(
            timestamp_s=-1, speech_active=False, level_dbfs=-120.0,
        )
