"""Unit tests for SpeakerRegistry, stable colors, renaming, and acoustic baselines."""
import pytest
import numpy as np
from app.diarization.registry import SpeakerRegistry


def test_consistent_speaker_assignment():
    registry = SpeakerRegistry()
    spk1 = registry.get_or_create_speaker()
    assert spk1.speaker_id == "speaker_1"
    assert spk1.label == "Speaker A"
    assert spk1.color_index == 0

    spk2 = registry.get_or_create_speaker()
    assert spk2.speaker_id == "speaker_2"
    assert spk2.label == "Speaker B"
    assert spk2.color_index == 1

    # Re-retrieving existing speaker does not change id or color
    spk1_again = registry.get_or_create_speaker("speaker_1")
    assert spk1_again.speaker_id == "speaker_1"
    assert spk1_again.color_index == 0


def test_speaker_renaming():
    registry = SpeakerRegistry()
    spk1 = registry.get_or_create_speaker()
    assert spk1.display_name == "Speaker A"

    # User renames Speaker A to "Dr. Rao"
    renamed = registry.rename_speaker("speaker_1", "Dr. Rao")
    assert renamed is not None
    assert renamed.custom_name == "Dr. Rao"
    assert renamed.display_name == "Dr. Rao (Speaker A)"
    # Color MUST remain identical
    assert renamed.color_index == 0


def test_acoustic_baselines():
    registry = SpeakerRegistry()
    spk = registry.get_or_create_speaker("speaker_1")
    assert spk.get_mean_rms() == -24.0  # default nominal

    # Add realistic measurements
    spk.update_baseline(rms_db=-18.0, pitch_hz=140.0)
    spk.update_baseline(rms_db=-20.0, pitch_hz=145.0)

    assert pytest.approx(spk.get_mean_rms(), 0.1) == -19.0
    assert pytest.approx(spk.get_median_pitch(), 0.1) == 142.5
