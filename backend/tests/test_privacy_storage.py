"""Unit tests for Privacy Mode enforcement, SQLite opt-in, and transcript exports."""
import os
import tempfile
from pathlib import Path
import pytest
from app.models.events import UtteranceEvent, ProsodyFeatures
from app.storage.database import SessionStorage


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def test_privacy_mode_prevents_disk_write(temp_db):
    storage = SessionStorage(db_path=temp_db)
    event = UtteranceEvent(
        session_id="s1",
        start_time=0.0,
        end_time=1.5,
        speaker_id="speaker_1",
        speaker_label="Speaker A",
        text="Private discussion"
    )

    # Privacy Mode = True, save_transcript = False (default)
    storage.save_utterance(event, privacy_mode=True, save_transcript=False)

    # Should be in ephemeral memory
    mem_events = storage.get_session_utterances("s1")
    assert len(mem_events) == 1
    assert mem_events[0].text == "Private discussion"

    # Disk file MUST NOT be created
    assert not temp_db.exists()

    storage.save_utterance(event, privacy_mode=True, save_transcript=True)
    assert not temp_db.exists()


def test_opt_in_transcript_saving(temp_db):
    storage = SessionStorage(db_path=temp_db)
    event = UtteranceEvent(
        session_id="s2",
        start_time=0.0,
        end_time=2.0,
        speaker_id="speaker_1",
        speaker_label="Speaker A",
        text="Opted-in transcript text"
    )

    # Privacy Mode = False, save_transcript = True
    storage.save_utterance(event, privacy_mode=False, save_transcript=True)

    # Disk DB must exist now
    assert temp_db.exists()

    # Clear memory and verify SQLite table was written
    storage._memory_store.clear()
    import sqlite3
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM utterances WHERE session_id = 's2'")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "Opted-in transcript text"

    reopened_storage = SessionStorage(db_path=temp_db)
    restored = reopened_storage.get_session_utterances("s2")
    assert len(restored) == 1
    assert restored[0].text == "Opted-in transcript text"
    assert reopened_storage.export_transcript("s2", fmt="txt").endswith(
        "Speaker A: Opted-in transcript text"
    )

    reopened_storage.clear_session("s2")
    assert reopened_storage.get_session_utterances("s2") == []


def test_clear_session_purges_all(temp_db):
    storage = SessionStorage(db_path=temp_db)
    event = UtteranceEvent(
        session_id="s3",
        start_time=0.0,
        end_time=2.0,
        speaker_id="speaker_1",
        speaker_label="Speaker A",
        text="To be deleted"
    )
    storage.save_utterance(event, privacy_mode=False, save_transcript=True)
    assert len(storage.get_session_utterances("s3")) == 1

    # Clear session
    storage.clear_session("s3")
    assert len(storage.get_session_utterances("s3")) == 0


def test_export_formats(temp_db):
    storage = SessionStorage(db_path=temp_db)
    event = UtteranceEvent(
        session_id="s4",
        start_time=12.5,
        end_time=15.0,
        speaker_id="speaker_1",
        speaker_label="Speaker A",
        text="Clinical observation"
    )
    storage.save_utterance(event, privacy_mode=True, save_transcript=False)

    txt = storage.export_transcript("s4", fmt="txt")
    assert "[00:12] Speaker A: Clinical observation" in txt

    vtt = storage.export_transcript("s4", fmt="vtt")
    assert "WEBVTT" in vtt
    assert "00:00:12.500 --> 00:00:15.000" in vtt
    assert "<v Speaker A>Clinical observation" in vtt

    json_export = storage.export_transcript("s4", fmt="json")
    assert '"text": "Clinical observation"' in json_export
