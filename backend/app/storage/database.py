"""Storage and privacy management module.

Data handling defaults:
- Raw audio is not written to disk by this storage module.
- Transcripts stay in memory unless the user explicitly opts in.
- Privacy Mode blocks transcript persistence to disk.
"""
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Dict

from app.models.events import ConfidenceInfo, DirectionInfo, ProsodyFeatures, UtteranceEvent

logger = logging.getLogger(__name__)


class SessionStorage:
    """Manages ephemeral in-memory transcripts and optional opt-in SQLite persistence."""
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        # Ephemeral in-memory store: session_id -> list of UtteranceEvent
        self._memory_store: Dict[str, List[UtteranceEvent]] = {}
        self._db_initialized = False

    def _init_sqlite(self):
        if self._db_initialized or not self.db_path:
            return
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS utterances (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        start_time REAL NOT NULL,
                        end_time REAL NOT NULL,
                        speaker_id TEXT NOT NULL,
                        speaker_label TEXT NOT NULL,
                        speaker_color_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        overlap INTEGER NOT NULL,
                        overlapping_speakers TEXT,
                        prosody_json TEXT,
                        direction_json TEXT,
                        confidence_json TEXT,
                        created_at REAL NOT NULL
                    )
                """)
                columns = {
                    row[1] for row in cursor.execute("PRAGMA table_info(utterances)").fetchall()
                }
                if "confidence_json" not in columns:
                    cursor.execute("ALTER TABLE utterances ADD COLUMN confidence_json TEXT")
                conn.commit()
            self._db_initialized = True
            logger.info(f"SQLite transcript database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")

    def save_utterance(
        self,
        utterance: UtteranceEvent,
        privacy_mode: bool = True,
        save_transcript: bool = False
    ):
        """Keep utterances in session memory and persist only after explicit opt-in."""
        # 1. Ephemeral memory for current live session
        if utterance.session_id not in self._memory_store:
            self._memory_store[utterance.session_id] = []
        self._memory_store[utterance.session_id].append(utterance)

        # Privacy Mode takes precedence over the separate transcript-save option.
        if privacy_mode or not save_transcript:
            return

        # 3. Opt-in disk storage
        if self.db_path:
            self._init_sqlite()
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO utterances (
                            id, session_id, start_time, end_time,
                            speaker_id, speaker_label, speaker_color_index,
                            text, overlap, overlapping_speakers,
                            prosody_json, direction_json, confidence_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        utterance.id,
                        utterance.session_id,
                        utterance.start_time,
                        utterance.end_time,
                        utterance.speaker_id,
                        utterance.speaker_label,
                        utterance.speaker_color_index,
                        utterance.text,
                        1 if utterance.overlap else 0,
                        json.dumps(utterance.overlapping_speakers),
                        utterance.prosody.model_dump_json(),
                        utterance.direction.model_dump_json() if utterance.direction else None,
                        utterance.confidence.model_dump_json(),
                        utterance.created_at
                    ))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error persisting utterance to SQLite: {e}")

    def get_session_utterances(self, session_id: str) -> List[UtteranceEvent]:
        """Retrieve session utterances from memory or an opted-in SQLite archive."""
        if session_id in self._memory_store:
            return self._memory_store[session_id].copy()
        if not self.db_path or not self.db_path.exists():
            return []

        self._init_sqlite()
        if not self._db_initialized:
            return []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM utterances WHERE session_id = ? ORDER BY start_time, created_at, id",
                    (session_id,),
                ).fetchall()
            events = []
            for row in rows:
                events.append(UtteranceEvent(
                    id=row["id"],
                    session_id=row["session_id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    speaker_id=row["speaker_id"],
                    speaker_label=row["speaker_label"],
                    speaker_color_index=row["speaker_color_index"],
                    text=row["text"],
                    is_final=True,
                    overlap=bool(row["overlap"]),
                    overlapping_speakers=json.loads(row["overlapping_speakers"] or "[]"),
                    prosody=ProsodyFeatures(**json.loads(row["prosody_json"] or "{}")),
                    direction=DirectionInfo(**json.loads(row["direction_json"]))
                    if row["direction_json"] else None,
                    confidence=ConfidenceInfo(**json.loads(row["confidence_json"] or "{}")),
                    created_at=row["created_at"],
                ))
            return events
        except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"Failed to load persisted transcript for {session_id}: {e}")
            return []

    def clear_session(self, session_id: str):
        """Remove records for a session from memory and any initialized SQLite archive."""
        if session_id in self._memory_store:
            del self._memory_store[session_id]

        if self.db_path and (self._db_initialized or self.db_path.exists()):
            try:
                self._init_sqlite()
                if not self._db_initialized:
                    return
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM utterances WHERE session_id = ?", (session_id,))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error purging SQLite session: {e}")

    def export_transcript(self, session_id: str, fmt: str = "txt") -> str:
        """Export session transcript in text, json, or vtt format."""
        events = self.get_session_utterances(session_id)
        if fmt == "json":
            return json.dumps([e.model_dump() for e in events], indent=2)

        elif fmt == "vtt":
            lines = ["WEBVTT", ""]
            for idx, e in enumerate(events, start=1):
                start_str = self._format_vtt_time(e.start_time)
                end_str = self._format_vtt_time(e.end_time)
                lines.append(f"{idx}")
                lines.append(f"{start_str} --> {end_str}")
                lines.append(f"<v {e.speaker_label}>{e.text}")
                lines.append("")
            return "\n".join(lines)

        else:  # plain text
            lines = []
            for e in events:
                mins = int(e.start_time // 60)
                secs = int(e.start_time % 60)
                time_str = f"[{mins:02d}:{secs:02d}]"
                lines.append(f"{time_str} {e.speaker_label}: {e.text}")
            return "\n".join(lines)

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"
