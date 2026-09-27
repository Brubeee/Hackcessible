"""Session-level speaker registry for identity stability, renaming, and acoustic baselines."""
from typing import Dict, List, Optional
import numpy as np
from app.models.events import SpeakerInfo


DEFAULT_SPEAKER_LABELS = [
    "Speaker A", "Speaker B", "Speaker C", "Speaker D",
    "Speaker E", "Speaker F", "Speaker G", "Speaker H"
]

class SpeakerProfile:
    """Acoustic profile and history for an individual speaker."""
    def __init__(self, speaker_id: str, label: str, color_index: int):
        self.speaker_id = speaker_id
        self.label = label
        self.custom_name: Optional[str] = None
        self.color_index = color_index
        self.total_utterances = 0
        self.last_heard_time = 0.0
        self.is_active = False
        
        # Acoustic baselines for normalized prosody comparison
        self.baseline_rms_db: List[float] = []
        self.baseline_pitch_hz: List[float] = []
        # Centroid feature vector for acoustic clustering
        self.embedding_centroid: Optional[np.ndarray] = None
        self.embedding_count = 0

    @property
    def display_name(self) -> str:
        if self.custom_name:
            return f"{self.custom_name} ({self.label})"
        return self.label

    def update_baseline(self, rms_db: Optional[float], pitch_hz: Optional[float]):
        """Update running acoustic baselines (keeping last 20 utterances)."""
        if rms_db is not None and not np.isnan(rms_db):
            self.baseline_rms_db.append(rms_db)
            if len(self.baseline_rms_db) > 20:
                self.baseline_rms_db.pop(0)
                
        if pitch_hz is not None and not np.isnan(pitch_hz) and pitch_hz > 50:
            self.baseline_pitch_hz.append(pitch_hz)
            if len(self.baseline_pitch_hz) > 20:
                self.baseline_pitch_hz.pop(0)

    def get_mean_rms(self) -> float:
        if not self.baseline_rms_db:
            return -24.0  # Default nominal speech level
        return float(np.mean(self.baseline_rms_db))

    def get_median_pitch(self) -> Optional[float]:
        if not self.baseline_pitch_hz:
            return None
        return float(np.median(self.baseline_pitch_hz))

    def update_embedding(self, new_emb: np.ndarray):
        """Update centroid with new normalized embedding."""
        new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-8)
        if self.embedding_centroid is None:
            self.embedding_centroid = new_emb
            self.embedding_count = 1
        else:
            # Running weighted centroid
            self.embedding_centroid = (
                self.embedding_centroid * self.embedding_count + new_emb
            ) / (self.embedding_count + 1)
            self.embedding_centroid /= (np.linalg.norm(self.embedding_centroid) + 1e-8)
            self.embedding_count += 1

    def to_info(self) -> SpeakerInfo:
        return SpeakerInfo(
            id=self.speaker_id,
            label=self.display_name,
            color_index=self.color_index,
            custom_name=self.custom_name,
            is_active=self.is_active,
            total_utterances=self.total_utterances,
            last_heard_time=self.last_heard_time
        )


class SpeakerRegistry:
    """Registry maintaining session-wide speaker identities and preventing reassignment."""
    def __init__(self):
        self.speakers: Dict[str, SpeakerProfile] = {}
        self._next_index = 0

    def get_or_create_speaker(self, speaker_id: Optional[str] = None) -> SpeakerProfile:
        """Retrieve existing speaker or register a new one with a consistent label and color."""
        if speaker_id and speaker_id in self.speakers:
            return self.speakers[speaker_id]
            
        idx = self._next_index
        self._next_index += 1
        new_id = speaker_id or f"speaker_{idx + 1}"
        default_label = (
            DEFAULT_SPEAKER_LABELS[idx % len(DEFAULT_SPEAKER_LABELS)]
            if idx < len(DEFAULT_SPEAKER_LABELS)
            else f"Speaker {idx + 1}"
        )
        
        profile = SpeakerProfile(
            speaker_id=new_id,
            label=default_label,
            color_index=idx % 8  # 8 distinct high-contrast accessible colors
        )
        self.speakers[new_id] = profile
        return profile

    def rename_speaker(self, speaker_id: str, new_name: str) -> Optional[SpeakerProfile]:
        """Manually rename a speaker (e.g. Speaker A -> 'Professor'). Color remains fixed."""
        if speaker_id not in self.speakers:
            return None
        cleaned_name = new_name.strip()
        self.speakers[speaker_id].custom_name = cleaned_name if cleaned_name else None
        return self.speakers[speaker_id]

    def set_active_speaker(self, speaker_id: str, is_active: bool = True):
        for sid, p in self.speakers.items():
            p.is_active = (sid == speaker_id) if is_active else False

    def list_speakers(self) -> List[SpeakerInfo]:
        return [p.to_info() for p in self.speakers.values()]

    def reset(self):
        """Reset registry for a new conversation session."""
        self.speakers.clear()
        self._next_index = 0
