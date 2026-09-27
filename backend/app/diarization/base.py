"""Abstract interface and result types for speaker diarization."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np


class DiarizationResult:
    def __init__(
        self,
        speaker_id: str,
        speaker_label: str,
        color_index: int,
        confidence: Optional[float] = None,
        is_overlap: bool = False,
        overlapping_speakers: Optional[List[str]] = None
    ):
        self.speaker_id = speaker_id
        self.speaker_label = speaker_label
        self.color_index = color_index
        self.confidence = confidence
        self.is_overlap = is_overlap
        self.overlapping_speakers = overlapping_speakers or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "speaker_label": self.speaker_label,
            "color_index": self.color_index,
            "confidence": self.confidence,
            "is_overlap": self.is_overlap,
            "overlapping_speakers": self.overlapping_speakers
        }


class BaseDiarizer(ABC):
    """Abstract interface for speaker diarization and voice identity clustering."""

    @abstractmethod
    def diarize_segment(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        start_time: float = 0.0,
        end_time: float = 0.0
    ) -> DiarizationResult:
        """Assign speaker identity and detect overlap for an audio segment."""
        pass

    @abstractmethod
    def reset(self):
        """Reset speaker clusters for a new session."""
        pass
