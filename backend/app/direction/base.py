"""Direction of Arrival (DoA) provider interface and implementations."""
from abc import ABC, abstractmethod
from typing import Optional
from app.models.events import DirectionInfo


class DirectionProvider(ABC):
    """Abstract interface for spatial audio / Direction of Arrival estimation."""

    @abstractmethod
    def get_direction(self, speaker_id: str, audio_chunk: Optional[bytes] = None) -> Optional[DirectionInfo]:
        """Estimate or retrieve direction of arrival for a speaker."""
        pass

    @abstractmethod
    def is_hardware(self) -> bool:
        """Returns True only if backed by genuine multi-channel microphone array hardware."""
        pass


class NullDirectionProvider(DirectionProvider):
    """Default provider for single-microphone systems (honest no-data)."""

    def get_direction(self, speaker_id: str, audio_chunk: Optional[bytes] = None) -> Optional[DirectionInfo]:
        return None

    def is_hardware(self) -> bool:
        return False
