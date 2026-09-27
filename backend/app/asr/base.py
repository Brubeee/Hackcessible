"""Abstract base interface for speech recognition engines."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np


class ASRResult:
    def __init__(
        self,
        text: str,
        confidence: Optional[float] = None,
        words: Optional[List[Dict[str, Any]]] = None,
        duration_s: float = 0.0,
        language: str = "en"
    ):
        self.text = text.strip()
        self.confidence = confidence
        self.words = words or []
        self.duration_s = duration_s
        self.language = language

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "words": self.words,
            "duration_s": self.duration_s,
            "language": self.language
        }


class BaseASR(ABC):
    """Abstract speech-to-text interface."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """Transcribe an audio segment (mono float32 [-1.0, 1.0])."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """True if the underlying model is loaded and ready for inference."""
        pass
