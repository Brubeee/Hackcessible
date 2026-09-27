"""Simulated and hardware direction provider implementations."""
from typing import Optional, Dict, Any
from app.models.events import DirectionInfo
from app.direction.base import DirectionProvider


class SimulatedDirectionProvider(DirectionProvider):
    """Simulated spatial direction provider for development and hackathon demonstrations.
    
    CRITICAL: Always marks `simulated=True` so that UI and logs clearly communicate
    that spatial values are generated for demonstration rather than measured from hardware.
    """
    def __init__(self):
        # Deterministic spatial layout for 3+ speakers
        self._layout: Dict[str, Dict[str, Any]] = {
            "speaker_1": {"angle": -45.0, "label": "left"},
            "speaker_2": {"angle": 0.0, "label": "front"},
            "speaker_3": {"angle": 45.0, "label": "right"},
            "speaker_4": {"angle": -75.0, "label": "far-left"},
            "speaker_5": {"angle": 75.0, "label": "far-right"}
        }

    def get_direction(self, speaker_id: str, audio_chunk: Optional[bytes] = None) -> Optional[DirectionInfo]:
        info = self._layout.get(speaker_id, {"angle": 0.0, "label": "front"})
        return DirectionInfo(
            angle_degrees=info["angle"],
            label=info["label"],
            simulated=True,
            confidence=0.85
        )

    def is_hardware(self) -> bool:
        return False


class FutureMicrophoneArrayDirectionProvider(DirectionProvider):
    """Placeholder driver interface for multi-channel microphone arrays (e.g. ReSpeaker 4-Mic, miniDSP UMA-8).
    
    When hardware is connected via USB multichannel ASIO/ALSA, this class implements
    Generalized Cross Correlation with Phase Transform (GCC-PHAT) across mic pairs.
    """
    def __init__(self, num_channels: int = 4, mic_distance_m: float = 0.05):
        self.num_channels = num_channels
        self.mic_distance_m = mic_distance_m
        self.hardware_detected = False

    def get_direction(self, speaker_id: str, audio_chunk: Optional[bytes] = None) -> Optional[DirectionInfo]:
        if not self.hardware_detected:
            # Honest fallback when multi-channel stream is not physically present
            return None
            
        # Example calculation placeholder for actual multichannel buffer
        return None

    def is_hardware(self) -> bool:
        return self.hardware_detected
