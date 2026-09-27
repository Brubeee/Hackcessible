"""Voice Activity Detection using Silero VAD ONNX with energy-based fallback."""
import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

class VoiceActivityDetector:
    """Detects active speech in streaming audio chunks using Silero VAD ONNX.
    
    Falls back gracefully to energy/zero-crossing analysis if the neural model
    is unavailable.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.45,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 150
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_samples = int(sample_rate * speech_pad_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
        
        self.model = None
        self.is_neural = False
        self._init_silero()
        
        # State tracking
        self.is_speaking = False
        self.speech_start_sample: Optional[int] = None
        self.current_silence_samples = 0
        self.current_speech_samples = 0
        self.total_processed_samples = 0
        
        # Energy fallback baseline
        self.noise_floor_rms = -50.0

    def _init_silero(self):
        try:
            import silero_vad
            self.model = silero_vad.load_silero_vad(onnx=True)
            self.is_neural = True
            logger.info("Silero VAD ONNX successfully loaded.")
        except Exception as e:
            logger.warning(f"Silero VAD ONNX could not be loaded, using energy fallback: {e}")
            self.model = None
            self.is_neural = False

    def process_frame_512(self, frame_512: np.ndarray) -> Dict[str, Any]:
        """Process exactly 512 samples (32ms at 16kHz).
        
        Returns:
            dict containing:
                - speech_prob: float (0.0 to 1.0)
                - is_speaking: bool
                - segment_started: bool
                - segment_ended: bool
                - energy_db: float
        """
        if len(frame_512) != 512:
            # Pad or truncate to 512
            if len(frame_512) < 512:
                frame_512 = np.pad(frame_512, (0, 512 - len(frame_512)))
            else:
                frame_512 = frame_512[:512]

        rms = np.sqrt(np.mean(frame_512 ** 2) + 1e-12)
        energy_db = float(20.0 * np.log10(rms + 1e-12))
        
        # Calculate speech probability
        if self.is_neural and self.model is not None:
            try:
                import torch
                tensor_chunk = torch.from_numpy(frame_512).float()
                prob = float(self.model(tensor_chunk, self.sample_rate).item())
            except Exception:
                prob = self._energy_prob(energy_db)
        else:
            prob = self._energy_prob(energy_db)

        segment_started = False
        segment_ended = False
        
        is_speech_frame = prob >= self.threshold

        if is_speech_frame:
            self.current_speech_samples += 512
            self.current_silence_samples = 0
            if not self.is_speaking:
                self.is_speaking = True
                segment_started = True
                self.speech_start_sample = self.total_processed_samples
        else:
            if self.is_speaking:
                self.current_silence_samples += 512
                # Check if silence threshold reached to conclude segment
                if self.current_silence_samples >= self.min_silence_samples:
                    self.is_speaking = False
                    segment_ended = True
                    self.current_speech_samples = 0
                    self.current_silence_samples = 0

        self.total_processed_samples += 512

        return {
            "speech_prob": prob,
            "is_speaking": self.is_speaking,
            "segment_started": segment_started,
            "segment_ended": segment_ended,
            "energy_db": energy_db,
            "total_samples": self.total_processed_samples
        }

    def _energy_prob(self, energy_db: float) -> float:
        """Conservative sigmoid mapping of RMS energy to probability."""
        threshold_db = -38.0
        # Smooth sigmoid between -45 dB and -25 dB
        x = (energy_db - threshold_db) / 5.0
        return float(1.0 / (1.0 + np.exp(-x)))

    def reset(self):
        """Reset internal speech state for a new session."""
        if self.model and hasattr(self.model, "reset_states"):
            self.model.reset_states()
        self.is_speaking = False
        self.speech_start_sample = None
        self.current_silence_samples = 0
        self.current_speech_samples = 0
        self.total_processed_samples = 0
