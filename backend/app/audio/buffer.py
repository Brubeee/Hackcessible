"""Audio buffer management and conditioning for real-time streaming audio."""
import io
import struct
import numpy as np
from scipy.signal import butter, sosfilt
from typing import Optional, Tuple


def apply_audio_conditioning(samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Strip sub-audible mechanical rumble (<75Hz) and normalize dynamic range."""
    if len(samples) < 32:
        return samples
    try:
        sos = butter(4, 75.0, btype='highpass', fs=sample_rate, output='sos')
        filtered = sosfilt(sos, samples)
        # Soft peak limiting with headroom
        peak = np.max(np.abs(filtered))
        if peak > 0.01:
            filtered = filtered / max(peak, 0.75) * 0.75
        return filtered.astype(np.float32)
    except Exception:
        return samples


class AudioBuffer:
    """Thread-safe rolling buffer for incoming PCM audio streams (16kHz, mono, float32)."""
    def __init__(self, sample_rate: int = 16000, max_duration_s: float = 30.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_s)
        self._buffer = np.zeros(0, dtype=np.float32)
        self.total_samples_received = 0

    @property
    def duration_seconds(self) -> float:
        return len(self._buffer) / self.sample_rate

    def append_pcm16(self, pcm_bytes: bytes) -> int:
        """Append raw signed 16-bit little-endian PCM bytes."""
        num_samples = len(pcm_bytes) // 2
        if num_samples == 0:
            return 0
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.append_float32(samples)

    def append_float32(self, samples: np.ndarray) -> int:
        """Append float32 normalized samples [-1.0, 1.0]."""
        if len(samples) == 0:
            return 0
        conditioned = apply_audio_conditioning(samples, self.sample_rate)
        self._buffer = np.concatenate([self._buffer, conditioned])
        self.total_samples_received += len(conditioned)
        
        # Enforce maximum rolling duration
        if len(self._buffer) > self.max_samples:
            self._buffer = self._buffer[-self.max_samples:]
        return len(conditioned)

    def get_slice(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Extract a slice of samples safely."""
        start_idx = max(0, start_sample)
        end_idx = min(len(self._buffer), end_sample)
        if start_idx >= end_idx:
            return np.zeros(0, dtype=np.float32)
        return self._buffer[start_idx:end_idx].copy()

    def get_last_n_seconds(self, seconds: float) -> np.ndarray:
        """Extract the most recent N seconds of audio."""
        num_samples = int(seconds * self.sample_rate)
        if num_samples <= 0 or len(self._buffer) == 0:
            return np.zeros(0, dtype=np.float32)
        return self._buffer[-num_samples:].copy()

    def get_all(self) -> np.ndarray:
        return self._buffer.copy()

    def clear(self):
        """Release this buffer's array reference; this is not secure memory erasure."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self.total_samples_received = 0

    def calculate_rms(self, audio: Optional[np.ndarray] = None) -> float:
        """Calculate Root Mean Square (RMS) energy in dBFS."""
        target = audio if audio is not None else self._buffer
        if len(target) == 0:
            return -100.0
        rms = np.sqrt(np.mean(target ** 2) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)
        return float(db)
