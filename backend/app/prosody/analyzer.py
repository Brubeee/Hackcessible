"""Conservative, defensible acoustic prosody analyzer.

Strictly calculates measurable signal properties (F0 contour, RMS vs baseline,
speaking rate, pauses) without inferring psychological or emotional states.
Includes voiced-frame masking to prevent unvoiced consonant noise from corrupting intonation.
"""
import logging
from collections import deque
from typing import Optional, List, Dict, Any
import numpy as np
import scipy.signal

from app.models.events import LiveProsodyEvent, ProsodyFeatures
from app.diarization.registry import SpeakerProfile

logger = logging.getLogger(__name__)


class ProsodyAnalyzer:
    """Acoustic feature extractor computing conservative vocal cues."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._live_pitch_history = deque(maxlen=12)
        self._live_level_history = deque(maxlen=30)

    def reset_live(self) -> None:
        """Clear short-term live baselines when a new session starts."""
        self._live_pitch_history.clear()
        self._live_level_history.clear()

    def _estimate_voiced_f0(self, audio: np.ndarray) -> List[float]:
        """Estimate fundamental frequency (F0) trajectory strictly over voiced vowel frames."""
        frame_len = int(self.sample_rate * 0.03)  # 480 samples (30ms)
        hop = int(self.sample_rate * 0.01)        # 160 samples (10ms)
        pitches = []

        # Min and max period lags: 80 Hz to 380 Hz
        min_lag = int(self.sample_rate / 380)  # ~42 samples
        max_lag = int(self.sample_rate / 80)   # ~200 samples

        audio = np.asarray(audio, dtype=np.float32)
        for i in range(0, len(audio) - frame_len + 1, hop):
            frame = audio[i:i + frame_len]
            frame = frame - float(np.mean(frame))
            frame = frame * np.hamming(frame_len).astype(np.float32)
            # 1. Silence gating
            rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
            if rms < 0.004:
                continue

            # 2. Zero Crossing Rate masks many unvoiced fricatives and broadband noise frames.
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)
            if zcr > 0.24:
                continue

            # 3. Normalized autocorrelation
            corr = np.correlate(frame, frame, mode='full')[len(frame) - 1:]
            r0 = corr[0]
            if r0 <= 0:
                continue

            if max_lag < len(corr):
                segment = corr[min_lag:max_lag]
                peak_offset = int(np.argmax(segment))
                peak_val = segment[peak_offset]
                peak_idx = min_lag + peak_offset

                # 4. Must be a true interior local maximum (not an edge clip)
                is_local_max = (
                    0 < peak_offset < len(segment) - 1
                    and segment[peak_offset] > segment[peak_offset - 1]
                    and segment[peak_offset] > segment[peak_offset + 1]
                )

                # 5. Voicing threshold
                if is_local_max and (peak_val / r0) >= 0.42:
                    # Parabolic interpolation reduces integer-lag quantization at low F0.
                    left = float(segment[peak_offset - 1])
                    center = float(segment[peak_offset])
                    right = float(segment[peak_offset + 1])
                    denom = left - (2.0 * center) + right
                    offset = 0.5 * (left - right) / denom if abs(denom) > 1e-12 else 0.0
                    refined_lag = peak_idx + float(np.clip(offset, -0.5, 0.5))
                    f0 = float(self.sample_rate / refined_lag)
                    if 70.0 <= f0 <= 400.0:
                        pitches.append(f0)

        if len(pitches) >= 5:
            # Apply 5-point median filter to suppress spurious octave jump glitches
            k_size = 5 if len(pitches) >= 5 else 3
            pitches = list(scipy.signal.medfilt(pitches, kernel_size=k_size))

        return pitches

    def analyze_live_frame(
        self,
        audio: np.ndarray,
        *,
        timestamp_s: float,
        speech_active: bool,
        speech_duration_s: float,
        pause_duration_s: Optional[float],
        speech_rate_wpm: Optional[float],
    ) -> LiveProsodyEvent:
        """Summarize a short PCM window for the live visualizer.

        Input windows are about 0.2 seconds. The two rolling baselines are local
        session references, not calibrated sound-pressure measurements.
        """
        signal = np.asarray(audio, dtype=np.float32)
        if signal.size:
            rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)) + 1e-12))
            level_dbfs = float(20.0 * np.log10(max(rms, 1e-6)))
        else:
            level_dbfs = -120.0

        pitch_values = self._estimate_voiced_f0(signal) if speech_active and signal.size else []
        pitch_hz = float(np.median(pitch_values)) if pitch_values else None
        prior_pitch = float(np.median(self._live_pitch_history)) if self._live_pitch_history else None
        prior_level = float(np.median(self._live_level_history)) if len(self._live_level_history) >= 4 else None
        relative_level_db = level_dbfs - prior_level if prior_level is not None and speech_active else None

        pitch_direction = "unavailable"
        pitch_delta = None
        if pitch_hz is not None:
            if prior_pitch is not None:
                pitch_delta = 12.0 * float(np.log2(pitch_hz / prior_pitch))
                if pitch_delta >= 1.2:
                    pitch_direction = "rising"
                elif pitch_delta <= -1.2:
                    pitch_direction = "falling"
                else:
                    pitch_direction = "steady"
            else:
                pitch_direction = "steady"

        emphasis_candidate = bool(
            speech_active
            and relative_level_db is not None
            and relative_level_db >= 6.0
            and pitch_delta is not None
            and abs(pitch_delta) >= 1.5
        )

        if speech_active:
            if pitch_hz is not None:
                self._live_pitch_history.append(pitch_hz)
            # Track voiced level only, so long silences do not pull the baseline down.
            if level_dbfs > -55.0:
                self._live_level_history.append(level_dbfs)

        return LiveProsodyEvent(
            timestamp_s=max(0.0, float(timestamp_s)),
            speech_active=bool(speech_active),
            pitch_hz=pitch_hz,
            pitch_direction=pitch_direction,
            level_dbfs=level_dbfs,
            relative_level_db=relative_level_db,
            speech_rate_wpm=speech_rate_wpm if speech_active else None,
            speech_duration_s=max(0.0, float(speech_duration_s)),
            pause_duration_s=max(0.0, float(pause_duration_s)) if pause_duration_s is not None else None,
            emphasis_candidate=emphasis_candidate,
        )

    def _detect_emphasis(self, audio: np.ndarray, f0_contour: List[float], mean_f0: Optional[float]) -> bool:
        """Detect local acoustic emphasis via energy burst paired with pitch excursion."""
        if len(audio) < int(self.sample_rate * 0.3):
            return False

        frame_len = int(self.sample_rate * 0.05)
        hop = int(self.sample_rate * 0.025)
        energies = []
        for i in range(0, len(audio) - frame_len, hop):
            frame = audio[i:i + frame_len]
            energies.append(np.sum(frame ** 2))

        if not energies:
            return False

        mean_energy = float(np.mean(energies)) + 1e-12
        max_energy = float(np.max(energies))

        energy_burst = (max_energy / mean_energy) > 2.5
        pitch_jump = False
        if f0_contour and mean_f0 and mean_f0 > 0:
            max_f0 = max(f0_contour)
            if (max_f0 / mean_f0) > 1.25:
                pitch_jump = True

        return energy_burst and (pitch_jump or len(f0_contour) < 3)

    def analyze(
        self,
        audio: np.ndarray,
        words: List[Dict[str, Any]],
        duration_s: float,
        time_since_last_utterance: float,
        speaker_profile: Optional[SpeakerProfile] = None
    ) -> ProsodyFeatures:
        """Extract conservative prosodic cues from utterance audio and timing."""
        if len(audio) == 0 or duration_s <= 0.0:
            return ProsodyFeatures()

        # 1. Energy & Relative Volume vs Speaker Baseline
        rms_val = np.sqrt(np.mean(audio ** 2) + 1e-12)
        rms_db = float(20.0 * np.log10(rms_val + 1e-12))

        relative_volume = "normal"
        if speaker_profile is not None:
            baseline_rms = speaker_profile.get_mean_rms()
            delta_db = rms_db - baseline_rms
            if delta_db > 4.5:
                relative_volume = "loud"
            elif delta_db < -5.0:
                relative_volume = "quiet"
        else:
            if rms_db > -18.0:
                relative_volume = "loud"
            elif rms_db < -32.0:
                relative_volume = "quiet"

        # 2. Pitch & Intonation Contour
        f0_list = self._estimate_voiced_f0(audio)
        mean_pitch = float(np.mean(f0_list)) if f0_list else None

        rising = False
        falling = False
        semitone_delta: Optional[float] = None

        if len(f0_list) >= 4:
            # Split into nucleus (preceding 60%) and boundary tone (terminal 40%)
            split_idx = max(2, int(len(f0_list) * 0.60))
            preceding_f0 = float(np.median(f0_list[:split_idx]))
            terminal_f0 = float(np.median(f0_list[split_idx:]))

            if preceding_f0 > 0:
                # Calculate excursion in semitones: 12 * log2(terminal / preceding)
                semitone_delta = float(12.0 * np.log2(terminal_f0 / preceding_f0))

                # Direction describes the measured acoustic contour, independent of punctuation.
                rising = semitone_delta >= 1.2
                falling = semitone_delta <= -1.2

        # 3. Vocal Emphasis
        emphasis = self._detect_emphasis(audio, f0_list, mean_pitch)

        # 4. Speaking Rate
        num_words = len(words)
        wpm = None
        speech_rate = "normal"

        if num_words > 0 and duration_s > 0.4:
            words_per_sec = num_words / duration_s
            wpm = float(words_per_sec * 60.0)
            if words_per_sec >= 3.6:
                speech_rate = "fast"
            elif words_per_sec <= 1.6 and num_words >= 2:
                speech_rate = "slow"

        # 5. Long Pause Before Utterance
        long_pause_before = (time_since_last_utterance >= 1.2)

        # Update speaker profile baseline for future relative comparisons
        if speaker_profile is not None:
            speaker_profile.update_baseline(rms_db, mean_pitch)

        return ProsodyFeatures(
            rising_intonation=rising,
            falling_intonation=falling,
            emphasis=emphasis,
            long_pause_before=long_pause_before,
            speech_rate=speech_rate,
            relative_volume=relative_volume,
            mean_pitch_hz=mean_pitch,
            min_pitch_hz=float(min(f0_list)) if f0_list else None,
            max_pitch_hz=float(max(f0_list)) if f0_list else None,
            pitch_range_semitones=(
                float(12.0 * np.log2(max(f0_list) / min(f0_list)))
                if f0_list and min(f0_list) > 0 else None
            ),
            rms_db=rms_db,
            words_per_minute=wpm,
            pitch_semitone_excursion=semitone_delta
        )
