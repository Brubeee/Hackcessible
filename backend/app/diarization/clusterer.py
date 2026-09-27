"""Neural and acoustic speaker clusterer for real-time conversation diarization."""
import logging
from typing import Optional, List, Tuple
import numpy as np
from scipy.signal import spectrogram

from app.diarization.base import BaseDiarizer, DiarizationResult
from app.diarization.registry import SpeakerRegistry, SpeakerProfile
from app.diarization.ecapa_embedder import ECAPASpeakerEmbedder

logger = logging.getLogger(__name__)


class AcousticClusterDiarizer(BaseDiarizer):
    """High-accuracy speaker clusterer using 192-dim deep embeddings and running centroids."""

    def __init__(
        self,
        registry: SpeakerRegistry,
        similarity_threshold: float = 0.25,
        min_cluster_duration_s: float = 0.35
    ):
        self.registry = registry
        self.similarity_threshold = similarity_threshold
        self.min_cluster_duration_s = min_cluster_duration_s
        self.embedder = ECAPASpeakerEmbedder()

    def _check_acoustic_overlap(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Heuristic check for multi-speaker overlap using spectral entropy and multi-pitch."""
        if len(audio) < int(sample_rate * 0.4):
            return False

        n_fft = 512
        f, t, sxx = spectrogram(audio, fs=sample_rate, nperseg=n_fft, noverlap=256)
        mag = np.abs(sxx) + 1e-10

        # Spectral energy distribution entropy
        p = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-10)
        entropy = -np.sum(p * np.log2(p + 1e-10), axis=0)
        mean_entropy = float(np.mean(entropy))

        # Check dual fundamental frequency presence
        frame_len = int(sample_rate * 0.03)
        dual_pitch_count = 0
        total_voiced = 0

        for i in range(0, len(audio) - frame_len, frame_len):
            frame = audio[i:i + frame_len]
            if np.max(np.abs(frame)) < 0.02:
                continue
            corr = np.correlate(frame, frame, mode='full')[len(frame) - 1:]
            low = int(sample_rate / 400)
            high = int(sample_rate / 80)
            if high < len(corr):
                seg = corr[low:high]
                peaks = []
                for p_idx in range(1, len(seg) - 1):
                    if seg[p_idx] > seg[p_idx - 1] and seg[p_idx] > seg[p_idx + 1] and seg[p_idx] > 0.35 * corr[0]:
                        peaks.append(seg[p_idx])
                if len(peaks) >= 2:
                    dual_pitch_count += 1
                if peaks:
                    total_voiced += 1

        is_overlap = (mean_entropy > 6.9 and total_voiced > 3 and (dual_pitch_count / total_voiced) > 0.40)
        return is_overlap

    def diarize_segment(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        start_time: float = 0.0,
        end_time: float = 0.0
    ) -> DiarizationResult:
        """Assign speaker identity and detect overlap using neural voiceprint."""
        # Extract 192-dim normalized embedding
        voiceprint = self.embedder.extract_embedding(audio, sample_rate)
        is_overlap = self._check_acoustic_overlap(audio, sample_rate)

        best_speaker: Optional[SpeakerProfile] = None
        best_similarity = -1.0

        for speaker in self.registry.speakers.values():
            if speaker.embedding_centroid is not None:
                sim = float(np.dot(voiceprint, speaker.embedding_centroid))
                if sim > best_similarity:
                    best_similarity = sim
                    best_speaker = speaker

        confidence = None

        # Compare against calibrated threshold (0.25 for ECAPA-TDNN)
        threshold = self.similarity_threshold if self.embedder.is_neural() else 0.85

        if best_speaker is not None and best_similarity >= threshold:
            # Matched to enrolled session speaker
            best_speaker.update_embedding(voiceprint)
            best_speaker.total_utterances += 1
            best_speaker.last_heard_time = end_time
            confidence = float(np.clip((best_similarity + 0.2) / 0.8, 0.0, 1.0))
            matched_profile = best_speaker
        else:
            # Register new distinct speaker (e.g. Speaker B, Speaker C)
            matched_profile = self.registry.get_or_create_speaker()
            matched_profile.update_embedding(voiceprint)
            matched_profile.total_utterances += 1
            matched_profile.last_heard_time = end_time
            confidence = 1.0 if len(self.registry.speakers) == 1 else 0.85

        return DiarizationResult(
            speaker_id=matched_profile.speaker_id,
            speaker_label=matched_profile.display_name,
            color_index=matched_profile.color_index,
            confidence=confidence,
            is_overlap=is_overlap,
            overlapping_speakers=[]
        )

    def reset(self):
        """Reset speaker profiles in registry."""
        self.registry.reset()
