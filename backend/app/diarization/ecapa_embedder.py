"""Neural speaker embedding extractor using SpeechBrain ECAPA-TDNN with robust fallback."""
import logging
import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Suppress HF symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class ECAPASpeakerEmbedder:
    """Extracts 192-dimensional deep speaker voiceprints using ECAPA-TDNN."""

    def __init__(self, model_dir: Optional[Path] = None, device: str = "cpu"):
        self.device = device
        self.model_dir = model_dir or (Path(__file__).resolve().parent.parent.parent / "models" / "ecapa_voxceleb")
        self.classifier = None
        self._load_model()

    def _load_model(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
            self.model_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Initializing ECAPA-TDNN speaker embedding model...")
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.model_dir),
                run_opts={"device": self.device},
                local_strategy=LocalStrategy.COPY
            )
            logger.info("ECAPA-TDNN speaker embedding model ready.")
        except Exception as e:
            logger.warning(f"Could not load neural ECAPA-TDNN, using CMN-MFCC fallback: {e}")
            self.classifier = None

    def is_neural(self) -> bool:
        return self.classifier is not None

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a 192-dimensional unit-normalized speaker embedding."""
        if len(audio) < int(sample_rate * 0.35):
            # Pad to at least 350ms for reliable spectral analysis
            pad_len = int(sample_rate * 0.35) - len(audio)
            audio = np.pad(audio, (0, pad_len))

        if self.classifier is not None:
            try:
                t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
                emb = self.classifier.encode_batch(t).squeeze().detach().cpu().numpy()
                norm = np.linalg.norm(emb) + 1e-8
                return emb / norm
            except Exception as e:
                logger.error(f"ECAPA extraction error, falling back: {e}")

        return self._extract_cmn_mfcc_fallback(audio, sample_rate)

    def _extract_cmn_mfcc_fallback(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Cepstral Mean-Normalized MFCC feature vector + pitch offset."""
        from scipy.fftpack import dct
        from scipy.signal import spectrogram

        n_fft = 512
        hop = 256
        f, t, sxx = spectrogram(audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
        mag = np.abs(sxx) + 1e-10

        # Simple 26 Mel bank
        mel_min = 2595.0 * np.log10(1.0 + 100.0 / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + (sample_rate / 2) / 700.0)
        mels = np.linspace(mel_min, mel_max, 28)
        hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)

        fb = np.zeros((26, n_fft // 2 + 1), dtype=np.float32)
        for i in range(26):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            if center > left:
                fb[i, left:center] = (np.arange(left, center) - left) / (center - left)
            if right > center:
                fb[i, center:right] = (right - np.arange(center, right)) / (right - center)

        mel_energy = np.dot(fb, mag)
        log_mel = np.log(mel_energy + 1e-6)
        mfcc = dct(log_mel, type=2, axis=0, norm='ortho')[1:21]  # coefficients 1-20

        # Mean subtraction across time (removes static channel/mic filter)
        mfcc_cmn = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        mean_feat = np.mean(mfcc_cmn, axis=1)
        std_feat = np.std(mfcc_cmn, axis=1)

        vec = np.hstack([mean_feat, std_feat]).astype(np.float32)
        # Pad to 192 dimensions to match ECAPA schema
        if len(vec) < 192:
            vec = np.pad(vec, (0, 192 - len(vec)))
        else:
            vec = vec[:192]
        return vec / (np.linalg.norm(vec) + 1e-8)
