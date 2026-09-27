"""Faster-Whisper implementation of BaseASR for local-first speech recognition."""
import logging
import time
from typing import Optional, List, Dict, Any
import numpy as np

from app.asr.base import BaseASR, ASRResult

logger = logging.getLogger(__name__)


class WhisperASR(BaseASR):
    """Local ASR engine using Faster-Whisper (CTranslate2).
    
    Fast, quantized local inference without cloud round-trips.
    """
    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Faster-Whisper model '{self.model_size}' on {self.device} ({self.compute_type})...")
            start = time.time()
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            elapsed = time.time() - start
            logger.info(f"Faster-Whisper model ready in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}")
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """Transcribe audio chunk (float32 array, 16kHz)."""
        if not self.is_ready():
            return ASRResult(text="[ASR engine unavailable]", confidence=None)

        if len(audio) < int(sample_rate * 0.25):  # Shorter than 250ms
            return ASRResult(text="", confidence=None, duration_s=len(audio)/sample_rate)

        duration_s = float(len(audio) / sample_rate)
        
        try:
            # Faster-whisper expects float32 array normalized to [-1.0, 1.0]
            segments, info = self.model.transcribe(
                audio,
                beam_size=1,  # Greedy search for lowest latency
                language="en",
                vad_filter=False,  # Our upstream VAD already segmented speech
                word_timestamps=True
            )

            collected_texts = []
            collected_words = []
            logprobs = []

            for segment in segments:
                text_clean = segment.text.strip()
                if text_clean:
                    collected_texts.append(text_clean)
                if segment.avg_logprob is not None:
                    logprobs.append(segment.avg_logprob)
                    
                if segment.words:
                    for w in segment.words:
                        collected_words.append({
                            "word": w.word.strip(),
                            "start": float(w.start),
                            "end": float(w.end),
                            "probability": float(w.probability) if hasattr(w, "probability") else None
                        })

            full_text = " ".join(collected_texts).strip()

            # Confidence estimate from token log-probabilities
            confidence = None
            if logprobs:
                avg_lp = float(np.mean(logprobs))
                # Convert log probability to normalized score [0, 1]
                confidence = float(np.clip(np.exp(avg_lp), 0.0, 1.0))

            return ASRResult(
                text=full_text,
                confidence=confidence,
                words=collected_words,
                duration_s=duration_s,
                language=info.language if hasattr(info, "language") else "en"
            )

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return ASRResult(text="", confidence=None, duration_s=duration_s)
