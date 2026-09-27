"""Event fusion and processing pipeline combining VAD, ASR, Diarization, Prosody, and Direction."""
import logging
import time
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np

from app.models.events import LiveProsodyEvent, UtteranceEvent, ProsodyFeatures, DirectionInfo, ConfidenceInfo, DebugMetrics
from app.models.settings import AccessibilitySettings
from app.audio.buffer import AudioBuffer
from app.audio.vad import VoiceActivityDetector
from app.asr.base import BaseASR
from app.diarization.base import BaseDiarizer
from app.diarization.registry import SpeakerRegistry
from app.prosody.analyzer import ProsodyAnalyzer
from app.direction.base import DirectionProvider

logger = logging.getLogger(__name__)


class FusionPipeline:
    """Loose-coupled conversation processor producing structured UtteranceEvents."""

    def __init__(
        self,
        asr: BaseASR,
        diarizer: BaseDiarizer,
        registry: SpeakerRegistry,
        prosody_analyzer: ProsodyAnalyzer,
        direction_provider: DirectionProvider,
        sample_rate: int = 16000
    ):
        self.asr = asr
        self.diarizer = diarizer
        self.registry = registry
        self.prosody = prosody_analyzer
        self.direction = direction_provider
        self.sample_rate = sample_rate

        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        self.audio_buffer = AudioBuffer(sample_rate=sample_rate)
        
        # State
        self.session_id = "live_session"
        self.session_start_time = time.time()
        self.last_utterance_end_time = 0.0
        self.recent_utterances: List[UtteranceEvent] = []
        self._live_prosody_frames: List[np.ndarray] = []
        self._last_live_prosody_sample = 0
        self._last_voiced_sample: Optional[int] = None
        self._live_speech_rate_wpm: Optional[float] = None
        
        # Segment accumulator
        self.segment_samples: List[np.ndarray] = []
        self.segment_start_timestamp = 0.0
        self.is_collecting_speech = False
        
        # Rolling partial transcription interval tracking
        self.last_partial_sample_count = 0
        self.active_interim_id: Optional[str] = None

    def reset_session(self, session_id: str):
        """Reset pipeline for a fresh conversation session."""
        self.session_id = session_id
        self.session_start_time = time.time()
        self.last_utterance_end_time = 0.0
        self.recent_utterances.clear()
        self.segment_samples.clear()
        self.is_collecting_speech = False
        self.active_interim_id = None
        self._live_prosody_frames.clear()
        self._last_live_prosody_sample = 0
        self._last_voiced_sample = None
        self._live_speech_rate_wpm = None
        self.prosody.reset_live()
        self.audio_buffer.clear()
        self.vad.reset()
        self.diarizer.reset()

    def process_pcm_frame(
        self,
        samples_512: np.ndarray,
        settings: AccessibilitySettings
    ) -> Dict[str, Any]:
        """Ingest 512 samples (32ms at 16kHz) from incoming audio stream.
        
        Returns dict with:
            - vad_state: dict
            - utterance_event: Optional[UtteranceEvent] (interim or final)
            - debug_metrics: Optional[DebugMetrics]
        """
        now = time.time()
        session_elapsed = now - self.session_start_time
        
        # 1. VAD step
        vad_start = time.perf_counter()
        vad_info = self.vad.process_frame_512(samples_512)
        vad_duration_ms = (time.perf_counter() - vad_start) * 1000.0

        utterance_event: Optional[UtteranceEvent] = None
        utterance_events: List[UtteranceEvent] = []
        metrics: Optional[DebugMetrics] = None
        prosody_state: Optional[LiveProsodyEvent] = None

        if vad_info["is_speaking"]:
            if not self.is_collecting_speech:
                self.is_collecting_speech = True
                self.segment_start_timestamp = max(0.0, session_elapsed - 0.15)  # include slight pre-roll
                self.segment_samples.clear()
                self.last_partial_sample_count = 0
                self.active_interim_id = None
                self._live_speech_rate_wpm = None

            self.segment_samples.append(samples_512)

            # Re-run interim ASR only after another 1.2 seconds of new speech
            # samples. A wall-clock gate can fire on every next frame when ASR
            # itself takes longer than the gate interval.
            current_speech_samples = len(self.segment_samples) * 512
            partial_interval_samples = int(1.2 * self.sample_rate)
            if current_speech_samples - self.last_partial_sample_count >= partial_interval_samples:
                self.last_partial_sample_count = current_speech_samples
                interim_audio = np.concatenate(self.segment_samples)
                # Quick partial ASR
                asr_res = self.asr.transcribe(interim_audio, self.sample_rate)
                if asr_res.text:
                    elapsed_s = self.vad.current_speech_samples / self.sample_rate
                    word_count = len(asr_res.words) or len(asr_res.text.split())
                    self._live_speech_rate_wpm = (
                        float(word_count * 60.0 / elapsed_s) if elapsed_s > 0.4 else None
                    )
                    if self.active_interim_id is None:
                        import uuid
                        self.active_interim_id = str(uuid.uuid4())
                        
                    utterance_event = UtteranceEvent(
                        id=self.active_interim_id,
                        session_id=self.session_id,
                        start_time=self.segment_start_timestamp,
                        end_time=session_elapsed,
                        speaker_id="speaker_current",
                        speaker_label="Speaking...",
                        speaker_color_index=0,
                        text=asr_res.text,
                        is_final=False,
                        overlap=False
                    )
                    utterance_events = [utterance_event]

        elif self.is_collecting_speech and vad_info["segment_ended"]:
            # Conclude speech segment
            self.is_collecting_speech = False
            total_samples = sum(len(c) for c in self.segment_samples)
            segment_dur = total_samples / self.sample_rate

            if total_samples > int(self.sample_rate * 0.3):  # at least 300ms
                audio_segment = np.concatenate(self.segment_samples)
                segment_end_time = session_elapsed

                # Run full processing pipeline on finalized segment (with turn partitioning)
                utterance_events, metrics = self._process_final_segment(
                    audio_segment=audio_segment,
                    start_time=self.segment_start_timestamp,
                    end_time=segment_end_time,
                    settings=settings,
                    vad_duration_ms=vad_duration_ms,
                    interim_id=self.active_interim_id,
                )
                if utterance_events:
                    utterance_event = utterance_events[0]
                    self.last_utterance_end_time = utterance_events[-1].end_time
                    rates = [event.prosody.words_per_minute for event in utterance_events if event.prosody.words_per_minute is not None]
                    self._live_speech_rate_wpm = float(np.mean(rates)) if rates else None

            # Release the segment accumulator references after processing. Python
            # does not guarantee secure erasure of the underlying memory.
            self.segment_samples.clear()
            self.active_interim_id = None

        # Aggregate six 32 ms VAD frames and publish about six live prosody updates
        # per second. This keeps the WebSocket/UI load bounded while retaining a
        # useful trend line. The timestamp is based on audio position, not wall time.
        self._live_prosody_frames.append(np.asarray(samples_512, dtype=np.float32).copy())
        sample_position = int(vad_info["total_samples"])
        frame_voiced = float(vad_info["speech_prob"]) >= float(self.vad.threshold)
        if frame_voiced:
            self._last_voiced_sample = sample_position

        prosody_interval = max(512, self.sample_rate // 6)
        if sample_position - self._last_live_prosody_sample >= prosody_interval:
            live_window = np.concatenate(self._live_prosody_frames) if self._live_prosody_frames else np.empty(0, dtype=np.float32)
            self._live_prosody_frames.clear()
            pause_duration = (
                max(0.0, (sample_position - self._last_voiced_sample) / self.sample_rate)
                if self._last_voiced_sample is not None else None
            )
            prosody_state = self.prosody.analyze_live_frame(
                live_window,
                timestamp_s=sample_position / self.sample_rate,
                speech_active=frame_voiced,
                speech_duration_s=self.vad.current_speech_samples / self.sample_rate,
                pause_duration_s=pause_duration,
                speech_rate_wpm=self._live_speech_rate_wpm,
            )
            self._last_live_prosody_sample = sample_position

        return {
            "vad_state": vad_info,
            "prosody_state": prosody_state,
            "utterance_event": utterance_event,
            "utterance_events": utterance_events,
            "debug_metrics": metrics
        }

    def _partition_into_turns(
        self,
        audio_segment: np.ndarray,
        words: List[Dict[str, Any]],
        start_time: float,
        end_time: float
    ) -> List[Tuple[np.ndarray, List[Dict[str, Any]], str, float, float]]:
        """Partition speech segment into speaker turns if an intra-segment speaker change is detected.
        
        Returns list of (sub_audio, sub_words, sub_text, sub_start_time, sub_end_time).
        """
        dur_s = len(audio_segment) / self.sample_rate
        if len(words) < 4 or dur_s < 1.0 or not hasattr(self.diarizer, "embedder"):
            full_text = " ".join([w.get("word", "") for w in words]).strip()
            return [(audio_segment, words, full_text, start_time, end_time)]

        # Search for speaker change point
        for i in range(len(words) - 1):
            w = words[i].get("word", "")
            gap = words[i + 1].get("start", 0.0) - words[i].get("end", 0.0)
            if w.endswith(("?", ".", "!")) or gap >= 0.20:
                split_t = (words[i].get("end", 0.0) + words[i + 1].get("start", 0.0)) / 2.0
                left_dur = split_t
                right_dur = dur_s - split_t
                
                if left_dur >= 0.35 and right_dur >= 0.35:
                    split_idx = int(split_t * self.sample_rate)
                    left_audio = audio_segment[:split_idx]
                    right_audio = audio_segment[split_idx:]
                    
                    emb_left = self.diarizer.embedder.extract_embedding(left_audio, self.sample_rate)
                    emb_right = self.diarizer.embedder.extract_embedding(right_audio, self.sample_rate)
                    sim = float(np.dot(emb_left, emb_right))
                    
                    threshold = getattr(self.diarizer, "similarity_threshold", 0.25)
                    if sim < threshold:
                        logger.info(f"Intra-segment speaker transition detected at {split_t:.2f}s (sim: {sim:.3f}). Partitioning turns.")
                        left_words = words[:i + 1]
                        right_words = []
                        for rw in words[i + 1:]:
                            rw_copy = dict(rw)
                            rw_copy["start"] = max(0.0, rw["start"] - split_t)
                            rw_copy["end"] = max(0.0, rw["end"] - split_t)
                            right_words.append(rw_copy)

                        left_text = " ".join([w.get("word", "") for w in left_words]).strip()
                        right_text = " ".join([w.get("word", "") for w in right_words]).strip()

                        left_turn = (left_audio, left_words, left_text, start_time, start_time + split_t)
                        
                        # Recursively check right turn for further transitions
                        right_turns = self._partition_into_turns(
                            right_audio,
                            right_words,
                            start_time + split_t,
                            end_time
                        )
                        return [left_turn] + right_turns

        # No speaker transition detected; return whole segment as single turn
        full_text = " ".join([w.get("word", "") for w in words]).strip()
        return [(audio_segment, words, full_text, start_time, end_time)]

    def _process_final_segment(
        self,
        audio_segment: np.ndarray,
        start_time: float,
        end_time: float,
        settings: AccessibilitySettings,
        vad_duration_ms: float,
        interim_id: Optional[str] = None,
    ) -> Tuple[List[UtteranceEvent], DebugMetrics]:
        """Run ASR, Diarization, Prosody, Direction, and Overlap Fusion on finalized segment."""
        t_start = time.perf_counter()
        dur_s = end_time - start_time

        # 1. ASR
        t_asr = time.perf_counter()
        asr_res = self.asr.transcribe(audio_segment, self.sample_rate)
        asr_ms = (time.perf_counter() - t_asr) * 1000.0

        # Partition into turns if multi-speaker transition exists
        turns = self._partition_into_turns(
            audio_segment=audio_segment,
            words=asr_res.words,
            start_time=start_time,
            end_time=end_time
        )

        utterance_events: List[UtteranceEvent] = []
        total_diar_ms = 0.0
        total_pros_ms = 0.0

        for sub_audio, sub_words, sub_text, sub_start, sub_end in turns:
            sub_dur = sub_end - sub_start

            # 2. Diarization
            t_diar = time.perf_counter()
            diar_res = self.diarizer.diarize_segment(
                audio=sub_audio,
                sample_rate=self.sample_rate,
                start_time=sub_start,
                end_time=sub_end
            )
            total_diar_ms += (time.perf_counter() - t_diar) * 1000.0

            # 3. Prosody Extraction
            t_pros = time.perf_counter()
            time_since_last = max(0.0, sub_start - self.last_utterance_end_time)
            speaker_prof = self.registry.speakers.get(diar_res.speaker_id)
            
            prosody_features = self.prosody.analyze(
                audio=sub_audio,
                words=sub_words,
                duration_s=sub_dur,
                time_since_last_utterance=time_since_last,
                speaker_profile=speaker_prof
            )
            total_pros_ms += (time.perf_counter() - t_pros) * 1000.0

            # 4. Direction
            direction_info = None
            if settings.show_speaker_direction:
                direction_info = self.direction.get_direction(diar_res.speaker_id)

            # 5. Overlap Detection
            is_temporal_overlap = False
            overlapping_speakers = set()
            for prev in self.recent_utterances[-5:]:
                if sub_start < (prev.end_time - 0.2):
                    if prev.speaker_id != diar_res.speaker_id:
                        is_temporal_overlap = True
                        overlapping_speakers.add(prev.speaker_label)

            is_overlap = is_temporal_overlap or diar_res.is_overlap
            if diar_res.overlapping_speakers:
                overlapping_speakers.update(diar_res.overlapping_speakers)

            text_display = sub_text if sub_text else (asr_res.text if asr_res.text else "[Speech detected]")

            utterance = UtteranceEvent(
                session_id=self.session_id,
                start_time=sub_start,
                end_time=sub_end,
                speaker_id=diar_res.speaker_id,
                speaker_label=diar_res.speaker_label,
                speaker_color_index=diar_res.color_index,
                text=text_display,
                is_final=True,
                overlap=is_overlap,
                overlapping_speakers=list(overlapping_speakers),
                prosody=prosody_features,
                direction=direction_info,
                confidence=ConfidenceInfo(
                    transcription=asr_res.confidence,
                    speaker=diar_res.confidence
                )
            )

            self.recent_utterances.append(utterance)
            if len(self.recent_utterances) > 50:
                self.recent_utterances.pop(0)

            self.last_utterance_end_time = sub_end
            utterance_events.append(utterance)

        # Replace the rolling interim card with the finalized first turn instead
        # of leaving an identical interim card beside a second final caption.
        if interim_id and utterance_events:
            utterance_events[0].id = interim_id

        total_e2e_ms = (time.perf_counter() - t_start) * 1000.0

        metrics = DebugMetrics(
            chunk_duration_ms=dur_s * 1000.0,
            vad_latency_ms=vad_duration_ms,
            asr_latency_ms=asr_ms,
            diarization_latency_ms=total_diar_ms,
            prosody_latency_ms=total_pros_ms,
            pipeline_processing_ms=total_e2e_ms,
            active_speakers_count=len(self.registry.speakers),
            queue_backlog=0
        )

        return utterance_events, metrics
