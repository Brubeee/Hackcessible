"""Data models for conversation utterances, prosody, speakers, and events."""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time
import uuid


class ProsodyFeatures(BaseModel):
    """Measurable, conservative acoustic features.
    
    CRITICAL: Never classify psychological or emotional states (e.g. angry, happy, nervous).
    Only output defensible acoustic measurements and derived directional cues.
    """
    rising_intonation: bool = Field(
        default=False,
        description="True if terminal pitch contour exhibits a significant upward slope (>15% increase)"
    )
    falling_intonation: bool = Field(
        default=False,
        description="True if terminal pitch contour exhibits a significant downward slope (>15% decrease)"
    )
    emphasis: bool = Field(
        default=False,
        description="True if a local syllable exhibits high intensity burst combined with pitch excursion"
    )
    long_pause_before: bool = Field(
        default=False,
        description="True if preceded by an unvoiced/silent gap longer than 1.2 seconds"
    )
    speech_rate: str = Field(
        default="normal",
        description="Relative speech rate: 'slow', 'normal', 'fast' based on words per second"
    )
    relative_volume: str = Field(
        default="normal",
        description="Relative volume: 'quiet', 'normal', 'loud' compared to speaker's running baseline"
    )
    mean_pitch_hz: Optional[float] = Field(
        default=None,
        description="Estimated mean fundamental frequency (F0) in Hz over voiced frames"
    )
    min_pitch_hz: Optional[float] = Field(default=None, description="Lowest reliable voiced F0 estimate in the utterance")
    max_pitch_hz: Optional[float] = Field(default=None, description="Highest reliable voiced F0 estimate in the utterance")
    pitch_range_semitones: Optional[float] = Field(default=None, description="Pitch range across voiced frames, in semitones")
    rms_db: Optional[float] = Field(
        default=None,
        description="Root mean square energy level in decibels relative to full scale"
    )
    words_per_minute: Optional[float] = Field(
        default=None,
        description="Estimated speaking rate in words per minute"
    )
    pitch_semitone_excursion: Optional[float] = Field(
        default=None,
        description="Measured boundary tone pitch excursion in semitones relative to preceding syllable nucleus"
    )


class LiveProsodyEvent(BaseModel):
    """Low-rate live acoustic measurements; deliberately contains no emotion labels."""
    timestamp_s: float = Field(ge=0, description="Audio position in the current session")
    speech_active: bool
    pitch_hz: Optional[float] = Field(default=None, gt=0, description="Voiced-frame F0 estimate in Hz")
    pitch_direction: str = Field(default="unavailable", description="Short-term measured F0 trend")
    level_dbfs: float = Field(description="RMS level in dBFS; depends on microphone gain")
    relative_level_db: Optional[float] = Field(default=None, description="Level relative to this session's recent voiced baseline")
    speech_rate_wpm: Optional[float] = Field(default=None, ge=0, description="Estimated from available interim/final ASR words")
    speech_duration_s: float = Field(default=0, ge=0)
    pause_duration_s: Optional[float] = Field(default=None, ge=0)
    emphasis_candidate: bool = Field(default=False, description="Possible local level and pitch excursion; heuristic only")


class DirectionInfo(BaseModel):
    """Direction of arrival information (hardware or clearly labeled simulation)."""
    angle_degrees: float = Field(
        ...,
        description="Estimated horizontal angle in degrees (-90 left to +90 right, 0 front)"
    )
    label: str = Field(
        ...,
        description="Categorical position: 'left', 'front-left', 'front', 'front-right', 'right'"
    )
    simulated: bool = Field(
        default=False,
        description="Must be True if generated in simulation/development mode without hardware"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score between 0.0 and 1.0 if provided by hardware"
    )


class ConfidenceInfo(BaseModel):
    """Model confidence reporting - null where not genuinely measurable."""
    transcription: Optional[float] = Field(
        default=None,
        description="Whisper average token log probability or confidence score"
    )
    speaker: Optional[float] = Field(
        default=None,
        description="Diarization embedding cosine similarity score"
    )


class SpeakerInfo(BaseModel):
    """Persistent speaker identity in session registry."""
    id: str = Field(..., description="Internal stable identifier, e.g. 'speaker_1'")
    label: str = Field(..., description="Display label, e.g. 'Speaker A' or renamed 'Professor'")
    color_index: int = Field(default=0, description="Index into accessible color palette")
    custom_name: Optional[str] = Field(default=None, description="User-assigned custom name")
    is_active: bool = Field(default=False, description="Currently speaking")
    total_utterances: int = Field(default=0, description="Total utterances by this speaker")
    last_heard_time: float = Field(default=0.0, description="Session relative timestamp in seconds")


class UtteranceEvent(BaseModel):
    """Core utterance event data model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default="default_session")
    start_time: float = Field(..., description="Start timestamp in seconds relative to session start")
    end_time: float = Field(..., description="End timestamp in seconds relative to session start")
    speaker_id: str = Field(..., description="Internal speaker id (e.g. 'speaker_1')")
    speaker_label: str = Field(..., description="Human-readable label (e.g. 'Speaker A' or 'Professor')")
    speaker_color_index: int = Field(default=0, description="Consistent color palette index")
    text: str = Field(..., description="Transcribed speech text")
    is_final: bool = Field(default=True, description="True if finalized segment, False if interim rolling")
    overlap: bool = Field(default=False, description="True if simultaneous overlapping speech detected")
    overlapping_speakers: List[str] = Field(
        default_factory=list,
        description="List of other speaker labels overlapping this segment"
    )
    prosody: ProsodyFeatures = Field(default_factory=ProsodyFeatures)
    direction: Optional[DirectionInfo] = Field(default=None)
    confidence: ConfidenceInfo = Field(default_factory=ConfidenceInfo)
    created_at: float = Field(default_factory=time.time)


class OverlapEvent(BaseModel):
    """Detailed overlap notification when two or more speakers talk at once."""
    start_time: float
    end_time: float
    speaker_ids: List[str]
    speaker_labels: List[str]
    description: str


class VADStateEvent(BaseModel):
    """Real-time Voice Activity Detection state."""
    speaking: bool
    energy: float
    speech_probability: float
    timestamp: float = Field(default_factory=time.time)


class DebugMetrics(BaseModel):
    """Pipeline metrics; demo values are explicitly marked as simulated."""
    chunk_duration_ms: float
    vad_latency_ms: float
    asr_latency_ms: float
    diarization_latency_ms: float
    prosody_latency_ms: float
    pipeline_processing_ms: float
    active_speakers_count: int
    queue_backlog: int
    simulated: bool = Field(
        default=False,
        description="True when values are illustrative demo data rather than measured processing times",
    )
    timestamp: float = Field(default_factory=time.time)
