"""User settings and accessibility configurations."""
from pydantic import BaseModel, Field


class AccessibilitySettings(BaseModel):
    """User preferences for accessibility controls."""
    show_captions: bool = Field(default=True, description="Master captions display")
    show_speaker_identities: bool = Field(default=True, description="Display speaker tags and distinct colors")
    show_overlap_indicators: bool = Field(default=True, description="Highlight simultaneous speech overlaps")
    show_prosody_cues: bool = Field(default=True, description="Display acoustic/prosodic badges (pitch, pace, etc.)")
    show_speaker_direction: bool = Field(default=False, description="Display available spatial direction indicators")
    save_transcript: bool = Field(default=False, description="Opt-in to local transcript saving (OFF by default)")
    privacy_mode: bool = Field(default=True, description="Prevent transcript persistence to disk")
    
    font_size: str = Field(default="medium", description="'small', 'medium', 'large', 'xlarge'")
    contrast_theme: str = Field(default="normal", description="'normal', 'high_contrast'")
    reduced_motion: bool = Field(default=False, description="Disable transitions and animated indicators")
    compact_mode: bool = Field(default=False, description="Compact view with reduced padding")
    simulated_direction: bool = Field(default=True, description="Allow simulated direction in demo/dev mode")


class AudioConfig(BaseModel):
    """Internal audio stream configuration."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size_samples: int = 512  # For Silero VAD frame (32ms at 16kHz)
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 600
    max_speech_duration_s: float = 12.0
