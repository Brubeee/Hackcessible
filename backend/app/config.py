"""Application configuration and environment variables."""
import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

class AppConfig(BaseModel):
    app_name: str = "Hackcessible India 2026 - Conversational Visual Layer"
    version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    
    # Audio settings
    sample_rate: int = 16000
    vad_threshold: float = 0.5
    min_speech_duration_ms: int = 300
    min_silence_duration_ms: int = 600
    
    # ASR settings
    whisper_model_size: str = os.getenv("WHISPER_MODEL", "tiny.en")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    
    # Diarization settings
    similarity_threshold: float = 0.68  # Cosine similarity threshold for clustering
    max_active_speakers: int = 10
    
    # Storage settings
    db_path: Path = BASE_DIR / "session_storage.db"
    default_privacy_mode: bool = True  # Raw audio never persisted by default
    save_transcript_default: bool = False  # Opt-in only

config = AppConfig()
