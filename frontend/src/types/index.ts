export interface ProsodyFeatures {
  rising_intonation: boolean;
  falling_intonation: boolean;
  emphasis: boolean;
  long_pause_before: boolean;
  speech_rate: 'slow' | 'normal' | 'fast';
  relative_volume: 'quiet' | 'normal' | 'loud';
  mean_pitch_hz?: number | null;
  min_pitch_hz?: number | null;
  max_pitch_hz?: number | null;
  pitch_range_semitones?: number | null;
  rms_db?: number | null;
  words_per_minute?: number | null;
  pitch_semitone_excursion?: number | null;
}

export interface LiveProsodyEvent {
  timestamp_s: number;
  speech_active: boolean;
  pitch_hz: number | null;
  pitch_direction: 'rising' | 'falling' | 'steady' | 'unavailable';
  level_dbfs: number;
  relative_level_db: number | null;
  speech_rate_wpm: number | null;
  speech_duration_s: number;
  pause_duration_s: number | null;
  emphasis_candidate: boolean;
}

export interface DirectionInfo {
  angle_degrees: number;
  label: string;
  simulated: boolean;
  confidence?: number | null;
}

export interface ConfidenceInfo {
  transcription?: number | null;
  speaker?: number | null;
}

export interface SpeakerInfo {
  id: string;
  label: string;
  color_index: number;
  custom_name?: string | null;
  is_active: boolean;
  total_utterances: number;
  last_heard_time: number;
}

export interface UtteranceEvent {
  id: string;
  session_id: string;
  start_time: number;
  end_time: number;
  speaker_id: string;
  speaker_label: string;
  speaker_color_index: number;
  text: string;
  is_final: boolean;
  overlap: boolean;
  overlapping_speakers: string[];
  prosody: ProsodyFeatures;
  direction?: DirectionInfo | null;
  confidence: ConfidenceInfo;
  created_at?: number;
}

export interface AccessibilitySettings {
  show_captions: boolean;
  show_speaker_identities: boolean;
  show_overlap_indicators: boolean;
  show_prosody_cues: boolean;
  show_speaker_direction: boolean;
  save_transcript: boolean;
  privacy_mode: boolean;
  font_size: 'small' | 'medium' | 'large' | 'xlarge';
  contrast_theme: 'normal' | 'high_contrast';
  reduced_motion: boolean;
  compact_mode: boolean;
  simulated_direction: boolean;
}

export interface VADState {
  speaking: boolean;
  energy: number;
  probability: number;
}

export interface DebugMetrics {
  chunk_duration_ms: number;
  vad_latency_ms: number;
  asr_latency_ms: number;
  diarization_latency_ms: number;
  prosody_latency_ms: number;
  pipeline_processing_ms: number;
  active_speakers_count: number;
  queue_backlog: number;
  simulated: boolean;
  timestamp?: number;
}
