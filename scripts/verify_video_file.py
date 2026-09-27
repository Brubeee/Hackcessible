"""End-to-end verification script testing the live streaming conversation pipeline
on the user's uploaded video file ('Conversations between two friends').
"""
import sys
import os
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import soundfile as sf
import numpy as np

from app.asr.whisper_asr import WhisperASR
from app.diarization.clusterer import AcousticClusterDiarizer
from app.diarization.registry import SpeakerRegistry
from app.prosody.analyzer import ProsodyAnalyzer
from app.direction.simulated import SimulatedDirectionProvider
from app.fusion.pipeline import FusionPipeline
from app.models.settings import AccessibilitySettings
from app.models.events import UtteranceEvent


def verify_audio_file(audio_path: Path):
    print("=" * 80)
    print(f"VERIFYING CONVERSATION PIPELINE ON: {audio_path.name}")
    print("=" * 80)

    if not audio_path.exists():
        print(f"ERROR: Audio file not found at {audio_path}")
        return False

    audio, sr = sf.read(str(audio_path))
    duration_s = len(audio) / sr
    print(f"Loaded {duration_s:.2f} seconds of audio ({len(audio)} samples, {sr} Hz)")

    # Normalize audio to float32 [-1.0, 1.0]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Initialize full production pipeline components
    asr = WhisperASR(model_size="tiny.en", device="cpu", compute_type="int8")
    registry = SpeakerRegistry()
    diarizer = AcousticClusterDiarizer(registry=registry, similarity_threshold=0.25)
    prosody = ProsodyAnalyzer(sample_rate=sr)
    direction = SimulatedDirectionProvider()

    pipeline = FusionPipeline(
        asr=asr,
        diarizer=diarizer,
        registry=registry,
        prosody_analyzer=prosody,
        direction_provider=direction,
        sample_rate=sr
    )
    settings = AccessibilitySettings(
        show_prosody_cues=True,
        show_speaker_direction=True,
        privacy_mode=True
    )

    pipeline.reset_session("verification_session")

    final_utterances = []
    chunk_size = 512  # 32ms streaming frames

    start_wall = time.perf_counter()
    print("\nStreaming 32ms frames through VoiceActivityDetector, ASR, ECAPA Diarizer, and Prosody...\n")

    for i in range(0, len(audio), chunk_size):
        frame = audio[i:i + chunk_size]
        if len(frame) < chunk_size:
            frame = np.pad(frame, (0, chunk_size - len(frame)))

        res = pipeline.process_pcm_frame(frame, settings)

        events = res.get("utterance_events") or []
        single = res.get("utterance_event")
        if not events and single:
            events = [single]

        for ev in events:
            if ev.is_final:
                final_utterances.append(ev)
                intonation = "Rising" if ev.prosody.rising_intonation else ("Falling" if ev.prosody.falling_intonation else "Flat")
                excursion_str = f"{ev.prosody.pitch_semitone_excursion:+.1f} ST" if ev.prosody.pitch_semitone_excursion is not None else "N/A"
                print(f"[{ev.start_time:5.2f}s - {ev.end_time:5.2f}s] "
                      f"{ev.speaker_label:<10s} (id: {ev.speaker_id}) | "
                      f"Pitch: {intonation:<10s} ({excursion_str}) | "
                      f"Vol: {ev.prosody.relative_volume:<6s} | "
                      f"Rate: {ev.prosody.speech_rate:<6s} | "
                      f"Text: '{ev.text}'")

    total_processing_time = time.perf_counter() - start_wall
    realtime_factor = total_processing_time / duration_s

    print("\n" + "=" * 80)
    print("VERIFICATION ANALYSIS & METRICS")
    print("=" * 80)
    print(f"Total Audio Duration:      {duration_s:.2f}s")
    print(f"Total Processing Time:     {total_processing_time:.2f}s")
    print(f"Real-Time Factor (RTF):    {realtime_factor:.2f}x (lower than 1.0x means faster than real-time)")
    print(f"Total Utterances Emitted:  {len(final_utterances)}")
    print(f"Distinct Speakers Found:   {len(registry.speakers)}")
    for spk_id, spk in registry.speakers.items():
        print(f"  - {spk.display_name} ({spk_id}): {spk.total_utterances} turns")

    # Verification Checks
    print("\n--- CRITERIA EVALUATION ---")
    
    # 1. Multi-speaker presence check
    multi_speaker_ok = len(registry.speakers) >= 2
    print(f"[CHECK 1] Multi-speaker distinction (>= 2 speakers identified): {'PASS' if multi_speaker_ok else 'FAIL'}")

    # 2. Alternating turn check
    speakers_sequence = [u.speaker_id for u in final_utterances]
    transitions = sum(1 for a, b in zip(speakers_sequence, speakers_sequence[1:]) if a != b)
    turn_taking_ok = transitions >= 3
    print(f"[CHECK 2] Natural turn-taking transitions ({transitions} speaker shifts): {'PASS' if turn_taking_ok else 'FAIL'}")

    # 3. Question Intonation Check (at least one question has rising intonation)
    question_utterances = [u for u in final_utterances if "?" in u.text]
    rising_questions = [u for u in question_utterances if u.prosody.rising_intonation]
    question_intonation_ok = len(rising_questions) >= 1
    print(f"[CHECK 3] Rising intonation on questions ({len(rising_questions)}/{len(question_utterances)} questions with rising pitch): {'PASS' if question_intonation_ok else 'FAIL'}")

    # 4. Defensible non-emotional prosody output check
    all_prosody_valid = all(
        u.prosody.speech_rate in ("slow", "normal", "fast") and
        u.prosody.relative_volume in ("quiet", "normal", "loud")
        for u in final_utterances
    )
    print(f"[CHECK 4] Ethical, non-emotional prosody compliance: {'PASS' if all_prosody_valid else 'FAIL'}")

    all_passed = multi_speaker_ok and turn_taking_ok and question_intonation_ok and all_prosody_valid
    print("=" * 80)
    print(f"OVERALL PIPELINE VERIFICATION: {'SUCCESS (ALL PASSED)' if all_passed else 'FAILURE'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    audio_file = Path("samples/test_video_60s.wav")
    success = verify_audio_file(audio_file)
    sys.exit(0 if success else 1)
