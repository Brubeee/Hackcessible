"""Controlled noise stress checks using the real PCM16 WebSocket caption path.

Noise is synthesized from deterministic generators; these results are not a
substitute for recorded room-noise tests or a ground-truth transcription set.
"""
import asyncio
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from verify_websocket_audio import PROJECT_ROOT, verify_audio_websocket
from app.main import ws_manager


def active_mask(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_len = max(1, sample_rate // 50)  # 20 ms
    count = len(audio) // frame_len
    if count == 0:
        return np.ones(len(audio), dtype=bool)
    blocks = audio[:count * frame_len].reshape(count, frame_len)
    levels = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1))
    threshold = max(0.008, float(np.percentile(levels, 85)) * 0.12)
    voiced = levels >= threshold
    mask = np.repeat(voiced, frame_len)
    return np.pad(mask, (0, len(audio) - len(mask)), constant_values=False)


def normalized_noise(kind: str, length: int, sample_rate: int, rng: np.random.Generator, speech: np.ndarray) -> np.ndarray:
    if kind == "white":
        noise = rng.normal(0.0, 1.0, length)
    elif kind == "pink":
        white = rng.normal(0.0, 1.0, length)
        spectrum = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(length, 1 / sample_rate)
        weights = np.ones_like(frequencies)
        weights[1:] = 1.0 / np.sqrt(frequencies[1:])
        noise = np.fft.irfft(spectrum * weights, n=length)
    elif kind == "fan_hum":
        t = np.arange(length, dtype=np.float64) / sample_rate
        noise = 0.6 * rng.normal(0.0, 1.0, length)
        noise += 0.8 * np.sin(2 * np.pi * 60 * t)
        noise += 0.35 * np.sin(2 * np.pi * 120 * t + 0.2)
        noise += 0.18 * np.sin(2 * np.pi * 240 * t + 0.7)
    elif kind == "speech_babble_proxy":
        # A delayed, circularly shifted copy approximates competing speech energy.
        # It is deliberately labeled a proxy, not a recording of independent talkers.
        shift = int(sample_rate * 7.3) % max(1, length)
        noise = np.roll(speech, shift).astype(np.float64)
        noise += 0.015 * rng.normal(0.0, 1.0, length)
    else:
        raise ValueError(f"Unknown noise type: {kind}")

    mask = active_mask(speech, sample_rate)
    noise_active = noise[mask]
    if noise_active.size == 0:
        noise_active = noise
    rms = float(np.sqrt(np.mean(noise_active ** 2) + 1e-12))
    return (noise / max(rms, 1e-9)).astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, sample_rate: int, snr_db: float) -> np.ndarray:
    mask = active_mask(speech, sample_rate)
    speech_rms = float(np.sqrt(np.mean(speech[mask].astype(np.float64) ** 2) + 1e-12))
    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    mixed = speech.astype(np.float64) + noise.astype(np.float64) * target_noise_rms
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.98:
        mixed *= 0.98 / peak
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def token_edit_rate(reference: str, candidate: str) -> float | None:
    """Text consistency vs the clean model output; this is not WER/accuracy."""
    ref = tokens(reference)
    got = tokens(candidate)
    if not ref:
        return None
    previous = list(range(len(got) + 1))
    for index, ref_word in enumerate(ref, start=1):
        current = [index]
        for column, got_word in enumerate(got, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (ref_word != got_word),
            ))
        previous = current
    return round(previous[-1] / len(ref), 3)


async def main() -> dict:
    source_path = PROJECT_ROOT / "samples" / "test_video_60s.wav"
    full_audio, sample_rate = sf.read(str(source_path), dtype="float32")
    if full_audio.ndim > 1:
        full_audio = full_audio.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz source audio, received {sample_rate} Hz.")

    # Use a speech-rich, repeatable 20-second section to keep this diagnostic practical.
    start = 20 * sample_rate
    speech = full_audio[start:start + 20 * sample_rate].copy()
    if len(speech) != 20 * sample_rate:
        raise ValueError("Bundled test audio is too short for the selected stress-test window.")

    rng = np.random.default_rng(20260927)
    noise_cases = [
        ("pink_20dB", "pink", 20.0),
        ("pink_10dB", "pink", 10.0),
        ("fan_hum_10dB", "fan_hum", 10.0),
        ("speech_babble_proxy_5dB", "speech_babble_proxy", 5.0),
    ]
    clean_result = await verify_audio_websocket(
        audio_override=speech,
        sample_rate_override=sample_rate,
        source_name="clean sample section (20-40s)",
        session_id="noise_clean_baseline",
        minimum_final_captions=0,
        minimum_recognized_captions=0,
        include_caption_texts=True,
    )
    clean_text = " ".join(clean_result.get("caption_texts", []))
    results = [{
        "condition": "clean_baseline",
        "snr_db": None,
        "noise_source": "none",
        "clean_model_consistency_edit_rate": 0.0,
        **clean_result,
    }]

    for label, noise_kind, snr in noise_cases:
        noise = normalized_noise(noise_kind, len(speech), sample_rate, rng, speech)
        mixed = mix_at_snr(speech, noise, sample_rate, snr)
        result = await verify_audio_websocket(
            audio_override=mixed,
            sample_rate_override=sample_rate,
            source_name=f"synthetic {noise_kind} at {snr:g} dB SNR",
            session_id=f"noise_{label}",
            minimum_final_captions=0,
            minimum_recognized_captions=0,
            include_caption_texts=True,
        )
        text = " ".join(result.get("caption_texts", []))
        results.append({
            "condition": label,
            "snr_db": snr,
            "noise_source": noise_kind,
            "clean_model_consistency_edit_rate": token_edit_rate(clean_text, text),
            **result,
        })

    return {
        "source": source_path.name,
        "source_window_seconds": [20, 40],
        "sample_rate_hz": sample_rate,
        "evaluation_note": "Synthetic noise over a bundled recording. Text edit rate compares noisy output with the clean model output, not a human transcript; it is not WER or accuracy.",
        "physical_microphone_test": False,
        "conditions": results,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="hackcessible-noise-check-") as temp_dir:
        original_db_path = ws_manager.storage.db_path
        ws_manager.storage.db_path = Path(temp_dir) / "temporary-noise-check.sqlite"
        try:
            print(json.dumps(asyncio.run(main()), indent=2))
        finally:
            ws_manager.storage.db_path = original_db_path
