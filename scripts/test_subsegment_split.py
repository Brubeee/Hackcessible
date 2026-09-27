import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import soundfile as sf
import numpy as np
from app.asr.whisper_asr import WhisperASR
from app.diarization.clusterer import AcousticClusterDiarizer
from app.diarization.registry import SpeakerRegistry

def main():
    data, sr = sf.read("samples/test_video_60s.wav")
    different_speaker_detected = False
    same_speaker_matched = False
    
    print("--- Test 1: Different Speakers (Friend 1 + Friend 2) ---")
    chunk1 = data[int(21.5 * sr):int(23.0 * sr)] # Friend 1 ("Hey, how's it going?")
    chunk2 = data[int(25.5 * sr):int(27.5 * sr)] # Friend 2 ("I'm good, thanks.")
    combined = np.concatenate([chunk1, np.zeros(int(0.15 * sr), dtype=np.float32), chunk2])

    asr = WhisperASR()
    res = asr.transcribe(combined, sr)
    diar = AcousticClusterDiarizer(SpeakerRegistry(), similarity_threshold=0.25)

    words = res.words
    for i in range(len(words) - 1):
        w = words[i]["word"]
        gap = words[i + 1]["start"] - words[i]["end"]
        if w.endswith(("?", ".", "!")) or gap >= 0.20:
            split_time = (words[i]["end"] + words[i + 1]["start"]) / 2.0
            left_audio = combined[:int(split_time * sr)]
            right_audio = combined[int(split_time * sr):]
            if len(left_audio) >= int(sr * 0.4) and len(right_audio) >= int(sr * 0.4):
                emb1 = diar.embedder.extract_embedding(left_audio, sr)
                emb2 = diar.embedder.extract_embedding(right_audio, sr)
                sim = float(np.dot(emb1, emb2))
                print(f"Candidate split after '{w}' at {split_time:.2f}s: similarity = {sim:.4f}")
                if sim < 0.25:
                    different_speaker_detected = True
                    print(f"==> Speaker change correctly detected! Sim ({sim:.4f}) < 0.25")

    print("\n--- Test 2: Same Speaker (Friend 2 + Friend 2) ---")
    # Friend 2: "I'm good, thanks." + "Just another day..."
    same_spk = data[int(25.5 * sr):int(31.0 * sr)]
    res_same = asr.transcribe(same_spk, sr)
    print("Same speaker text:", res_same.text)
    words_same = res_same.words
    for i in range(len(words_same) - 1):
        w = words_same[i]["word"]
        gap = words_same[i + 1]["start"] - words_same[i]["end"]
        if w.endswith(("?", ".", "!")) or gap >= 0.20:
            split_time = (words_same[i]["end"] + words_same[i + 1]["start"]) / 2.0
            left_audio = same_spk[:int(split_time * sr)]
            right_audio = same_spk[int(split_time * sr):]
            if len(left_audio) >= int(sr * 0.4) and len(right_audio) >= int(sr * 0.4):
                emb1 = diar.embedder.extract_embedding(left_audio, sr)
                emb2 = diar.embedder.extract_embedding(right_audio, sr)
                sim = float(np.dot(emb1, emb2))
                print(f"Candidate split after '{w}' at {split_time:.2f}s: similarity = {sim:.4f}")
                if sim >= 0.25:
                    same_speaker_matched = True
                    print(f"==> Correctly recognized as same speaker! Sim ({sim:.4f}) >= 0.25")

    if not different_speaker_detected:
        raise AssertionError("No different-speaker candidate split passed the 0.25 threshold.")
    if not same_speaker_matched:
        raise AssertionError("No same-speaker candidate match passed the 0.25 threshold.")

if __name__ == "__main__":
    main()
