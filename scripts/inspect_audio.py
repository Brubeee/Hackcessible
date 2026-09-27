import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import soundfile as sf
from app.asr.whisper_asr import WhisperASR

def main():
    wav_path = Path("samples/test_video_60s.wav")
    if not wav_path.exists():
        print(f"File {wav_path} not found")
        return
    data, sr = sf.read(str(wav_path))
    print(f"Loaded {wav_path}: {len(data)/sr:.2f}s, sample rate: {sr}")
    
    asr = WhisperASR()
    res = asr.transcribe(data, sr)
    print("Full Transcribed Text:\n", res.text)
    print("\nWords with timestamps:")
    for w in res.words:
        print(f"  {w['start']:6.2f}s - {w['end']:6.2f}s : {w['word']}")

if __name__ == "__main__":
    main()
