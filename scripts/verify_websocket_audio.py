"""Replay the bundled WAV through the microphone WebSocket PCM16 input path.

This is a prerecorded-audio integration check, not a physical microphone test.
The reported latency values come from measured live pipeline events, not demo data.
"""
import asyncio
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import uvicorn
import websockets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.main import app  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("websocket-audio-verifier")

HOST = "127.0.0.1"
PORT = 8012
SESSION_ID = "audio_websocket_verification"
DRAIN_SESSION_ID = "audio_websocket_verification_drained"
PACKET_SAMPLES = 2048  # Matches the browser ScriptProcessorNode buffer size.


async def wait_for_server(server: uvicorn.Server, server_task: asyncio.Task) -> None:
    async with httpx.AsyncClient(base_url=f"http://{HOST}:{PORT}") as client:
        for _ in range(100):
            if server.started:
                return
            if server_task.done():
                await server_task
            try:
                response = await client.get("/api/health", timeout=1.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
    raise RuntimeError("Audio verification server did not become healthy within 10 seconds.")


async def verify_audio_websocket(
    wav_path: Path | None = None,
    *,
    audio_override: np.ndarray | None = None,
    sample_rate_override: int | None = None,
    source_name: str | None = None,
    session_id: str = SESSION_ID,
    minimum_final_captions: int = 2,
    minimum_recognized_captions: int = 2,
    include_caption_texts: bool = False,
) -> dict:
    if audio_override is None:
        wav_path = wav_path or PROJECT_ROOT / "samples" / "test_video_60s.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Required verification audio is missing: {wav_path}")
        audio, sample_rate = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        source_name = source_name or wav_path.name
    else:
        audio = np.asarray(audio_override, dtype=np.float32).reshape(-1).copy()
        sample_rate = int(sample_rate_override or 16000)
        source_name = source_name or "in-memory PCM clip"

    if sample_rate != 16000:
        raise ValueError(f"Expected a 16 kHz WAV for direct PCM streaming, got {sample_rate} Hz.")

    # Include trailing silence to let VAD finalize the last speech segment.
    audio = np.concatenate((audio, np.zeros(sample_rate, dtype=np.float32)))
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    server = uvicorn.Server(uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        log_level="warning",
        access_log=False,
    ))
    server_task = asyncio.create_task(server.serve())
    try:
        await wait_for_server(server, server_task)
        counters = {
            "final_captions": 0,
            "interim_captions": 0,
            "recognized_captions": 0,
            "live_directionless_captions": 0,
            "live_prosody_updates": 0,
            "vad_updates": 0,
            "caption_texts": [],
            "speaker_ids": set(),
            "latencies_ms": [],
        }
        session_ready = asyncio.Event()
        stream_drained = asyncio.Event()
        target_session_id = session_id
        drain_session_id = f"{target_session_id}_drained"
        uri = f"ws://{HOST}:{PORT}/ws/conversation"

        async with websockets.connect(uri, max_queue=2048) as ws:
            initial_state = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            assert initial_state.get("type") == "session_state", "Missing initial WebSocket session state."

            async def receive_events() -> None:
                async for raw_message in ws:
                    message = json.loads(raw_message)
                    message_type = message.get("type")
                    data = message.get("data", {})

                    if message_type == "session_state":
                        received_session_id = data.get("session_id")
                        if received_session_id == target_session_id:
                            session_ready.set()
                        elif received_session_id == drain_session_id:
                            stream_drained.set()
                            return
                    elif message_type == "vad_state":
                        counters["vad_updates"] += 1
                    elif message_type == "utterance":
                        if data.get("is_final"):
                            counters["final_captions"] += 1
                            counters["speaker_ids"].add(data.get("speaker_id"))
                            if data.get("direction") is not None:
                                raise AssertionError("Live mono-microphone mode returned a spatial direction.")
                            counters["live_directionless_captions"] += 1
                            text = (data.get("text") or "").strip()
                            if text and text != "[Speech detected]":
                                counters["recognized_captions"] += 1
                                counters["caption_texts"].append(text)
                        else:
                            counters["interim_captions"] += 1
                    elif message_type == "prosody_state":
                        counters["live_prosody_updates"] += 1
                    elif message_type == "debug_metrics":
                        if data.get("simulated", False):
                            raise AssertionError("A live PCM run emitted simulated debug metrics.")
                        counters["latencies_ms"].append(float(data["pipeline_processing_ms"]))

            receiver_task = asyncio.create_task(receive_events())
            await ws.send(json.dumps({
                "type": "start_session",
                "session_id": target_session_id,
                "settings": {
                    "privacy_mode": True,
                    "save_transcript": False,
                    "show_speaker_direction": True,
                },
            }))
            await asyncio.wait_for(session_ready.wait(), timeout=5.0)

            started = time.perf_counter()
            packet_bytes = PACKET_SAMPLES * 2
            for offset in range(0, len(pcm16), packet_bytes):
                await ws.send(pcm16[offset:offset + packet_bytes])

            # Ordered messages make this start_session event a drain barrier.
            await ws.send(json.dumps({"type": "start_session", "session_id": drain_session_id}))
            drain_waiter = asyncio.create_task(stream_drained.wait())
            done, _ = await asyncio.wait(
                {drain_waiter, receiver_task},
                timeout=180.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                receiver_task.cancel()
                drain_waiter.cancel()
                raise TimeoutError("PCM replay did not reach the WebSocket drain barrier within 180 seconds.")
            if receiver_task in done and not stream_drained.is_set():
                await receiver_task
                raise RuntimeError("WebSocket event receiver stopped before the drain barrier.")
            await drain_waiter
            wall_seconds = time.perf_counter() - started
            await receiver_task

        assert counters["vad_updates"] > 0, "No VAD updates were received for the streamed audio."
        assert counters["final_captions"] >= minimum_final_captions, "Fewer than the required final captions were produced."
        assert counters["recognized_captions"] >= minimum_recognized_captions, "Fewer than the required recognized captions were produced."
        assert counters["live_directionless_captions"] == counters["final_captions"]
        assert counters["live_prosody_updates"] > 0, "The live pipeline emitted no real prosody updates."
        if minimum_final_captions > 0:
            assert len(counters["speaker_ids"]) >= 1, "No speaker identities were assigned."
        if minimum_recognized_captions > 0:
            assert counters["latencies_ms"], "The live pipeline emitted no measured debug metrics."

        latencies = counters["latencies_ms"]
        result = {
            "source": source_name,
            "physical_microphone_test": False,
            "sample_rate_hz": sample_rate,
            "streamed_audio_seconds": round(len(audio) / sample_rate, 2),
            "wall_clock_seconds": round(wall_seconds, 2),
            "final_captions": counters["final_captions"],
            "interim_captions": counters["interim_captions"],
            "recognized_captions": counters["recognized_captions"],
            "live_direction_returned_none": True,
            "speaker_count": len(counters["speaker_ids"]),
            "vad_updates": counters["vad_updates"],
            "live_prosody_updates": counters["live_prosody_updates"],
            "measured_pipeline_processing_samples": len(latencies),
            "measured_pipeline_processing_median_ms": round(statistics.median(latencies), 2) if latencies else None,
            "measured_pipeline_processing_max_ms": round(max(latencies), 2) if latencies else None,
        }
        if include_caption_texts:
            result["caption_texts"] = counters["caption_texts"]
        return result
    finally:
        server.should_exit = True
        if not server_task.done():
            try:
                await asyncio.wait_for(server_task, timeout=10.0)
            except asyncio.TimeoutError:
                server.force_exit = True
                await server_task


if __name__ == "__main__":
    try:
        result = asyncio.run(verify_audio_websocket())
    except Exception:
        logger.exception("WebSocket PCM audio verification failed.")
        raise
    print(json.dumps(result, indent=2))
