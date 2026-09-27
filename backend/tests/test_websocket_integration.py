"""Integration test for ConnectionManager logic and message handling."""
import pytest
import asyncio
import numpy as np
from app.models.settings import AccessibilitySettings
from app.api.websocket import ConnectionManager
from app.fusion.pipeline import FusionPipeline
from app.storage.database import SessionStorage
from app.diarization.registry import SpeakerRegistry
from app.asr.base import BaseASR, ASRResult
from app.diarization.base import BaseDiarizer, DiarizationResult
from app.prosody.analyzer import ProsodyAnalyzer
from app.direction.simulated import SimulatedDirectionProvider


class MockASR(BaseASR):
    def transcribe(self, audio, sample_rate=16000):
        return ASRResult(text="Mocked speech", confidence=0.95)
    def is_ready(self):
        return True


class MockDiarizer(BaseDiarizer):
    def diarize_segment(self, audio, sample_rate=16000, start_time=0.0, end_time=0.0):
        return DiarizationResult(speaker_id="speaker_1", speaker_label="Speaker A", color_index=0)
    def reset(self):
        pass


class FakeWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.closed = False

    async def accept(self):
        pass

    async def send_json(self, data):
        self.sent_messages.append(data)


@pytest.mark.anyio
async def test_connection_manager_flow():
    reg = SpeakerRegistry()
    pipeline = FusionPipeline(
        asr=MockASR(),
        diarizer=MockDiarizer(),
        registry=reg,
        prosody_analyzer=ProsodyAnalyzer(),
        direction_provider=SimulatedDirectionProvider()
    )
    storage = SessionStorage()
    mgr = ConnectionManager(pipeline=pipeline, storage=storage)

    ws = FakeWebSocket()
    await mgr.connect(ws)
    assert len(ws.sent_messages) == 1
    assert ws.sent_messages[0]["type"] == "session_state"

    # Test rename speaker
    reg.get_or_create_speaker("speaker_1")
    await mgr.handle_message(ws, '{"type": "rename_speaker", "speaker_id": "speaker_1", "new_name": "Dr. Rao"}')
    assert any(m.get("type") == "speaker_list" for m in ws.sent_messages)

    # Test clear history
    await mgr.handle_message(ws, '{"type": "clear_history"}')
    assert any(m.get("type") == "history_cleared" for m in ws.sent_messages)

    # Test stop demo when idle
    await mgr.handle_message(ws, '{"type": "stop_demo"}')
    assert any(m.get("type") == "demo_stopped" for m in ws.sent_messages)


@pytest.mark.anyio
async def test_binary_pcm_stream_broadcasts_live_prosody_events():
    reg = SpeakerRegistry()
    pipeline = FusionPipeline(
        asr=MockASR(),
        diarizer=MockDiarizer(),
        registry=reg,
        prosody_analyzer=ProsodyAnalyzer(),
        direction_provider=SimulatedDirectionProvider(),
    )
    frames = 0

    def fake_vad(frame):
        nonlocal frames
        frames += 1
        return {
            "is_speaking": True,
            "segment_ended": False,
            "speech_prob": 0.9,
            "energy_db": -20.0,
            "total_samples": frames * 512,
            "current_speech_samples": frames * 512,
        }

    pipeline.vad.process_frame_512 = fake_vad
    manager = ConnectionManager(pipeline=pipeline, storage=SessionStorage())
    socket = FakeWebSocket()
    await manager.connect(socket)

    tone = 0.2 * np.sin(2 * np.pi * 180 * np.arange(512 * 12) / 16000)
    pcm16 = np.clip(tone * 32768, -32768, 32767).astype("<i2").tobytes()
    await manager.handle_message(socket, pcm16)

    states = [message for message in socket.sent_messages if message.get("type") == "prosody_state"]
    assert len(states) == 2
    assert all(message["data"]["speech_active"] for message in states)
    assert all(message["data"]["pitch_hz"] == pytest.approx(180, abs=8) for message in states)
    assert states[0]["data"]["timestamp_s"] < states[1]["data"]["timestamp_s"]
