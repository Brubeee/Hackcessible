"""WebSocket connection manager and conversation session handler."""
import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Dict, Set, Optional, Any, List
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from app.models.events import UtteranceEvent, ProsodyFeatures, DirectionInfo, ConfidenceInfo, DebugMetrics
from app.models.settings import AccessibilitySettings
from app.fusion.pipeline import FusionPipeline
from app.storage.database import SessionStorage

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSockets and dispatches events."""
    def __init__(self, pipeline: FusionPipeline, storage: SessionStorage):
        self.pipeline = pipeline
        self.storage = storage
        self.active_connections: Set[WebSocket] = set()
        self.settings = AccessibilitySettings()
        self._demo_task: Optional[asyncio.Task] = None
        self._is_demo_running = False
        self._pipeline_lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")
        
        # Send initial state
        await self.send_personal_message({
            "type": "session_state",
            "data": {
                "session_id": self.pipeline.session_id,
                "is_demo": self._is_demo_running,
                "settings": self.settings.model_dump(),
                "speakers": [s.model_dump() for s in self.pipeline.registry.list_speakers()]
            }
        }, websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        dead = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error broadcasting to client: {e}")
                dead.append(connection)
        for d in dead:
            self.active_connections.discard(d)

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Error sending message to client: {e}")

    async def handle_message(self, websocket: WebSocket, raw_data: Any):
        """Handle incoming text JSON or binary audio."""
        if isinstance(raw_data, bytes):
            # Browser microphone input and the verification client send PCM16 little-endian.
            await self._process_binary_audio(raw_data)
            return

        try:
            msg = json.loads(raw_data)
        except Exception:
            await self.send_personal_message({"type": "error", "message": "Invalid JSON"}, websocket)
            return

        msg_type = msg.get("type", "")

        if msg_type == "start_session":
            session_id = msg.get("session_id", f"session_{int(time.time())}")
            self.pipeline.reset_session(session_id)
            if "settings" in msg:
                self.settings = AccessibilitySettings(**msg["settings"])
            await self.broadcast({
                "type": "session_state",
                "data": {
                    "session_id": session_id,
                    "is_demo": False,
                    "settings": self.settings.model_dump(),
                    "speakers": [s.model_dump() for s in self.pipeline.registry.list_speakers()]
                }
            })

        elif msg_type == "audio_frame":
            # PCM16 base64 encoded
            b64_str = msg.get("pcm16_base64", "")
            if b64_str:
                raw_bytes = base64.b64decode(b64_str)
                await self._process_binary_audio(raw_bytes)

        elif msg_type == "rename_speaker":
            speaker_id = msg.get("speaker_id")
            new_name = msg.get("new_name", "")
            if speaker_id and new_name:
                profile = self.pipeline.registry.rename_speaker(speaker_id, new_name)
                if profile:
                    await self.broadcast({
                        "type": "speaker_list",
                        "data": [s.model_dump() for s in self.pipeline.registry.list_speakers()]
                    })

        elif msg_type == "update_settings":
            if "settings" in msg:
                self.settings = AccessibilitySettings(**msg["settings"])
                await self.broadcast({
                    "type": "settings_updated",
                    "data": self.settings.model_dump()
                })

        elif msg_type == "clear_history":
            self.pipeline.reset_session(self.pipeline.session_id)
            self.storage.clear_session(self.pipeline.session_id)
            await self.broadcast({
                "type": "history_cleared",
                "session_id": self.pipeline.session_id
            })

        elif msg_type == "export_transcript":
            fmt = msg.get("format", "txt")
            content = self.storage.export_transcript(self.pipeline.session_id, fmt)
            await self.send_personal_message({
                "type": "export_ready",
                "data": {"format": fmt, "content": content}
            }, websocket)

        elif msg_type == "start_demo":
            await self.start_demo_playback()

        elif msg_type == "stop_demo":
            await self.stop_demo_playback()

    async def _process_binary_audio(self, raw_bytes: bytes):
        """Process binary audio chunk from microphone."""
        if self._is_demo_running:
            return  # Ignore live audio during demo replay

        # The binary protocol is PCM16 little-endian. Float32 is ambiguous here
        # because its byte length is also divisible by two.
        if len(raw_bytes) % 2 != 0:
            return
        samples = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32) / 32768.0

        # Chunk into 512-sample frames for VAD & pipeline
        for i in range(0, len(samples), 512):
            frame = samples[i:i + 512]
            if len(frame) < 512:
                frame = np.pad(frame, (0, 512 - len(frame)))

            async with self._pipeline_lock:
                res = await asyncio.to_thread(
                    self.pipeline.process_pcm_frame,
                    frame,
                    self.settings,
                )

            # Broadcast VAD state
            vad_state = res.get("vad_state")
            if vad_state:
                await self.broadcast({
                    "type": "vad_state",
                    "data": {
                        "speaking": vad_state["is_speaking"],
                        "energy": vad_state["energy_db"],
                        "probability": vad_state["speech_prob"]
                    }
                })

            prosody_state = res.get("prosody_state")
            if prosody_state is not None:
                await self.broadcast({
                    "type": "prosody_state",
                    "data": prosody_state.model_dump()
                    if hasattr(prosody_state, "model_dump") else prosody_state,
                })

            # Broadcast finalized or interim utterance event(s)
            events: List[UtteranceEvent] = res.get("utterance_events") or []
            single_event: Optional[UtteranceEvent] = res.get("utterance_event")
            if not events and single_event:
                events = [single_event]

            for event in events:
                if event.is_final:
                    # Save to storage (subject to privacy mode rules)
                    self.storage.save_utterance(
                        event,
                        privacy_mode=self.settings.privacy_mode,
                        save_transcript=self.settings.save_transcript
                    )
                    # Broadcast updated speaker list
                    await self.broadcast({
                        "type": "speaker_list",
                        "data": [s.model_dump() for s in self.pipeline.registry.list_speakers()]
                    })

                await self.broadcast({
                    "type": "utterance",
                    "data": event.model_dump()
                })

            # Broadcast debug performance metrics if available
            metrics: Optional[DebugMetrics] = res.get("debug_metrics")
            if metrics:
                await self.broadcast({
                    "type": "debug_metrics",
                    "data": metrics.model_dump()
                })

    async def start_demo_playback(self):
        """Replay realistic multi-speaker scenario through the identical event pipeline."""
        if self._is_demo_running:
            return

        self._is_demo_running = True
        self.pipeline.reset_session("demo_session")
        self.storage.clear_session("demo_session")

        await self.broadcast({
            "type": "demo_started",
            "message": "Demo mode activated. Replaying 3-speaker conversation."
        })

        self._demo_task = asyncio.create_task(self._run_demo_loop())

    async def stop_demo_playback(self):
        """Halt demo playback."""
        if self._demo_task and not self._demo_task.done():
            self._demo_task.cancel()
        self._is_demo_running = False
        await self.broadcast({
            "type": "demo_stopped",
            "message": "Demo mode deactivated."
        })

    async def _run_demo_loop(self):
        """Loop demo events with realistic timing and interim words."""
        sample_path = Path(__file__).resolve().parent.parent.parent.parent / "samples" / "demo_conversation.json"
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                demo_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load demo conversation file: {e}")
            self._is_demo_running = False
            return

        session_clock = 0.0
        try:
            for item in demo_data:
                if not self._is_demo_running:
                    break

                delay_s = item.get("delay_ms", 1500) / 1000.0
                await asyncio.sleep(delay_s)
                session_clock += delay_s

                # Emulate interim typing / speech-in-progress first
                full_text = item["text"]
                words = full_text.split()
                if len(words) > 3:
                    half_text = " ".join(words[: len(words) // 2]) + "..."
                    interim_event = UtteranceEvent(
                        session_id="demo_session",
                        start_time=round(session_clock, 2),
                        end_time=round(session_clock + 0.8, 2),
                        speaker_id=item["speaker_id"],
                        speaker_label=item["speaker_label"],
                        speaker_color_index=item["speaker_color_index"],
                        text=half_text,
                        is_final=False,
                        overlap=False
                    )
                    await self.broadcast({
                        "type": "utterance",
                        "data": interim_event.model_dump()
                    })
                    await asyncio.sleep(0.5)
                    session_clock += 0.5

                # Finalized utterance event
                final_event = UtteranceEvent(
                    session_id="demo_session",
                    start_time=round(session_clock, 2),
                    end_time=round(session_clock + 1.6, 2),
                    speaker_id=item["speaker_id"],
                    speaker_label=item["speaker_label"],
                    speaker_color_index=item["speaker_color_index"],
                    text=full_text,
                    is_final=True,
                    overlap=item["overlap"],
                    overlapping_speakers=item.get("overlapping_speakers", []),
                    prosody=ProsodyFeatures(**item["prosody"]),
                    direction=DirectionInfo(**item["direction"]) if item.get("direction") else None,
                    confidence=ConfidenceInfo(**item.get("confidence", {}))
                )

                # Update speaker registry
                spk = self.pipeline.registry.get_or_create_speaker(item["speaker_id"])
                spk.label = item["speaker_label"]
                spk.color_index = item["speaker_color_index"]
                spk.total_utterances += 1
                spk.last_heard_time = session_clock + 1.6

                # Save to ephemeral memory
                self.storage.save_utterance(
                    final_event,
                    privacy_mode=self.settings.privacy_mode,
                    save_transcript=self.settings.save_transcript
                )

                await self.broadcast({
                    "type": "speaker_list",
                    "data": [s.model_dump() for s in self.pipeline.registry.list_speakers()]
                })

                await self.broadcast({
                    "type": "utterance",
                    "data": final_event.model_dump()
                })

                # Demo timings are illustrative only and must never be presented as measured latency.
                await self.broadcast({
                    "type": "debug_metrics",
                    "data": DebugMetrics(
                        chunk_duration_ms=1600.0,
                        vad_latency_ms=4.2,
                        asr_latency_ms=78.5,
                        diarization_latency_ms=14.3,
                        prosody_latency_ms=6.1,
                        pipeline_processing_ms=103.1,
                        active_speakers_count=3,
                        queue_backlog=0,
                        simulated=True
                    ).model_dump()
                })

            await asyncio.sleep(1.0)
            await self.broadcast({
                "type": "demo_completed",
                "message": "Demo sequence completed."
            })

        except asyncio.CancelledError:
            pass
        finally:
            self._is_demo_running = False
