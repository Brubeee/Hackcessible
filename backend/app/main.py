"""Main FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import config
from app.storage.database import SessionStorage
from app.diarization.registry import SpeakerRegistry
from app.asr.whisper_asr import WhisperASR
from app.diarization.clusterer import AcousticClusterDiarizer
from app.prosody.analyzer import ProsodyAnalyzer
from app.direction.base import NullDirectionProvider
from app.fusion.pipeline import FusionPipeline
from app.api.websocket import ConnectionManager
from app.api.routes import get_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hackcessible")
ALLOWED_BROWSER_ORIGINS = {
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
}

# Global instances
storage = SessionStorage(db_path=config.db_path)
speaker_registry = SpeakerRegistry()
asr_engine = WhisperASR(
    model_size=config.whisper_model_size,
    device=config.whisper_device,
    compute_type=config.whisper_compute_type
)
diarizer = AcousticClusterDiarizer(registry=speaker_registry)
prosody_analyzer = ProsodyAnalyzer(sample_rate=config.sample_rate)
# A mono laptop microphone cannot provide a real direction estimate. Demo events
# carry their own explicitly simulated direction values.
direction_provider = NullDirectionProvider()

pipeline = FusionPipeline(
    asr=asr_engine,
    diarizer=diarizer,
    registry=speaker_registry,
    prosody_analyzer=prosody_analyzer,
    direction_provider=direction_provider,
    sample_rate=config.sample_rate
)

ws_manager = ConnectionManager(pipeline=pipeline, storage=storage)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Hackcessible Conversation Accessibility Service...")
    logger.info(f"Local ASR Ready: {asr_engine.is_ready()}")
    logger.info(f"Default Privacy Mode: {config.default_privacy_mode}")
    yield
    logger.info("Shutting down service...")


app = FastAPI(
    title=config.app_name,
    version=config.version,
    lifespan=lifespan
)

# CORS for the served app and the documented local Vite development server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_BROWSER_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Mount REST routes
app.include_router(get_router(speaker_registry, storage))


@app.websocket("/ws/conversation")
async def websocket_endpoint(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if origin and origin not in ALLOWED_BROWSER_ORIGINS:
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    await ws_manager.connect(websocket)
    try:
        while True:
            # Can receive text JSON or raw binary PCM
            message = await websocket.receive()
            if "text" in message and message["text"]:
                await ws_manager.handle_message(websocket, message["text"])
            elif "bytes" in message and message["bytes"]:
                await ws_manager.handle_message(websocket, message["bytes"])
    except (WebSocketDisconnect, RuntimeError):
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket unhandled error: {e}")
        ws_manager.disconnect(websocket)


# Mount frontend dist if exists
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
