"""REST API routes for session management, speaker renaming, and exports."""
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.models.events import SpeakerInfo
from app.models.settings import AccessibilitySettings
from app.diarization.registry import SpeakerRegistry
from app.storage.database import SessionStorage

router = APIRouter(prefix="/api")


class RenameSpeakerRequest(BaseModel):
    new_name: str


def get_router(registry: SpeakerRegistry, storage: SessionStorage) -> APIRouter:
    @router.get("/health")
    def health():
        return {
            "status": "healthy",
            "service": "Hackcessible Visual Conversation Layer",
            "version": "1.0.0"
        }

    @router.get("/speakers", response_model=List[SpeakerInfo])
    def get_speakers():
        return registry.list_speakers()

    @router.post("/speakers/{speaker_id}/rename")
    def rename_speaker(speaker_id: str, req: RenameSpeakerRequest):
        profile = registry.rename_speaker(speaker_id, req.new_name)
        if not profile:
            raise HTTPException(status_code=404, detail="Speaker not found")
        return profile.to_info()

    @router.get("/transcript/{session_id}")
    def get_transcript(session_id: str, format: str = "json"):
        content = storage.export_transcript(session_id, format)
        return {"session_id": session_id, "format": format, "content": content}

    return router
