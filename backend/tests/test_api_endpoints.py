"""Unit tests for FastAPI REST API endpoints."""
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from app.main import app, speaker_registry


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Hackcessible" in data["service"]


def test_speakers_and_renaming(client):
    # Register speaker first
    spk = speaker_registry.get_or_create_speaker("speaker_test")
    assert spk.speaker_id == "speaker_test"

    # Fetch speakers list
    res = client.get("/api/speakers")
    assert res.status_code == 200
    speakers = res.json()
    assert any(s["id"] == "speaker_test" for s in speakers)

    # Rename speaker
    rename_res = client.post(
        "/api/speakers/speaker_test/rename",
        json={"new_name": "Professor Sharma"}
    )
    assert rename_res.status_code == 200
    data = rename_res.json()
    assert data["custom_name"] == "Professor Sharma"
    assert "Professor Sharma" in data["label"]


def test_websocket_rejects_untrusted_browser_origin(client):
    with pytest.raises(WebSocketDisconnect) as disconnect:
        with client.websocket_connect(
            "/ws/conversation",
            headers={"origin": "https://untrusted.example"},
        ):
            pass

    assert disconnect.value.code == 1008


def test_websocket_accepts_local_app_origin(client):
    with client.websocket_connect(
        "/ws/conversation",
        headers={"origin": "http://127.0.0.1:8000"},
    ) as websocket:
        assert websocket.receive_json()["type"] == "session_state"
