"""End-to-End System Verification Script for Hackcessible.

Starts the FastAPI server, tests REST endpoints, establishes a WebSocket connection,
streams the scripted demo, verifies demo event labeling and speaker renaming, and checks privacy rules.
"""
import asyncio
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
import httpx
import uvicorn
import websockets

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.main import app, ws_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e-verifier")


def create_server():
    return uvicorn.Server(uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="warning",
        access_log=False
    ))


async def run_e2e_verification():
    logger.info("==================================================")
    logger.info("STARTING HACKCESSIBLE E2E VERIFICATION SUITE")
    logger.info("==================================================")

    # Keep this check isolated from any transcript database the user may have.
    test_storage_dir = tempfile.TemporaryDirectory(prefix="hackcessible-e2e-")
    db_file = Path(test_storage_dir.name) / "session_storage.db"
    original_db_path = ws_manager.storage.db_path
    ws_manager.storage.db_path = db_file

    # 1. Start server in background task
    server = create_server()
    server_task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        if server_task.done():
            await server_task
        await asyncio.sleep(0.1)
    else:
        raise RuntimeError("E2E server did not start within 10 seconds.")

    results = {
        "rest_health": False,
        "websocket_connected": False,
        "initial_state_received": False,
        "demo_started": False,
        "multi_speaker_received": False,
        "scripted_overlap_event_received": False,
        "prosody_cues_received": False,
        "demo_metrics_marked_simulated": False,
        "speaker_renamed": False,
        "privacy_enforced": False
    }

    try:
        # 2. Test REST /api/health
        logger.info("[1/7] Testing REST /api/health...")
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
            resp = await client.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            results["rest_health"] = True
            logger.info("✓ REST /api/health returned healthy.")

        # 3. Test WebSocket connection
        logger.info("[2/7] Connecting to ws://127.0.0.1:8000/ws/conversation...")
        uri = "ws://127.0.0.1:8000/ws/conversation"
        async with websockets.connect(uri) as ws:
            results["websocket_connected"] = True
            logger.info("✓ WebSocket connected successfully.")

            # 4. Receive session_state
            logger.info("[3/7] Waiting for session_state...")
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            state_data = json.loads(msg)
            assert state_data["type"] == "session_state"
            results["initial_state_received"] = True
            logger.info("✓ Initial session state received.")

            # 5. Start Demo Mode
            logger.info("[4/7] Triggering Demo Mode via WebSocket...")
            await ws.send(json.dumps({"type": "start_demo"}))

            demo_start_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            parsed_start = json.loads(demo_start_msg)
            assert parsed_start["type"] == "demo_started"
            results["demo_started"] = True
            logger.info("✓ Demo mode confirmed started by server.")

            # Collect demo events
            logger.info("[5/7] Streaming scripted conversation events...")
            received_speakers = set()
            start_stream = time.perf_counter()

            while len(received_speakers) < 3 and (time.perf_counter() - start_stream) < 15.0:
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=6.0)
                event = json.loads(raw_msg)
                evt_type = event.get("type")

                if evt_type == "utterance":
                    utt = event["data"]
                    if utt.get("is_final"):
                        received_speakers.add(utt["speaker_id"])
                        logger.info(
                            f"  ↳ Utterance from {utt['speaker_label']}: '{utt['text'][:45]}...'"
                        )
                        if utt.get("overlap"):
                            results["scripted_overlap_event_received"] = True
                            logger.info(f"  ↳ SCRIPTED OVERLAP EVENT: {utt.get('overlapping_speakers')}")
                        prosody = utt.get("prosody", {})
                        if prosody.get("rising_intonation") or prosody.get("emphasis") or prosody.get("speech_rate") != "normal":
                            results["prosody_cues_received"] = True
                            logger.info(f"  ↳ Prosody cues present: rising={prosody.get('rising_intonation')}, rate={prosody.get('speech_rate')}")

                elif evt_type == "debug_metrics":
                    m = event["data"]
                    assert m.get("simulated") is True, "Demo metrics must be marked simulated."
                    results["demo_metrics_marked_simulated"] = True

            assert len(received_speakers) >= 3
            results["multi_speaker_received"] = True
            logger.info(f"✓ Successfully received utterances from {len(received_speakers)} distinct speakers.")

            # 6. Test Speaker Renaming
            logger.info("[6/7] Testing speaker renaming...")
            await ws.send(json.dumps({
                "type": "rename_speaker",
                "speaker_id": "speaker_1",
                "new_name": "Professor Sharma"
            }))

            # Look for updated speaker list
            rename_verified = False
            for _ in range(5):
                r_msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                p = json.loads(r_msg)
                if p.get("type") == "speaker_list":
                    for spk in p["data"]:
                        if spk["id"] == "speaker_1" and "Professor Sharma" in spk["label"]:
                            rename_verified = True
                            break
                if rename_verified:
                    break

            assert rename_verified
            results["speaker_renamed"] = True
            logger.info("✓ Speaker renamed and broadcast across session.")

            # Stop demo
            await ws.send(json.dumps({"type": "stop_demo"}))

        # 7. Test Privacy Mode
        logger.info("[7/7] Testing Privacy Mode disk isolation...")
        # In privacy mode, SQLite file should NOT exist or contain 0 rows
        if not db_file.exists():
            results["privacy_enforced"] = True
            logger.info("✓ Privacy verified: SQLite file was NOT created (memory-only mode).")
        else:
            import sqlite3
            with sqlite3.connect(db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM utterances")
                count = cursor.fetchone()[0]
                assert count == 0
                results["privacy_enforced"] = True
                logger.info("✓ Privacy verified: 0 utterances persisted to disk.")

    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=10.0)
        except asyncio.TimeoutError:
            server.force_exit = True
            await server_task
        ws_manager.storage.db_path = original_db_path
        test_storage_dir.cleanup()

    logger.info("==================================================")
    logger.info("VERIFICATION RESULTS SUMMARY:")
    for k, v in results.items():
        logger.info(f"  - {k}: {'PASSED ✓' if v else 'FAILED ✗'}")
    logger.info("  - Demo latency values are simulated; use verify_websocket_audio.py for measured processing metrics.")
    logger.info("==================================================")

    # Return overall success
    all_passed = all(results.values())
    if not all_passed:
        sys.exit(1)
    logger.info("ALL END-TO-END VERIFICATIONS PASSED!")


if __name__ == "__main__":
    asyncio.run(run_e2e_verification())
