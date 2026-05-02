"""FastAPI app — LiveRecall control plane.

  - GET  /healthz                       → quick liveness probe
  - POST /token                         → LiveKit access token (phone or worker)
  - POST /snap                          → single-image retrieval (Vision sync, optional question)
  - GET  /scene-context/recent          → dashboard convenience read
  - GET  /trace/:question_id            → full reasoning chain
  - GET  /answers/:question_id          → final answer text
  - WS   /stream                        → change-stream fan-out for dashboard
  - POST /ask                           → text-only pipeline kick

Architecture: the four downstream agents (router/retrievers/reranker/answerer)
are NOT wired through Mongo change streams anymore. /snap, /ask, and the
LiveKit worker call `orchestrator.run_pipeline(question_doc)` directly. The
only change-stream consumer kept here is the dashboard fan-out hub on the
`/stream` websocket. See DECISIONS.md (h).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants
from pydantic import BaseModel

from shared.types import DEFAULT_CAPTURE_MODE, CaptureMode

from .agents.vision import process_frame
from .change_streams import hub, websocket_endpoint
from .config import settings
from .mongo import collection, init_collections
from .orchestrator import schedule_pipeline
from .util import new_id, now_ms

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_collections()
    await hub.start()
    log.info("api ready (direct-call orchestrator; dashboard hub started)")
    try:
        yield
    finally:
        await hub.stop()


app = FastAPI(title="LiveRecall", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas -----------------------------------------------------------------

class TokenReq(BaseModel):
    identity: str
    room: str
    can_publish: bool = True
    can_subscribe: bool = True
    capture_mode: CaptureMode | None = None


class TokenResp(BaseModel):
    token: str
    url: str
    room: str
    capture_mode: CaptureMode


class AskReq(BaseModel):
    text: str
    session_id: str = "demo"


class AskResp(BaseModel):
    question_id: str


class SnapReq(BaseModel):
    """Single-image retrieval. Send a frame and (optionally) a question.

    `image_b64` is the JPEG image as a plain base64 string (no data: prefix).
    If `question` is empty, only `scene_context` is written; the next
    `/ask` call will be grounded against this fresh frame. If `question` is
    present, the full pipeline runs in the background and the client should
    poll `/answers/{question_id}`.
    """

    image_b64: str
    question: str | None = None
    session_id: str = "demo"
    capture_mode: CaptureMode | None = None


class SnapResp(BaseModel):
    scene_context_id: str
    question_id: str | None
    objects: list[str]
    apparatus: list[str]
    text_visible: list[str]
    text_summary: str
    capture_mode: CaptureMode


# --- Endpoints ---------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "ts": now_ms()}


@app.post("/token", response_model=TokenResp)
async def token(req: TokenReq) -> TokenResp:
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        raise HTTPException(500, "LiveKit credentials not configured (.env: LIVEKIT_API_KEY / LIVEKIT_API_SECRET)")
    grants = VideoGrants(
        room=req.room,
        room_join=True,
        can_publish=req.can_publish,
        can_subscribe=req.can_subscribe,
        can_publish_data=True,
    )
    at = (
        AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(req.identity)
        .with_name(req.identity)
        .with_grants(grants)
        .with_ttl(timedelta(hours=2))
    )
    capture_mode: CaptureMode = req.capture_mode or DEFAULT_CAPTURE_MODE
    session_id = req.room.removeprefix("liverecall-") or req.room
    await collection("sessions").update_one(
        {"_id": session_id},
        {
            "$set": {"room": req.room, "capture_mode": capture_mode},
            "$setOnInsert": {"_id": session_id, "started_at": now_ms(), "ended_at": None},
        },
        upsert=True,
    )
    return TokenResp(
        token=at.to_jwt(),
        url=settings.livekit_url,
        room=req.room,
        capture_mode=capture_mode,
    )


@app.post("/snap", response_model=SnapResp)
async def snap(req: SnapReq) -> SnapResp:
    """Single-image retrieval — runs Vision *synchronously* on the supplied
    frame, writes `scene_context` (with apparatus + cache prefetch), and if a
    question was supplied, kicks the full pipeline as a background task.

    Returns immediately with the scene description and (if applicable) the
    question_id for the client to poll `/answers/{qid}`. End-to-end snap →
    answer is roughly Vision (1.5–3 s) + Router/Retrievers/Reranker/Answerer
    (~2–4 s) ≈ 4–7 s.
    """
    if not req.image_b64:
        raise HTTPException(400, "image_b64 is required")
    image = req.image_b64
    if image.startswith("data:"):
        image = image.split(",", 1)[1]

    session_id = req.session_id or "demo"
    t0 = now_ms()
    capture_mode: CaptureMode = req.capture_mode or DEFAULT_CAPTURE_MODE

    await collection("sessions").update_one(
        {"_id": session_id},
        {
            "$set": {"capture_mode": capture_mode},
            "$setOnInsert": {"_id": session_id, "started_at": now_ms(), "ended_at": None},
        },
        upsert=True,
    )

    frame_id = new_id("vf")
    frame_doc = {
        "_id": frame_id,
        "session_id": session_id,
        "timestamp": now_ms(),
        "image_b64": image,
        "width": 0,
        "height": 0,
        "source": "snap",
        "capture_mode": capture_mode,
    }
    await collection("video_frames").insert_one(frame_doc)

    # `process_frame` runs Vision, writes a complete scene_context (objects,
    # apparatus, text_visible, capture_mode), and schedules the local-cache
    # prefetch from both OCR'd text and recognised apparatus shapes.
    sc_id = await process_frame(frame_doc)
    sc_doc = await collection("scene_context").find_one({"_id": sc_id}, {"text_embedding": 0}) or {}

    qid: str | None = None
    if req.question and req.question.strip():
        qid = new_id("q")
        qdoc = {
            "_id": qid,
            "session_id": session_id,
            "transcript_id": "",
            "text": req.question.strip(),
            "asked_at": now_ms(),
        }
        await collection("questions").insert_one(qdoc)
        # Fire-and-forget — client polls /answers/{qid} for the result.
        schedule_pipeline(qdoc)

    log.info(
        "/snap completed in %dms (sc=%s, qid=%s, apparatus=%s, capture_mode=%s)",
        now_ms() - t0,
        sc_id,
        qid,
        sc_doc.get("apparatus"),
        capture_mode,
    )
    return SnapResp(
        scene_context_id=sc_id or "",
        question_id=qid,
        objects=sc_doc.get("objects") or [],
        apparatus=sc_doc.get("apparatus") or [],
        text_visible=sc_doc.get("text_visible") or [],
        text_summary=sc_doc.get("text_summary") or "",
        capture_mode=capture_mode,
    )


@app.get("/scene-context/recent")
async def scene_recent(seconds: int = 30, session_id: str | None = None) -> dict[str, Any]:
    cutoff = now_ms() - seconds * 1000
    q: dict[str, Any] = {"timestamp": {"$gte": cutoff}}
    if session_id:
        q["session_id"] = session_id
    cur = collection("scene_context").find(q, {"text_embedding": 0}).sort("timestamp", -1).limit(20)
    return {"items": [d async for d in cur]}


@app.get("/trace/{question_id}")
async def trace(question_id: str) -> dict[str, Any]:
    plan = await collection("retrieval_plans").find_one({"question_id": question_id})
    results = [d async for d in collection("retrieval_results").find({"question_id": question_id})]
    final = await collection("final_context").find_one({"question_id": question_id})
    answer = await collection("answers").find_one({"question_id": question_id})
    traces = [
        d
        async for d in collection("agent_traces")
        .find({"question_id": question_id})
        .sort("timestamp", 1)
    ]
    return {
        "question_id": question_id,
        "plan": plan,
        "results": results,
        "final_context": final,
        "answer": answer,
        "traces": traces,
    }


@app.get("/answers/{question_id}")
async def answer(question_id: str) -> dict[str, Any]:
    a = await collection("answers").find_one({"question_id": question_id})
    if not a:
        raise HTTPException(404, "answer not yet ready")
    return a


@app.post("/ask", response_model=AskResp)
async def ask(req: AskReq) -> AskResp:
    """Text-only pipeline kick. Drops a question into Mongo + schedules the
    full direct-call pipeline. Client polls `/answers/{question_id}`.
    """
    qid = new_id("q")
    qdoc = {
        "_id": qid,
        "session_id": req.session_id,
        "transcript_id": "",
        "text": req.text,
        "asked_at": now_ms(),
    }
    await collection("questions").insert_one(qdoc)
    schedule_pipeline(qdoc)
    return AskResp(question_id=qid)


@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    await websocket_endpoint(ws)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=False,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
