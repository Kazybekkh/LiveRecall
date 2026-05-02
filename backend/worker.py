"""LiveKit Agents worker — bridges the room to the direct-call orchestrator.

Per session/room:
  - Subscribes to remote audio track  → ElevenLabs Scribe v2 Realtime STT.
  - On STT-final question              → calls orchestrator.run_pipeline AND
                                          streams the answerer tokens out as
                                          ElevenLabs Flash v2.5 TTS into the
                                          published LiveKit audio track.
  - Subscribes to remote video track  → 1 fps frame sampler. Every sampled
                                          frame is run through Vision directly
                                          (process_frame), which also schedules
                                          the local-cache prefetch.

The Mongo change-stream agent bus is gone (see backend/orchestrator.py +
DECISIONS.md (h)). The worker is now an in-process pipeline driver.

Run as: `python -m backend.worker dev`
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from collections.abc import AsyncIterator
from typing import Any

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
)
from PIL import Image

from .agents.answerer import stream_tokens
from .agents.vision import process_frame
from .config import settings
from .mongo import collection, init_collections
from .orchestrator import run_pipeline
from .stt import ScribeSession
from .tracing import trace_event
from .tts import publish_to_room
from .util import new_id, now_ms

log = logging.getLogger("worker")

# session_id -> {"room": rtc.Room, "audio_source": rtc.AudioSource}
ROOM_REGISTRY: dict[str, dict[str, Any]] = {}


def get_audio_source(session_id: str) -> rtc.AudioSource | None:
    entry = ROOM_REGISTRY.get(session_id)
    return entry.get("audio_source") if entry else None


async def entrypoint(ctx: JobContext) -> None:
    await init_collections()
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_AND_VIDEO)

    room = ctx.room
    session_id = _session_id_from_room(room.name)
    log.info("worker joined room=%s session=%s", room.name, session_id)
    # /token writes capture_mode + started_at — don't clobber them.
    await collection("sessions").update_one(
        {"_id": session_id},
        {
            "$set": {"room": room.name, "ended_at": None},
            "$setOnInsert": {"_id": session_id, "started_at": now_ms()},
        },
        upsert=True,
    )

    # --- Outbound audio track ------------------------------------------------
    audio_source = rtc.AudioSource(sample_rate=16_000, num_channels=1)
    out_track = rtc.LocalAudioTrack.create_audio_track("liverecall-answer", audio_source)
    await room.local_participant.publish_track(
        out_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    ROOM_REGISTRY[session_id] = {"room": room, "audio_source": audio_source}

    # --- Subscribe handlers --------------------------------------------------
    @room.on("track_subscribed")
    def _on_track(track: rtc.Track, _pub, _participant) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(_consume_audio(track, session_id))
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(_consume_video(track, session_id))

    # If tracks were already published before we attached the handler.
    for participant in room.remote_participants.values():
        for pub in participant.tracks.values():
            if pub.track:
                _on_track(pub.track, pub, participant)

    log.info("worker ready (session=%s); waiting on tracks", session_id)
    await asyncio.Future()  # keep the worker running until the room ends.


# --- Audio (STT → orchestrator → TTS back into the room) --------------------

async def _consume_audio(track: rtc.Track, session_id: str) -> None:
    log.info("audio track subscribed (session=%s)", session_id)

    async def _on_question(qdoc: dict[str, Any]) -> None:
        """Handle every STT-detected question by running the full pipeline AND
        streaming the spoken answer back into the LiveKit audio track."""
        log.info("STT question detected qid=%s text=%s", qdoc["_id"], qdoc.get("text", "")[:60])
        try:
            await _answer_into_room(qdoc, session_id)
        except Exception as e:  # noqa: BLE001
            log.exception("worker failed to answer qid=%s: %s", qdoc["_id"], e)

    stt = ScribeSession(session_id=session_id, on_question=_on_question)
    await stt.start()
    try:
        astream = rtc.AudioStream(track)
        async for ev in astream:
            frame: rtc.AudioFrame = ev.frame
            if frame.sample_rate != stt.sample_rate or frame.num_channels != 1:
                frame = frame.remix_and_resample(stt.sample_rate, 1)
            await stt.send(bytes(frame.data))
    finally:
        await stt.close()


async def _answer_into_room(qdoc: dict[str, Any], session_id: str) -> None:
    """Run the orchestrator up through Reranker, then stream Answerer tokens
    straight into ElevenLabs TTS → the LiveKit AudioSource. We bypass
    `orchestrator.run_pipeline`'s text-only Answerer call because we want the
    stream to start firing audio as soon as the first token lands.
    """
    audio_source = get_audio_source(session_id)
    if audio_source is None:
        log.warning("no audio_source for session=%s; falling back to text-only", session_id)
        await run_pipeline(qdoc)
        return

    # Stages 1–3: router → retrievers → reranker (writes final_context).
    from .agents.reranker import rerank
    from .agents.retrievers import run_plan as run_retrieval_plan
    from .agents.router import plan as build_plan

    t_pipeline = now_ms()
    plan = await build_plan(qdoc)
    await run_retrieval_plan(plan)
    final_ctx = await rerank(plan)

    # Stage 4: answerer streams tokens; we pump them into ElevenLabs WS → PCM
    # → LiveKit audio frames.
    text = qdoc.get("text", "")
    tokens = stream_tokens(text, final_ctx)
    bytes_published = await publish_to_room(audio_source, tokens)

    await trace_event(
        agent="worker.answer_tts",
        stage="end",
        question_id=qdoc["_id"],
        session_id=session_id,
        latency_ms=now_ms() - t_pipeline,
        payload={"audio_bytes": bytes_published, "rerank_passes": final_ctx.get("rerank_passes")},
    )


# --- Video (frame sampler → Vision direct) ----------------------------------

async def _consume_video(track: rtc.Track, session_id: str) -> None:
    log.info("video track subscribed (session=%s)", session_id)
    interval_s = 1.0 / max(settings.frame_sample_hz, 0.1)
    last_t = 0.0
    vstream = rtc.VideoStream(track)
    async for ev in vstream:
        now = asyncio.get_event_loop().time()
        if now - last_t < interval_s:
            continue
        last_t = now
        try:
            await _ingest_frame(ev.frame, session_id)
        except Exception as e:  # noqa: BLE001
            log.exception("frame ingest failed: %s", e)


async def _ingest_frame(frame: rtc.VideoFrame, session_id: str) -> None:
    rgb = frame.convert(rtc.VideoBufferType.RGB24)
    img = Image.frombytes("RGB", (rgb.width, rgb.height), bytes(rgb.data))
    img.thumbnail((640, 640))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    doc = {
        "_id": new_id("vf"),
        "session_id": session_id,
        "timestamp": now_ms(),
        "image_b64": b64,
        "width": img.width,
        "height": img.height,
        "source": "livekit",
    }
    await collection("video_frames").insert_one(doc)
    await trace_event(
        agent="frame_sampler",
        stage="end",
        session_id=session_id,
        payload={"frame_id": doc["_id"], "size": len(b64)},
    )
    # Direct invocation: scene_context insert + apparatus + cache prefetch all
    # happen inside process_frame. Fire-and-forget so the next sampled frame
    # isn't blocked by the LLM call.
    asyncio.create_task(_safe_process_frame(doc))


async def _safe_process_frame(doc: dict[str, Any]) -> None:
    try:
        await process_frame(doc)
    except Exception as e:  # noqa: BLE001
        log.exception("vision process_frame failed: %s", e)


def _session_id_from_room(room_name: str) -> str:
    return room_name.removeprefix("liverecall-") or room_name


def main() -> None:
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    main()
