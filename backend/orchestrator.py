"""Direct-call orchestrator.

Replaces the change-stream agent bus. Every entry point (POST /snap, POST /ask,
LiveKit worker STT-detected question) calls `run_pipeline(question_doc)` which
runs Router → Retrievers → Reranker → Answerer sequentially in-process. Each
stage still writes its document to Mongo (so the dashboard's existing
/trace/{qid} reads keep working), but we no longer rely on Atlas change-stream
fan-out to drive the pipeline.

Why direct calls instead of change streams:
  - Change streams are great for fan-out (dashboard websocket), but for a
    request/response pipeline they add wall-clock latency (each consumer polls
    its own change stream) and silent-failure surface (a crashed loop doesn't
    block the request). With direct calls the failure shows up in the request
    log, latency is the sum of stage costs, and we can run a single uvicorn
    worker per process without spawning N background tasks.
  - LiveKit-driven path: on every STT-final question the worker calls
    `run_pipeline` and then publishes the streaming TTS back to the room. Same
    function, same Mongo trace.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .agents.answerer import answer_text_only
from .agents.reranker import rerank
from .agents.retrievers import run_plan as run_retrieval_plan
from .agents.router import plan as build_plan
from .mongo import collection
from .tracing import trace_event
from .util import now_ms

log = logging.getLogger("orchestrator")


async def run_pipeline(question_doc: dict[str, Any]) -> dict[str, Any]:
    """End-to-end: router → retrievers → reranker → answerer.

    Returns a dict with the artefact ids so callers (e.g. the worker) can
    publish TTS, log latency, etc.

    The function is best-effort. Each stage failure is logged + traced; we
    proceed with whatever we have rather than aborting.
    """
    qid = question_doc["_id"]
    sid = question_doc.get("session_id")
    text = question_doc.get("text") or ""
    pipeline_t0 = now_ms()
    out: dict[str, Any] = {
        "question_id": qid,
        "session_id": sid,
        "stages": {},
    }

    # 1. Router → retrieval_plans
    try:
        t0 = now_ms()
        plan = await build_plan(question_doc)
        out["plan_id"] = plan["_id"]
        out["stages"]["router"] = now_ms() - t0
    except Exception as e:  # noqa: BLE001
        log.exception("orchestrator: router failed for qid=%s: %s", qid, e)
        await trace_event(agent="router", stage="error", question_id=qid, session_id=sid, payload={"error": str(e)})
        return out

    # 2. Retrievers → 3 retrieval_results
    try:
        t0 = now_ms()
        await run_retrieval_plan(plan)
        out["stages"]["retrievers"] = now_ms() - t0
    except Exception as e:  # noqa: BLE001
        log.exception("orchestrator: retrievers failed for qid=%s: %s", qid, e)
        await trace_event(agent="retrievers", stage="error", question_id=qid, session_id=sid, payload={"error": str(e)})

    # 3. Reranker → final_context (may also fire active follow-ups)
    final_ctx: dict[str, Any] | None = None
    try:
        t0 = now_ms()
        final_ctx = await rerank(plan)
        out["final_context_id"] = final_ctx["_id"]
        out["rerank_passes"] = final_ctx.get("rerank_passes", 1)
        out["stages"]["reranker"] = now_ms() - t0
    except Exception as e:  # noqa: BLE001
        log.exception("orchestrator: reranker failed for qid=%s: %s", qid, e)
        await trace_event(agent="reranker", stage="error", question_id=qid, session_id=sid, payload={"error": str(e)})

    # 4. Answerer → answers (text-only here; LiveKit worker wraps with TTS)
    if final_ctx:
        try:
            t0 = now_ms()
            answer_text = await answer_text_only(text, final_ctx)
            out["answer_text"] = answer_text
            out["stages"]["answerer"] = now_ms() - t0
        except Exception as e:  # noqa: BLE001
            log.exception("orchestrator: answerer failed for qid=%s: %s", qid, e)
            await trace_event(agent="answerer", stage="error", question_id=qid, session_id=sid, payload={"error": str(e)})

    out["total_ms"] = now_ms() - pipeline_t0
    log.info(
        "pipeline qid=%s done in %dms (stages=%s)",
        qid, out["total_ms"], out["stages"],
    )
    return out


def schedule_pipeline(question_doc: dict[str, Any]) -> asyncio.Task:
    """Fire-and-forget version. Caller doesn't await; the result is materialised
    in Mongo and read back via /answers/{qid} or change streams.
    """
    return asyncio.create_task(run_pipeline(question_doc))


async def ask_text(text: str, *, session_id: str = "demo") -> str:
    """Drop a question into Mongo, run the full pipeline, return the qid.
    Callers that want the answer should poll /answers/{qid}.
    """
    from .util import new_id  # local import to keep module load order clean

    qid = new_id("q")
    qdoc = {
        "_id": qid,
        "session_id": session_id,
        "transcript_id": "",
        "text": text,
        "asked_at": now_ms(),
    }
    await collection("questions").insert_one(qdoc)
    schedule_pipeline(qdoc)
    return qid
