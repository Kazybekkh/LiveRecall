"""Router agent — GPT-4o-mini.

Reads a fresh question + last 30s of `scene_context`, emits a `retrieval_plan`
with three differentiated queries (references, events, notes). One short JSON
call.

This is the *adaptive retrieval* surface — the system rewrites the query based
on what the clinician just saw (drug name on a bottle, MRN on a wristband, lab
on a monitor).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from ..mongo import collection
from ..tracing import MongoTraceCallback
from ..util import new_id, now_ms

log = logging.getLogger("router")

SYSTEM = """You are the Router for an adaptive clinical retrieval system grounded
in real-time visual memory. The user is a clinician wearing camera glasses or
holding a phone; we just extracted recent scene context (objects + visible text
+ recognised apparatus + environment). Now they asked a question.

The References retriever queries a UNIFIED apparatus catalog where each row
has (name, context, image) and is one of:
  - category="medication"  (FDA SPL drug labels with product photos)
  - category="equipment"   (medical devices with photos)
You can target it by `name` (canonical lowercase, e.g. "metformin",
"infusion pump") and/or `category`. Prefer `name` when scene context gives
you a confident match; fall back to `category` when the question is about a
device class but you don't have the exact model.

Output STRICT JSON, no prose, with this schema:
{
  "scene_summary": "one sentence",
  "queries": [
    {
      "source": "references",
      "filter": {"name"?: "metformin", "category"?: "medication"},
      "vector_query": "phrase to embed for monographs / device safety / protocols",
      "weight": 1.0
    },
    {
      "source": "events",
      "filter": {"patient_id": "P-204"},
      "vector_query": "",
      "weight": 1.0
    },
    {
      "source": "notes",
      "filter": {},
      "vector_query": "phrase to embed for past clinical handoff notes",
      "weight": 0.7
    }
  ]
}

Rules:
- Always emit exactly 3 queries, one per source: references, events, notes.
- If visible text contains a patient identifier like P-201..P-205, set
  filter.patient_id on the events query.
- If `apparatus` from scene context is non-empty, set filter.name on the
  references query to the most-relevant apparatus name (one only). Add
  filter.category ("medication" or "equipment") to match.
- If visible text contains a medication name not yet in `apparatus`, lowercase
  it and put it in filter.name on the references query.
- Vector queries should differ across sources — references = clinical/spec
  terms (dosing, contraindications, device safety alerts), notes =
  experiential phrasing (handoff, prior visit).
- Bias filters toward what the clinician *just saw*. That is the adaptation.
"""


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=350,
        api_key=settings.openai_api_key,
        timeout=15,
    )


async def recent_scene_context(session_id: str, *, seconds: int = 30) -> list[dict[str, Any]]:
    cutoff = now_ms() - seconds * 1000
    cur = collection("scene_context").find(
        {"session_id": session_id, "timestamp": {"$gte": cutoff}}
    ).sort("timestamp", -1).limit(5)
    return [d async for d in cur]


def _scene_blob(scenes: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for s in scenes:
        parts.append(
            f"- t={s.get('timestamp')} env={s.get('environment')} "
            f"objects={s.get('objects')} apparatus={s.get('apparatus') or []} "
            f"visible_text={s.get('text_visible')} summary={s.get('text_summary')}"
        )
    return "\n".join(parts) or "(no recent scene context)"


async def plan(question: dict[str, Any]) -> dict[str, Any]:
    """Build and persist a retrieval_plan from a question doc."""
    qid = question["_id"]
    session_id = question["session_id"]
    scenes = await recent_scene_context(session_id, seconds=30)
    cb = MongoTraceCallback(agent="router", question_id=qid, session_id=session_id)

    prompt = (
        f"Question: {question['text']}\n\n"
        f"Recent scene context (newest first):\n{_scene_blob(scenes)}"
    )
    resp = await _llm().ainvoke(
        [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)],
        config={"callbacks": [cb]},
    )
    raw = (resp.content or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("router json parse failed, falling back. raw=%s", raw[:200])
        parsed = {"queries": _fallback_queries(question["text"])}

    queries = parsed.get("queries") or _fallback_queries(question["text"])
    plan_doc = {
        "_id": new_id("rp"),
        "question_id": qid,
        "session_id": session_id,
        "question_text": question["text"],
        "scene_context_ids": [s["_id"] for s in scenes],
        "queries": queries[:3],
        "created_at": now_ms(),
    }
    await collection("retrieval_plans").insert_one(plan_doc)
    return plan_doc


def _fallback_queries(question_text: str) -> list[dict[str, Any]]:
    return [
        {"source": "references", "filter": {}, "vector_query": question_text, "weight": 1.0},
        {"source": "events", "filter": {}, "vector_query": "", "weight": 1.0},
        {"source": "notes", "filter": {}, "vector_query": question_text, "weight": 0.7},
    ]
