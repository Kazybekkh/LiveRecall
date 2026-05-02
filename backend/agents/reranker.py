"""Reranker agent — GPT-4o-mini, with optional Active Retrieval pass.

Pass 1 (always):
  Reads scene_context + every retrieval_results bundle, ranks the top results,
  AND may emit up to 2 `active_followups` if it spots an information gap that
  matters clinically (e.g. has a renal-contraindication monograph chunk but no
  recent eGFR in the events list).

Active Retrieval (sometimes):
  If the Reranker requested follow-ups, we execute them in parallel via
  backend/agents/active_tools.py (~30–80 ms each) and run Pass 2 — a tight
  second rerank that folds the new facts in.

Hard cap: one round of follow-ups, max 2 tools. Keeps latency bounded.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from ..mongo import collection
from ..tracing import MongoTraceCallback, trace_event
from ..util import new_id, now_ms
from . import active_tools

log = logging.getLogger("reranker")

EXPECTED_SOURCES = 3
MAX_FOLLOWUPS = 2

SYSTEM = """You rerank retrieval results for an adaptive clinical decision-support
system. The user is a clinician wearing camera glasses or holding a phone;
recent scene context is provided (drug names visible, MRNs visible, vitals on
monitors). Reorder the results to favor anything that connects to what the
clinician just SAW.

You may also request up to 2 short follow-up tool calls when there is a
concrete, clinically meaningful information gap in the candidate results
(e.g. you have a renal-contraindication chunk for metformin but no recent
eGFR in the events). Skip follow-ups whenever the existing candidates
already answer the question well — a clean Pass-1 is the fast path.

Return STRICT JSON only:
{
  "ranked_results": [
    {
      "document_id": "...",
      "snippet": "shortened to <= 220 chars",
      "source": "references|events|notes",
      "boosted_score": 0.0-1.0,
      "boost_reason": "explicit reference to a scene object, drug name, MRN, or lab value — or 'baseline relevance'"
    }
  ],
  "active_followups": [
    {
      "tool": "get_latest_lab" | "get_last_administration" | "get_monograph_section",
      "args": { ... },                  // see tool schema below
      "reason": "why this gap matters for the clinician's question"
    }
  ]
}

Tool schemas (use these exact arg names):
  get_latest_lab:           { "patient_id": "P-204", "lab_name": "eGFR" }
  get_last_administration:  { "patient_id": "P-204", "medication": "metformin" }
  get_monograph_section:    { "medication": "metformin", "section_keyword": "contraindications" }

Rules:
- Return at most 5 ranked_results.
- "active_followups" must be [] if Pass-1 candidates already cover the gap.
- At least one boost_reason MUST cite a concrete scene object, drug name, MRN,
  or visible lab value if any are present in the scene.
- Prefer the most clinically actionable item near the top.
- Do not invent facts; only rank what was given.
"""

REPASS_SYSTEM = """You are doing PASS 2 of reranking. New facts have arrived from
targeted follow-up tool calls. Re-emit the ranked_results JSON ONLY (no
active_followups this turn — Pass-2 has no further follow-ups).

Same shape as before:
{
  "ranked_results": [
    { "document_id", "snippet", "source", "boosted_score", "boost_reason" }
  ]
}

Rules:
- Treat each follow-up result as a high-confidence fact about the patient or
  the medication. If a follow-up directly answers a missing piece (e.g. the
  eGFR value), include it near the top with a boost_reason that names the
  follow-up tool.
- Keep the same 5-item cap.
"""


def _llm(*, max_tokens: int = 700) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=max_tokens,
        api_key=settings.openai_api_key,
        timeout=20,
    )


async def _wait_for_all_results(plan_id: str, timeout_s: float = 4.0) -> list[dict[str, Any]]:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        docs = [d async for d in collection("retrieval_results").find({"plan_id": plan_id})]
        if len(docs) >= EXPECTED_SOURCES:
            return docs
        if asyncio.get_event_loop().time() >= deadline:
            return docs
        await asyncio.sleep(0.05)


def _trim(s: str, n: int = 220) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# Fields we strip from the LLM-facing copy of metadata. image_b64 is the big
# offender (~10–20 KB per chunk × 5 chunks = ~100 KB of base64 inside the
# prompt → token blow-up + OpenAI read timeout). Image data still lives on the
# stored final_context / retrieval_results so the dashboard can render it.
_LLM_STRIP_META_KEYS = {
    "image_b64",
    "image_mime",
    "image_attribution",
    "image_source_url",
    "text_embedding",
}


def _llm_safe_metadata(md: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (md or {}).items() if k not in _LLM_STRIP_META_KEYS}


async def _scene_blob(plan: dict[str, Any]) -> str:
    ids = plan.get("scene_context_ids") or []
    if not ids:
        return "(no recent scene context)"
    docs = [d async for d in collection("scene_context").find({"_id": {"$in": ids}})]
    parts = []
    for s in docs[:3]:
        parts.append(
            f"objects={s.get('objects')} visible_text={s.get('text_visible')} "
            f"summary={s.get('text_summary')}"
        )
    return " ; ".join(parts)


def _parse(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _validate_followups(raw_followups: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_followups, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw_followups:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool")
        if tool not in active_tools.ALLOWED_TOOLS:
            continue
        out.append({
            "tool": tool,
            "args": item.get("args") or {},
            "reason": item.get("reason") or "",
        })
        if len(out) >= MAX_FOLLOWUPS:
            break
    return out


async def _execute_followups(
    followups: list[dict[str, Any]],
    *,
    qid: str,
    session_id: str | None,
) -> list[dict[str, Any]]:
    async def _one(f: dict[str, Any]) -> dict[str, Any]:
        result = await active_tools.execute(f["tool"], f["args"])
        await trace_event(
            agent=f"active:{f['tool']}",
            stage="end",
            question_id=qid,
            session_id=session_id,
            latency_ms=result.get("latency_ms", 0),
            payload={
                "args": f["args"],
                "reason": f["reason"],
                "snippet": result.get("snippet"),
                "found": (result.get("metadata") or {}).get("found"),
            },
        )
        return {
            "tool": f["tool"],
            "args": f["args"],
            "reason": f["reason"],
            "snippet": result.get("snippet", ""),
            "metadata": result.get("metadata") or {},
            "latency_ms": result.get("latency_ms", 0),
        }
    return await asyncio.gather(*[_one(f) for f in followups])


def _ranked_with_metadata(ranked: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in ranked:
        base = by_id.get(r.get("document_id"), {})
        meta = dict(base.get("metadata") or {})
        out.append({**r, "metadata": meta})
    return out[:5]


async def rerank(plan: dict[str, Any]) -> dict[str, Any]:
    qid = plan["question_id"]
    session_id = plan.get("session_id")
    cb = MongoTraceCallback(agent="reranker", question_id=qid, session_id=session_id)

    results = await _wait_for_all_results(plan["_id"])
    # `bundle` is what we send to the LLM — it MUST NOT include image_b64 or
    # other heavy/binary fields, or OpenAI will time out parsing the prompt.
    # `by_id` is what we use after ranking to re-attach the FULL metadata
    # (including image_b64) for storage + dashboard rendering.
    bundle = []
    by_id: dict[str, dict[str, Any]] = {}
    for r in results:
        for item in r.get("results", [])[:5]:
            full_meta = item.get("metadata", {}) or {}
            doc_id = item.get("document_id")
            entry = {
                "document_id": doc_id,
                "source": r.get("source"),
                "snippet": _trim(item.get("snippet", "")),
                "score": item.get("score"),
                "metadata": _llm_safe_metadata(full_meta),
            }
            bundle.append(entry)
            by_id[doc_id] = {**entry, "metadata": full_meta}

    scene = await _scene_blob(plan)

    # ---- Pass 1 -----------------------------------------------------------
    pass1_prompt = (
        f"Question: {plan['question_text']}\n\n"
        f"Recent scene: {scene}\n\n"
        f"Candidate results:\n{json.dumps(bundle, ensure_ascii=False, default=str)}"
    )
    pass1_resp = await _llm().ainvoke(
        [SystemMessage(content=SYSTEM), HumanMessage(content=pass1_prompt)],
        config={"callbacks": [cb]},
    )
    parsed = _parse(pass1_resp.content or "")
    ranked = parsed.get("ranked_results") or []
    followups_req = _validate_followups(parsed.get("active_followups"))

    if not ranked:
        log.warning("reranker pass-1 json missing; using raw order")
        ranked = [
            {
                "document_id": b["document_id"],
                "source": b["source"],
                "snippet": b["snippet"],
                "boosted_score": float(b.get("score") or 0.0),
                "boost_reason": "baseline relevance",
            }
            for b in bundle[:5]
        ]

    rerank_passes = 1
    followup_results: list[dict[str, Any]] = []

    # ---- Active Retrieval (Pass 2) ----------------------------------------
    if followups_req:
        await trace_event(
            agent="reranker",
            stage="active_followups_requested",
            question_id=qid,
            session_id=session_id,
            payload={"followups": followups_req},
        )
        followup_results = await _execute_followups(followups_req, qid=qid, session_id=session_id)

        # Inject as synthetic "active:" candidates so Pass-2 can rank them.
        active_bundle = []
        for fr in followup_results:
            doc_id = f"active:{fr['tool']}:{json.dumps(fr['args'], sort_keys=True)}"
            active_bundle.append({
                "document_id": doc_id,
                "source": "events" if fr["tool"] != "get_monograph_section" else "references",
                "snippet": _trim(fr["snippet"]),
                "score": 0.95,
                "metadata": {**(fr["metadata"] or {}), "from_active_followup": True, "tool": fr["tool"]},
            })
            by_id[doc_id] = active_bundle[-1]

        repass_prompt = (
            f"Question: {plan['question_text']}\n\n"
            f"Recent scene: {scene}\n\n"
            f"Pass-1 candidates:\n{json.dumps(bundle, ensure_ascii=False, default=str)}\n\n"
            f"Follow-up facts from tool calls:\n"
            f"{json.dumps(active_bundle, ensure_ascii=False, default=str)}"
        )
        pass2_resp = await _llm(max_tokens=550).ainvoke(
            [SystemMessage(content=REPASS_SYSTEM), HumanMessage(content=repass_prompt)],
            config={"callbacks": [cb]},
        )
        parsed2 = _parse(pass2_resp.content or "")
        ranked2 = parsed2.get("ranked_results") or []
        if ranked2:
            ranked = ranked2
            rerank_passes = 2

    final = {
        "_id": new_id("fc"),
        "question_id": qid,
        "session_id": session_id,
        "ranked_results": _ranked_with_metadata(ranked, by_id),
        "active_followups": followup_results,
        "rerank_passes": rerank_passes,
        "created_at": now_ms(),
    }
    await collection("final_context").insert_one(final)
    return final
