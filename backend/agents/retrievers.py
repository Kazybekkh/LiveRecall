"""Retrievers — pure Mongo aggregation pipelines, NO LLM.

Three sources run in parallel via asyncio.gather:
  1. references — Atlas $vectorSearch on `documents`. Now a unified MULTIMODAL
                  apparatus catalog: medication monograph chunks (DailyMed, with
                  FDA product photos) AND equipment entries (Wikimedia, with
                  device thumbnails). Each row carries (name, context, image),
                  and metadata propagates `image_b64` + `category` so the
                  dashboard can render the photo next to the snippet.
  2. events     — Time Series filter + sort on `clinical_events` (per-patient).
  3. notes      — Atlas $vectorSearch + recency boost on `transcripts`.

Every plan triggers exactly 3 result writes. Latency target: <400 ms total.

LOCAL RETRIEVAL — Vision schedules background prefetches into `local_cache`
keyed on patient_id, medication, AND recognised apparatus name. Retrievers
consult that cache first; hits return in ~5 ms. Each result item carries
`metadata.from_cache` so the dashboard can show the optimisation working.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..config import settings
from ..embeddings import embed
from ..local_cache import (
    cache,
    medication_refs_key,
    patient_events_key,
)
from ..mongo import collection
from ..tracing import trace_event
from ..util import new_id, now_ms
from shared.types import VEC_IDX_DOCS, VEC_IDX_TRANSCRIPTS

log = logging.getLogger("retrievers")


# --- References (vector search on drug monographs / protocols) ---------------

def _doc_metadata(d: dict[str, Any]) -> dict[str, Any]:
    """Build the standard metadata blob for a `documents` row, including the
    apparatus name + category and any product/device image we have on file.
    Image bytes (~5–20 KB b64) only ride on metadata — never inside `snippet`
    or any field that a downstream LLM call might re-tokenize.
    """
    md: dict[str, Any] = {
        "source_doc": d.get("source_doc"),
        "name": d.get("name") or d.get("medication"),
        "category": d.get("category") or "medication",
        "section": d.get("section"),
        "medication": d.get("medication"),
    }
    if d.get("image_b64"):
        md["image_b64"] = d["image_b64"]
        md["image_mime"] = d.get("image_mime") or "image/jpeg"
        md["image_attribution"] = d.get("image_attribution")
        md["image_source_url"] = d.get("image_source_url")
    return md


async def _query_references_mongo(query: dict[str, Any]) -> list[dict[str, Any]]:
    vec = await embed(query.get("vector_query") or "")
    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": VEC_IDX_DOCS,
                "queryVector": vec,
                "path": "text_embedding",
                "numCandidates": settings.num_candidates,
                "limit": settings.retrieval_limit,
            }
        },
        {
            "$project": {
                "snippet": "$text",
                "score": {"$meta": "vectorSearchScore"},
                "name": 1,
                "category": 1,
                "source_doc": 1,
                "section": 1,
                "medication": 1,
                "image_b64": 1,
                "image_mime": 1,
                "image_attribution": 1,
                "image_source_url": 1,
            }
        },
    ]
    f = query.get("filter") or {}
    post_match: dict[str, Any] = {}
    if f.get("medication"):
        post_match["medication"] = f["medication"]
    if f.get("name"):
        post_match["name"] = f["name"].lower() if isinstance(f["name"], str) else f["name"]
    if f.get("category"):
        post_match["category"] = f["category"]
    if post_match:
        pipeline.append({"$match": post_match})

    items: list[dict[str, Any]] = []
    async for d in collection("documents").aggregate(pipeline):
        items.append({
            "document_id": str(d.get("_id")),
            "score": float(d.get("score") or 0.0),
            "snippet": d.get("snippet") or "",
            "metadata": _doc_metadata(d),
        })
    if not items:
        items = await _references_fallback(query)
    return items


async def _query_apparatus_refs_mongo(name: str, limit: int = 5) -> list[dict[str, Any]]:
    """Filter-only loader used by the prefetch path (no question yet).

    Matches on `name` (covers both medication and equipment rows) with a
    fallback to legacy `medication` for older fixtures.
    """
    n = name.lower()
    cur = collection("documents").find(
        {"$or": [{"name": n}, {"medication": n}]}
    ).limit(limit)
    items: list[dict[str, Any]] = []
    async for d in cur:
        items.append({
            "document_id": str(d.get("_id")),
            "score": 0.7,   # neutral-but-relevant; Reranker will re-score
            "snippet": d.get("text", ""),
            "metadata": _doc_metadata(d),
        })
    return items


# Backwards-compat alias — older callers (vision.py prefetch path) imported
# `_query_medication_refs_mongo`. We keep the name pointing at the new
# apparatus-aware loader so nothing breaks if a stale checkout still calls it.
_query_medication_refs_mongo = _query_apparatus_refs_mongo


async def _references_fallback(query: dict[str, Any]) -> list[dict[str, Any]]:
    """When Vector Search isn't ready (cold cluster), keyword-match instead."""
    q = (query.get("vector_query") or "").strip()
    f = query.get("filter") or {}
    match: dict[str, Any] = {}
    if f.get("medication"):
        match["medication"] = f["medication"]
    if f.get("name"):
        match["name"] = f["name"].lower() if isinstance(f["name"], str) else f["name"]
    if f.get("category"):
        match["category"] = f["category"]
    if q:
        match["$text"] = {"$search": q}
    cur = collection("documents").find(match).limit(settings.retrieval_limit)
    items: list[dict[str, Any]] = []
    async for d in cur:
        items.append({
            "document_id": str(d.get("_id")),
            "score": 0.5,
            "snippet": d.get("text", ""),
            "metadata": _doc_metadata(d),
        })
    return items


async def retrieve_references(query: dict[str, Any], *, session_id: str | None = None) -> tuple[list[dict[str, Any]], bool]:
    """Cache-first multimodal reference retrieval. Returns (items, from_cache).

    Cache key is the apparatus name (medication OR equipment), not just the
    medication. So the same cache machinery serves "metformin" lookups AND
    "infusion pump" lookups.
    """
    f = query.get("filter") or {}
    name = f.get("name") or f.get("medication")
    if name and session_id:
        cached = cache.get(session_id, medication_refs_key(name))
        if cached:
            return _stamp_cache(cached, True), True
    items = await _query_references_mongo(query)
    if name and session_id and items:
        cache.put(session_id, medication_refs_key(name), items)
    return _stamp_cache(items, False), False


# --- Events (Time Series, per-patient) ---------------------------------------

async def _query_patient_events_mongo(patient_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
    pipeline = [
        {"$match": {"patient_id": patient_id}},
        {"$sort": {"timestamp": -1}},
        {"$limit": limit},
    ]
    out: list[dict[str, Any]] = []
    async for d in collection("clinical_events").aggregate(pipeline):
        raw_ts = d.get("timestamp")
        if hasattr(raw_ts, "timestamp"):       # datetime → ms since epoch
            ts = int(raw_ts.timestamp() * 1000)
        else:
            try:
                ts = int(raw_ts or 0)
            except (TypeError, ValueError):
                ts = 0
        med = d.get("medication")
        snippet_bits = [
            f"[{d.get('event_type')} • {d.get('severity')}]",
            d.get("notes") or "",
        ]
        if med:
            snippet_bits.append(f"(med: {med} {d.get('dose') or ''})".strip())
        snippet_bits.append(f"(patient {d.get('patient_id')})")
        out.append({
            "document_id": f"evt:{d.get('patient_id')}:{ts}",
            "score": 1.0,
            "snippet": " ".join(s for s in snippet_bits if s),
            "metadata": {
                "patient_id": d.get("patient_id"),
                "event_type": d.get("event_type"),
                "severity": d.get("severity"),
                "medication": med,
                "lab_name": d.get("lab_name"),
                "lab_value": d.get("lab_value"),
                "lab_unit": d.get("lab_unit"),
                "timestamp": ts,
            },
        })
    return out


async def retrieve_events(query: dict[str, Any], *, session_id: str | None = None) -> tuple[list[dict[str, Any]], bool]:
    f = query.get("filter") or {}
    pid = f.get("patient_id")
    if pid and session_id:
        cached = cache.get(session_id, patient_events_key(pid))
        if cached:
            return _stamp_cache(cached, True), True
    if not pid:
        return [], False
    items = await _query_patient_events_mongo(pid, limit=settings.retrieval_limit)
    if session_id and items:
        cache.put(session_id, patient_events_key(pid), items)
    return _stamp_cache(items, False), False


# --- Notes (vector + recency boost on past transcripts) ----------------------

async def _query_notes_mongo(query: dict[str, Any]) -> list[dict[str, Any]]:
    vec = await embed(query.get("vector_query") or "")
    now = now_ms()
    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": VEC_IDX_TRANSCRIPTS,
                "queryVector": vec,
                "path": "text_embedding",
                "numCandidates": settings.num_candidates,
                "limit": 10,
            }
        },
        {
            "$addFields": {
                "vec_score": {"$meta": "vectorSearchScore"},
                "recency_boost": {
                    "$divide": [
                        1,
                        {
                            "$add": [
                                1,
                                {
                                    "$divide": [
                                        {"$subtract": [now, "$timestamp"]},
                                        86_400_000,
                                    ]
                                },
                            ]
                        },
                    ]
                },
            }
        },
        {
            "$addFields": {
                "boosted": {"$multiply": ["$vec_score", {"$add": [1, "$recency_boost"]}]}
            }
        },
        {"$sort": {"boosted": -1}},
        {"$limit": settings.retrieval_limit},
        {
            "$project": {
                "snippet": "$text",
                "score": "$boosted",
                "metadata": {
                    "timestamp": "$timestamp",
                    "session_id": "$session_id",
                    "kind": "clinical_note",
                },
            }
        },
    ]
    items: list[dict[str, Any]] = []
    async for d in collection("transcripts").aggregate(pipeline):
        items.append({
            "document_id": str(d.get("_id")),
            "score": float(d.get("score") or 0.0),
            "snippet": d.get("snippet") or "",
            "metadata": d.get("metadata") or {},
        })
    return items


async def retrieve_notes(query: dict[str, Any], *, session_id: str | None = None) -> tuple[list[dict[str, Any]], bool]:
    # Notes aren't pre-fetchable by patient/medication alone — they vector-search
    # on the question. Always go to Mongo. We still report from_cache=False so
    # the dashboard shows the contrast.
    items = await _query_notes_mongo(query)
    return _stamp_cache(items, False), False


# --- Cache-warm prefetchers (called from Vision) ----------------------------

def prefetch_patient_events(session_id: str, patient_id: str) -> None:
    async def loader() -> list[dict[str, Any]]:
        return await _query_patient_events_mongo(patient_id, limit=settings.retrieval_limit)
    asyncio.create_task(cache.prefetch(session_id, patient_events_key(patient_id), loader))


def prefetch_medication_refs(session_id: str, name: str) -> None:
    """Warm the cache for any apparatus name (medication or equipment).
    Function name is kept for backwards compatibility; semantically it's now
    `prefetch_apparatus_refs`.
    """
    async def loader() -> list[dict[str, Any]]:
        return await _query_apparatus_refs_mongo(name, limit=settings.retrieval_limit)
    asyncio.create_task(cache.prefetch(session_id, medication_refs_key(name), loader))


# --- Helpers ----------------------------------------------------------------

def _stamp_cache(items: list[dict[str, Any]], hit: bool) -> list[dict[str, Any]]:
    out = []
    for it in items:
        meta = dict(it.get("metadata") or {})
        meta["from_cache"] = hit
        out.append({**it, "metadata": meta})
    return out


# --- Fan-out -----------------------------------------------------------------

DISPATCH = {
    "references": retrieve_references,
    "events": retrieve_events,
    "notes": retrieve_notes,
}


async def _run_one(plan_id: str, qid: str, session_id: str | None, query: dict[str, Any]) -> None:
    src = query.get("source", "references")
    fn = DISPATCH.get(src)
    if not fn:
        return
    t0 = now_ms()
    try:
        results, from_cache = await fn(query, session_id=session_id)
    except Exception as e:  # noqa: BLE001
        log.exception("retriever %s failed: %s", src, e)
        results, from_cache = [], False
    latency = now_ms() - t0
    doc = {
        "_id": new_id("rr"),
        "plan_id": plan_id,
        "question_id": qid,
        "source": src,
        "results": results,
        "latency_ms": latency,
        "from_cache": from_cache,
        "created_at": now_ms(),
    }
    await collection("retrieval_results").insert_one(doc)
    await trace_event(
        agent=f"retriever:{src}",
        stage="end",
        question_id=qid,
        session_id=session_id,
        latency_ms=latency,
        payload={"hits": len(results), "from_cache": from_cache},
    )


async def run_plan(plan_doc: dict[str, Any]) -> None:
    plan_id = plan_doc["_id"]
    qid = plan_doc["question_id"]
    sid = plan_doc.get("session_id")
    queries = plan_doc.get("queries") or []
    await asyncio.gather(*[_run_one(plan_id, qid, sid, q) for q in queries])
