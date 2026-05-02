"""Vision agent — GPT-4o (vision).

Triggered on every sampled video frame written to `video_frames`. Extracts
structured `scene_context` (objects, visible text, environment, activity), then
embeds a one-line summary for vector search.

Runs OFFLINE w.r.t. the question hot path. Latency target: <1500 ms per frame.

After writing scene_context, also kicks off **local-retrieval prefetches**
(see backend/local_cache.py): every detected patient_id (P-2xx) and medication
name triggers a background pre-load of the data the Router is most likely to
ask for. By the time the question arrives, the Retrievers serve from in-memory
cache instead of round-tripping Mongo.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from shared.types import DEFAULT_CAPTURE_MODE, CaptureMode

from ..config import settings
from ..embeddings import embed
from ..mongo import collection
from ..tracing import MongoTraceCallback, trace_event
from ..util import new_id, now_ms

log = logging.getLogger("vision")


# --- Apparatus catalog vocab (loaded from committed fixtures) --------------
# Vision must emit `apparatus` strings that *exactly* match `name` values in
# the documents collection — otherwise the Router/Retrievers can't filter by
# them. We load the canonical set from the JSONL fixtures at import time so
# the vocab never drifts from the catalog the seeder loads. Falls back to a
# hand-curated list if the fixtures aren't present (e.g. unit-test imports).

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "fixtures"
_FIXTURES = (
    _FIXTURE_DIR / "dailymed_sample.jsonl",
    _FIXTURE_DIR / "wikimedia_equipment_sample.jsonl",
)


def _load_catalog_vocab() -> tuple[list[str], list[str]]:
    """Returns (medication_names, equipment_names), each lowercase + dedup'd."""
    meds: dict[str, None] = {}
    equip: dict[str, None] = {}
    for path in _FIXTURES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = (rec.get("name") or rec.get("medication") or "").strip().lower()
                if not name:
                    continue
                if (rec.get("category") or "medication") == "equipment":
                    equip.setdefault(name, None)
                else:
                    meds.setdefault(name, None)
    if not meds and not equip:
        # hand-curated safety net so the prompt stays useful in test envs
        meds = dict.fromkeys([
            "metformin", "insulin glargine", "lisinopril", "amlodipine",
            "warfarin", "atorvastatin", "albuterol", "omeprazole",
            "levothyroxine", "amoxicillin",
        ])
        equip = dict.fromkeys([
            "infusion pump", "pulse oximeter", "stethoscope", "glucose meter",
            "defibrillator", "mechanical ventilator", "syringe", "insulin pen",
            "patient wristband",
        ])
    return list(meds), list(equip)


_MED_VOCAB, _EQUIP_VOCAB = _load_catalog_vocab()


def _vocab_block() -> str:
    """Render the catalog vocab into a compact prompt section."""
    return (
        "  Medications (when you see the labelled bottle / blister pack / pen):\n"
        f"    {', '.join(_MED_VOCAB)}\n\n"
        "  Equipment / devices:\n"
        f"    {', '.join(_EQUIP_VOCAB)}"
    )

SYSTEM = """You extract structured scene context from a single first-person frame
captured by a clinician's camera (glasses or phone) at the point of care.

Return STRICT JSON with these keys, nothing else:
{
  "objects": ["short noun phrases, max 6 — e.g. pill bottle, wristband, IV pump"],
  "apparatus": ["canonical machinery / apparatus names from the catalog when you can identify the physical object — see list below"],
  "text_visible": ["any legible labels, drug names with dose, MRNs like P-204, lab values, gauge readings"],
  "environment": "one of: hospital_room | clinic | pharmacy | medication_cart | exam_table | nursing_station | outdoors | unknown",
  "activity": "one short verb phrase, e.g. 'reviewing medication' or 'checking wristband'",
  "text_summary": "one short sentence describing the clinical scene for retrieval"
}

`apparatus` should be the **canonical lowercase name** of any recognised
piece of medical machinery, equipment, or medication packaging. Use these
exact strings when applicable — they map 1:1 to entries in the apparatus
catalog so the References retriever can pull (name, context, image) for them:

__APPARATUS_VOCAB__

If unsure of identity, leave `apparatus` empty rather than guessing.

Rules:
- If you see a drug name, include it with its dose verbatim (e.g. "METFORMIN 500 mg") in `text_visible`, AND its canonical name in `apparatus`.
- If you see a wristband or chart with a patient identifier (P-201..P-205 in this demo), include the MRN in `text_visible` and add "patient wristband" to `apparatus`.
- If you see a lab value or vital reading on a monitor, include it (e.g. "eGFR 38", "BP 152/94") and add the device name (e.g. "vital signs monitor") to `apparatus`.
- Be concise. No prose, no markdown, no preamble."""


# Short POV hint appended at call time depending on the capture device the
# clinician is using. Two parallel entry points feed the same prompt:
#   - "glasses": Meta Ray-Ban / first-person, hands-free continuous capture.
#   - "phone":   universal fallback — every clinician already has one.
# Kept tiny so we don't fight prompt caching: the bulk of SYSTEM above is
# constant. See DECISIONS.md (g).
_POV_HINTS: dict[CaptureMode, str] = {
    "glasses": (
        "\n\nCapture POV: scene comes from glasses (first-person, the "
        "clinician's natural view). What their hands are touching is in the "
        "lower half of the frame; treat reaching forearms as theirs and "
        "ignore them as 'objects'."
    ),
    "phone": (
        "\n\nCapture POV: scene comes from a phone held by the clinician or "
        "a colleague at bedside. Framing is over-the-shoulder or arm's-length "
        "and the framed object (pill bottle, wristband, monitor) is the "
        "subject of the question."
    ),
}


def _system_prompt_for(capture_mode: CaptureMode) -> str:
    """Surgical prompt extension — append a POV hint, do not rewrite SYSTEM."""
    base = SYSTEM.replace("__APPARATUS_VOCAB__", _vocab_block())
    return base + _POV_HINTS.get(capture_mode, _POV_HINTS["phone"])


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=400,
        api_key=settings.openai_api_key,
        timeout=8,
    )


async def extract_scene(
    image_b64: str,
    *,
    session_id: str,
    capture_mode: CaptureMode = DEFAULT_CAPTURE_MODE,
) -> dict[str, Any]:
    """One Vision call. Returns parsed JSON dict (best-effort).

    `capture_mode` toggles a tiny POV hint appended to the system prompt so
    GPT-4o knows whether the frame came from glasses (first-person) or a
    phone (over-the-shoulder). Defaults to "phone" — the safer fallback.
    """
    cb = MongoTraceCallback(agent="vision", question_id=None, session_id=session_id)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Extract scene context as JSON."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
    )
    resp = await _llm().ainvoke(
        [SystemMessage(content=_system_prompt_for(capture_mode)), msg],
        config={"callbacks": [cb]},
    )
    raw = (resp.content or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "objects": [],
            "apparatus": [],
            "text_visible": [],
            "environment": "unknown",
            "activity": "",
            "text_summary": raw[:200],
        }
    data.setdefault("objects", [])
    data.setdefault("apparatus", [])
    data.setdefault("text_visible", [])
    data.setdefault("environment", "unknown")
    data.setdefault("activity", "")
    data.setdefault("text_summary", "")
    seen: set[str] = set()
    norm_app: list[str] = []
    for a in data["apparatus"]:
        s = (a or "").strip().lower()
        if s and s not in seen:
            seen.add(s)
            norm_app.append(s)
    data["apparatus"] = norm_app[:8]
    return data


async def process_frame(frame_doc: dict[str, Any]) -> str | None:
    """Run Vision on one `video_frames` doc, write `scene_context`. Returns id."""
    t0 = now_ms()
    session_id = frame_doc["session_id"]
    image_b64 = frame_doc.get("image_b64")
    if not image_b64:
        return None

    # The frame writer (/snap, or any future ingress) may stamp capture_mode
    # directly on the frame doc; otherwise fall back to whatever was persisted
    # on the session, then to the safer phone default.
    capture_mode: CaptureMode = (
        frame_doc.get("capture_mode")
        or await _get_session_capture_mode(session_id)
        or DEFAULT_CAPTURE_MODE
    )

    data = await extract_scene(
        image_b64, session_id=session_id, capture_mode=capture_mode
    )
    summary = data["text_summary"] or " ".join(data["objects"])
    vec = await embed(summary)

    doc = {
        "_id": new_id("sc"),
        "session_id": session_id,
        "timestamp": now_ms(),
        "source_frame_id": frame_doc["_id"],
        "objects": data["objects"][:8],
        "apparatus": data["apparatus"][:8],
        "text_visible": data["text_visible"][:8],
        "environment": data["environment"],
        "activity": data["activity"],
        "text_summary": summary,
        "text_embedding": vec,
        "capture_mode": capture_mode,
    }
    await collection("scene_context").insert_one(doc)

    # Local-retrieval prefetch: warm the cache from BOTH the OCR'd text (drug
    # names, MRNs printed on labels) AND the canonical apparatus list (which
    # covers cases where Vision recognised the device shape but no text was
    # legible — e.g. an unlabelled IV pump).
    schedule_prefetch_from_scene(session_id, doc["text_visible"], doc["apparatus"])

    await trace_event(
        agent="vision",
        stage="end",
        session_id=session_id,
        latency_ms=now_ms() - t0,
        payload={
            "frame_id": frame_doc["_id"],
            "objects": doc["objects"],
            "apparatus": doc["apparatus"],
        },
    )
    return doc["_id"]


# --- Capture-mode lookup ----------------------------------------------------
# Tiny in-process cache so the per-frame Vision call doesn't round-trip Mongo
# just to read `sessions.capture_mode`. Capture mode is written once per
# session (on first /snap) and never changes mid-session, so a never-evicted
# dict is fine for hackathon scope.
_SESSION_CAPTURE_MODE: dict[str, CaptureMode] = {}


async def _get_session_capture_mode(session_id: str) -> CaptureMode | None:
    cached = _SESSION_CAPTURE_MODE.get(session_id)
    if cached is not None:
        return cached
    sess = await collection("sessions").find_one(
        {"_id": session_id}, {"capture_mode": 1}
    )
    mode = (sess or {}).get("capture_mode") if sess else None
    if mode in ("glasses", "phone"):
        _SESSION_CAPTURE_MODE[session_id] = mode  # type: ignore[assignment]
        return mode  # type: ignore[return-value]
    return None


# --- Prefetch helpers -------------------------------------------------------
# These live next to Vision because Vision is the only writer of scene_context
# and therefore the only place where we know "an MRN was just seen." Importing
# from retrievers (rather than the reverse) keeps the dependency direction
# clean: agents/retrievers.py already depends on local_cache.

_PATIENT_ID_RE = re.compile(r"\bP-?\d{3}\b", re.IGNORECASE)
# Names of medications we have monographs for. Kept in sync with the DailyMed
# DRUG_LIST so any OCR'd brand/generic in `text_visible` triggers prefetch
# even when Vision didn't recognise the apparatus by shape.
_KNOWN_MEDS = (
    "metformin", "insulin glargine", "insulin", "lisinopril", "amlodipine",
    "warfarin", "albuterol", "atorvastatin", "omeprazole", "levothyroxine",
    "amoxicillin",
)


def _normalise_patient_id(raw: str) -> str:
    s = raw.upper().replace(" ", "")
    if s.startswith("P") and not s.startswith("P-"):
        s = "P-" + s[1:]
    return s


def schedule_prefetch_from_scene(
    session_id: str,
    visible_tokens: list[str],
    apparatus: list[str] | None = None,
) -> None:
    """Fire-and-forget cache warm-up driven by what's currently in view.

    Two signal paths feed the prefetcher:
      1. `visible_tokens` — OCR'd labels (drug names + MRNs printed on
         wristbands / bottles). Catches branded items via regex/substring.
      2. `apparatus`      — canonical names that Vision recognised by shape
         (e.g. an unlabelled "infusion pump"). Catches devices that have no
         OCR-legible text on them.

    The SessionCache dedups in-flight + already-cached prefetches so we can
    safely call this on every frame.
    """
    # Imported lazily to break the import cycle: retrievers depends on local_cache,
    # vision schedules prefetch via retrievers.
    from .retrievers import prefetch_medication_refs, prefetch_patient_events

    blob = " ".join(visible_tokens or [])
    blob_l = blob.lower()

    for m in _PATIENT_ID_RE.findall(blob):
        pid = _normalise_patient_id(m)
        prefetch_patient_events(session_id, pid)

    for med in _KNOWN_MEDS:
        if med in blob_l:
            prefetch_medication_refs(session_id, med)

    # Apparatus path: any catalog name → prefetch. We treat every recognised
    # apparatus as a candidate catalog row first (cheap if missing); the cache
    # tolerates empty results gracefully.
    for name in apparatus or []:
        n = (name or "").strip().lower()
        if not n:
            continue
        prefetch_medication_refs(session_id, n)
