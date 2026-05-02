"""Seed Mongo for the bedside-medication-safety demo.

Loads from real public datasets when their fixtures are present, otherwise
falls back to a small inline mock so the demo always works offline / on a
fresh checkout that hasn't run the ingest scripts.

Datasets (see data/README.md + DECISIONS.md (e), (f)):
  - DailyMed (FDA SPL labels + product images)  → documents (medication rows
    with text monograph chunks AND inlined product photos for visual match)
  - Wikimedia Commons + Wikipedia (CC-BY-SA)    → documents (equipment rows
    with article extract + downsized device photo, category="equipment")
  - Synthea (synthetic FHIR patients)           → patients + clinical_events
  - MTSamples (CC0 medical notes)               → transcripts (past notes)

The unified `documents` collection now holds the multimodal apparatus
catalog — every row carries (name, context, image) so the References
retriever can serve real product/device imagery alongside the text snippet.

The demo headline patient `P-204` is constructed by:
  1. Picking the most demo-relevant Synthea patient (T2DM + CKD if available)
  2. Re-labelling them as `P-204` with mock name "Sarah Chen"
  3. Injecting two hand-shaped headline events on top of their real events:
       - metformin 500 mg PO administered ~47 hours ago
       - eGFR = 38 mL/min/1.73m^2 measured ~18 hours ago
This guarantees the demo beats land deterministically while the surrounding
context is real-shaped Synthea data.

All data is mock or synthetic. No real PHI.

Run:
    python -m scripts.seed_mongo
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.embeddings import embed_batch
from backend.mongo import collection, init_collections
from backend.util import new_id, now_ms

log = logging.getLogger("seed")
random.seed(42)

FIXTURES = Path("data/fixtures")
DAILYMED_FIXTURE = FIXTURES / "dailymed_sample.jsonl"
WIKIMEDIA_FIXTURE = FIXTURES / "wikimedia_equipment_sample.jsonl"
SYNTHEA_FIXTURE = FIXTURES / "synthea_sample.jsonl"
MTSAMPLES_FIXTURE = FIXTURES / "mtsamples_sample.jsonl"

EMBED_BATCH_SIZE = 50
DEMO_PATIENT_ID = "P-204"
DEMO_PATIENT_NAME = "Sarah Chen"


# --- Inline mock fallbacks (used when a fixture is missing) ----------------

MOCK_PATIENTS = [
    {
        "patient_id": "P-201",
        "name": "Mock Patient 201",
        "age": 58,
        "weight_kg": 82.0,
        "allergies": ["penicillin"],
        "active_conditions": ["type 2 diabetes", "hypertension"],
    },
    {
        "patient_id": "P-202",
        "name": "Mock Patient 202",
        "age": 71,
        "weight_kg": 64.5,
        "allergies": [],
        "active_conditions": ["atrial fibrillation"],
    },
    {
        "patient_id": "P-203",
        "name": "Mock Patient 203",
        "age": 44,
        "weight_kg": 91.2,
        "allergies": ["sulfa"],
        "active_conditions": ["asthma"],
    },
    {
        "patient_id": "P-205",
        "name": "Mock Patient 205",
        "age": 35,
        "weight_kg": 70.0,
        "allergies": [],
        "active_conditions": [],
    },
]

MOCK_REFERENCES = [
    {
        "name": "metformin",
        "category": "medication",
        "medication": "metformin",
        "source_doc": "Metformin Monograph (mock fallback)",
        "section": "Renal Contraindications",
        "text": (
            "Metformin is contraindicated when eGFR < 30 mL/min/1.73m^2 due to risk "
            "of lactic acidosis. eGFR 30-44 is a CAUTION zone — do NOT initiate; if "
            "already on therapy, consider dose reduction and reassess."
        ),
    },
    {
        "name": "general_safety",
        "category": "other",
        "medication": "general_safety",
        "source_doc": "Five Rights of Medication (mock fallback)",
        "section": "The Five Rights",
        "text": (
            "Right patient, right drug, right dose, right route, right time. "
            "Verify the patient identifier (wristband MRN) at every administration."
        ),
    },
]

MOCK_EQUIPMENT = [
    {
        "name": "infusion pump",
        "category": "equipment",
        "source_doc": "Infusion Pump (mock fallback)",
        "text": (
            "An infusion pump infuses fluids, medication or nutrients into the "
            "circulatory system. Verify drug, dose, rate, and patient before start."
        ),
    },
    {
        "name": "patient wristband",
        "category": "equipment",
        "source_doc": "Patient Wristband (mock fallback)",
        "text": (
            "Bedside identification band printed with patient MRN, name, and DOB. "
            "Mandatory check at every medication or procedure event."
        ),
    },
]

MOCK_NOTES = [
    "Handoff for P-204: known CKD stage 3b, baseline eGFR around 40. On metformin 500 mg BID with caution. Recheck renal function next week.",
    "P-204 family meeting today: discussed renal trajectory and medication risks. Patient understands metformin caution.",
]


# --- Fixture loaders --------------------------------------------------------

def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("skipping bad line in %s: %s", p, e)
    return out


def _load_dailymed() -> list[dict]:
    rows = _load_jsonl(DAILYMED_FIXTURE)
    if rows:
        with_img = sum(1 for r in rows if r.get("image_b64"))
        log.info("loaded %d DailyMed chunks (%d with product images)", len(rows), with_img)
        return rows
    log.warning("no %s — using mock medication references", DAILYMED_FIXTURE)
    return []


def _load_wikimedia_equipment() -> list[dict]:
    rows = _load_jsonl(WIKIMEDIA_FIXTURE)
    if rows:
        with_img = sum(1 for r in rows if r.get("image_b64"))
        log.info("loaded %d Wikimedia equipment entries (%d with images)", len(rows), with_img)
        return rows
    log.warning("no %s — using mock equipment fallback", WIKIMEDIA_FIXTURE)
    return []


def _load_synthea() -> tuple[list[dict], list[dict]]:
    rows = _load_jsonl(SYNTHEA_FIXTURE)
    if not rows:
        log.warning("no %s — using mock patients only", SYNTHEA_FIXTURE)
        return [], []
    patients = [r for r in rows if r.get("kind") == "patient"]
    events = [r for r in rows if r.get("kind") == "event"]
    log.info("loaded %d Synthea patients + %d events", len(patients), len(events))
    return patients, events


def _load_mtsamples() -> list[dict]:
    rows = _load_jsonl(MTSAMPLES_FIXTURE)
    if rows:
        log.info("loaded %d MTSamples notes", len(rows))
        return rows
    log.warning("no %s — using mock notes", MTSAMPLES_FIXTURE)
    return []


# --- Demo-patient construction ---------------------------------------------

def _pick_demo_patient(synthea_patients: list[dict]) -> dict | None:
    """Pick the Synthea patient that best matches T2DM + CKD; fall back to
    any with diabetes; fall back to first overall."""
    def score(p: dict) -> int:
        conds = " ".join(p.get("active_conditions") or []).lower()
        s = 0
        if "diabet" in conds:
            s += 2
        if "kidney" in conds or "renal" in conds:
            s += 3
        return -s    # negative so sort asc gives best first
    if not synthea_patients:
        return None
    return sorted(synthea_patients, key=score)[0]


# --- Wipe + seed ------------------------------------------------------------

async def _wipe() -> None:
    for c in (
        "patients",
        "documents",
        "transcripts",
        "clinical_events",
        "scene_context",
        "video_frames",
        "questions",
        "retrieval_plans",
        "retrieval_results",
        "final_context",
        "answers",
        "agent_traces",
        "sessions",
    ):
        await collection(c).delete_many({})
    log.info("wiped existing demo data")


async def _seed_patients(synthea_patients: list[dict]) -> str | None:
    """Returns the synthea_id of the patient we re-labelled as P-204, if any."""
    docs: list[dict] = []
    demo_synthea_id: str | None = None

    if synthea_patients:
        demo = _pick_demo_patient(synthea_patients)
        demo_synthea_id = demo.get("synthea_id") if demo else None

        # Demo patient first, with the canonical P-204 ID + Sarah Chen name.
        if demo:
            docs.append({
                "_id": new_id("p"),
                "patient_id": DEMO_PATIENT_ID,
                "name": DEMO_PATIENT_NAME,
                "age": demo.get("age") or 67,
                "weight_kg": demo.get("weight_kg") or 78.0,
                "allergies": demo.get("allergies") or [],
                "active_conditions": list({
                    *(demo.get("active_conditions") or []),
                    "type 2 diabetes mellitus",
                    "chronic kidney disease",
                }),
                "enrolled_at": now_ms() - random.randint(60, 720) * 86_400_000,
                "_provenance": "synthea (relabelled as P-204 for demo)",
            })

        # The rest of the Synthea patients keep their generated short IDs.
        for p in synthea_patients:
            if p.get("synthea_id") == demo_synthea_id:
                continue
            docs.append({
                "_id": new_id("p"),
                "patient_id": p["patient_id"],
                "name": p["name"],
                "age": p.get("age") or 0,
                "weight_kg": p.get("weight_kg") or 0.0,
                "allergies": p.get("allergies") or [],
                "active_conditions": p.get("active_conditions") or [],
                "enrolled_at": now_ms() - random.randint(60, 720) * 86_400_000,
                "_provenance": "synthea v3.3 (Apache 2.0)",
            })
    else:
        # Fallback: use the inline mock patients + add the demo P-204 directly.
        docs.append({
            "_id": new_id("p"),
            "patient_id": DEMO_PATIENT_ID,
            "name": DEMO_PATIENT_NAME,
            "age": 67,
            "weight_kg": 78.0,
            "allergies": [],
            "active_conditions": ["type 2 diabetes", "chronic kidney disease stage 3b"],
            "enrolled_at": now_ms() - random.randint(60, 720) * 86_400_000,
            "_provenance": "mock (no synthea fixture)",
        })
        for p in MOCK_PATIENTS:
            docs.append({
                "_id": new_id("p"),
                **p,
                "enrolled_at": now_ms() - random.randint(60, 720) * 86_400_000,
                "_provenance": "mock fallback",
            })

    await collection("patients").insert_many(docs)
    log.info("seeded %d patients (demo patient: %s)", len(docs), DEMO_PATIENT_ID)
    return demo_synthea_id


async def _seed_clinical_events(
    synthea_events: list[dict],
    synthea_patients: list[dict],
    demo_synthea_id: str | None,
) -> None:
    """Seeds clinical_events with:
       - Two hand-shaped headline events on P-204 (metformin admin, eGFR=38 lab)
       - All Synthea events, with the demo patient's events re-labelled to P-204
       - Falls back to small mock if no Synthea events
    """
    docs: list[dict] = []

    # 1. Headline events for the demo, deterministic and recent.
    docs.append({
        "timestamp": datetime.now(timezone.utc) - timedelta(hours=47),
        "patient_id": DEMO_PATIENT_ID,
        "event_type": "med_administration",
        "severity": "low",
        "notes": "Metformin 500 mg PO administered with breakfast. Tolerated well.",
        "medication": "metformin",
        "dose": "500 mg PO",
        "lab_name": None,
        "lab_value": None,
        "lab_unit": None,
        "_provenance": "demo headline (hand-shaped)",
    })
    docs.append({
        "timestamp": datetime.now(timezone.utc) - timedelta(hours=18),
        "patient_id": DEMO_PATIENT_ID,
        "event_type": "lab_result",
        "severity": "high",
        "notes": "eGFR 38 mL/min/1.73m^2 — low; CKD stage 3b; metformin caution.",
        "medication": None,
        "dose": None,
        "lab_name": "eGFR",
        "lab_value": 38.0,
        "lab_unit": "mL/min/1.73m^2",
        "_provenance": "demo headline (hand-shaped)",
    })
    docs.append({
        "timestamp": datetime.now(timezone.utc) - timedelta(hours=18),
        "patient_id": DEMO_PATIENT_ID,
        "event_type": "lab_result",
        "severity": "medium",
        "notes": "Creatinine 1.7 mg/dL — elevated for CKD.",
        "medication": None,
        "dose": None,
        "lab_name": "creatinine",
        "lab_value": 1.7,
        "lab_unit": "mg/dL",
        "_provenance": "demo headline (hand-shaped)",
    })

    # 2. Synthea events. Re-map the demo patient's events to P-204.
    short_demo = next(
        (p["patient_id"] for p in synthea_patients if p.get("synthea_id") == demo_synthea_id),
        None,
    )
    for ev in synthea_events:
        pid = ev["patient_id"]
        if short_demo and pid == short_demo:
            pid = DEMO_PATIENT_ID
        ts_ms = ev.get("timestamp") or 0
        if ts_ms <= 0:
            continue
        when = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        docs.append({
            "timestamp": when,
            "patient_id": pid,
            "event_type": ev["event_type"],
            "severity": ev.get("severity") or "low",
            "notes": ev.get("notes") or "",
            "medication": ev.get("medication"),
            "dose": ev.get("dose"),
            "lab_name": ev.get("lab_name"),
            "lab_value": ev.get("lab_value"),
            "lab_unit": ev.get("lab_unit"),
            "_provenance": ev.get("_provenance") or "synthea",
        })

    # 3. If Synthea didn't load, sprinkle in tiny mock filler.
    if not synthea_events:
        for i, mock_pid in enumerate([p["patient_id"] for p in MOCK_PATIENTS]):
            docs.append({
                "timestamp": datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30)),
                "patient_id": mock_pid,
                "event_type": "vitals",
                "severity": "low",
                "notes": "BP 130/82, HR 75, SpO2 98%.",
                "medication": None,
                "dose": None,
                "lab_name": None,
                "lab_value": None,
                "lab_unit": None,
                "_provenance": "mock fallback",
            })

    await collection("clinical_events").insert_many(docs)
    log.info("seeded %d clinical events into time-series", len(docs))


def _image_fields(c: dict) -> dict:
    """Extract the image columns from an ingest record (DailyMed or Wikimedia)."""
    out: dict[str, Any] = {}
    for k in ("image_b64", "image_mime", "image_attribution", "image_source_url"):
        v = c.get(k)
        if v:
            out[k] = v
    return out


async def _seed_references(
    dailymed_chunks: list[dict],
    equipment_rows: list[dict],
) -> None:
    """Seeds the unified apparatus catalog: medication monograph chunks
    (with FDA product photos) + medical-equipment entries (with Wikimedia
    device photos). Every row carries (name, category, context, image)."""
    chunks_to_seed: list[dict] = []

    # 1. Medication chunks from DailyMed (or mock if fixture missing)
    if dailymed_chunks:
        for c in dailymed_chunks:
            chunks_to_seed.append({
                "_id": new_id("doc"),
                "name": c.get("name") or c["medication"],
                "category": c.get("category") or "medication",
                "medication": c["medication"],
                "source_doc": c["source_doc"],
                "section": c["section"],
                "text": c["text"],
                "loinc": c.get("loinc"),
                "_provenance": c.get("_provenance") or "dailymed",
                **_image_fields(c),
            })
    else:
        for c in MOCK_REFERENCES:
            chunks_to_seed.append({
                "_id": new_id("doc"),
                **c,
                "_provenance": "mock fallback",
            })

    # 2. Equipment from Wikimedia/Wikipedia (or mock if fixture missing)
    if equipment_rows:
        for c in equipment_rows:
            chunks_to_seed.append({
                "_id": new_id("doc"),
                "name": c["name"],
                "category": c.get("category") or "equipment",
                "source_doc": c["source_doc"],
                "section": c.get("wiki_title") or c["name"],
                "text": c["text"],
                "wiki_title": c.get("wiki_title"),
                "_provenance": c.get("_provenance") or "wikimedia",
                **_image_fields(c),
            })
    else:
        for c in MOCK_EQUIPMENT:
            chunks_to_seed.append({
                "_id": new_id("doc"),
                **c,
                "section": c.get("name", "Equipment"),
                "_provenance": "mock fallback",
            })

    # Embed in batches. Build the embedding text from name + section + body so
    # equipment rows (no LOINC section) still produce useful vectors.
    texts = [
        f"{c.get('name','')} — {c.get('section','')}: {c['text']}"
        for c in chunks_to_seed
    ]
    log.info("embedding %d catalog rows (batched)…", len(texts))
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vectors.extend(await embed_batch(batch))
    for c, v in zip(chunks_to_seed, vectors):
        c["text_embedding"] = v
    await collection("documents").insert_many(chunks_to_seed)
    try:
        await collection("documents").create_index([("text", "text")])
    except Exception:  # noqa: BLE001
        pass
    by_cat: dict[str, int] = {}
    for c in chunks_to_seed:
        by_cat[c.get("category", "?")] = by_cat.get(c.get("category", "?"), 0) + 1
    img_count = sum(1 for c in chunks_to_seed if c.get("image_b64"))
    log.info(
        "seeded %d catalog rows (%s); %d carry images",
        len(chunks_to_seed),
        ", ".join(f"{k}={v}" for k, v in by_cat.items()),
        img_count,
    )


async def _seed_past_notes(mtsamples_notes: list[dict]) -> None:
    docs_to_seed: list[dict] = []
    if mtsamples_notes:
        for n in mtsamples_notes:
            docs_to_seed.append({
                "_id": new_id("ts"),
                "session_id": n.get("session_id") or "historical",
                "timestamp": now_ms() - random.randint(1, 60) * 86_400_000,
                "text": n["text"],
                "is_final": True,
                "is_question": False,
                "medical_specialty": n.get("medical_specialty"),
                "_provenance": n.get("_provenance") or "mtsamples",
            })
    else:
        for t in MOCK_NOTES:
            docs_to_seed.append({
                "_id": new_id("ts"),
                "session_id": "historical",
                "timestamp": now_ms() - random.randint(1, 30) * 86_400_000,
                "text": t,
                "is_final": True,
                "is_question": False,
                "_provenance": "mock fallback",
            })

    texts = [d["text"] for d in docs_to_seed]
    log.info("embedding %d past notes (batched)…", len(texts))
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vectors.extend(await embed_batch(batch))
    for d, v in zip(docs_to_seed, vectors):
        d["text_embedding"] = v
    await collection("transcripts").insert_many(docs_to_seed)
    log.info("seeded %d past clinical notes", len(docs_to_seed))


async def _seed_session() -> None:
    await collection("sessions").update_one(
        {"_id": "demo"},
        {"$set": {"_id": "demo", "room": "liverecall-demo", "started_at": now_ms(), "ended_at": None}},
        upsert=True,
    )


# --- Provenance summary -----------------------------------------------------

def _provenance_summary() -> dict[str, str]:
    parts = []
    if DAILYMED_FIXTURE.exists():
        parts.append("DailyMed (FDA SPL + product images)")
    else:
        parts.append("mock medications")
    if WIKIMEDIA_FIXTURE.exists():
        parts.append("Wikimedia equipment (CC-BY-SA)")
    else:
        parts.append("mock equipment")
    return {
        "documents": " + ".join(parts),
        "patients+clinical_events": (
            "Synthea v3.3 (Apache 2.0, MITRE)" if SYNTHEA_FIXTURE.exists() else "mock fallback"
        ),
        "transcripts": (
            "MTSamples (CC0)" if MTSAMPLES_FIXTURE.exists() else "mock fallback"
        ),
    }


async def main() -> None:
    await init_collections()
    await _wipe()

    dailymed = _load_dailymed()
    equipment = _load_wikimedia_equipment()
    synthea_patients, synthea_events = _load_synthea()
    mtsamples = _load_mtsamples()

    demo_synthea_id = await _seed_patients(synthea_patients)
    await _seed_clinical_events(synthea_events, synthea_patients, demo_synthea_id)
    await _seed_references(dailymed, equipment)
    await _seed_past_notes(mtsamples)
    await _seed_session()

    log.info("seed complete — provenance: %s", _provenance_summary())
    log.info(
        "try: 'Is it safe to give this dose now?' with %s + METFORMIN in scene",
        DEMO_PATIENT_ID,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
