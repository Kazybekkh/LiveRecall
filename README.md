# LiveRecall
MongoDB Agentic Evolution Hackathon

> **✅ Demo state (2026-05-02): full multi-source seed live, Path A verified.**
> Atlas carries the multimodal apparatus catalog (DailyMed medications +
> Wikimedia equipment) in `documents`, the 25-patient Synthea cohort in
> `patients` (incl. the headline `P-204` "Sarah Chen"), ~900 time-series
> `clinical_events`, and the MTSamples `transcripts`. All three Atlas
> Vector Search indexes (`doc_text_vec`, `scene_text_vec`,
> `transcript_text_vec`) are `READY`. Path A (`POST /snap` with
> `capture_mode="glasses"`) has been run end-to-end three times:
> metformin label → `apparatus=['metformin']` → ranked references with
> image thumbnails + P-204 clinical events → grounded CKD/eGFR answer,
> ~15–30 s round trip, repeatable.
>
> - Smoke-test vector search + exact current counts any time — see [§ Verify](#verify).
> - Optional equipment-only slice: `[EQUIPMENT_SCOPE.md](./EQUIPMENT_SCOPE.md)` + `make seed-equipment`. Restore full multi-source with `make seed`.

**Clinical decision support grounded in live visual memory.** A clinician
wears camera glasses (or holds a phone). They look at a pill bottle, a
patient wristband, a vital-signs monitor — and ask one question. The system
pulls drug references, the patient's clinical timeline, and prior notes,
*reweighted by what the clinician is looking at right now*. **MongoDB
Atlas** is the substrate, **LiveKit Cloud** is the transport.

> Pitch: *"Multi-source agentic retrieval that reweights answers based on
> what the clinician is looking at right now."* Decision support, not
> diagnosis. Not RAG. Not an image analyzer.

**Two ways in:**

1. **Continuous capture** — glasses or phone stream POV; the system passively builds scene context.
2. **Single-image snap** — one tap, one frame, one question. `POST /snap`.

**Capture devices** — parallel entry points, same Vision pipeline:


| Mode                                      | Entry point                                                                          | When to use it                                                                                              |
| ----------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| 🕶️ **Meta Ray-Ban v2 (POV, headline)**   | `[scripts/bridge_rayban_snap.py](./scripts/bridge_rayban_snap.py)` → `/snap`         | Real glasses. Mirror the Meta AI iPhone app to Mac, screenshot every 3 s. Deterministic, Path A. See § 4.1. |
| 🎬 **Ray-Ban pre-recorded replay**        | `[scripts/bridge_rayban.py](./scripts/bridge_rayban.py)` → LiveKit room as `glasses` | Publish a recorded Ray-Ban MP4 into the demo room for a fully-deterministic walkthrough. See § 4.2.         |
| 🧪 **Webcam stand-in**                    | `[phone/glasses.html](./phone/glasses.html)` — `capture_mode="glasses"`              | Laptop webcam / phone in a head strap when the real glasses aren't around. Zero setup.                      |
| 📱 **Phone browser (universal fallback)** | `[phone/index.html](./phone/index.html)` — `capture_mode="phone"`                    | Day-1 adoption: every clinician on shift already has one, no hardware purchase needed.                      |


All four paths converge on the same Vision prompt (POV hint flips on
`capture_mode`) and the same downstream Router → Retrievers → Reranker →
Answerer pipeline. The branch happens at the ingestion seam only, so
whichever capture wins in the field, the rest of the system is unchanged.
The path to a *live* Meta Live AI → LiveKit ingress integration (no
screen-mirror in the middle) is documented in
`[DECISIONS.md](./DECISIONS.md)` entry (g). Compliance rules apply to every
capture path: mock patients only, no real PHI.

---

## Quickstart (≈5 min — assumes `.env` is filled in)

```bash
# 1. One-time setup
cp .env.example .env                # fill in keys, see § 1
make install                        # venv + pip + npm
make seed                           # load full multi-source fixtures into Atlas

# 2. Run the demo — three terminals
make backend                        # FastAPI on :8000 (or your $BACKEND_PORT)
make dashboard                      # Next.js on :3000

# 3. Pick your capture path:
# 3a. Real Ray-Bans (primary). Mirror Meta AI on the iPhone to your Mac first.
make bridge-rayban-pick             # drag a rectangle over the mirror window
# copy the REGION line, then:
make bridge-rayban REGION=120,90,960,540 INTERVAL=3 SESSION=demo

# 3b. No glasses handy? Serve the webcam stand-in page:
make phone                          # http.server on :8080 → open phone/glasses.html

# 4. Point at a metformin bottle + P-204 wristband, ask:
#    "Is it safe to give this dose now?"
#    → ranked references + events + grounded CKD/eGFR answer in ~15–30 s.
```

---

## What's in here

```
shared/         types.py + types.ts — the API contract
backend/        FastAPI + LiveKit Agents worker + 5 LangChain agents + Mongo
  agents/       vision · router · retrievers · reranker · answerer
  worker.py     LiveKit Agents entrypoint (joins room, samples frames, pipes audio)
  main.py       FastAPI app + change-stream WebSocket + REST endpoints + /snap
  mongo.py      8 collections, vector indexes, time series, TTL, change streams
  stt.py        ElevenLabs Scribe v2 Realtime streaming STT (writes transcripts + questions)
  tts.py        ElevenLabs Flash v2.5 streaming TTS (PCM into LiveKit AudioSource)
  embeddings.py text-embedding-3-small batched + cached
  tracing.py    LangChain MongoTraceCallback → agent_traces
phone/          Single-file LiveKit client for any phone browser (incl. Snap & ask)
dashboard/      Next.js + Tailwind judge-facing dashboard
scripts/        seed_mongo.py           — 5 patients, 50 clinical events (TS), 3 drug references, 5 past notes
                bridge_rayban_snap.py   — Meta Ray-Ban bridge: periodic screenshots → /snap (primary)
                bridge_rayban.py        — Meta Ray-Ban bridge: pre-recorded MP4 → LiveKit (fallback)
```

## Architecture (one screen)

```
[Phone]──audio+video──>[LiveKit room]──>[Worker]
   │                                      │
   │   tap "Snap & ask" ──POST /snap──────┤ (synchronous Vision; bypass continuous sampling)
   │                                      │
        ┌─────── frame sampler ──────────►│ video_frames ──► Vision (GPT-4o) ──► scene_context (vec)
        │                                 │
        │   Scribe v2 Realtime STT ◄──────┤ transcripts ──┐
        │                                 │               └─► questions ──► Router (GPT-4o-mini)
        │                                 │                                       │
        │                                 │                                  retrieval_plans
        │                                 │                                       │
        │                                 │                  ┌──────── 3 Retrievers (no LLM)
        │                                 │                  │   references  ($vectorSearch on documents)
        │                                 │                  │   events      (Time Series filter+sort, per patient_id)
        │                                 │                  │   notes       ($vectorSearch + recency)
        │                                 │                  │
        │                                 │                  │   ↑ Local Retrieval: cache pre-warmed by Vision the
        │                                 │                  │     moment a wristband / pill label is seen.
        │                                 │                  │     Hit ≈5 ms vs cold Mongo ≈150–300 ms.
        │                                 │                  │
        │                                 │                  └──► retrieval_results (with from_cache flag)
        │                                 │                                       │
        │                                 │                              Reranker Pass-1 (GPT-4o-mini)
        │                                 │                              may emit ≤2 active_followups
        │                                 │                                       │
        │                                 │                  ┌── Active Retrieval (when needed) ──┐
        │                                 │                  │   get_latest_lab(patient, lab)     │
        │                                 │                  │   get_last_administration(...)     │
        │                                 │                  │   get_monograph_section(...)       │
        │                                 │                  └──► Reranker Pass-2 folds in facts ──┘
        │                                 │                                       │
        │                                 │                                final_context
        │                                 │                                       │
        │                                 │                          Answerer (GPT-4o-mini, streaming)
        │                                 │                                       │
        │                                 │                ElevenLabs Flash v2.5  │
        │                                 │                       │               │
        │                                 │◄──── publish audio ───┘               │
        ▼                                 │                                       │
[LiveKit room]──audio──>[Phone speaker]   └──── change streams ──► /stream WS ──► Dashboard
```

Every state change is a Mongo write. The agent bus is Mongo change streams.

## Datasets — multimodal apparatus catalog + clinical context

Four real public datasets back the three retrievers. Everything runs
**zero-shot** through pre-trained models — no fine-tuning per hospital —
because the entities are canonical (USAN drug names, LOINC labs, SNOMED
conditions, common bedside device shapes).


| Retriever                | Dataset                                                                                      | License                     | What we use                                                                                                                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| References (medications) | [DailyMed](https://dailymed.nlm.nih.gov/) (FDA SPL labels + product images)                  | US Government work / public | 10 drugs, 136 LOINC-coded chunks, **100% with real product photos** from `/spls/{setid}/media`                                                                                               |
| References (equipment)   | [Wikipedia + Wikimedia Commons](https://en.wikipedia.org)                                    | CC-BY-SA 4.0                | 60–110 bedside devices (infusion pump, pulse oximeter, defibrillator, insulin pen, ECG machine, …), thumbnails downscaled to ≤256 px. Count depends on `make seed` vs `make seed-equipment`. |
| Events                   | [Synthea v3.3](https://github.com/synthetichealth/synthea) (synthetic FHIR)                  | Apache 2.0                  | 25 patients (T2DM/CKD-biased) + ~917 medications, vitals, labs                                                                                                                               |
| Notes                    | [MTSamples](https://huggingface.co/datasets/harishnair04/mtsamples) (medical transcriptions) | CC0 (public domain)         | 35 notes across 7 specialties (endo, nephro, discharge…)                                                                                                                                     |


The References sources (DailyMed + Wikimedia) are unioned into the
`documents` collection as a single multimodal apparatus catalog where every
row has `(name, context, image)` and `category ∈ {medication, equipment}`.
This is what lets the Vision agent recognise *both* labelled medications
*and* unlabelled equipment, and what lets the dashboard show the actual
product photo next to the matched snippet.

- Fixtures live in `[data/fixtures/](./data/fixtures/)` (committed; ~2.2 MB total — ~700 KB of inlined image thumbnails).
- Ingest scripts in `[scripts/ingest_*.py](./scripts/)` re-fetch from source on demand.
- The seeder picks the most demo-relevant Synthea patient and **re-labels them
as `P-204` "Sarah Chen"** with two hand-shaped headline events injected on
top (metformin admin 47 h ago + eGFR=38 lab 18 h ago) so the demo lands
deterministically. Surrounding context is real-shaped Synthea data.
- See `[DECISIONS.md](./DECISIONS.md)` entries **(e)** + **(f)** for rationale +
`[data/README.md](./data/README.md)` for refresh workflow + license details.

## Latency target


| Path                                                     | End-of-question → first audio byte |
| -------------------------------------------------------- | ---------------------------------- |
| **Streaming** (continuous capture)                       | **≤2 s target / 2.5 s ceiling**    |
| **Streaming + warm cache** (Local Retrieval hit)         | **≈1.4 s**                         |
| **Streaming + active follow-up** (Reranker Pass-2 fired) | **≈1.8 s**                         |
| **Snap & ask** (Vision in critical path)                 | **≤3.5 s**                         |



| Stage                                         | Budget                                                   |
| --------------------------------------------- | -------------------------------------------------------- |
| Phone → LiveKit                               | 50–100 ms                                                |
| ElevenLabs Scribe v2 Realtime STT             | ~150 ms partial · ~300 ms commit                         |
| Vision (GPT-4o, *only* on snap path)          | 1000–1500 ms                                             |
| Router (GPT-4o-mini)                          | 300–500 ms                                               |
| 3 Retrievers (parallel Mongo) — cold          | 200–400 ms                                               |
| 3 Retrievers — **Local Retrieval cache hit**  | **≈5 ms**                                                |
| Reranker Pass-1 (GPT-4o-mini)                 | 300–500 ms                                               |
| Reranker Pass-2 (when active follow-up fires) | +120–250 ms LLM + 30–80 ms tools (parallel, capped at 2) |
| Answerer first token (streaming)              | 200–400 ms                                               |
| ElevenLabs Flash v2.5 first byte              | 150–300 ms                                               |
| LiveKit → phone                               | 50–100 ms                                                |


> **Local + Active Retrieval.** Vision drives a per-session prefetch cache the
> moment it reads a wristband or pill label, so the Retrievers usually serve
> from in-process memory by the time the question arrives. The Reranker can
> additionally fire up to 2 targeted follow-up tool calls when it spots a
> concrete information gap (e.g. has a renal-contraindication chunk but no
> recent eGFR). Both layered on **pre-trained models** — the medical domain is
> canonical (USAN drug names, LOINC labs, MRN wristbands), so we don't need to
> fine-tune anything per hospital. See `[DECISIONS.md](./DECISIONS.md)` entry
> (d).

> **STT vendor note.** The original spec called for Deepgram Nova-2. We swapped
> to ElevenLabs Scribe v2 Realtime (Jan 2026) — same first-token latency
> (~150 ms), one fewer vendor + key, single ElevenLabs SDK for both speech
> directions. See `[DECISIONS.md](./DECISIONS.md)`.

## Setup

### 0. Prereqs

- Python 3.11+
- Node 20+
- A **MongoDB Atlas Sandbox** cluster (use the hackathon link — *not* a personal Atlas).
- A **LiveKit Cloud** project (free tier).
- API keys for **OpenAI** and **ElevenLabs** (one ElevenLabs key powers both Scribe v2 Realtime STT and Flash v2.5 TTS).
- **JDK 11+** *only* if you want to refresh the Synthea fixture from source. The committed fixture works without Java.

### 1. Configure env

```bash
cp .env.example .env
# fill in LIVEKIT_*, MONGODB_URI, OPENAI_API_KEY, ELEVENLABS_API_KEY
```

### 2. Backend

```bash
cd backend
python -m venv ../.venv && source ../.venv/bin/activate
pip install -r requirements.txt
cd ..
python -m backend.mongo          # creates collections + vector indexes
python -m scripts.seed_mongo     # loads from data/fixtures/, falls back to mock if any are missing
python -m backend.main           # starts FastAPI + agent loops on :8000
# in a second terminal:
python -m backend.worker dev     # starts the LiveKit Agents worker
```

> **Atlas Vector Search** indexes can take a couple of minutes to come online
> after creation. The References retriever falls back to a `$text` search if
> the vector index isn't ready, so the system still works — just less adaptive.

### 2.1. (Optional) Refresh the dataset fixtures

The repo ships pre-built JSONL fixtures under `data/fixtures/` so the seed
step works offline. To rebuild them from the public sources:

```bash
python -m scripts.ingest_dailymed             # ~40 s — FDA SPL labels + product images (REST API)
python -m scripts.ingest_wikimedia_equipment  # ~3–5 min — Wikipedia/Commons device thumbnails (rate-limited)
python -m scripts.ingest_mtsamples            # ~10 s — HuggingFace parquet (CC0)
python -m scripts.ingest_synthea              # ~5 min — runs the Synthea jar (needs JDK 11+)
```

See `[data/README.md](./data/README.md)` for layout, licenses, and the
provenance details for all four datasets.

### 3. Dashboard

```bash
cd dashboard
npm install
npm run dev    # http://localhost:3000
```

### 4. Phone / webcam stand-in (any browser)

Serve `phone/` with anything (simplest):

```bash
cd phone && python -m http.server 8080     # open http://<laptop-ip>:8080 on any phone
# or:
make phone
```

Two pages, two capture modes, same backend:

- `[phone/index.html](./phone/index.html)` — **phone fallback** (universal "safety mode"). Announces `capture_mode="phone"`.
- `[phone/glasses.html](./phone/glasses.html)` — **webcam Ray-Ban stand-in**. Announces `capture_mode="glasses"`. First-person framing; works as a stand-in via laptop webcam or a phone in a head strap when the real glasses + the § 4.1 bridge aren't around.

Important:

- The pages ask for your backend URL on first launch — point it at the laptop IP, not `localhost`, when hitting from another device.
- iOS Safari requires HTTPS for camera/mic on remote hosts. Easiest path: tether the phone to the laptop via hotspot and wrap your local backend in HTTPS via `[localhost.run](https://localhost.run/)` or `ngrok http 8080`.

### 4.1. Meta Ray-Ban bridge — screenshot loop (primary, macOS)

Drives the already-verified Path A (`POST /snap` with `capture_mode="glasses"`)
on a timer. End-to-end:

1. Put on the Ray-Ban Meta v2. Open **Meta AI** on the paired iPhone so the glasses' POV is visible in the app.
2. Mirror the iPhone to the Mac — QuickTime Player → *File → New Movie Recording* → down-arrow next to the record button → pick the iPhone. (AirPlay to a Mac receiver works too.) Arrange the mirror window so the glasses POV fills it.
3. Pick the screen region to capture (one-time):
  ```bash
   make bridge-rayban-pick
   # a translucent overlay appears → drag a rectangle over the mirror window
   # → prints: REGION 120,90,960,540
  ```
4. Run the bridge. Every `INTERVAL` seconds (default **3 s**) it screenshots that region, downsizes to ≤640 px long-edge, JPEG-encodes, and POSTs to `/snap`:
  ```bash
   make bridge-rayban REGION=120,90,960,540 INTERVAL=3 SESSION=demo
   # or directly:
   python -m scripts.bridge_rayban_snap \
     --region 120,90,960,540 --interval 3 \
     --backend http://localhost:8000 --session-id demo
  ```

Useful flags:

- `--fullscreen` — capture the whole main display instead of a region.
- `--question "Is it safe to give this dose now?"` — fire a question with every snap (full Path A). Omit for scene-only updates and ask separately through `/ask` or voice.
- `--dedup` — skip POSTing if the JPEG bytes are byte-identical to the previous frame (cheap frozen-window guard).
- `-v` — debug logging.

Expected log cadence:

```
#0001 sent in 3.2s · objects=['pill bottle'] · visible=['METFORMIN 500 mg', 'P-204'] · qid=-
#0002 sent in 2.8s · objects=['pill bottle'] · visible=['METFORMIN 500 mg'] · qid=-
```

`Ctrl-C` to stop.

Each POST stamps `scene_context.capture_mode = "glasses"`, so Vision picks
the first-person POV prompt, the Router/Retrievers pull P-204 events and
metformin monograph chunks, and the dashboard lights up the `GLASSES` tag.
Why this shape instead of a live video bridge: Ray-Ban Meta v2 doesn't
expose a public live-video endpoint we can subscribe to from outside the
Meta AI app, and `/snap` is already the deterministic, judge-proof entry
point. See `[DECISIONS.md](./DECISIONS.md)` entry (g) for the live-API
upgrade path.

**macOS first-run gotcha.** The first time `screencapture` grabs your
screen, macOS will prompt for **Screen Recording** permission on whichever
app is running Python (Terminal, iTerm, VS Code, etc.). Approve it in
*System Settings → Privacy & Security → Screen Recording*, **quit and
relaunch** that app, then re-run the bridge. After that it's silent.

**Loop cadence note.** The loop is synchronous — each iteration waits for
`/snap` to return before sleeping. If Vision takes longer than `INTERVAL`
the real cadence becomes Vision-bound (no request overlap, no pileup; just
effective interval = `max(INTERVAL, /snap_time)`). That's usually what you
want; if you need strictly-every-3s regardless of Vision latency, switch to
`--dedup` + a shorter `INTERVAL` or ask for the async variant.

### 4.2. Meta Ray-Ban bridge (pre-recorded MP4 — fallback)

If you'd rather replay the same clip deterministically through LiveKit
(e.g. for a rehearsed video walkthrough), the original file bridge is still
here. It publishes an MP4 into the demo room as the `glasses` participant,
and the Worker sees the same audio + video tracks a real Ray-Ban Live AI
feed would send:

```bash
python -m scripts.bridge_rayban path/to/rayban_clip.mp4 \
  --backend http://localhost:8000 --room liverecall-demo --loop
# or
make bridge-rayban-mp4 MP4=path/to/rayban_clip.mp4 ROOM=liverecall-demo
```

Requires `av>=13.0` (PyAV; already in `backend/requirements.txt`). Any
standard H.264/AAC MP4 works; frames are decoded at the file's native fps
and handed to LiveKit's `AVSynchronizer`, which keeps audio and video in
lock-step.

### 5. Try it

- Open `http://localhost:3000` (dashboard).
- Open either the phone page (§ 4) or start the Ray-Ban screenshot bridge (§ 4.1).
- Either: (a) speak a question while pointing the camera, or (b) tap **Snap & ask** / pass `--question` to the bridge, and frame a pill bottle / wristband.
- Watch the lanes light up: Vision → Router → Retrievers (references / events / notes) → Reranker → Answerer.
- The phone plays the answer audio back through its speaker (or the dashboard shows the text if you're on the bridge).

## Verify (smoke tests)

Paste these into a terminal with `.venv` activated — they hit the live
Atlas cluster in your `.env` and print green/red in seconds.

**1. MongoDB vector search + collection health:**

```bash
python -c "
import asyncio
from dotenv import load_dotenv; load_dotenv()
from backend.mongo import collection, get_db

async def main():
    db = get_db()
    print('db:', db.name)
    for c in ['documents','patients','clinical_events','transcripts','scene_context']:
        print(f'  {c:<18}', await collection(c).count_documents({}))
    print('  P-204 exists:', bool(await collection('patients').find_one({'patient_id':'P-204'})))
    for coll in ['documents','scene_context','transcripts']:
        idx = [i async for i in db[coll].list_search_indexes()]
        print(f'  {coll}.indexes', [(i.get(\"name\"), i.get(\"status\")) for i in idx])

asyncio.run(main())"
```

Expect: `documents 242`, `patients 25`, `clinical_events 920`, `transcripts 35`,
`P-204 exists: True`, and all three vector indexes `READY`.

**2. Live `$vectorSearch` on the apparatus catalog:**

```bash
python -c "
import asyncio
from dotenv import load_dotenv; load_dotenv()
from backend.embeddings import embed
from backend.mongo import collection

async def q(text):
    v = await embed(text)
    pipe = [
      {'\$vectorSearch': {'index':'doc_text_vec','queryVector':v,'path':'text_embedding','numCandidates':50,'limit':3}},
      {'\$project': {'name':1,'category':1,'score':{'\$meta':'vectorSearchScore'}}},
    ]
    print(f'Q: {text!r}')
    async for d in collection('documents').aggregate(pipe):
        print(f'  [{d[\"score\"]:.3f}] {d.get(\"category\",\"?\"):<10} {d.get(\"name\")}')

async def main():
    for t in ['IV drip pump','metformin renal contraindication','airway intubation device']:
        await q(t)

asyncio.run(main())"
```

Expect the top hit to be the semantically-matching row in each case
(e.g. `infusion pump`, `metformin`, `laryngeal tube`) with scores roughly
0.74+ on device queries and 0.82+ on the medication query.

**3. End-to-end Path A against a running backend:**

```bash
# with backend running — use your actual $BACKEND_PORT (default 8000)
BACKEND=${PUBLIC_BACKEND_URL:-http://localhost:8000}

curl -s -X POST $BACKEND/ask \
  -H 'content-type: application/json' \
  -d '{"session_id":"demo","text":"Is it safe to give metformin to P-204?"}' | jq
# copy the returned question_id, then:
curl -s $BACKEND/trace/<question_id> | jq '.plan, .results[].source, .answer.text'
```

Should return a plan hitting all three retrievers, non-empty `results[]`,
and an answer that mentions renal function / eGFR / hold-this-dose.

## Demo question

> *"Is it safe to give this dose now? When did they last receive it?"*

Expected behavior: while the phone is pointed at a pill-bottle label reading
**METFORMIN 500 mg** with patient wristband **P-204** visible, the Router
pulls `medication="metformin"` and `patient_id="P-204"` into the retrieval
plan. The References retriever returns the metformin monograph chunk on
renal contraindications. The Events retriever returns the last administration
(~47 hours ago) and a recent eGFR=38 lab from the time-series collection.
The Notes retriever returns prior handoff notes about P-204. The Reranker
boosts the renal-contraindication chunk with a `boost_reason` citing the
visible "METFORMIN" token *and* the patient's eGFR. The Answerer says
something like:

> *"Hold this dose — P-204's most recent eGFR is 38, which is borderline for
> metformin. Last administration was 47 hours ago. Recommend rechecking renal
> function before giving."*

**Compliance language**: this is **clinical decision support, not diagnosis**.
The clinician is in the loop; the system never auto-administers anything. All
seed data is mock — no real PHI. Use that wording verbatim if a judge asks.

## MongoDB features used (8 load-bearing)

1. **Document model** — every collection (`patients`, `sessions`, `answers`, …)
2. **Atlas Vector Search** — `scene_context`, `documents`, `transcripts`
3. **Time Series collection** — `clinical_events` (`timeField=timestamp`, `metaField=patient_id`, granularity=hours)
4. **Change Streams** — the agent bus (`video_frames` → Vision; `questions` → Router; etc.)
5. `**$vectorSearch` in aggregation** — References + Notes retrievers
6. **GridFS** — `clips` bucket (24h TTL)
7. **TTL indexes** — `clips.files.uploadDate`
8. **Aggregation pipelines** — Retrievers (vector+filter+sort, recency boost) + Reranker

The dashboard sidebar lights each one up the moment it's actually exercised.

## Endpoints (backend — port from `BACKEND_PORT`, default `:8000`)


| Method | Path                               | Purpose                                                                                                                                                                                                                                    |
| ------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GET    | `/healthz`                         | liveness probe.                                                                                                                                                                                                                            |
| POST   | `/token`                           | LiveKit access token (phone, worker, or bridge). Body: `{identity, room, capture_mode?}`. `capture_mode` is `"glasses"` or `"phone"` — persisted on the session doc and threaded onto every `scene_context`. Defaults to `"phone"`.        |
| POST   | `/snap`                            | Single-image retrieval. Body: `{session_id, image_b64, question?, capture_mode?}`. Runs Vision synchronously, writes `video_frames` + `scene_context`, optionally inserts a `questions` doc. This is the bridge's entry point (see § 4.1). |
| POST   | `/ask`                             | Text-only pipeline kick (debug / CLI). Body: `{session_id, text}`. Inserts a `questions` doc; Router picks it up via change stream.                                                                                                        |
| GET    | `/scene-context/recent?seconds=30` | Recent `scene_context` docs, for dashboard convenience reads.                                                                                                                                                                              |
| GET    | `/trace/:question_id`              | Full reasoning chain: plan, per-source results, final context, answer, ordered agent traces.                                                                                                                                               |
| GET    | `/answers/:question_id`            | Just the final answer text.                                                                                                                                                                                                                |
| WS     | `/stream`                          | Change-stream fan-out for the dashboard.                                                                                                                                                                                                   |


## Troubleshooting


| Symptom                                                       | Probable cause & fix                                                                                                                                                                                                          |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `screencapture` returns a blank / black image from the bridge | macOS Screen Recording permission not granted. *System Settings → Privacy & Security → Screen Recording* → approve the app running Python → quit & relaunch it → re-run.                                                      |
| `/snap` takes 6–10 s and objects come back empty              | Vision got an IDE screenshot / non-clinical frame. Check that the `--region` actually covers the Meta AI mirror window. Run `make bridge-rayban-pick` again to re-select.                                                     |
| "Address already in use" when starting the backend            | A prior `python -m backend.main` is still bound. `lsof -iTCP:$BACKEND_PORT -sTCP:LISTEN` to find the PID, kill it, relaunch.                                                                                                  |
| Vector search returns 0 hits                                  | Atlas Vector Search index isn't `READY` yet (can take a minute after `python -m backend.mongo`). The References retriever falls back to `$text` — still works, less adaptive. Re-run § Verify step 1 to confirm index status. |
| Bridge exits immediately with "cancelled"                     | You hit ESC in the region picker instead of dragging. Re-run `make bridge-rayban-pick`.                                                                                                                                       |
| Bridge runs but no `scene_context` appears                    | Backend `/snap` isn't actually receiving the POST. Check `--backend` URL matches your `BACKEND_PORT`, and that `curl http://localhost:$BACKEND_PORT/healthz` returns `{"ok": true}`.                                          |
| Dashboard lanes don't light up                                | Dashboard is a WS consumer of `/stream`. Check `NEXT_PUBLIC_BACKEND_WS` matches your actual backend port.                                                                                                                     |


## Cuts list (ordered, if time slips)

1. Bluetooth Ray-Bans audio out → phone speaker.
2. Live Meta Ray-Ban Live AI → the already-shipped screenshot bridge (§ 4.1) or MP4 bridge (§ 4.2). Both give first-person POV, fully deterministic for demo reruns.
3. Streaming Answerer → non-streaming (+500 ms).
4. Vision per-frame → scene-change-only (every ~5 s).
5. Latency monitor on dashboard (cosmetic).

Never cut Mongo features. Never cut LiveKit.

## License

MIT (hackathon prototype).
