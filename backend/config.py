"""Centralised env config. Import `settings` everywhere."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val or ""


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    mongodb_db: str
    openai_api_key: str
    backend_host: str
    backend_port: int
    public_backend_url: str

    # LiveKit (optional — only needed for /token + worker streaming path).
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    # ElevenLabs (optional — only needed for streaming TTS/STT).
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

    # Hard latency targets (used for the dashboard latency monitor).
    target_total_ms: int = 2000
    ceiling_total_ms: int = 2500

    # Vision sampling.
    scene_context_ttl_s: int = 10         # reuse vision output for up to 10s
    frame_sample_hz: float = 1.0          # how often the worker grabs a frame

    # Vector search.
    num_candidates: int = 50
    retrieval_limit: int = 5


def load_settings() -> Settings:
    return Settings(
        mongodb_uri=_env("MONGODB_URI"),
        mongodb_db=_env("MONGODB_DB", "liverecall"),
        openai_api_key=_env("OPENAI_API_KEY"),
        backend_host=_env("BACKEND_HOST", "0.0.0.0"),
        backend_port=int(_env("BACKEND_PORT", "8000")),
        public_backend_url=_env("PUBLIC_BACKEND_URL", "http://localhost:8000"),
        livekit_url=_env("LIVEKIT_URL", ""),
        livekit_api_key=_env("LIVEKIT_API_KEY", ""),
        livekit_api_secret=_env("LIVEKIT_API_SECRET", ""),
        elevenlabs_api_key=_env("ELEVENLABS_API_KEY", ""),
        elevenlabs_voice_id=_env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
    )


settings = load_settings()
