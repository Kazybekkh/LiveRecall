"use client";

import type { SceneContext } from "../lib/useTrace";

// Small badge that shows which capture device drove the most recent scene —
// Meta Ray-Ban POV (preferred) or phone fallback. Reads `capture_mode` off
// the latest scene_context. See DECISIONS.md (g) for why we run two
// parallel capture paths.
export function CaptureModePill({ scene }: { scene: SceneContext | null }) {
  const captureMode = scene?.capture_mode;

  if (captureMode === "glasses") {
    return (
      <span
        className="text-[11px] uppercase tracking-widest px-2 py-1 rounded-full bg-gradient-to-r from-violet-500/25 to-blue-500/20 text-violet-200 border border-violet-400/40"
        title="Headline POV: Meta Ray-Ban first-person capture (or any first-person-framed stand-in)."
      >
        🕶 glasses
      </span>
    );
  }

  if (captureMode === "phone") {
    return (
      <span
        className="text-[11px] uppercase tracking-widest px-2 py-1 rounded-full bg-slate-500/20 text-slate-200 border border-slate-400/30"
        title="Phone fallback — universal safety mode."
      >
        📱 phone <span className="ml-1 text-slate-400 normal-case tracking-normal">· fallback</span>
      </span>
    );
  }

  return (
    <span
      className="text-[11px] uppercase tracking-widest px-2 py-1 rounded-full bg-slate-700/30 text-slate-500"
      title="Awaiting first scene_context — capture mode will resolve to GLASSES or PHONE."
    >
      capture · —
    </span>
  );
}
