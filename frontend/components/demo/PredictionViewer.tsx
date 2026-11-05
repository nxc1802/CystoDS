"use client";

import React from "react";
import {
  PredictResponse,
  InferenceStatus,
} from "@/types";
import {
  ShieldAlert,
  CheckCircle,
  Activity,
  Cpu,
  Clock,
} from "lucide-react";

interface PredictionViewerProps {
  status: InferenceStatus;
  result: PredictResponse | null;
  error: string | null;
  onRetry?: () => void;
  latencyTimer?: number;
}

const COARSE_COLORS: Record<string, { bg: string; fill: string; text: string }> = {
  Malignant: {
    bg: "bg-rose-500/10",
    fill: "bg-rose-500",
    text: "text-rose-400",
  },
  "Non-malignant": {
    bg: "bg-amber-500/10",
    fill: "bg-amber-500",
    text: "text-amber-400",
  },
  "Normal mucosa": {
    bg: "bg-emerald-500/10",
    fill: "bg-emerald-500",
    text: "text-emerald-400",
  },
  "Anatomical landmarks": {
    bg: "bg-cyan-500/10",
    fill: "bg-cyan-500",
    text: "text-cyan-400",
  },
  "Foreign bodies": {
    bg: "bg-purple-500/10",
    fill: "bg-purple-500",
    text: "text-purple-400",
  },
};

export default function PredictionViewer({
  status,
  result,
  error,
  onRetry,
  latencyTimer = 0,
}: PredictionViewerProps) {
  // ── IDLE STATE ────────────────────────────────────────────────────────
  if (status === "idle") {
    return (
      <div className="flex flex-col items-center justify-center text-center p-8 md:p-12 h-full min-h-[420px] rounded-2xl border border-white/5 bg-white/[0.01]">
        <div className="w-14 h-14 rounded-full bg-white/[0.03] border border-white/10 flex items-center justify-center mb-4 text-neutral-400">
          <Activity className="w-6 h-6 text-brand-400" strokeWidth={1.25} />
        </div>
        <h3 className="text-base font-semibold text-white mb-2">
          Ready for Hierarchical Inference
        </h3>
        <p className="text-xs text-neutral-400 max-w-sm leading-relaxed mb-6">
          Select one of the 6 curated clinical reference cases or upload your own high-resolution cystoscopy image to trigger real-time multi-level analysis.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-2 text-[10px] font-mono text-neutral-500">
          <span className="px-2 py-1 rounded-md bg-white/[0.02] border border-white/5">
            Level 1: Binary ROI
          </span>
          <span className="px-2 py-1 rounded-md bg-white/[0.02] border border-white/5">
            Level 2: Coarse 5-Class
          </span>
          <span className="px-2 py-1 rounded-md bg-white/[0.02] border border-white/5">
            Level 3: Fine 22-Class
          </span>
        </div>
      </div>
    );
  }

  // ── RUNNING / LOADING STATE ──────────────────────────────────────────
  if (status === "uploading" || status === "running") {
    return (
      <div className="flex flex-col items-center justify-center text-center p-8 md:p-12 h-full min-h-[420px] rounded-2xl border border-brand-500/20 bg-brand-500/[0.02]">
        <div className="relative w-16 h-16 flex items-center justify-center mb-6">
          <div className="absolute inset-0 rounded-full border-2 border-brand-500/20 border-t-brand-400 animate-spin" />
          <Cpu className="w-6 h-6 text-brand-400 animate-pulse" strokeWidth={1.25} />
        </div>

        <h3 className="text-base font-semibold text-white mb-1">
          {status === "uploading"
            ? "Transferring Endoscopy Frame..."
            : "Executing 3S-HFT Neural Network..."}
        </h3>

        <p className="text-xs text-neutral-400 max-w-xs leading-relaxed mb-4">
          Running CenterFractionCrop(0.92) fov normalization, Swin-Tiny feature encoding, and hierarchical probability decoding.
        </p>

        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 border border-white/10 text-xs font-mono text-brand-300">
          <Clock className="w-3.5 h-3.5 text-brand-400" strokeWidth={1.25} />
          <span>Elapsed: {(latencyTimer / 1000).toFixed(2)}s</span>
        </div>
      </div>
    );
  }

  // ── ERROR STATE ──────────────────────────────────────────────────────
  if (status === "error" || error) {
    return (
      <div className="flex flex-col items-center justify-center text-center p-8 h-full min-h-[420px] rounded-2xl border border-rose-500/20 bg-rose-500/[0.02]">
        <div className="w-12 h-12 rounded-full bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mb-4 text-rose-400">
          <ShieldAlert className="w-6 h-6" strokeWidth={1.25} />
        </div>
        <h3 className="text-base font-semibold text-white mb-2">
          Inference Execution Error
        </h3>
        <p className="text-xs text-rose-300/80 max-w-sm leading-relaxed mb-6 font-mono">
          {error || "An unexpected error occurred during model computation."}
        </p>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="px-5 py-2.5 rounded-full bg-white text-black text-xs font-semibold hover:bg-neutral-100 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98]"
          >
            Retry Inference
          </button>
        )}
      </div>
    );
  }

  // ── SUCCESS STATE ────────────────────────────────────────────────────
  if (status === "success" && result) {
    const isBinaryROI = result.binary.prediction === "ROI";
    const binaryConfPct = (result.binary.confidence * 100).toFixed(1);

    // Coarse probabilities sorted
    const coarseEntries = Object.entries(result.coarse.probabilities || {}).sort(
      ([, a], [, b]) => b - a
    );

    const urgencyColor =
      result.urgency === "CRITICAL"
        ? "border-rose-500/30 bg-rose-500/10 text-rose-400"
        : result.urgency === "HIGH"
        ? "border-amber-500/30 bg-amber-500/10 text-amber-400"
        : result.urgency === "MEDIUM"
        ? "border-yellow-500/30 bg-yellow-500/10 text-yellow-300"
        : "border-emerald-500/30 bg-emerald-500/10 text-emerald-400";

    return (
      <div className="flex flex-col gap-6">
        {/* Header: Clinical Urgency Badge & Clinical Recommendation */}
        <div
          className={`p-4 rounded-2xl border ${urgencyColor} flex flex-col gap-2`}
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono uppercase tracking-widest font-semibold">
              Clinical Urgency Triage Level: {result.urgency}
            </span>
            <span className="text-[10px] font-mono text-neutral-400">
              Request ID: {result.request_id}
            </span>
          </div>

          <h4 className="text-sm sm:text-base font-bold text-white tracking-tight">
            {result.clinical_assessment?.title || `${result.coarse.prediction} Detected`}
          </h4>

          <p className="text-xs text-neutral-200/90 leading-relaxed">
            {result.clinical_assessment?.action ||
              "Follow established urological endoscopy protocol."}
          </p>
        </div>

        {/* ── LEVEL 1: BINARY ROI GAUGE ─────────────────────────────────── */}
        <div className="p-4 rounded-2xl border border-white/5 bg-black/40 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-white/10 text-white">
                Stage 1
              </span>
              <span className="text-xs font-semibold text-white">
                Binary Lesion Discrimination
              </span>
            </div>
            <span
              className={`text-xs font-mono font-bold px-2.5 py-0.5 rounded-full border ${
                isBinaryROI
                  ? "bg-rose-500/10 border-rose-500/30 text-rose-400"
                  : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
              }`}
            >
              {result.binary.prediction} ({binaryConfPct}%)
            </span>
          </div>

          {/* Probability Bar */}
          <div className="w-full h-2 rounded-full bg-white/5 overflow-hidden flex">
            <div
              style={{
                width: `${(result.binary.probabilities["ROI"] || 0) * 100}%`,
              }}
              className="h-full bg-rose-500 transition-all duration-700 ease-[cubic-bezier(0.32,0.72,0,1)]"
            />
            <div
              style={{
                width: `${(result.binary.probabilities["Non-ROI"] || 0) * 100}%`,
              }}
              className="h-full bg-emerald-500 transition-all duration-700 ease-[cubic-bezier(0.32,0.72,0,1)]"
            />
          </div>

          <div className="flex justify-between text-[10px] font-mono text-neutral-400">
            <span>
              ROI (Abnormal):{" "}
              {((result.binary.probabilities["ROI"] || 0) * 100).toFixed(2)}%
            </span>
            <span>
              Non-ROI (Benign/Landmark):{" "}
              {((result.binary.probabilities["Non-ROI"] || 0) * 100).toFixed(2)}%
            </span>
          </div>
        </div>

        {/* ── LEVEL 2: COARSE 5 CATEGORIES PROBABILITY BARS ────────────── */}
        <div className="p-4 rounded-2xl border border-white/5 bg-black/40 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-brand-500/20 text-brand-300">
                Stage 2
              </span>
              <span className="text-xs font-semibold text-white">
                Coarse Pathological Groups
              </span>
            </div>
            <span className="text-xs font-mono text-neutral-400">
              Primary: <strong className="text-white">{result.coarse.prediction}</strong>
            </span>
          </div>

          <div className="flex flex-col gap-2 mt-1">
            {coarseEntries.map(([groupName, prob]) => {
              const pct = (prob * 100).toFixed(1);
              const colorConfig = COARSE_COLORS[groupName] || {
                bg: "bg-neutral-500/10",
                fill: "bg-neutral-400",
                text: "text-neutral-300",
              };
              const isSelected = result.coarse.prediction === groupName;

              return (
                <div key={groupName} className="flex flex-col gap-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span
                      className={`${
                        isSelected ? "text-white font-semibold" : "text-neutral-400"
                      }`}
                    >
                      {groupName}
                    </span>
                    <span className={colorConfig.text}>{pct}%</span>
                  </div>
                  <div className="w-full h-1.5 rounded-full bg-white/5 overflow-hidden">
                    <div
                      style={{ width: `${pct}%` }}
                      className={`h-full rounded-full transition-all duration-700 ease-[cubic-bezier(0.32,0.72,0,1)] ${colorConfig.fill}`}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ── LEVEL 3: FINE 22-SUBCLASS TOP PREDICTIONS ────────────────── */}
        <div className="p-4 rounded-2xl border border-white/5 bg-black/40 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-accent-purple/20 text-accent-purple">
                Stage 3
              </span>
              <span className="text-xs font-semibold text-white">
                Fine Subclass Ranking (Top 3)
              </span>
            </div>
            <span className="text-xs font-mono text-accent-purple font-semibold">
              Top: {result.fine.prediction}
            </span>
          </div>

          <div className="flex flex-col gap-2 mt-1">
            {result.fine.top_3.map((item, idx) => (
              <div
                key={item.class}
                className="p-2.5 rounded-xl bg-white/[0.02] border border-white/5 flex items-center justify-between"
              >
                <div className="flex items-center gap-2.5">
                  <span className="w-5 h-5 rounded-full bg-white/5 flex items-center justify-center text-[10px] font-mono text-neutral-400">
                    #{idx + 1}
                  </span>
                  <div className="flex flex-col">
                    <span className="text-xs font-medium text-white">
                      {item.class}
                    </span>
                    <span className="text-[10px] font-mono text-neutral-500">
                      Parent: {item.parent}
                    </span>
                  </div>
                </div>
                <span className="text-xs font-mono font-semibold text-brand-300">
                  {(item.probability * 100).toFixed(2)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* ── HIERARCHICAL CONSISTENCY & PROVENANCE ──────────────────────── */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Consistency Indicator */}
          <div className="p-3.5 rounded-2xl border border-white/5 bg-black/30 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle
                className={`w-4 h-4 ${
                  result.consistency.is_consistent
                    ? "text-emerald-400"
                    : "text-amber-400"
                }`}
                strokeWidth={1.25}
              />
              <span className="text-xs text-neutral-300">
                Hierarchical Consistency
              </span>
            </div>
            <span
              className={`text-xs font-mono font-medium ${
                result.consistency.is_consistent
                  ? "text-emerald-400"
                  : "text-amber-400"
              }`}
            >
              {result.consistency.is_consistent ? "Valid (Aligned)" : "Discrepancy"}
            </span>
          </div>

          {/* Model Inference Latency & Device */}
          <div className="p-3.5 rounded-2xl border border-white/5 bg-black/30 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-brand-400" strokeWidth={1.25} />
              <span className="text-xs text-neutral-300">Compute Speed</span>
            </div>
            <span className="text-xs font-mono text-neutral-200">
              {result.inference_time_ms.toFixed(1)} ms (
              {(1000 / Math.max(result.inference_time_ms, 1)).toFixed(0)} FPS)
            </span>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
