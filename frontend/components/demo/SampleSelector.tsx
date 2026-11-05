"use client";

import React from "react";
import { SampleItem } from "@/types";
import { Check } from "lucide-react";

interface SampleSelectorProps {
  samples: SampleItem[];
  selectedSampleId: string | null;
  onSelect: (sample: SampleItem) => void;
  disabled?: boolean;
}

export default function SampleSelector({
  samples,
  selectedSampleId,
  onSelect,
  disabled = false,
}: SampleSelectorProps) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-neutral-300 uppercase tracking-wider font-mono">
          Curated Reference Cases (6)
        </span>
        <span className="text-[10px] font-mono text-neutral-500">
          Gold-standard histopathology
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        {samples.map((sample) => {
          const isSelected = selectedSampleId === sample.id;
          return (
            <button
              key={sample.id}
              type="button"
              disabled={disabled}
              onClick={() => onSelect(sample)}
              className={`p-3 rounded-2xl border text-left transition-all duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] relative flex flex-col justify-between gap-2 ${
                isSelected
                  ? "border-brand-500/50 bg-brand-500/[0.08] shadow-[0_0_20px_rgba(38,143,255,0.15)] ring-1 ring-brand-500/30"
                  : "border-white/5 hover:border-white/15 bg-white/[0.02] hover:bg-white/[0.04]"
              } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded-md bg-white/5 text-neutral-400">
                    {sample.stage !== "None" ? sample.stage : "Benign"}
                  </span>
                  <span
                    className={`text-[9px] font-mono uppercase px-1.5 py-0.5 rounded-md border ${
                      sample.urgency === "CRITICAL"
                        ? "bg-rose-500/10 border-rose-500/30 text-rose-400"
                        : sample.urgency === "HIGH"
                        ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                        : sample.urgency === "MEDIUM"
                        ? "bg-yellow-500/10 border-yellow-500/30 text-yellow-300"
                        : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                    }`}
                  >
                    {sample.urgency}
                  </span>
                </div>

                {isSelected && (
                  <div className="w-4 h-4 rounded-full bg-brand-500 flex items-center justify-center text-white shrink-0">
                    <Check className="w-2.5 h-2.5" strokeWidth={1.25} />
                  </div>
                )}
              </div>

              <div>
                <span className="text-xs font-medium text-white line-clamp-1">
                  {sample.name}
                </span>
                <p className="text-[11px] text-neutral-400 line-clamp-1 mt-0.5">
                  {sample.ground_truth.coarse} • {sample.ground_truth.fine}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
