import React from "react";
import { Users } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="flex flex-col gap-12 pt-28 md:pt-36 pb-20 px-4 md:px-8 max-w-4xl mx-auto overflow-x-hidden">
      <div className="flex flex-col gap-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-mono uppercase tracking-[0.2em] font-medium border border-white/10 bg-white/[0.04] text-brand-400 w-max">
          <Users className="w-3.5 h-3.5" strokeWidth={1.25} />
          <span>Research Consortium</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-white leading-tight">
          About CystoDS Research
        </h1>
        <p className="text-sm sm:text-base text-neutral-400 leading-relaxed">
          The CystoDS project is dedicated to engineering reproducible, explainable, and biologically coherent deep learning architectures for computer-aided cystoscopy and early bladder cancer detection.
        </p>
      </div>

      <div className="double-bezel-card">
        <div className="double-bezel-core p-6 md:p-8 flex flex-col gap-6">
          <h2 className="text-xl font-bold text-white">Scientific Mission</h2>
          <p className="text-xs sm:text-sm text-neutral-300 leading-relaxed">
            Bladder cancer is one of the most resource-intensive malignancies due to lifelong endoscopic surveillance requirements. Missed flat lesions (Tis) lead to preventable muscle invasion, while false-positive alarms result in unnecessary invasive biopsies. Our goal is providing objective diagnostic assistance with mathematically bounded hierarchical consistency.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 hairline-divider text-xs font-mono">
            <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5">
              <span className="text-neutral-500 block mb-1">Study Focus</span>
              <span className="text-white font-semibold">Long-Tail Hierarchy</span>
            </div>
            <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5">
              <span className="text-neutral-500 block mb-1">Backbone Network</span>
              <span className="text-brand-300 font-semibold">Swin-Tiny 224</span>
            </div>
            <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5">
              <span className="text-neutral-500 block mb-1">Validation</span>
              <span className="text-emerald-400 font-semibold">Patient-Disjoint</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
