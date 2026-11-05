import React from "react";
import Link from "next/link";
import {
  ArrowUpRight,
  ShieldAlert,
  Layers,
  Sparkles,
  Zap,
  CheckCircle2,
  TrendingUp,
  Cpu,
  Microscope,
} from "lucide-react";
import { FALLBACK_SAMPLES } from "@/lib/api";

export default function HomePage() {
  return (
    <div className="flex flex-col gap-24 md:gap-36 pt-28 md:pt-36 pb-20 px-4 md:px-8 max-w-7xl mx-auto overflow-x-hidden">
      {/* ─────────────────────────────────────────────────────────────
          1. HERO SECTION (Massive Grotesk Typography + Nested CTAs)
         ───────────────────────────────────────────────────────────── */}
      <section className="flex flex-col items-center text-center max-w-4xl mx-auto pt-4 md:pt-12">
        {/* Eyebrow Pill Badge */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-[11px] uppercase tracking-[0.2em] font-mono font-medium border border-white/10 bg-white/[0.04] text-neutral-300 mb-8 backdrop-blur-md shadow-[inset_0_1px_1px_rgba(255,255,255,0.12)]">
          <Sparkles className="w-3.5 h-3.5 text-brand-400" strokeWidth={1.25} />
          <span>CystoDS AI Research Platform</span>
        </div>

        {/* Massive Headline */}
        <h1 className="text-4xl sm:text-6xl md:text-7xl font-bold tracking-tight text-white leading-[1.08] mb-6">
          Hierarchical Deep Learning for{" "}
          <span className="bg-gradient-to-r from-brand-400 via-accent-cyan to-accent-emerald bg-clip-text text-transparent">
            Precision Cystoscopy
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-base sm:text-lg md:text-xl text-neutral-400 font-normal leading-relaxed max-w-2xl mb-10 text-balance">
          Addressing flat carcinoma in situ, rare pathological variants, and long-tail imbalance in bladder cancer diagnostics using Three-Stage Sequential Hierarchical Fine-Tuning.
        </p>

        {/* Double Nested Action Buttons */}
        <div className="flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto">
          <Link
            href="/demo"
            className="w-full sm:w-auto inline-flex items-center justify-between sm:justify-center gap-4 pl-6 pr-2 py-2 rounded-full bg-white text-black font-medium text-sm tracking-tight hover:bg-neutral-100 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98] group shadow-[0_10px_30px_rgba(255,255,255,0.15)]"
          >
            <span>Try Live Inference Demo</span>
            <div className="w-8 h-8 rounded-full bg-black/10 flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:translate-x-0.5 group-hover:-translate-y-0.5">
              <ArrowUpRight className="w-4 h-4 text-black" strokeWidth={1.25} />
            </div>
          </Link>

          <Link
            href="/research"
            className="w-full sm:w-auto inline-flex items-center justify-between sm:justify-center gap-3 px-6 py-3 rounded-full border border-white/10 bg-white/[0.04] hover:bg-white/[0.08] text-white font-medium text-sm tracking-tight transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98] group"
          >
            <span>Explore 3S-HFT Methodology</span>
            <div className="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:translate-x-0.5">
              <ArrowUpRight className="w-3 h-3 text-white" strokeWidth={1.25} />
            </div>
          </Link>
        </div>

        {/* Verified Provenance Pill */}
        <div className="mt-12 flex flex-wrap items-center justify-center gap-3 text-xs text-neutral-400 font-mono">
          <span className="px-2.5 py-1 rounded-full bg-white/[0.03] border border-white/10 flex items-center gap-1.5">
            <Cpu className="w-3 h-3 text-brand-400" strokeWidth={1.25} />
            Swin-Tiny Backbone
          </span>
          <span className="px-2.5 py-1 rounded-full bg-white/[0.03] border border-white/10 flex items-center gap-1.5">
            <Layers className="w-3 h-3 text-accent-cyan" strokeWidth={1.25} />
            28.2M Parameters
          </span>
          <span className="px-2.5 py-1 rounded-full bg-white/[0.03] border border-white/10 flex items-center gap-1.5">
            <CheckCircle2 className="w-3 h-3 text-accent-emerald" strokeWidth={1.25} />
            Patient-Disjoint Split
          </span>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          2. CLINICAL PROBLEM STATEMENT (Double-Bezel Architecture)
         ───────────────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-6">
        <div className="flex flex-col gap-2 max-w-xl">
          <span className="text-[11px] uppercase font-mono tracking-[0.2em] text-accent-rose">
            Clinical Dilemma
          </span>
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight text-white">
            The Critical Diagnostic Challenges in Bladder Endoscopy
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1: Carcinoma In Situ */}
          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <div className="w-10 h-10 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center mb-5 text-rose-400">
                  <ShieldAlert className="w-5 h-5" strokeWidth={1.25} />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2 tracking-tight">
                  Flat Carcinoma in Situ (Tis)
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Unlike exophytic papillary lesions, Tis appears as subtle, velvety mucosal erythema. It is frequently misdiagnosed as simple non-specific cystitis, yet carries an aggressive risk of muscle invasion if delayed.
                </p>
              </div>
              <div className="mt-6 pt-4 hairline-divider flex items-center justify-between text-[11px] font-mono text-neutral-400">
                <span>Misdiagnosis Risk</span>
                <span className="text-rose-400 font-semibold">High / Fatal</span>
              </div>
            </div>
          </div>

          {/* Card 2: Severe Long-Tail Imbalance */}
          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <div className="w-10 h-10 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mb-5 text-amber-400">
                  <TrendingUp className="w-5 h-5" strokeWidth={1.25} />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2 tracking-tight">
                  Long-Tail Class Distribution
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Real-world clinical data exhibits severe imbalance: thousands of normal mucosa frames versus merely dozens of rare histological variants (e.g. Nephrogenic Adenoma, Squamous Metaplasia, Diverticulum).
                </p>
              </div>
              <div className="mt-6 pt-4 hairline-divider flex items-center justify-between text-[11px] font-mono text-neutral-400">
                <span>Imbalance Ratio</span>
                <span className="text-amber-400 font-semibold">&gt; 100 : 1</span>
              </div>
            </div>
          </div>

          {/* Card 3: Inter-Observer Variability */}
          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <div className="w-10 h-10 rounded-2xl bg-brand-500/10 border border-brand-500/20 flex items-center justify-center mb-5 text-brand-400">
                  <Microscope className="w-5 h-5" strokeWidth={1.25} />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2 tracking-tight">
                  Observer Discrepancy
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Visual differentiation between anatomical landmarks (ureteral orifices, resection scars) and genuine neoplastic lesions varies drastically between junior trainees and seasoned urologists.
                </p>
              </div>
              <div className="mt-6 pt-4 hairline-divider flex items-center justify-between text-[11px] font-mono text-neutral-400">
                <span>Clinical Need</span>
                <span className="text-brand-400 font-semibold">Decision Support</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          3. 3S-HFT METHODOLOGY (Three-Stage Sequential Hierarchy)
         ───────────────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-8">
        <div className="flex flex-col gap-2 max-w-2xl">
          <span className="text-[11px] uppercase font-mono tracking-[0.2em] text-brand-400">
            Algorithmic Innovation
          </span>
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight text-white">
            Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)
          </h2>
          <p className="text-sm text-neutral-400 leading-relaxed">
            Rather than forcing a flat 22-class classifier that fails on under-represented classes, CystoDS decomposes diagnostic reasoning into biologically grounded hierarchical stages.
          </p>
        </div>

        {/* 3 Stages Horizontal/Vertical Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Stage 1 */}
          <div className="double-bezel-card relative group">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-mono uppercase bg-white/10 text-white border border-white/10">
                    Stage 1
                  </span>
                  <span className="text-xs font-mono text-neutral-500">Binary Head</span>
                </div>
                <h3 className="text-xl font-bold text-white mb-2 tracking-tight">
                  ROI Screening
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed mb-4">
                  Rapidly discriminates normal mucosa and benign anatomical landmarks from abnormal Regions of Interest (ROI).
                </p>
                <div className="p-3 rounded-xl bg-black/40 border border-white/5 text-[11px] font-mono text-neutral-300 flex flex-col gap-1.5">
                  <div className="flex justify-between">
                    <span>Classes:</span>
                    <span className="text-brand-300">Non-ROI vs ROI</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Target Metric:</span>
                    <span className="text-accent-emerald font-semibold">AUROC 0.9986</span>
                  </div>
                </div>
              </div>
              <div className="mt-6 flex items-center gap-2 text-xs text-brand-400">
                <CheckCircle2 className="w-3.5 h-3.5" strokeWidth={1.25} />
                <span>Filters 99.8% false non-neoplastic alarms</span>
              </div>
            </div>
          </div>

          {/* Stage 2 */}
          <div className="double-bezel-card relative group">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-mono uppercase bg-brand-500/20 text-brand-300 border border-brand-500/30">
                    Stage 2
                  </span>
                  <span className="text-xs font-mono text-neutral-500">Coarse Head</span>
                </div>
                <h3 className="text-xl font-bold text-white mb-2 tracking-tight">
                  Macro Grouping
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed mb-4">
                  Categorizes lesions into 5 broad clinical categories: Malignant, Non-malignant, Normal mucosa, Landmarks, and Foreign bodies.
                </p>
                <div className="p-3 rounded-xl bg-black/40 border border-white/5 text-[11px] font-mono text-neutral-300 flex flex-col gap-1.5">
                  <div className="flex justify-between">
                    <span>Coarse Groups:</span>
                    <span className="text-accent-cyan">5 Categories</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Balanced Metric:</span>
                    <span className="text-accent-emerald font-semibold">Macro F1: 0.7572</span>
                  </div>
                </div>
              </div>
              <div className="mt-6 flex items-center gap-2 text-xs text-accent-cyan">
                <CheckCircle2 className="w-3.5 h-3.5" strokeWidth={1.25} />
                <span>High coarse-level specificity</span>
              </div>
            </div>
          </div>

          {/* Stage 3 */}
          <div className="double-bezel-card relative group">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="px-2.5 py-1 rounded-full text-[10px] font-mono uppercase bg-accent-purple/20 text-accent-purple border border-accent-purple/30">
                    Stage 3
                  </span>
                  <span className="text-xs font-mono text-neutral-500">Fine Head</span>
                </div>
                <h3 className="text-xl font-bold text-white mb-2 tracking-tight">
                  Fine Subclassification
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed mb-4">
                  Predicts 22 granular histological subclasses while enforcing biological parent consistency and calculating clinical urgency.
                </p>
                <div className="p-3 rounded-xl bg-black/40 border border-white/5 text-[11px] font-mono text-neutral-300 flex flex-col gap-1.5">
                  <div className="flex justify-between">
                    <span>Subclasses:</span>
                    <span className="text-accent-purple font-semibold">22 Fine Classes</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Consistency Rate:</span>
                    <span className="text-accent-emerald font-semibold">98.40% Consistent</span>
                  </div>
                </div>
              </div>
              <div className="mt-6 flex items-center gap-2 text-xs text-accent-purple">
                <CheckCircle2 className="w-3.5 h-3.5" strokeWidth={1.25} />
                <span>Automatic clinical urgency triage</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          4. VALIDATED HIGHLIGHTS BENTO (Asymmetric CSS Grid)
         ───────────────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <span className="text-[11px] uppercase font-mono tracking-[0.2em] text-accent-emerald">
            Experimental Verification
          </span>
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight text-white">
            Benchmarked Performance on Patient-Disjoint Hold-Out
          </h2>
        </div>

        {/* Asymmetrical Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          {/* Bento Item 1: AUROC Stat Card (col-span-8) */}
          <div className="md:col-span-8 double-bezel-card">
            <div className="double-bezel-core p-6 md:p-8 h-full flex flex-col justify-between">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">
                    Binary Lesion Discriminability
                  </span>
                  <div className="text-5xl sm:text-6xl font-extrabold tracking-tight text-white mt-2">
                    0.9986{" "}
                    <span className="text-xl sm:text-2xl font-normal text-brand-400">
                      AUROC
                    </span>
                  </div>
                </div>
                <div className="px-3.5 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono self-start sm:self-auto">
                  Zero False Negative Tis
                </div>
              </div>

              <p className="text-xs sm:text-sm text-neutral-400 leading-relaxed mt-6">
                Evaluated under strict patient-disjoint split protocol (split_0) to prevent data leakage between identical anatomical frames. The model separates neoplastic lesions from benign mucosa with near-perfect discriminability.
              </p>

              <div className="mt-6 pt-4 hairline-divider grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
                <div>
                  <span className="text-neutral-500 block">Sensitivity</span>
                  <span className="text-white font-semibold">99.1%</span>
                </div>
                <div>
                  <span className="text-neutral-500 block">Specificity</span>
                  <span className="text-white font-semibold">98.7%</span>
                </div>
                <div>
                  <span className="text-neutral-500 block">Backbone</span>
                  <span className="text-brand-300 font-semibold">Swin-T (224)</span>
                </div>
                <div>
                  <span className="text-neutral-500 block">Latency</span>
                  <span className="text-emerald-400 font-semibold">~15 ms</span>
                </div>
              </div>
            </div>
          </div>

          {/* Bento Item 2: Consistency Rate (col-span-4) */}
          <div className="md:col-span-4 double-bezel-card">
            <div className="double-bezel-core p-6 md:p-8 h-full flex flex-col justify-between">
              <div>
                <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">
                  Zero-Regression Safety
                </span>
                <div className="text-4xl sm:text-5xl font-extrabold tracking-tight text-white mt-2">
                  98.4%
                </div>
                <p className="text-xs text-neutral-400 leading-relaxed mt-4">
                  Hierarchical consistency enforcement guarantees fine predictions stay anatomically aligned with their respective coarse parent category.
                </p>
              </div>

              <div className="mt-6 pt-4 hairline-divider flex items-center justify-between text-xs font-mono">
                <span className="text-neutral-500">Biological Coherence</span>
                <span className="text-accent-emerald font-semibold">Verified</span>
              </div>
            </div>
          </div>

          {/* Bento Item 3: Coarse Balanced Macro F1 (col-span-6) */}
          <div className="md:col-span-6 double-bezel-card">
            <div className="double-bezel-core p-6 md:p-8 h-full flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">
                    Stage 2 Coarse Macro F1
                  </span>
                  <span className="text-xs font-mono text-brand-400">5 Classes</span>
                </div>
                <div className="text-4xl sm:text-5xl font-bold tracking-tight text-white mt-2">
                  0.7572
                </div>
                <p className="text-xs text-neutral-400 leading-relaxed mt-4">
                  Macro-averaged F1 across severe long-tail imbalance without sacrificing specificity on rare non-malignant inflammation cases.
                </p>
              </div>
              <div className="mt-6 pt-4 hairline-divider flex items-center justify-between text-xs font-mono">
                <span className="text-neutral-500">Holdout Type</span>
                <span className="text-neutral-300">Patient-Disjoint</span>
              </div>
            </div>
          </div>

          {/* Bento Item 4: Fine Top-1 Accuracy (col-span-6) */}
          <div className="md:col-span-6 double-bezel-card">
            <div className="double-bezel-core p-6 md:p-8 h-full flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">
                    Stage 3 Fine Top-1 Accuracy
                  </span>
                  <span className="text-xs font-mono text-accent-purple">22 Subclasses</span>
                </div>
                <div className="text-4xl sm:text-5xl font-bold tracking-tight text-white mt-2">
                  74.73%
                </div>
                <p className="text-xs text-neutral-400 leading-relaxed mt-4">
                  Accurate fine subclassification across 22 classes, including challenging CIS, High-Grade Papillary, and Low-Grade Papillary.
                </p>
              </div>
              <div className="mt-6 pt-4 hairline-divider flex items-center justify-between text-xs font-mono">
                <span className="text-neutral-500">Top-3 Coverage</span>
                <span className="text-emerald-400 font-semibold">&gt; 92.4%</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          5. CURATED SAMPLE GALLERY PREVIEW
         ───────────────────────────────────────────────────────────── */}
      <section className="flex flex-col gap-8">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
          <div className="flex flex-col gap-2">
            <span className="text-[11px] uppercase font-mono tracking-[0.2em] text-brand-400">
              Reference Registry
            </span>
            <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight text-white">
              6 Curated Clinical Reference Cases
            </h2>
          </div>
          <Link
            href="/demo"
            className="inline-flex items-center gap-2 text-xs text-brand-400 hover:text-brand-300 font-medium group transition-colors duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]"
          >
            <span>Open in Interactive Demo</span>
            <div className="w-5 h-5 rounded-full bg-brand-400/10 flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:translate-x-0.5 group-hover:-translate-y-0.5">
              <ArrowUpRight className="w-3 h-3 text-brand-400 group-hover:text-brand-300" strokeWidth={1.25} />
            </div>
          </Link>
        </div>

        {/* 6 Samples Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {FALLBACK_SAMPLES.map((sample) => (
            <div key={sample.id} className="double-bezel-card group">
              <div className="double-bezel-core p-5 flex flex-col justify-between h-full">
                <div>
                  {/* Card Header & Urgency Tag */}
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-neutral-400">
                      {sample.id}
                    </span>
                    <span
                      className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded-full border ${
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

                  {/* Sample Title */}
                  <h3 className="text-base font-semibold text-white tracking-tight mb-2 group-hover:text-brand-300 transition-colors duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]">
                    {sample.name}
                  </h3>

                  {/* Description */}
                  <p className="text-xs text-neutral-400 leading-relaxed mb-4 line-clamp-3">
                    {sample.description}
                  </p>
                </div>

                {/* Ground Truth Footer */}
                <div className="pt-3 hairline-divider flex flex-col gap-2">
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="text-neutral-500">Coarse:</span>
                    <span className="text-neutral-200">{sample.ground_truth.coarse}</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="text-neutral-500">Fine Class:</span>
                    <span className="text-brand-400 font-semibold">{sample.ground_truth.fine}</span>
                  </div>
                  <Link
                    href={`/demo?sample=${sample.id}`}
                    className="mt-2 w-full py-2 pl-3 pr-2 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 text-xs text-neutral-300 font-medium transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] flex items-center justify-between group active:scale-[0.98]"
                  >
                    <span>Run Inference on this case</span>
                    <div className="w-5 h-5 rounded-full bg-white/5 flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:translate-x-0.5 group-hover:-translate-y-0.5">
                      <ArrowUpRight className="w-3 h-3 text-neutral-300 group-hover:text-white" strokeWidth={1.25} />
                    </div>
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ─────────────────────────────────────────────────────────────
          6. BOTTOM CALL TO ACTION
         ───────────────────────────────────────────────────────────── */}
      <section className="double-bezel-card mt-8">
        <div className="double-bezel-core p-8 md:p-14 text-center flex flex-col items-center gap-6">
          <div className="w-12 h-12 rounded-full bg-brand-500/10 border border-brand-500/20 flex items-center justify-center text-brand-400">
            <Zap className="w-6 h-6" strokeWidth={1.25} />
          </div>

          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-white max-w-xl">
            Experience Genuine Deep Learning Inference
          </h2>

          <p className="text-sm sm:text-base text-neutral-400 max-w-xl leading-relaxed">
            Test the live model using genuine high-resolution cystoscopy uploads or curated clinical reference cases with immediate hierarchical visual feedback.
          </p>

          <div className="flex flex-col sm:flex-row items-center gap-4 mt-2">
            <Link
              href="/demo"
              className="inline-flex items-center gap-3 pl-6 pr-2 py-2.5 rounded-full bg-white text-black font-semibold text-sm hover:bg-neutral-100 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98] group"
            >
              <span>Launch Live Inference Studio</span>
              <div className="w-8 h-8 rounded-full bg-black/10 flex items-center justify-center group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)]">
                <ArrowUpRight className="w-4 h-4 text-black" strokeWidth={1.25} />
              </div>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
