"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  BookOpen,
  ArrowUpRight,
  ShieldAlert,
  Layers,
  GitBranch,
  Cpu,
  Activity,
  Sparkles,
  Filter,
  AlertCircle,
} from "lucide-react";

export default function ResearchPage() {
  const [activeStage, setActiveStage] = useState<1 | 2 | 3>(1);
  const [selectedCoarseFilter, setSelectedCoarseFilter] = useState<string>("ALL");

  const STAGES = [
    {
      id: 1,
      title: "Stage 1: Representation Learning",
      subtitle: "Multi-Task Feature Learning + 128-d SupCon Head",
      backboneStatus: "Trainable (Full Backbone + Heads)",
      lossFormula: "L_Stage1 = α · L_binary + β · L_coarse + γ · L_fine + μ · L_SupCon",
      lossDescription:
        "Joint optimization combining hierarchical multi-task cross-entropy (α=0.2, β=0.3, γ=0.5) with supervised contrastive loss (μ=0.1) over a 128-d hyperspherical projection head. Endoscopic circular borders are stripped via CenterFractionCrop(0.92) to eliminate vignette bias.",
      target: "Swin-Tiny 224×224 Hierarchical Multi-Scale Features (768-d)",
      keyOutcome: "Robust cluster separation between high-grade malignancies and normal urothelium.",
    },
    {
      id: 2,
      title: "Stage 2: Coarse Category Alignment",
      subtitle: "Smoothed Balanced Softmax with Frozen Backbone",
      backboneStatus: "Frozen (∇θ = 0, Backbone weights locked)",
      lossFormula: "L_Stage2 = - log [ (π_k^τ · e^{z_k}) / ∑ (π_j^τ · e^{z_j}) ]",
      lossDescription:
        "Smoothed Balanced Softmax addresses severe class imbalance across the 5 coarse groupings. Class frequencies π_k are smoothed with temperature τ = 0.5 to prevent majority classes (Normal/Artifact) from drowning rare pathology gradients.",
      target: "5 Coarse Decision Boundaries (Malignant, Non-Malignant, Landmark, Foreign Body, Normal)",
      keyOutcome: "Coarse Balanced Accuracy reaches 91.4% with zero feature manifold degradation.",
    },
    {
      id: 3,
      title: "Stage 3: Fine Head Tuning",
      subtitle: "Decoupled Fine Tuning + Prior & Consistency Regularization",
      backboneStatus: "Frozen Backbone, Active Fine Head Optimization",
      lossFormula: "ŷ_fine = argmax_c [ z_c - τ · log(π_c) - λ · I(CoarseParent(c) ≠ ŷ_coarse) ]",
      lossDescription:
        "Decoupled fine classification head calibrated with class-prior shift. A hierarchical consistency penalty (λ = 0.2) strictly forbids contradictory biological predictions (e.g. predicting HighGradePapillary under a Benign parent).",
      target: "22 Granular Histology Subclasses",
      keyOutcome: "Biological Coherence Rate hits 98.40%, Fine Top-1 Accuracy reaches 74.73%.",
    },
  ];

  const TAXONOMY_DATA = [
    // Malignant
    {
      id: "cis",
      fine: "Carcinoma in situ (Tis)",
      coarse: "Malignant",
      urgency: "CRITICAL",
      count: 48,
      description: "Flat, high-grade non-invasive urothelial malignancy. Velvety erythematous mucosal patches mimicking benign inflammation.",
      surgicalImplication: "Urgent biopsy under Blue-Light / PDD cystoscopy; BCG induction immunotherapy. High progression risk (>54%).",
      badgeColor: "rose",
    },
    {
      id: "high_grade_pap",
      fine: "High-Grade Papillary Carcinoma",
      coarse: "Malignant",
      urgency: "CRITICAL",
      count: 142,
      description: "Exophytic papillary tumor with nuclear pleomorphism, high architectural atypia, and delicate fibrovascular stalks.",
      surgicalImplication: "Immediate Transurethral Resection of Bladder Tumor (TURBT) with detrusor muscle sampling + single-dose chemo.",
      badgeColor: "rose",
    },
    {
      id: "low_grade_pap",
      fine: "Low-Grade Papillary Carcinoma",
      coarse: "Malignant",
      urgency: "HIGH",
      count: 186,
      description: "Ordered papillary branching with minimal cytological atypia; stalks lined by urothelium showing uniform polarity.",
      surgicalImplication: "TURBT resection or outpatient cystoscopic fulguration followed by active endoscopic surveillance schedule.",
      badgeColor: "rose",
    },
    {
      id: "premalignant",
      fine: "PreMalignant / Dysplasia",
      coarse: "Malignant",
      urgency: "HIGH",
      count: 34,
      description: "Urothelial dysplasia with marked cytological atypia lacking the full epithelial thickness involvement of Tis.",
      surgicalImplication: "Quadrantic mapping biopsies, urine cytology, and repeat surveillance cystoscopy at 3-month intervals.",
      badgeColor: "rose",
    },

    // Non-Malignant
    {
      id: "ccg",
      fine: "Cystitis Cystica et Glandularis",
      coarse: "Non-Malignant",
      urgency: "MEDIUM",
      count: 98,
      description: "Common benign reactive proliferation of von Brunn nests undergoing glandular or cystic lumina metaplasia.",
      surgicalImplication: "Conservative surveillance; biopsy only if nodular morphology cannot be differentiated from adenocarcinoma.",
      badgeColor: "amber",
    },
    {
      id: "inflammation_nos",
      fine: "Inflammation NOS (Erythema)",
      coarse: "Non-Malignant",
      urgency: "MEDIUM",
      count: 215,
      description: "Non-specific mucosal hyperemia, vascular congestion, or chronic interstitial inflammatory infiltration.",
      surgicalImplication: "Urinalysis and antibiotic or anti-inflammatory management; follow-up cystoscopy if redness persists >6 weeks.",
      badgeColor: "amber",
    },
    {
      id: "denuded_mucosa",
      fine: "Denuded Mucosa",
      coarse: "Non-Malignant",
      urgency: "MEDIUM",
      count: 64,
      description: "Loss of umbrella cell layer or full-thickness epithelial shedding due to catheterization or mechanical shear.",
      surgicalImplication: "Careful selective cytology to exclude discohesive shedding of underlying Carcinoma in situ cells.",
      badgeColor: "amber",
    },
    {
      id: "papilloma",
      fine: "Urothelial Papilloma",
      coarse: "Non-Malignant",
      urgency: "LOW",
      count: 28,
      description: "True benign neoplasm composed of delicate papillary fronds lined by cytologically normal mature urothelium.",
      surgicalImplication: "Complete transurethral excision; benign natural course with low recurrence rate (<8%).",
      badgeColor: "amber",
    },
    {
      id: "nephrogenic_adenoma",
      fine: "Nephrogenic Adenoma",
      coarse: "Non-Malignant",
      urgency: "MEDIUM",
      count: 18,
      description: "Rare benign metaplastic lesion composed of tubules, cysts, and papillae, often post-instrumentation or renal transplant.",
      surgicalImplication: "Biopsy confirmation to rule out clear cell urothelial carcinoma; conservative transurethral resection.",
      badgeColor: "amber",
    },
    {
      id: "benign_nos",
      fine: "Benign NOS / Quiescent",
      coarse: "Non-Malignant",
      urgency: "LOW",
      count: 340,
      description: "Normal-appearing resting urothelium without architectural distortion, prominent capillaries, or mucosal edema.",
      surgicalImplication: "Routine endoscopic surveillance documented; no surgical biopsy or intervention indicated.",
      badgeColor: "amber",
    },

    // Landmarks
    {
      id: "ureteral_orifice",
      fine: "Ureteral Orifice (Right/Left)",
      coarse: "Landmarks",
      urgency: "NORMAL",
      count: 190,
      description: "Slit-like or horseshoe opening on the interureteric ridge emitting periodic peristaltic urine jets.",
      surgicalImplication: "Critical anatomical reference; preserve at least 5mm safety margin during resection to prevent ureteral stenosis.",
      badgeColor: "cyan",
    },
    {
      id: "resection_bed",
      fine: "Active Resection Bed",
      coarse: "Landmarks",
      urgency: "NORMAL",
      count: 76,
      description: "Raw circular detrusor muscle surface immediately post-cauterization or transurethral resection.",
      surgicalImplication: "Hemostasis verification; inspection of muscle fibers to confirm adequate depth staging without perforation.",
      badgeColor: "cyan",
    },
    {
      id: "resection_scar",
      fine: "Healed Resection Scar",
      coarse: "Landmarks",
      urgency: "NORMAL",
      count: 154,
      description: "Pale, fibrotic, stellate mucosal contracture at previous TURBT site with re-epithelialized glistening surface.",
      surgicalImplication: "Targeted inspection of scar periphery to detect early marginal cancer recurrence.",
      badgeColor: "cyan",
    },
    {
      id: "trabeculation",
      fine: "Muscular Trabeculation",
      coarse: "Landmarks",
      urgency: "NORMAL",
      count: 122,
      description: "Hypertrophied criss-crossing detrusor muscle bundles characteristic of chronic bladder outlet obstruction.",
      surgicalImplication: "Check for hidden tumor fronds within deep pseudodiverticula created between trabecular ridges.",
      badgeColor: "cyan",
    },

    // Foreign Bodies
    {
      id: "air_bubble",
      fine: "Air Bubble (Optical Artifact)",
      coarse: "Foreign Bodies",
      urgency: "NORMAL",
      count: 110,
      description: "Trapped gas bubble at the anterior bladder dome displaying specular white reflection and circular black meniscus.",
      surgicalImplication: "Optical artifact to be rejected by the AI perception filter; no biological tissue correlation.",
      badgeColor: "purple",
    },
    {
      id: "resection_loop",
      fine: "Resection Loop / Electrode",
      coarse: "Foreign Bodies",
      urgency: "NORMAL",
      count: 45,
      description: "Tungsten wire semicircular cutting loop extending through resectoscope sheath for electrocautery tissue shaving.",
      surgicalImplication: "Instrument tracking; visual confirmation of tool-tissue contact during active surgical cut phase.",
      badgeColor: "purple",
    },
    {
      id: "stent",
      fine: "Double-J Ureteral Stent",
      coarse: "Foreign Bodies",
      urgency: "NORMAL",
      count: 38,
      description: "Polyurethane or silicone coiled stent curl extending from ureteral orifice into bladder lumen.",
      surgicalImplication: "Verify stent positioning, evaluate for encrustation, and schedule cystoscopic stent retrieval.",
      badgeColor: "purple",
    },
  ];

  const filteredTaxonomy =
    selectedCoarseFilter === "ALL"
      ? TAXONOMY_DATA
      : TAXONOMY_DATA.filter((item) => item.coarse.toUpperCase() === selectedCoarseFilter);

  return (
    <div className="flex flex-col gap-16 md:gap-24 pt-28 md:pt-36 pb-24 px-4 md:px-8 max-w-6xl mx-auto overflow-x-hidden">
      {/* ── Page Header ────────────────────────────────────────────── */}
      <div className="flex flex-col gap-4 max-w-3xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-mono uppercase tracking-[0.2em] font-medium border border-white/10 bg-white/[0.04] text-brand-400 w-max">
          <BookOpen className="w-3.5 h-3.5" strokeWidth={1.25} />
          <span>Scientific Specification & Methodology</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-white leading-tight">
          The 3S-HFT Architecture & Protocol
        </h1>
        <p className="text-sm sm:text-base text-neutral-400 leading-relaxed">
          Three-Stage Sequential Hierarchical Fine-Tuning resolves severe class imbalance, rare histology variants, and flat lesion under-detection in computer-aided cystoscopy with mathematically bounded consistency.
        </p>
      </div>

      {/* ── 1. Problem Formulation & Clinical Dilemma ───────────────── */}
      <section className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <span className="text-[11px] uppercase font-mono tracking-[0.2em] text-brand-400">
            Section 01
          </span>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            Clinical Problem Formulation & The Diagnostic Dilemma
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <div className="w-8 h-8 rounded-full bg-rose-500/10 flex items-center justify-center text-rose-400 mb-4">
                  <AlertCircle className="w-4 h-4" strokeWidth={1.25} />
                </div>
                <h3 className="text-base font-bold text-white mb-2">
                  Flat Lesion (Tis / CIS) Vulnerability
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Unlike exophytic papillary tumors (Ta/T1) that protrude into the lumen, Carcinoma in situ (CIS) manifests as flat, velvety erythematous mucosal patches that closely mimic benign follicular cystitis. Missed CIS carries a &gt;54% risk of progression to muscle invasion (T2+).
                </p>
              </div>
              <div className="mt-6 pt-3 hairline-divider flex items-center justify-between text-[11px] font-mono">
                <span className="text-neutral-500">Recurrence Risk</span>
                <span className="text-rose-400 font-semibold">&gt; 70%</span>
              </div>
            </div>
          </div>

          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <div className="w-8 h-8 rounded-full bg-amber-500/10 flex items-center justify-center text-amber-400 mb-4">
                  <Layers className="w-4 h-4" strokeWidth={1.25} />
                </div>
                <h3 className="text-base font-bold text-white mb-2">
                  Severe Long-Tailed Distribution
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Across 22 histological subclasses, over 85% of cystoscopic surveillance video frames consist of quiescent urothelium. Rare variants such as Nephrogenic Adenoma, Inverted Papilloma, and CIS account for &lt;1% of frames, causing standard cross-entropy models to collapse on tail classes.
                </p>
              </div>
              <div className="mt-6 pt-3 hairline-divider flex items-center justify-between text-[11px] font-mono">
                <span className="text-neutral-500">Fine Subclasses</span>
                <span className="text-amber-400 font-semibold">22 Granular</span>
              </div>
            </div>
          </div>

          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <div className="w-8 h-8 rounded-full bg-cyan-500/10 flex items-center justify-center text-cyan-400 mb-4">
                  <Activity className="w-4 h-4" strokeWidth={1.25} />
                </div>
                <h3 className="text-base font-bold text-white mb-2">
                  Asymmetric Clinical Costs (FN vs FP)
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  A False Negative (FN) delays cancer resection, necessitating morbid radical cystectomy and urinary diversion. A False Positive (FP) results in unnecessary invasive electrocautery biopsies under anesthesia, urethral strictures, and elevated clinical expenditure.
                </p>
              </div>
              <div className="mt-6 pt-3 hairline-divider flex items-center justify-between text-[11px] font-mono">
                <span className="text-neutral-500">Binary Sensitivity Target</span>
                <span className="text-cyan-400 font-semibold">&gt; 98.5%</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── 2. 3S-HFT Visual Architecture & Pipeline ────────────────── */}
      <section className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <span className="text-[11px] uppercase font-mono tracking-[0.2em] text-accent-cyan">
            Section 02
          </span>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT)
          </h2>
          <p className="text-xs sm:text-sm text-neutral-400">
            Interactive pipeline representation: click through each stage to inspect the loss formulation, backbone freeze state, and calibration strategy.
          </p>
        </div>

        {/* Stage Selector Pills */}
        <div className="flex flex-wrap items-center gap-2">
          {STAGES.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setActiveStage(s.id as 1 | 2 | 3)}
              className={`px-4 py-2 rounded-full text-xs font-mono transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98] ${
                activeStage === s.id
                  ? "bg-white text-black font-semibold shadow-[0_4px_20px_rgba(255,255,255,0.2)]"
                  : "bg-white/[0.04] text-neutral-400 hover:text-white border border-white/5"
              }`}
            >
              Stage {s.id}
            </button>
          ))}
        </div>

        {/* Stage Detail Card */}
        {(() => {
          const current = STAGES.find((s) => s.id === activeStage)!;
          return (
            <div className="double-bezel-card">
              <div className="double-bezel-core p-6 md:p-8 flex flex-col gap-6">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 hairline-divider pb-6">
                  <div>
                    <div className="flex items-center gap-2 text-xs font-mono text-brand-400 mb-1">
                      <Sparkles className="w-3.5 h-3.5" strokeWidth={1.25} />
                      <span>{current.subtitle}</span>
                    </div>
                    <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
                      {current.title}
                    </h3>
                  </div>
                  <div className="px-3 py-1.5 rounded-xl border border-white/10 bg-white/[0.02] text-xs font-mono text-neutral-300 w-max shrink-0">
                    <span className="text-neutral-500">Backbone: </span>
                    <span className="text-accent-emerald font-semibold">{current.backboneStatus}</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="flex flex-col gap-4">
                    <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">
                      Optimization Objective & Details
                    </span>
                    <p className="text-xs sm:text-sm text-neutral-300 leading-relaxed">
                      {current.lossDescription}
                    </p>
                    <div className="p-4 rounded-xl bg-white/[0.02] border border-white/5 flex flex-col gap-1">
                      <span className="text-[11px] font-mono text-neutral-500">Primary Transformation Target:</span>
                      <span className="text-xs font-mono text-white font-medium">{current.target}</span>
                    </div>
                    <div className="p-4 rounded-xl bg-emerald-500/[0.03] border border-emerald-500/20 flex flex-col gap-1">
                      <span className="text-[11px] font-mono text-emerald-400 font-semibold">Key Empirical Outcome:</span>
                      <span className="text-xs text-neutral-300">{current.keyOutcome}</span>
                    </div>
                  </div>

                  <div className="flex flex-col gap-3">
                    <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">
                      Mathematical Formulation
                    </span>
                    <div className="p-5 rounded-2xl bg-black/80 border border-white/10 font-mono text-xs text-brand-300 leading-relaxed overflow-x-auto shadow-inner">
                      <div className="text-neutral-500 text-[10px] uppercase mb-2"># Objective Function</div>
                      <div>{current.lossFormula}</div>
                    </div>

                    <div className="p-4 rounded-2xl border border-white/5 bg-white/[0.02] flex flex-col gap-2 mt-2 text-xs font-mono text-neutral-400">
                      <div className="flex items-center gap-2 text-neutral-200">
                        <Cpu className="w-3.5 h-3.5 text-brand-400" strokeWidth={1.25} />
                        <span>Swin-Tiny Backbone Architecture</span>
                      </div>
                      <p className="text-[11px] text-neutral-400 font-sans leading-relaxed">
                        Hierarchical feature maps constructed via 4 stages of shifted window self-attention (W-MSA / SW-MSA), patch size 4×4, window size 7, feature channels [96, 192, 384, 768].
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}
      </section>

      {/* ── 3. Patient-Disjoint Cross-Validation Protocol ──────────── */}
      <section className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <span className="text-[11px] uppercase font-mono tracking-[0.2em] text-accent-emerald">
            Section 03
          </span>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            Patient-Disjoint Cross-Validation Protocol
          </h2>
          <p className="text-xs sm:text-sm text-neutral-400">
            Why random frame splitting is scientifically invalid in endoscopic video analysis: preventing data leakage between consecutive frames.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">
                  Partition Split 0 (Canonical)
                </span>
                <h3 className="text-lg font-bold text-white mt-1 mb-2">
                  Training Split (70%)
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Cohort A patients exclusively. Applied for backbone feature extraction, RandAugment transformations, and 3S-HFT sequential loss updates.
                </p>
              </div>
              <div className="mt-6 pt-3 hairline-divider flex items-center justify-between text-[11px] font-mono">
                <span className="text-neutral-500">Patient Isolation</span>
                <span className="text-white font-semibold">100% Disjoint</span>
              </div>
            </div>
          </div>

          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">
                  Partition Split 1
                </span>
                <h3 className="text-lg font-bold text-white mt-1 mb-2">
                  Validation Split (15%)
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Cohort B patients. Monitored strictly for early stopping, learning rate warmup cosine annealing, and prior smoothing parameter (τ = 0.5) selection.
                </p>
              </div>
              <div className="mt-6 pt-3 hairline-divider flex items-center justify-between text-[11px] font-mono">
                <span className="text-neutral-500">Early Stopping</span>
                <span className="text-accent-cyan font-semibold">Patience 15</span>
              </div>
            </div>
          </div>

          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 h-full flex flex-col justify-between">
              <div>
                <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">
                  Partition Split 2
                </span>
                <h3 className="text-lg font-bold text-white mt-1 mb-2">
                  Holdout Test Benchmark (15%)
                </h3>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Cohort C patients. Zero frames from this patient group are ever seen during parameter optimization, hyperparameter tuning, or prompt design.
                </p>
              </div>
              <div className="mt-6 pt-3 hairline-divider flex items-center justify-between text-[11px] font-mono">
                <span className="text-neutral-500">Generalization</span>
                <span className="text-emerald-400 font-semibold">Gold Standard</span>
              </div>
            </div>
          </div>
        </div>

        {/* Video Frame Leakage Warning Card */}
        <div className="p-6 rounded-2xl border border-white/10 bg-white/[0.02] flex flex-col md:flex-row items-start md:items-center justify-between gap-4 text-xs font-mono">
          <div className="flex items-start gap-3">
            <GitBranch className="w-4 h-4 text-brand-400 shrink-0 mt-0.5" strokeWidth={1.25} />
            <div className="flex flex-col gap-1">
              <span className="text-white font-semibold">Data Leakage Prevention Guardrail</span>
              <p className="text-neutral-400 font-sans text-xs max-w-3xl leading-relaxed">
                Adjacent video frames taken 33ms apart share mucosal vascular arborization, lighting geometry, and instrument angle. Random frame splitting leaks identity artifacts, artificially inflating benchmark accuracy by up to 18%. CystoDS enforces strict patient-level isolation across all 3 cross-validation splits.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── 4. 3-Level Clinical Taxonomy Explorer ───────────────────── */}
      <section className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <span className="text-[11px] uppercase font-mono tracking-[0.2em] text-accent-purple">
            Section 04
          </span>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            3-Level Clinical Taxonomy & Surgical Workflow
          </h2>
          <p className="text-xs sm:text-sm text-neutral-400">
            A comprehensive mapping from binary screening to 5 coarse groups and 22 fine-grained histology subclasses, annotated with clinical recommendations.
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono text-neutral-400 mr-2">
            <Filter className="w-3.5 h-3.5" strokeWidth={1.25} />
            <span>Filter Coarse:</span>
          </div>
          {["ALL", "MALIGNANT", "NON-MALIGNANT", "LANDMARKS", "FOREIGN BODIES"].map((cat) => (
            <button
              key={cat}
              type="button"
              onClick={() => setSelectedCoarseFilter(cat)}
              className={`px-3 py-1.5 rounded-full text-xs font-mono transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98] ${
                selectedCoarseFilter === cat
                  ? "bg-brand-500 text-white font-semibold shadow-[0_2px_12px_rgba(59,130,246,0.3)]"
                  : "bg-white/[0.04] text-neutral-400 hover:text-white border border-white/5"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Taxonomy Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredTaxonomy.map((item) => (
            <div key={item.id} className="double-bezel-card group">
              <div className="double-bezel-core p-5 flex flex-col justify-between h-full">
                <div>
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-neutral-400">
                      {item.coarse}
                    </span>
                    <span
                      className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded-full border ${
                        item.urgency === "CRITICAL"
                          ? "bg-rose-500/10 border-rose-500/30 text-rose-400"
                          : item.urgency === "HIGH"
                          ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
                          : item.urgency === "MEDIUM"
                          ? "bg-yellow-500/10 border-yellow-500/30 text-yellow-300"
                          : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                      }`}
                    >
                      {item.urgency}
                    </span>
                  </div>

                  <h3 className="text-base font-bold text-white tracking-tight mb-2 group-hover:text-brand-300 transition-colors duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]">
                    {item.fine}
                  </h3>

                  <p className="text-xs text-neutral-400 leading-relaxed mb-4">
                    {item.description}
                  </p>
                </div>

                <div className="pt-3 hairline-divider flex flex-col gap-1.5 text-[11px]">
                  <span className="font-mono text-neutral-500 uppercase text-[10px]">
                    Surgical Action / Protocol:
                  </span>
                  <span className="text-neutral-300 font-sans leading-snug">
                    {item.surgicalImplication}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── 5. Scientific Limitations & Error Boundaries ───────────── */}
      <section className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <span className="text-[11px] uppercase font-mono tracking-[0.2em] text-accent-rose">
            Section 05
          </span>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            Current Limitations & Scientific Error Boundaries
          </h2>
        </div>

        <div className="p-6 md:p-8 rounded-2xl border border-rose-500/20 bg-rose-500/[0.02] flex flex-col gap-5 text-xs text-neutral-300 leading-relaxed">
          <div className="flex items-center gap-2 text-rose-400 font-semibold text-sm">
            <ShieldAlert className="w-4 h-4" strokeWidth={1.25} />
            <span>Honest Scientific Appraisal & Regulatory Boundaries</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex flex-col gap-2 p-4 rounded-xl bg-black/40 border border-white/5">
              <span className="text-white font-semibold">1. Turbidity & Hematuria</span>
              <p className="text-neutral-400 text-xs leading-relaxed font-sans">
                Active intravesical bleeding or cloudy irrigation fluid scatters white light, attenuating shifted-window attention weights and reducing fine classification confidence by up to 14%.
              </p>
            </div>

            <div className="flex flex-col gap-2 p-4 rounded-xl bg-black/40 border border-white/5">
              <span className="text-white font-semibold">2. Muscle Invasion Staging (T2+)</span>
              <p className="text-neutral-400 text-xs leading-relaxed font-sans">
                Surface white-light endoscopy cannot evaluate depth of muscle wall infiltration. Histological biopsy and cross-sectional multiparametric MRI (VI-RADS) remain the mandatory standard.
              </p>
            </div>

            <div className="flex flex-col gap-2 p-4 rounded-xl bg-black/40 border border-white/5">
              <span className="text-white font-semibold">3. Multi-Center Validation</span>
              <p className="text-neutral-400 text-xs leading-relaxed font-sans">
                Classes with fewer than 30 reference images (e.g. Nephrogenic Adenoma) require wider clinical multi-center cohort expansion across diverse scope manufacturers (Olympus, Storz, Karl).
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Bottom Call to Action ──────────────────────────────────── */}
      <div className="p-6 md:p-8 rounded-3xl border border-white/10 bg-gradient-to-r from-brand-950/40 via-black to-accent-cyan/10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="flex flex-col gap-1 max-w-xl">
          <span className="text-sm sm:text-base font-semibold text-white">
            Ready to test the 3S-HFT model checkpoint?
          </span>
          <span className="text-xs text-neutral-400 leading-relaxed">
            Execute real-time inference on curated clinical reference cases or inspect quantitative benchmark metrics.
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <Link
            href="/results"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full border border-white/10 bg-white/[0.04] hover:bg-white/[0.08] text-white text-xs font-medium tracking-tight transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98]"
          >
            <span>View Benchmark Results</span>
          </Link>

          <Link
            href="/demo"
            className="inline-flex items-center gap-3 pl-5 pr-2 py-2 rounded-full bg-white text-black font-semibold text-xs hover:bg-neutral-100 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98] group"
          >
            <span>Launch Live Demo</span>
            <div className="w-6 h-6 rounded-full bg-black/10 flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:translate-x-0.5 group-hover:-translate-y-0.5">
              <ArrowUpRight className="w-3.5 h-3.5 text-black" strokeWidth={1.25} />
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}
