import React from "react";
import { Shield, Lock, EyeOff } from "lucide-react";

export default function PrivacyPage() {
  return (
    <div className="flex flex-col gap-12 pt-28 md:pt-36 pb-20 px-4 md:px-8 max-w-4xl mx-auto overflow-x-hidden">
      <div className="flex flex-col gap-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-mono uppercase tracking-[0.2em] font-medium border border-white/10 bg-white/[0.04] text-brand-400 w-max">
          <Shield className="w-3.5 h-3.5" strokeWidth={1.25} />
          <span>Data Governance & Ethics</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-white leading-tight">
          Privacy Policy & Research Data Governance
        </h1>
        <p className="text-sm sm:text-base text-neutral-400 leading-relaxed">
          How image data, inference payloads, and anonymized reference samples are processed under the CystoDS scientific platform.
        </p>
      </div>

      <div className="double-bezel-card">
        <div className="double-bezel-core p-6 md:p-8 flex flex-col gap-6 text-xs sm:text-sm text-neutral-300 leading-relaxed">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-emerald-500/10 flex items-center justify-center shrink-0 text-emerald-400">
              <EyeOff className="w-4 h-4" strokeWidth={1.25} />
            </div>
            <div>
              <h2 className="text-base font-bold text-white mb-1">
                Zero Permanent Retention on Demo Uploads
              </h2>
              <p className="text-neutral-400 leading-relaxed">
                Images uploaded through the Live Demo interface are held strictly in ephemeral RAM for the duration of model forward computation and are immediately discarded. No patient identifiers, metadata headers, or pixel data are saved to disk or persistent databases.
              </p>
            </div>
          </div>

          <div className="hairline-divider" />

          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-brand-500/10 flex items-center justify-center shrink-0 text-brand-400">
              <Lock className="w-4 h-4" strokeWidth={1.25} />
            </div>
            <div>
              <h2 className="text-base font-bold text-white mb-1">
                Anonymized Reference Case Compliance
              </h2>
              <p className="text-neutral-400 leading-relaxed">
                All 6 reference cases in the public gallery originate from anonymized open research archives with explicit publication approval. All identifying patient demographics have been stripped at the source.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
