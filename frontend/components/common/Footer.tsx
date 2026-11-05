import React from "react";
import Link from "next/link";
import { ShieldAlert, Cpu, GitBranch, Database, FileText } from "lucide-react";

export default function Footer() {
  return (
    <footer className="relative z-10 border-t border-white/10 bg-[#050507] pt-16 pb-12 px-4 md:px-8 mt-24">
      <div className="max-w-6xl mx-auto flex flex-col gap-12">
        {/* Research Disclaimer Banner (PRD Mandatory) */}
        <div className="p-4 md:p-5 rounded-2xl bg-amber-500/[0.04] border border-amber-500/20 flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-amber-500/10 flex items-center justify-center shrink-0">
            <ShieldAlert className="w-4 h-4 text-amber-400" strokeWidth={1.25} />
          </div>
          <div className="text-xs text-neutral-300 leading-relaxed">
            <strong className="text-amber-300 font-medium mr-1.5">
              Research & Scientific Demonstration Use Only:
            </strong>
            The CystoDS platform and 3S-HFT neural network outputs are designed exclusively for algorithmic evaluation, computer vision benchmark analysis, and scientific research. This tool is not an FDA/CE-cleared diagnostic device and should never replace qualified urological examination or histopathological biopsy.
          </div>
        </div>

        {/* Main Footer Navigation Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Col 1: Brand & Model Info */}
          <div className="flex flex-col gap-3 md:col-span-2">
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold tracking-tight text-white">
                CystoDS AI
              </span>
              <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-neutral-400">
                Phase 3 / Split 0
              </span>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed max-w-md">
              Three-Stage Sequential Hierarchical Fine-Tuning (3S-HFT) deep learning architecture powered by Swin Transformer for high-precision cystoscopy bladder cancer detection and clinical urgency classification.
            </p>
            <div className="flex items-center gap-4 mt-2">
              <div className="flex items-center gap-1.5 text-[11px] text-neutral-400 font-mono">
                <Cpu className="w-3.5 h-3.5 text-brand-400" strokeWidth={1.25} />
                <span>SHA-256: 6c103665...816846c</span>
              </div>
            </div>
          </div>

          {/* Col 2: Platform Links */}
          <div className="flex flex-col gap-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-neutral-300 font-mono">
              Platform
            </span>
            <ul className="flex flex-col gap-2 text-xs text-neutral-400">
              <li>
                <Link
                  href="/"
                  className="hover:text-white transition-colors duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]"
                >
                  Overview & Methodology
                </Link>
              </li>
              <li>
                <Link
                  href="/demo"
                  className="hover:text-white transition-colors duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] text-brand-400"
                >
                  Interactive Live Demo
                </Link>
              </li>
              <li>
                <Link
                  href="/research"
                  className="hover:text-white transition-colors duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]"
                >
                  Research Specification
                </Link>
              </li>
              <li>
                <Link
                  href="/results"
                  className="hover:text-white transition-colors duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]"
                >
                  Benchmark Performance
                </Link>
              </li>
            </ul>
          </div>

          {/* Col 3: Research Artifacts */}
          <div className="flex flex-col gap-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-neutral-300 font-mono">
              Scientific Assets
            </span>
            <ul className="flex flex-col gap-2 text-xs text-neutral-400">
              <li className="flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-neutral-500" strokeWidth={1.25} />
                <span>Model Audit Manifest v1</span>
              </li>
              <li className="flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-neutral-500" strokeWidth={1.25} />
                <span>OSF Cystoscopy Archive</span>
              </li>
              <li className="flex items-center gap-1.5">
                <GitBranch className="w-3.5 h-3.5 text-neutral-500" strokeWidth={1.25} />
                <span>Patient-Disjoint Protocol</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Hairline & Copyright */}
        <div className="hairline-divider pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-neutral-500">
          <span>
            © {new Date().getFullYear()} CystoDS Research Platform. All scientific rights reserved.
          </span>
          <div className="flex items-center gap-6">
            <span>FastAPI Backend Parity Verified</span>
            <span>TypeScript + Next.js App Router</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
