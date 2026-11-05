"use client";

import React, { useState, useEffect, Suspense, useRef } from "react";
import { useSearchParams } from "next/navigation";
import {
  SampleItem,
  PredictResponse,
  InferenceStatus,
} from "@/types";
import {
  getSamples,
  predictByFile,
  predictBySampleId,
  FALLBACK_SAMPLES,
} from "@/lib/api";
import Dropzone from "@/components/demo/Dropzone";
import SampleSelector from "@/components/demo/SampleSelector";
import PredictionViewer from "@/components/demo/PredictionViewer";
import {
  ArrowRight,
  Sparkles,
  Activity,
  RefreshCw,
} from "lucide-react";

function DemoContent() {
  const searchParams = useSearchParams();
  const requestedSampleId = searchParams.get("sample");

  const [samples, setSamples] = useState<SampleItem[]>(FALLBACK_SAMPLES);
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [status, setStatus] = useState<InferenceStatus>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch samples on load
  useEffect(() => {
    async function loadInitial() {
      try {
        const fetched = await getSamples();
        if (fetched && fetched.length > 0) {
          setSamples(fetched);
          if (requestedSampleId) {
            const found = fetched.find((s) => s.id === requestedSampleId);
            if (found) {
              setSelectedSample(found);
            }
          }
        }
      } catch {
        // Fallback already assigned
      }
    }
    loadInitial();
  }, [requestedSampleId]);

  // Handle selecting a reference sample
  const handleSelectSample = (sample: SampleItem) => {
    setSelectedSample(sample);
    setSelectedFile(null); // Clear custom upload if selecting sample
    setErrorMsg(null);
  };

  // Handle uploading a file
  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setSelectedSample(null); // Clear sample if uploading custom file
    setErrorMsg(null);
  };

  const handleClearFile = () => {
    setSelectedFile(null);
  };

  // Run Inference Execution
  const handleRunInference = async () => {
    if (!selectedFile && !selectedSample) {
      setErrorMsg("Please select a curated reference case or upload a cystoscopy image.");
      return;
    }

    setStatus(selectedFile ? "uploading" : "running");
    setErrorMsg(null);
    setResult(null);
    setElapsedMs(0);

    const startTime = Date.now();
    timerRef.current = setInterval(() => {
      setElapsedMs(Date.now() - startTime);
    }, 50);

    try {
      let res: PredictResponse;
      if (selectedFile) {
        setStatus("running");
        res = await predictByFile(selectedFile);
      } else if (selectedSample) {
        res = await predictBySampleId(selectedSample.id);
      } else {
        throw new Error("No input specified");
      }

      setResult(res);
      setStatus("success");
    } catch (err: unknown) {
      setStatus("error");
      const message =
        err instanceof Error
          ? err.message
          : "Failed to execute inference on backend model.";
      setErrorMsg(message);
    } finally {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }
  };

  const isBusy = status === "uploading" || status === "running";
  const canRun = (selectedFile !== null || selectedSample !== null) && !isBusy;

  return (
    <div className="flex flex-col gap-10 pt-28 md:pt-36 pb-20 px-4 md:px-8 max-w-7xl mx-auto overflow-x-hidden">
      {/* Header Eyebrow & Title */}
      <div className="flex flex-col gap-3 max-w-3xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-[11px] font-mono uppercase tracking-[0.2em] font-medium border border-white/10 bg-white/[0.04] text-brand-400 w-max">
          <Activity className="w-3.5 h-3.5" strokeWidth={1.25} />
          <span>Interactive Clinical Inference Studio</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-white leading-tight">
          Live Cystoscopy Diagnostic Assistant
        </h1>
        <p className="text-sm sm:text-base text-neutral-400 leading-relaxed">
          Evaluate endoscopic images in real-time. The 3S-HFT network executes 3-level hierarchical reasoning, generating binary lesion tags, 5 coarse groupings, and 22 fine histological subclasses.
        </p>
      </div>

      {/* Main Two-Panel Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* ── LEFT PANEL: INPUT WORKSPACE (col-span-5) ──────────────────── */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 flex flex-col gap-6">
              {/* Section Header */}
              <div className="flex items-center justify-between hairline-divider pb-4">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-brand-400 animate-pulse" />
                  <span className="text-sm font-semibold text-white tracking-tight">
                    Input Configuration
                  </span>
                </div>
                <span className="text-[11px] font-mono text-neutral-400">
                  {selectedFile
                    ? "Custom Image"
                    : selectedSample
                    ? `Case: ${selectedSample.id}`
                    : "No Input"}
                </span>
              </div>

              {/* Input Mode 1: Curated Clinical Cases */}
              <SampleSelector
                samples={samples}
                selectedSampleId={selectedSample?.id || null}
                onSelect={handleSelectSample}
                disabled={isBusy}
              />

              {/* Divider */}
              <div className="relative flex items-center justify-center">
                <div className="w-full border-t border-white/10" />
                <span className="absolute px-3 bg-[#0d0d12] text-[10px] font-mono text-neutral-400 uppercase tracking-widest">
                  OR UPLOAD FILE
                </span>
              </div>

              {/* Input Mode 2: Custom File Upload Dropzone */}
              <Dropzone
                selectedFile={selectedFile}
                onFileSelect={handleFileSelect}
                onClear={handleClearFile}
                disabled={isBusy}
              />

              {/* Active Selection Summary Card */}
              {(selectedSample || selectedFile) && (
                <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/5 flex flex-col gap-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-neutral-400">Target Source:</span>
                    <span className="font-medium text-white truncate max-w-[200px]">
                      {selectedFile ? selectedFile.name : selectedSample?.name}
                    </span>
                  </div>
                  {selectedSample && (
                    <div className="flex items-center justify-between text-[11px] font-mono text-neutral-400">
                      <span>Ground Truth:</span>
                      <span className="text-brand-300">
                        {selectedSample.ground_truth.coarse} •{" "}
                        {selectedSample.ground_truth.fine}
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Action Button: Button-in-Button */}
              <button
                type="button"
                onClick={handleRunInference}
                disabled={!canRun}
                className={`w-full py-2.5 pl-6 pr-2 rounded-full font-medium text-sm flex items-center justify-between transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98] group ${
                  canRun
                    ? "bg-white text-black hover:bg-neutral-100 shadow-[0_4px_20px_rgba(255,255,255,0.2)]"
                    : "bg-white/10 text-neutral-500 cursor-not-allowed border border-white/5"
                }`}
              >
                <span>
                  {isBusy ? "Computing Forward Pass..." : "Run Live Inference"}
                </span>
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
                    canRun
                      ? "bg-black/10 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 text-black"
                      : "bg-white/5 text-neutral-500"
                  }`}
                >
                  {isBusy ? (
                    <RefreshCw className="w-4 h-4 animate-spin text-neutral-400" strokeWidth={1.25} />
                  ) : (
                    <ArrowRight className="w-4 h-4" strokeWidth={1.25} />
                  )}
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* ── RIGHT PANEL: PREDICTION & VISUALIZATION (col-span-7) ───────── */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          <div className="double-bezel-card">
            <div className="double-bezel-core p-6 flex flex-col gap-6">
              {/* Output Header */}
              <div className="flex items-center justify-between hairline-divider pb-4">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-brand-400" strokeWidth={1.25} />
                  <span className="text-sm font-semibold text-white tracking-tight">
                    Multi-Level Diagnostic Output
                  </span>
                </div>
                <div className="flex items-center gap-2 text-[11px] font-mono text-neutral-400">
                  <span>Swin-Tiny v3.1</span>
                  <span>•</span>
                  <span>224×224 FOV</span>
                </div>
              </div>

              {/* Main Prediction Viewer Component */}
              <PredictionViewer
                status={status}
                result={result}
                error={errorMsg}
                onRetry={handleRunInference}
                latencyTimer={elapsedMs}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function DemoPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-[60vh] text-neutral-400">
          Loading Clinical Inference Studio...
        </div>
      }
    >
      <DemoContent />
    </Suspense>
  );
}
