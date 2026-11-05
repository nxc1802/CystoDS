"use client";

import React, { useRef, useState } from "react";
import { UploadCloud, X, FileImage, AlertCircle } from "lucide-react";

interface DropzoneProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  onClear: () => void;
  disabled?: boolean;
}

export default function Dropzone({
  onFileSelect,
  selectedFile,
  onClear,
  disabled = false,
}: DropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    setErrorMsg(null);

    // Validate size (15MB limit)
    if (file.size > 15 * 1024 * 1024) {
      setErrorMsg("File size exceeds 15MB limit. Please upload a smaller image.");
      return;
    }

    // Validate type
    const validTypes = ["image/png", "image/jpeg", "image/bmp", "image/webp"];
    if (!validTypes.includes(file.type)) {
      setErrorMsg("Unsupported image format. Please use PNG, JPEG, BMP, or WEBP.");
      return;
    }

    // Create object URL for preview
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    onFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    onClear();
  };

  return (
    <div className="flex flex-col gap-3">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/bmp,image/webp"
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
          }
        }}
      />

      {selectedFile && previewUrl ? (
        /* Selected Image Preview with Double-Bezel Nested Styling */
        <div className="relative rounded-2xl border border-white/10 bg-black/40 p-3 flex flex-col gap-3">
          <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-black/60 flex items-center justify-center border border-white/5">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Uploaded cystoscopy preview"
              className="w-full h-full object-contain"
            />
            <button
              type="button"
              onClick={handleClear}
              disabled={disabled}
              className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/70 hover:bg-black border border-white/20 text-white flex items-center justify-center transition-colors duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-95"
              title="Remove image"
            >
              <X className="w-3.5 h-3.5" strokeWidth={1.25} />
            </button>
          </div>

          <div className="flex items-center justify-between text-xs text-neutral-400 font-mono px-1">
            <span className="truncate max-w-[200px] text-white">
              {selectedFile.name}
            </span>
            <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
          </div>
        </div>
      ) : (
        /* Upload Drag-and-Drop Area */
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
          className={`cursor-pointer rounded-2xl border-2 border-dashed p-6 sm:p-8 flex flex-col items-center justify-center text-center transition-all duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] ${
            isDragging
              ? "border-brand-400 bg-brand-500/10 scale-[0.99]"
              : "border-white/10 hover:border-white/20 bg-white/[0.02] hover:bg-white/[0.04]"
          } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
        >
          <div className="w-12 h-12 rounded-full bg-white/[0.05] border border-white/10 flex items-center justify-center mb-3 text-neutral-300">
            <UploadCloud className="w-5 h-5 text-brand-400" strokeWidth={1.25} />
          </div>
          <span className="text-sm font-medium text-white mb-1">
            Upload Cystoscopy Image
          </span>
          <p className="text-xs text-neutral-400 max-w-xs leading-relaxed mb-3">
            Drag and drop or click to browse. Max 15MB.
          </p>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/10 text-[10px] font-mono text-neutral-400">
            <FileImage className="w-3 h-3 text-neutral-400" strokeWidth={1.25} />
            <span>PNG, JPEG, BMP, WEBP</span>
          </div>
        </div>
      )}

      {errorMsg && (
        <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" strokeWidth={1.25} />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
}
