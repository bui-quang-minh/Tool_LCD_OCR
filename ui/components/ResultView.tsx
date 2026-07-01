"use client";

import { useState, useEffect, useCallback } from "react";
import type { PredictResult } from "@/app/page";

interface Props {
  result: PredictResult;
  onReset: () => void;
}

export default function ResultView({ result, onReset }: Props) {
  const [lightbox, setLightbox] = useState<string | null>(null);

  const closeLightbox = useCallback(() => setLightbox(null), []);

  // Close on Escape key
  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") closeLightbox(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox, closeLightbox]);
  const isOK = result.verdict === "OK";
  const isNG = result.verdict === "NG";
  const isUnknown = result.verdict === "unknown";
  const displayValue = result.value_nm !== null ? result.value_nm.toFixed(1) : null;

  return (
    <div className="grid grid-cols-5 gap-4">

      {/* ── Left col (2/5): range bar + stats + reading + button ───────────── */}
      <div className="col-span-2 flex flex-col gap-4">

        {/* Range bar */}
        {displayValue && !isUnknown && (
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-4">
              Torque range (Nm)
            </p>
            <div className="relative h-3 mb-6">
              <div className="absolute inset-0 bg-gray-800 rounded-full" />
              <div
                className="absolute h-full bg-green-500/30 border border-green-500/40 rounded-full"
                style={{ left: "50%", width: "16.7%" }}
              />
              {result.value_nm !== null && (
                <div
                  className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full border-2 border-white ${
                    isOK ? "bg-green-400" : "bg-red-500"
                  } shadow-lg`}
                  style={{ left: `${Math.min(100, Math.max(0, (result.value_nm / 120) * 100))}%` }}
                />
              )}
            </div>
            <div className="flex justify-between text-xs text-gray-600">
              <span>0</span>
              <span className="text-green-500/70">60</span>
              <span className="text-green-500/70">80</span>
              <span>120</span>
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 space-y-4">
          <p className="text-xs text-gray-500 uppercase tracking-wider">Details</p>
          <StatRow label="OCR filter used" value={result.filter_used || "—"} />
          <StatRow label="Filter retries" value={`${result.n_tries} of 7`} />
          <StatRow label="Valid range" value="60 – 80 Nm" />
          <StatRow
            label="Status"
            value={isOK ? "Pass" : isNG ? "Fail" : "Unknown"}
            valueClass={isOK ? "text-green-400" : isNG ? "text-red-400" : "text-gray-400"}
          />
        </div>

        {/* Detected reading — sits under Details */}
        <div
          className={`flex items-center justify-between px-5 py-4 rounded-2xl border ${
            isOK
              ? "bg-green-500/10 border-green-500/30"
              : isNG
              ? "bg-red-500/10 border-red-500/30"
              : "bg-gray-800 border-gray-700"
          }`}
        >
          <div>
            <p className="text-gray-400 text-[10px] uppercase tracking-widest mb-1">
              Detected reading
            </p>
            {displayValue ? (
              <p className="text-4xl font-bold tracking-tight leading-none">
                {displayValue}
                <span className="text-xl text-gray-400 ml-2 font-light">Nm</span>
              </p>
            ) : (
              <p className="text-lg text-gray-400 font-mono">{result.reading}</p>
            )}
          </div>

          <div
            className={`flex flex-col items-center justify-center w-16 h-16 rounded-2xl font-black text-2xl shrink-0 ${
              isOK
                ? "bg-green-500 text-white shadow-lg shadow-green-500/30"
                : isNG
                ? "bg-red-600 text-white shadow-lg shadow-red-600/30"
                : "bg-gray-700 text-gray-300"
            }`}
          >
            {isUnknown ? "?" : result.verdict}
            <span className="text-[9px] font-normal opacity-75 mt-0.5 tracking-wide">
              {isOK ? "PASS" : isNG ? "FAIL" : "N/A"}
            </span>
          </div>
        </div>

        {/* Analyze again */}
        <button
          onClick={onReset}
          className="w-full py-3.5 rounded-2xl bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition-colors font-medium text-sm"
        >
          ← Analyze another image
        </button>
      </div>

      {/* Right col (3/5): images */}
      <div className="col-span-3 flex flex-col gap-4">
        <ImageCard
          src={result.annotated_image}
          label="Detection result"
          badge="YOLO OBB"
          tall
          onOpen={setLightbox}
        />
        {result.lcd_crop && (
          <ImageCard
            src={result.lcd_crop}
            label="LCD crop · OCR input"
            badge="OCR model"
            onOpen={setLightbox}
          />
        )}
      </div>

      {/* Lightbox */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm"
          onClick={closeLightbox}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={lightbox}
            alt="Full screen"
            className="max-w-[90vw] max-h-[90vh] object-contain rounded-xl shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
          <button
            onClick={closeLightbox}
            className="absolute top-5 right-5 w-10 h-10 flex items-center justify-center rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
          <p className="absolute bottom-5 text-gray-500 text-xs">
            Click outside or press Esc to close
          </p>
        </div>
      )}
    </div>
  );
}

function ImageCard({
  src,
  label,
  badge,
  tall = false,
  onOpen,
}: {
  src: string;
  label: string;
  badge: string;
  tall?: boolean;
  onOpen: (src: string) => void;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden group">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-800">
        <p className="text-xs text-gray-400 font-medium">{label}</p>
        <span className="text-[10px] bg-gray-800 text-gray-500 px-2 py-0.5 rounded font-mono">
          {badge}
        </span>
      </div>
      <div
        className="relative cursor-zoom-in"
        onClick={() => onOpen(src)}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={label}
          className={`w-full object-contain bg-black transition-opacity group-hover:opacity-90 ${tall ? "max-h-80" : "max-h-52"}`}
        />
        {/* Expand hint */}
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="bg-black/50 rounded-full p-2">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatRow({
  label,
  value,
  valueClass = "text-gray-200",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-500 text-sm">{label}</span>
      <span className={`text-sm font-medium ${valueClass}`}>{value}</span>
    </div>
  );
}
