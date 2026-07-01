"use client";

import { useState, useEffect, useCallback } from "react";
import { X, ArrowsOut } from "@phosphor-icons/react";
import type { PredictResult } from "@/app/page";

interface Props {
  result: PredictResult;
  onReset: () => void;
}

export default function ResultView({ result, onReset }: Props) {
  const [lightbox, setLightbox] = useState<string | null>(null);

  const closeLightbox = useCallback(() => setLightbox(null), []);

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
    <>
      <div className="grid grid-cols-5 gap-4">

        {/* ── Left col (2/5) ───────────────────────────────────────────── */}
        <div className="col-span-2 flex flex-col gap-3">

          {/* Verdict + reading */}
          <div className={`rounded-xl border p-5 ${
            isOK
              ? "border-emerald-900/60 bg-emerald-950/30"
              : isNG
              ? "border-red-900/60 bg-red-950/30"
              : "border-zinc-800 bg-zinc-900"
          }`}>
            <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest mb-3">
              Detected reading
            </p>
            <div className="flex items-end justify-between">
              <div>
                {displayValue ? (
                  <p className="text-4xl font-semibold tracking-tight text-zinc-100 leading-none">
                    {displayValue}
                    <span className="text-lg text-zinc-500 font-normal ml-1.5">Nm</span>
                  </p>
                ) : (
                  <p className="text-2xl font-mono text-zinc-400">{result.reading || "—"}</p>
                )}
              </div>
              <span className={`text-xs font-mono font-bold px-2.5 py-1 rounded ${
                isOK
                  ? "bg-emerald-500/20 text-emerald-400"
                  : isNG
                  ? "bg-red-500/20 text-red-400"
                  : "bg-zinc-800 text-zinc-500"
              }`}>
                {isUnknown ? "N/A" : result.verdict}
              </span>
            </div>
          </div>

          {/* Range bar */}
          {displayValue && !isUnknown && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900 px-5 py-4">
              <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest mb-4">
                Torque range
              </p>
              <div className="relative h-1.5 mb-3">
                <div className="absolute inset-0 rounded-full bg-zinc-800" />
                {/* OK band 60–80 of 120 = 50%–66.7% */}
                <div
                  className="absolute h-full rounded-full bg-emerald-900/60 border-x border-emerald-800/60"
                  style={{ left: "50%", width: "13.9%" }}
                />
                {result.value_nm !== null && (
                  <div
                    className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full ${
                      isOK ? "bg-emerald-400" : "bg-red-500"
                    }`}
                    style={{ left: `${Math.min(100, Math.max(0, (result.value_nm / 120) * 100))}%` }}
                  />
                )}
              </div>
              <div className="flex justify-between text-[10px] font-mono text-zinc-700">
                <span>0</span>
                <span className="text-emerald-700">60</span>
                <span className="text-emerald-700">80</span>
                <span>120</span>
              </div>
            </div>
          )}

          {/* Details */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900 px-5 py-4">
            <p className="text-[10px] font-mono text-zinc-500 uppercase tracking-widest mb-3">
              Details
            </p>
            <div className="divide-y divide-zinc-800/60">
              <StatRow label="Filter" value={result.filter_used || "—"} />
              <StatRow label="Retries" value={`${result.n_tries} / 7`} />
              <StatRow label="Valid range" value="60 – 80 Nm" />
              <StatRow
                label="Status"
                value={isOK ? "Pass" : isNG ? "Fail" : "Unknown"}
                valueClass={isOK ? "text-emerald-400" : isNG ? "text-red-400" : "text-zinc-500"}
              />
            </div>
          </div>

          {/* Reset */}
          <button
            onClick={onReset}
            className="w-full py-2.5 rounded-lg border border-zinc-800 bg-zinc-900 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 transition-colors text-sm"
          >
            Analyze another image
          </button>
        </div>

        {/* ── Right col (3/5): images ───────────────────────────────────── */}
        <div className="col-span-3 flex flex-col gap-3">
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
            className="max-w-[90vw] max-h-[90vh] object-contain rounded-lg"
            onClick={(e) => e.stopPropagation()}
          />
          <button
            onClick={closeLightbox}
            className="absolute top-4 right-4 w-9 h-9 flex items-center justify-center rounded-full bg-zinc-900/80 border border-zinc-700 hover:bg-zinc-800 text-zinc-300 transition-colors"
            aria-label="Close"
          >
            <X size={16} />
          </button>
          <p className="absolute bottom-4 text-zinc-600 text-xs font-mono">
            ESC to close
          </p>
        </div>
      )}
    </>
  );
}

function ImageCard({
  src, label, badge, tall = false, onOpen,
}: {
  src: string;
  label: string;
  badge: string;
  tall?: boolean;
  onOpen: (src: string) => void;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 overflow-hidden group">
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800">
        <span className="text-xs text-zinc-400">{label}</span>
        <span className="text-[10px] font-mono text-zinc-600 bg-zinc-800 px-2 py-0.5 rounded">
          {badge}
        </span>
      </div>
      <div className="relative cursor-zoom-in" onClick={() => onOpen(src)}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={label}
          className={`w-full object-contain bg-black transition-opacity group-hover:opacity-80 ${
            tall ? "max-h-80" : "max-h-52"
          }`}
        />
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="bg-zinc-950/70 rounded-full p-2 border border-zinc-700">
            <ArrowsOut size={18} className="text-zinc-300" />
          </div>
        </div>
      </div>
    </div>
  );
}

function StatRow({
  label, value, valueClass = "text-zinc-300",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex items-center justify-between py-2.5">
      <span className="text-xs text-zinc-500">{label}</span>
      <span className={`text-xs font-mono ${valueClass}`}>{value}</span>
    </div>
  );
}
