"use client";

import { useState, useCallback } from "react";
import UploadZone from "@/components/UploadZone";
import LoadingView from "@/components/LoadingView";
import ResultView from "@/components/ResultView";

type Step = "upload" | "loading" | "result" | "error";

export type PredictResult = {
  reading: string;
  verdict: "OK" | "NG" | "unknown";
  value_nm: number | null;
  n_tries: number;
  filter_used: string;
  annotated_image: string;
  lcd_crop: string | null;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const STEPS = ["Upload", "Analyzing", "Result"] as const;

export default function Home() {
  const [step, setStep] = useState<Step>("upload");
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    setStep("loading");
    setError(null);

    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body });

      if (!res.ok) {
        const msg = await res.text().catch(() => `Server error ${res.status}`);
        throw new Error(msg || `Server error ${res.status}`);
      }

      const data: PredictResult = await res.json();
      setResult(data);
      setStep("result");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error occurred");
      setStep("error");
    }
  }, []);

  const reset = useCallback(() => {
    setStep("upload");
    setResult(null);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
  }, [preview]);

  const isResult = step === "result";
  const activeIndex =
    step === "upload" || step === "error" ? 0 : step === "loading" ? 1 : 2;

  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
      <div className={`w-full transition-[max-width] duration-500 ease-out ${isResult ? "max-w-6xl" : "max-w-xl"}`}>

        {/* Header */}
        <div className={`${isResult ? "mb-6" : "mb-10"}`}>
          <p className="text-xs font-mono text-zinc-500 uppercase tracking-widest mb-1">
            Torque · 60–80 Nm
          </p>
          <h1 className="text-xl font-semibold text-zinc-100 tracking-tight">
            LCD OCR Inspector
          </h1>
        </div>

        {/* Step indicator: dots + lines */}
        <div className="flex items-center mb-8">
          {STEPS.map((label, i) => {
            const done = i < activeIndex;
            const active = i === activeIndex;
            return (
              <div key={label} className="flex items-center">
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2 h-2 rounded-full transition-colors duration-300 ${
                      done
                        ? "bg-zinc-400"
                        : active
                        ? "bg-zinc-100"
                        : "bg-zinc-700"
                    }`}
                  />
                  <span
                    className={`text-xs transition-colors duration-300 ${
                      active ? "text-zinc-200 font-medium" : done ? "text-zinc-500" : "text-zinc-700"
                    }`}
                  >
                    {label}
                  </span>
                </div>
                {i < STEPS.length - 1 && (
                  <div
                    className={`mx-3 h-px w-8 transition-colors duration-300 ${
                      done ? "bg-zinc-600" : "bg-zinc-800"
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>

        {/* Content */}
        {isResult && result ? (
          <div className="fade-up">
            <ResultView result={result} onReset={reset} />
          </div>
        ) : (
          <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
            {(step === "upload" || step === "error") && (
              <UploadZone onFile={handleFile} error={error} />
            )}
            {step === "loading" && <LoadingView preview={preview} />}
          </div>
        )}
      </div>
    </main>
  );
}
