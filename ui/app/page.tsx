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
  const activeStepIndex =
    step === "error" ? 0 : step === "upload" ? 0 : step === "loading" ? 1 : 2;

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-6">
      {/* Container expands when showing results */}
      <div className={`w-full transition-all duration-500 ${isResult ? "max-w-6xl" : "max-w-2xl"}`}>

        {/* Header — compact on result */}
        <div className={`text-center ${isResult ? "mb-4" : "mb-8"}`}>
          <h1 className={`font-bold tracking-tight ${isResult ? "text-xl" : "text-2xl"}`}>
            LCD OCR Inspector
          </h1>
          {!isResult && (
            <p className="text-gray-500 text-sm mt-1">
              Torque wrench reading detection · 60–80 Nm validation
            </p>
          )}
        </div>

        {/* Step indicator */}
        <div className={`flex items-center justify-center ${isResult ? "mb-6" : "mb-8"}`}>
          {(["Upload", "Analyzing", "Result"] as const).map((label, i) => (
            <div key={label} className="flex items-center">
              <div
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                  i === activeStepIndex
                    ? "bg-blue-600 text-white"
                    : i < activeStepIndex
                    ? "text-blue-400"
                    : "text-gray-600"
                }`}
              >
                <span
                  className={`w-5 h-5 rounded-full border flex items-center justify-center text-xs font-bold ${
                    i < activeStepIndex
                      ? "border-blue-400 bg-blue-400 text-gray-950"
                      : i === activeStepIndex
                      ? "border-white text-white"
                      : "border-gray-600 text-gray-600"
                  }`}
                >
                  {i < activeStepIndex ? "✓" : i + 1}
                </span>
                {label}
              </div>
              {i < 2 && (
                <div className={`w-10 h-px mx-1 ${i < activeStepIndex ? "bg-blue-400" : "bg-gray-800"}`} />
              )}
            </div>
          ))}
        </div>

        {/* Content — no card wrapper on result step so it can go full width */}
        {isResult && result ? (
          <ResultView result={result} onReset={reset} />
        ) : (
          <div className="rounded-2xl border border-gray-800 bg-gray-900/50 p-6 backdrop-blur-sm">
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
