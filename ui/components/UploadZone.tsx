"use client";

import { useCallback, useRef, useState } from "react";
import { UploadSimple, Warning } from "@phosphor-icons/react";

interface Props {
  onFile: (file: File) => void;
  error: string | null;
}

export default function UploadZone({ onFile, error }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = (file: File) => {
    if (file.type.startsWith("image/")) onFile(file);
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) accept(file);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [onFile]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) accept(file);
    e.target.value = "";
  };

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`flex flex-col items-center justify-center h-52 rounded-lg border border-dashed cursor-pointer select-none transition-colors duration-150 ${
          dragging
            ? "border-zinc-500 bg-zinc-100 dark:bg-zinc-800/60"
            : "border-zinc-300 dark:border-zinc-700 hover:border-zinc-400 dark:hover:border-zinc-600 hover:bg-zinc-100 dark:hover:bg-zinc-800/40"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleChange}
        />

        <UploadSimple
          size={28}
          weight="light"
          className={`mb-3 transition-colors ${
            dragging ? "text-zinc-700 dark:text-zinc-300" : "text-zinc-400 dark:text-zinc-500"
          }`}
        />

        <p className={`text-sm transition-colors ${
          dragging ? "text-zinc-800 dark:text-zinc-200" : "text-zinc-600 dark:text-zinc-400"
        }`}>
          {dragging ? "Drop to analyze" : "Drop image or click to browse"}
        </p>
        <p className="text-xs text-zinc-400 dark:text-zinc-600 mt-1">JPG · PNG · BMP · WEBP</p>
      </div>

      {error && (
        <div className="flex items-start gap-3 p-4 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/60 rounded-lg">
          <Warning size={16} weight="fill" className="text-red-500 mt-0.5 shrink-0" />
          <div>
            <p className="text-red-600 dark:text-red-400 text-sm font-medium">Analysis failed</p>
            <p className="text-red-500/70 dark:text-red-400/60 text-xs mt-0.5 font-mono break-all">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
