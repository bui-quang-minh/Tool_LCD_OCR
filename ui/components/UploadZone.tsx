"use client";

import { useCallback, useRef, useState } from "react";

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
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`flex flex-col items-center justify-center h-56 border-2 border-dashed rounded-2xl cursor-pointer select-none transition-all duration-200 ${
          dragging
            ? "border-blue-400 bg-blue-500/10 scale-[1.01]"
            : "border-gray-700 bg-gray-900 hover:border-gray-500 hover:bg-gray-800/60"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleChange}
        />

        {/* Icon */}
        <div
          className={`mb-4 p-4 rounded-2xl transition-colors ${
            dragging ? "bg-blue-500/20" : "bg-gray-800"
          }`}
        >
          <svg
            className={`w-10 h-10 transition-colors ${
              dragging ? "text-blue-400" : "text-gray-500"
            }`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
            />
          </svg>
        </div>

        <p className="font-medium text-gray-200">
          {dragging ? "Drop to analyze" : "Drop image here or click to browse"}
        </p>
        <p className="text-gray-500 text-sm mt-1">JPG · PNG · BMP · WEBP</p>
      </div>

      {error && (
        <div className="flex items-start gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
          <svg
            className="w-5 h-5 text-red-400 mt-0.5 shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
            />
          </svg>
          <div>
            <p className="text-red-400 font-medium text-sm">Analysis failed</p>
            <p className="text-red-300/70 text-xs mt-0.5 font-mono">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
