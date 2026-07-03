interface Props {
  preview: string | null;
}

const STEPS = [
  "Detecting LCD display region",
  "Warping and cropping LCD area",
  "Running OCR on digits",
  "Validating torque reading",
];

export default function LoadingView({ preview }: Props) {
  return (
    <div className="flex flex-col items-center gap-7">

      {/* Image with scan overlay */}
      {preview && (
        <div className="relative w-full rounded-lg overflow-hidden bg-zinc-950 border border-zinc-200 dark:border-zinc-800">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preview}
            alt="Uploaded image"
            className="w-full max-h-48 object-contain opacity-40"
          />
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="scan-line" />
          </div>
          {/* Corner marks */}
          <div className="absolute inset-2.5 pointer-events-none">
            <div className="absolute top-0 left-0 w-4 h-4 border-t border-l border-zinc-500" />
            <div className="absolute top-0 right-0 w-4 h-4 border-t border-r border-zinc-500" />
            <div className="absolute bottom-0 left-0 w-4 h-4 border-b border-l border-zinc-500" />
            <div className="absolute bottom-0 right-0 w-4 h-4 border-b border-r border-zinc-500" />
          </div>
        </div>
      )}

      {/* Spinner + label */}
      <div className="flex flex-col items-center gap-3">
        <div className="relative w-10 h-10">
          <div className="absolute inset-0 rounded-full border border-zinc-200 dark:border-zinc-800" />
          <div className="absolute inset-0 rounded-full border border-zinc-600 dark:border-zinc-400 border-t-transparent animate-spin" />
        </div>
        <div className="text-center">
          <p className="text-sm text-zinc-700 dark:text-zinc-300 font-medium">Analyzing image</p>
          <p className="text-xs text-zinc-400 dark:text-zinc-600 mt-0.5">YOLO · OCR pipeline</p>
        </div>
      </div>

      {/* Pipeline steps */}
      <div className="w-full divide-y divide-zinc-200 dark:divide-zinc-800/60">
        {STEPS.map((label, i) => (
          <div key={i} className="flex items-center gap-3 py-2.5 text-xs text-zinc-500">
            <span className="w-1 h-1 rounded-full bg-zinc-300 dark:bg-zinc-700 shrink-0" />
            {label}
          </div>
        ))}
      </div>
    </div>
  );
}
