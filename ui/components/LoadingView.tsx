interface Props {
  preview: string | null;
}

const PIPELINE_STEPS = [
  "Detecting LCD display region (YOLO OBB)",
  "Warping and cropping LCD area",
  "Running OCR model on digits",
  "Validating torque reading (60–80 Nm)",
];

export default function LoadingView({ preview }: Props) {
  return (
    <div className="flex flex-col items-center gap-8">
      {/* Image with scanning overlay */}
      {preview && (
        <div className="relative w-full rounded-xl overflow-hidden bg-black border border-gray-800">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preview}
            alt="Uploaded image"
            className="w-full max-h-52 object-contain opacity-50"
          />
          {/* Scanning line */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="scan-line" />
          </div>
          {/* Corner brackets */}
          <div className="absolute inset-3 pointer-events-none">
            <div className="absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 border-blue-400 rounded-tl" />
            <div className="absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 border-blue-400 rounded-tr" />
            <div className="absolute bottom-0 left-0 w-5 h-5 border-b-2 border-l-2 border-blue-400 rounded-bl" />
            <div className="absolute bottom-0 right-0 w-5 h-5 border-b-2 border-r-2 border-blue-400 rounded-br" />
          </div>
        </div>
      )}

      {/* Spinner */}
      <div className="flex flex-col items-center gap-4">
        <div className="relative w-14 h-14">
          <div className="absolute inset-0 rounded-full border-[3px] border-gray-800" />
          <div className="absolute inset-0 rounded-full border-[3px] border-blue-500 border-t-transparent animate-spin" />
          <div
            className="absolute inset-2 rounded-full border-2 border-blue-300/30 border-b-transparent animate-spin"
            style={{ animationDuration: "1.5s", animationDirection: "reverse" }}
          />
        </div>
        <div className="text-center">
          <p className="font-medium text-gray-200">Analyzing image</p>
          <p className="text-gray-500 text-sm mt-0.5">Running YOLO + OCR pipeline</p>
        </div>
      </div>

      {/* Pipeline steps */}
      <div className="w-full space-y-2">
        {PIPELINE_STEPS.map((label, i) => (
          <div key={i} className="flex items-center gap-3 text-sm">
            <span
              className="w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0"
              style={{
                animation: `pulse-dot 1.4s ease-in-out ${i * 0.25}s infinite`,
              }}
            />
            <span className="text-gray-400">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
