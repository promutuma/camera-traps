import { useRef, useState } from "react";
import { inspectImage } from "../api/client";

function parseBbox(raw: unknown): [number, number, number, number] | null {
  if (!raw) return null;
  try {
    const arr = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (Array.isArray(arr) && arr.length === 4) return arr as [number, number, number, number];
  } catch {}
  return null;
}

export default function Diagnostics() {
  const [file, setFile] = useState<File | null>(null);
  const [imgSrc, setImgSrc] = useState<string | null>(null);
  const [imgNatural, setImgNatural] = useState<{ w: number; h: number } | null>(null);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleInspect = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await inspectImage(file);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Inspection failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Deep Inspection Tool</h1>
        <p className="text-slate-500 mt-1">
          Run a single image through all AI components and visualize dual-model detections.
        </p>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-4">
        <div
          onClick={() => inputRef.current?.click()}
          className="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center cursor-pointer hover:border-green-400"
        >
          {file ? (
            <p className="text-slate-700 font-medium">{file.name}</p>
          ) : (
            <p className="text-slate-400">Click to select an image</p>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png"
            className="hidden"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                const selected = e.target.files[0];
                setFile(selected);
                setImgSrc(URL.createObjectURL(selected));
                setResult(null);
                setImgNatural(null);
              }
            }}
          />
        </div>
        <button
          onClick={handleInspect}
          disabled={!file || loading}
          className="w-full py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-slate-300 text-white font-semibold rounded-lg"
        >
          {loading ? "Inspecting…" : "Run Deep Inspection"}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">{error}</div>
      )}

      {imgSrc && (
        <div className="bg-slate-900 rounded-xl overflow-hidden relative border border-slate-800 shadow-md">
          {/* Bounding box Legend */}
          <div className="absolute top-2 left-2 z-10 flex gap-2">
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-black/70 backdrop-blur-md text-[10px] font-semibold text-blue-400 border border-blue-500/30">
              <span className="w-3 h-0.5 bg-blue-500 inline-block"></span>
              MDv5a (Primary)
            </span>
            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-black/70 backdrop-blur-md text-[10px] font-semibold text-violet-400 border border-violet-500/30">
              <span className="w-3 h-0.5 border-t border-dashed border-violet-500 inline-block"></span>
              MDv1000 (Redwood)
            </span>
          </div>

          <div className="relative w-full flex items-center justify-center bg-slate-950">
            <div className="relative inline-block w-full">
              <img
                src={imgSrc}
                alt="Inspection source"
                className="w-full h-auto block select-none"
                onLoad={(e) => {
                  const img = e.currentTarget as HTMLImageElement;
                  setImgNatural({ w: img.naturalWidth, h: img.naturalHeight });
                }}
              />
              {Array.isArray(result?.megadetector) && imgNatural && (
                <svg
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  viewBox={`0 0 ${imgNatural.w} ${imgNatural.h}`}
                >
                  {(result.megadetector as any[]).map((det, idx) => {
                    const bbox = parseBbox(det.bbox);
                    if (!bbox) return null;
                    const isV1000 = det.model === "MDv1000";
                    const color = isV1000 ? "#c084fc" : "#60a5fa"; // soft purple for v1000, soft blue for v5a
                    const label = det.label || "Animal";
                    const conf = Math.round((det.conf || 0) * 100);

                    return (
                      <g key={idx}>
                        <rect
                          x={bbox[0] * imgNatural.w}
                          y={bbox[1] * imgNatural.h}
                          width={bbox[2] * imgNatural.w}
                          height={bbox[3] * imgNatural.h}
                          fill="none"
                          stroke={color}
                          strokeWidth={Math.max(3, imgNatural.w / 250)}
                          strokeDasharray={isV1000 ? "6 4" : undefined}
                        />
                        <text
                          x={bbox[0] * imgNatural.w + 6}
                          y={bbox[1] * imgNatural.h - 8}
                          fill={color}
                          fontWeight="bold"
                          fontSize={Math.max(14, imgNatural.w / 40)}
                          style={{ filter: "drop-shadow(0 1px 3px #000)" }}
                        >
                          {label} {conf}% ({det.model})
                        </text>
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="space-y-4">
          {result.ocr != null && (
            <Section title="1. OCR Data Extraction" data={result.ocr as object} />
          )}
          {result.megadetector != null && (
            <Section title="2. MegaDetector Raw Candidates" data={result.megadetector as object} />
          )}
          {result.bioclip != null && (
            <Section title="3. BioClip Top-20 Predictions" data={result.bioclip as object} />
          )}
          {result.speciesnet != null && (
            <Section title="4. Google SpeciesNet Predictions" data={result.speciesnet as object} />
          )}
        </div>
      )}
    </div>
  );
}

function Section({ title, data }: { title: string; data: object }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
        <h2 className="font-semibold text-slate-700">{title}</h2>
      </div>
      <pre className="p-4 text-xs text-slate-700 overflow-x-auto whitespace-pre-wrap">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}
