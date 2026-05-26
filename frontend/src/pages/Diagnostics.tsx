import { useRef, useState } from "react";
import { inspectImage } from "../api/client";

export default function Diagnostics() {
  const [file, setFile] = useState<File | null>(null);
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
          Run a single image through all three AI components and see raw outputs.
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
            onChange={(e) => e.target.files && setFile(e.target.files[0])}
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
