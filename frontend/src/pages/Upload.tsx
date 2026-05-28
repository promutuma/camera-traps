import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadImages, startProcessing, getJobResults, getModelStatus, flagByFilenames } from "../api/client";
import { useConfigStore } from "../store/configStore";

// ── Types ─────────────────────────────────────────────────────────────────────

type JobState = {
  jobId: string;
  status: string;
  total: number;
  completed: number;
  error?: string;
};

type ModelStatus = { models_loaded: boolean; error: string | null } | null;

type Detection = { label: string; conf: number; bbox?: number[] };

type ModelEvent = {
  type: "model_event";
  image: string;
  image_index: number;
  model: string;
  // Detection models
  detections?: Detection[];
  merged_count?: number;
  sources_used?: string[];
  // Classification models
  top5?: [string, number][];
  skipped?: boolean;
  // Final result
  species?: string;
  confidence?: number;
  agreement?: "High" | "Medium" | "Low";
  all_candidates?: [string, number][];
};

type ImageRow = {
  name: string;
  index: number;
  events: ModelEvent[];
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtTime(secs: number) {
  return `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
}

function parseSnetLabel(raw: string): { display: string; tooltip: string } {
  if (raw.startsWith("{")) {
    try {
      const p = JSON.parse(raw);
      const display = p.common_name || p.display || raw;
      const tooltip = [
        p.common_name && `Common: ${p.common_name}`,
        p.scientific_name && `Scientific: ${p.scientific_name}`,
        p.hierarchy?.length && `Taxonomy: ${p.hierarchy.join(" › ")}`,
      ]
        .filter(Boolean)
        .join("\n");
      return { display, tooltip };
    } catch {}
  }
  return { display: raw.trim(), tooltip: "" };
}

// ── Small components ──────────────────────────────────────────────────────────

type FileStatus = "pending" | "processing" | "done";

function ThumbnailItem({
  file,
  onRemove,
  disabled,
  status = "pending",
}: {
  file: File;
  onRemove: () => void;
  disabled: boolean;
  status?: FileStatus;
}) {
  const [preview, setPreview] = useState<string>("");

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div className={`flex items-center justify-between p-2 rounded-xl border transition-colors ${
      status === "done"
        ? "border-emerald-200 dark:border-emerald-900/50 bg-emerald-50/30 dark:bg-emerald-950/10"
        : status === "processing"
        ? "border-indigo-200 dark:border-indigo-900/50 bg-indigo-50/20 dark:bg-indigo-950/10"
        : "border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-900/40 hover:bg-slate-100/50 dark:hover:bg-slate-900/70"
    }`}>
      <div className="flex items-center gap-3 overflow-hidden">
        <div className="relative shrink-0">
          {preview ? (
            <img
              src={preview}
              alt={file.name}
              className="w-10 h-10 rounded-lg object-cover border border-slate-200 dark:border-slate-800"
            />
          ) : (
            <div className="w-10 h-10 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
              <span className="material-symbols-outlined text-slate-400 select-none text-lg">image</span>
            </div>
          )}
          {status === "done" && (
            <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 border-2 border-white dark:border-slate-900 flex items-center justify-center">
              <span className="material-symbols-outlined text-white text-[9px] leading-none select-none">check</span>
            </div>
          )}
          {status === "processing" && (
            <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-indigo-400 border-2 border-white dark:border-slate-900 flex items-center justify-center">
              <span className="material-symbols-outlined text-white text-[9px] leading-none animate-spin select-none">sync</span>
            </div>
          )}
        </div>
        <div className="overflow-hidden">
          <p
            className="text-[10px] font-semibold text-slate-700 dark:text-slate-300 truncate max-w-[130px]"
            title={file.name}
          >
            {file.name}
          </p>
          <p className="text-[9px] text-slate-400 dark:text-slate-500">
            {(file.size / 1024).toFixed(0)} KB
            {status === "done" && <span className="ml-1 text-emerald-500 font-semibold">· done</span>}
            {status === "processing" && <span className="ml-1 text-indigo-400 font-semibold">· analysing…</span>}
          </p>
        </div>
      </div>
      <button
        onClick={onRemove}
        disabled={disabled}
        className="text-slate-300 dark:text-slate-600 hover:text-red-500 p-1 rounded-full hover:bg-red-50 dark:hover:bg-red-950/30 transition-all shrink-0 disabled:opacity-50 cursor-pointer"
      >
        <span className="material-symbols-outlined text-sm select-none leading-none block">close</span>
      </button>
    </div>
  );
}

function PipelineBadge({ status }: { status: ModelStatus }) {
  if (!status)
    return (
      <div className="flex items-center gap-1.5 px-3 py-1 bg-slate-50 dark:bg-slate-900 rounded-full border border-slate-200 dark:border-slate-800">
        <span className="w-2 h-2 rounded-full bg-slate-300 dark:bg-slate-600 animate-pulse" />
        <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">Checking…</span>
      </div>
    );
  if (status.error)
    return (
      <div
        className="flex items-center gap-1.5 px-3 py-1 bg-red-50 dark:bg-red-950/30 rounded-full border border-red-200 dark:border-red-900/50"
        title={status.error}
      >
        <span className="w-2 h-2 rounded-full bg-red-500" />
        <span className="text-xs font-semibold text-red-700 dark:text-red-400">Load Error</span>
      </div>
    );
  if (!status.models_loaded)
    return (
      <div className="flex items-center gap-1.5 px-3 py-1 bg-amber-50 dark:bg-amber-950/30 rounded-full border border-amber-200 dark:border-amber-900/50">
        <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
        <span className="text-xs font-semibold text-amber-700 dark:text-amber-400">Loading Models…</span>
      </div>
    );
  return (
    <div className="flex items-center gap-1.5 px-3 py-1 bg-emerald-50 dark:bg-emerald-950/30 rounded-full border border-emerald-200 dark:border-emerald-900/50">
      <span className="w-2 h-2 rounded-full bg-emerald-500" />
      <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-400">Pipeline Ready</span>
    </div>
  );
}

function AgreementBadge({ level }: { level?: string }) {
  if (!level) return null;
  const styles =
    level === "High"
      ? "bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-900/30"
      : level === "Medium"
      ? "bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-900/30"
      : "bg-red-100 dark:bg-red-950/60 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-900/30";
  return (
    <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full border ${styles}`}>
      {level}
    </span>
  );
}

function ModelTag({ name }: { name: string }) {
  const color =
    name === "MDv5a" || name === "MDv1000"
      ? "bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-400 border-blue-200/50 dark:border-blue-900/30"
      : name === "BioClip"
      ? "bg-violet-100 dark:bg-violet-950/60 text-violet-700 dark:text-violet-400 border-violet-200/50 dark:border-violet-900/30"
      : name === "SpeciesNet"
      ? "bg-teal-100 dark:bg-teal-950/60 text-teal-700 dark:text-teal-400 border-teal-200/50 dark:border-teal-900/30"
      : name === "Detection"
      ? "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700"
      : "bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 border-emerald-200/50 dark:border-emerald-900/30";
  return (
    <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${color}`}>
      {name}
    </span>
  );
}

/** Crops the detection bounding box from the source file onto a canvas. */
function DetectionCrop({ file, bbox }: { file?: File; bbox?: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!file || !canvasRef.current) return;
    let alive = true;
    const objUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      if (!alive || !canvasRef.current) return;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const [bx, by, bw, bh] =
        bbox && bbox.length >= 4 ? bbox : [0, 0, 1, 1];
      const pad = 0.06;
      const sx = Math.max(0, (bx - pad)) * img.naturalWidth;
      const sy = Math.max(0, (by - pad)) * img.naturalHeight;
      const sw = Math.min(img.naturalWidth - sx, (bw + pad * 2) * img.naturalWidth);
      const sh = Math.min(img.naturalHeight - sy, (bh + pad * 2) * img.naturalHeight);

      canvas.width = 100;
      canvas.height = 75;
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, 100, 75);
      if (alive) setReady(true);
    };
    img.src = objUrl;
    return () => {
      alive = false;
      URL.revokeObjectURL(objUrl);
    };
  }, [file, bbox]);

  if (!file) return null;

  return (
    <div className="shrink-0 relative" style={{ width: 100, height: 75 }}>
      {!ready && (
        <div className="absolute inset-0 rounded-lg bg-slate-100 dark:bg-slate-800 animate-pulse" />
      )}
      <canvas
        ref={canvasRef}
        className={`rounded-lg border border-slate-200 dark:border-slate-700 object-cover transition-opacity duration-300 ${
          ready ? "opacity-100" : "opacity-0"
        }`}
        style={{ width: 100, height: 75 }}
      />
    </div>
  );
}

/** Ranked species row with mini confidence bar. */
function SpeciesRow({
  label,
  conf,
  rank,
}: {
  label: string;
  conf: number;
  rank: number;
}) {
  const { display, tooltip } = parseSnetLabel(label);
  const pct = Math.round(conf * 100);
  return (
    <div className="flex items-center gap-1.5 min-w-0">
      <span className="text-[9px] text-slate-400 dark:text-slate-500 w-3 shrink-0 text-right">
        {rank}.
      </span>
      <span
        title={tooltip || undefined}
        className={`text-[10px] text-slate-600 dark:text-slate-300 truncate flex-1 min-w-0 ${
          tooltip ? "underline decoration-dotted underline-offset-2 decoration-slate-400/50 cursor-help" : ""
        }`}
      >
        {display}
      </span>
      <div className="w-10 h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden shrink-0">
        <div
          className="h-1.5 rounded-full bg-teal-400 dark:bg-teal-600"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[9px] font-mono text-slate-400 shrink-0 w-6 text-right">{pct}%</span>
    </div>
  );
}

/** One processed-image card in the live panel */
function ImageResultCard({
  row,
  file,
  flagged,
  onFlag,
}: {
  row: ImageRow;
  file?: File;
  flagged: boolean;
  onFlag: () => void;
}) {
  const detMDv5a = row.events.find((e) => e.model === "MDv5a");
  const detMDv1000 = row.events.find((e) => e.model === "MDv1000");
  const detFusion = row.events.find((e) => e.model === "Detection");
  const evBC = row.events.find((e) => e.model === "BioClip");
  const evSN = row.events.find((e) => e.model === "SpeciesNet");
  const evResult = row.events.find((e) => e.model === "Result");

  const isEmpty = !evResult || evResult.confidence === 0;
  const isLowConf = !isEmpty && evResult && (evResult.confidence ?? 0) < 0.4;

  // First bbox from merged detection or MDv5a
  const firstBbox =
    detFusion?.detections?.[0]?.bbox ??
    detMDv5a?.detections?.[0]?.bbox ??
    detMDv1000?.detections?.[0]?.bbox;

  // Compact row for empty/non-animal frames
  if (isEmpty) {
    const personLabels = [detMDv5a, detMDv1000]
      .flatMap((e) => e?.detections ?? [])
      .filter((d) => d.label.toLowerCase() !== "animal")
      .map((d) => `${d.label} ${d.conf.toFixed(2)}`);
    const isPersonOrVehicle = personLabels.length > 0;
    return (
      <div className="flex items-center gap-2.5 px-3 py-2 rounded-xl border border-slate-100 dark:border-slate-800/60 bg-slate-50/40 dark:bg-slate-900/30">
        <span className={`material-symbols-outlined text-sm select-none shrink-0 ${isPersonOrVehicle ? "text-blue-400" : "text-slate-300 dark:text-slate-600"}`}>
          {isPersonOrVehicle ? "person" : "hide_image"}
        </span>
        <p className="text-[10px] font-semibold text-slate-600 dark:text-slate-400 truncate flex-1 min-w-0" title={row.name}>
          {row.name}
        </p>
        <span className="text-[9px] text-slate-400 dark:text-slate-500 italic shrink-0">
          {isPersonOrVehicle ? personLabels.join(", ") : "empty"}
        </span>
        <button
          onClick={onFlag}
          title={flagged ? "Remove flag" : "Flag for review"}
          className={`p-0.5 rounded-full shrink-0 transition-colors cursor-pointer ${flagged ? "text-amber-500" : "text-slate-300 dark:text-slate-600 hover:text-amber-500"}`}
        >
          <span className="material-symbols-outlined text-sm select-none leading-none block">{flagged ? "flag" : "outlined_flag"}</span>
        </button>
      </div>
    );
  }

  return (
    <div
      className={`rounded-xl border overflow-hidden shadow-sm hover:shadow-md transition-shadow duration-200 ${
        isLowConf
          ? "border-amber-200 dark:border-amber-900/50 bg-amber-50/30 dark:bg-amber-950/10"
          : "border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60"
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-slate-50 dark:bg-slate-950/80 border-b border-slate-100 dark:border-slate-800">
        <p className="text-[10px] font-bold text-slate-700 dark:text-slate-300 truncate max-w-[160px]">
          {row.name}
        </p>
        <div className="flex items-center gap-1.5">
          {isLowConf && (
            <span
              className="material-symbols-outlined text-amber-500 text-sm select-none"
              title="Low confidence — verify manually"
            >
              warning
            </span>
          )}
          {evResult && !isEmpty && <AgreementBadge level={evResult.agreement} />}
          <button
            onClick={onFlag}
            title={flagged ? "Remove review flag" : "Flag for review"}
            className={`p-0.5 rounded-full transition-colors cursor-pointer ${
              flagged
                ? "text-amber-500 hover:text-amber-600"
                : "text-slate-300 dark:text-slate-600 hover:text-amber-500"
            }`}
          >
            <span className="material-symbols-outlined text-sm select-none leading-none block">
              {flagged ? "flag" : "outlined_flag"}
            </span>
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="p-3 space-y-2 text-[10px]">
        {/* Detection crop + detector row */}
        <div className="flex gap-2.5">
          {!isEmpty && (
            <DetectionCrop file={file} bbox={firstBbox} />
          )}
          <div className="flex-1 space-y-1.5 min-w-0">
            {/* Detectors */}
            <div className="flex flex-wrap items-center gap-1">
              <ModelTag name="MDv5a" />
              <span className="text-slate-600 dark:text-slate-400">
                {detMDv5a?.detections?.length
                  ? detMDv5a.detections
                      .map((d) => `${d.label} ${d.conf.toFixed(2)}`)
                      .join(", ")
                  : "—"}
              </span>
              {detMDv1000 && (
                <>
                  <span className="text-slate-300 dark:text-slate-700">|</span>
                  <ModelTag name="MDv1000" />
                  <span className="text-slate-600 dark:text-slate-400">
                    {detMDv1000.detections?.length
                      ? detMDv1000.detections
                          .map((d) => `${d.label} ${d.conf.toFixed(2)}`)
                          .join(", ")
                      : "—"}
                  </span>
                </>
              )}
            </div>
            {detFusion && (
              <span className="text-slate-400 dark:text-slate-500 italic">
                → {detFusion.merged_count ?? 1} detection(s) merged
              </span>
            )}

            {/* BioCLIP top results */}
            {!isEmpty && evBC?.top5?.length && (
              <div className="flex flex-wrap items-center gap-1">
                <ModelTag name="BioClip" />
                <span className="text-slate-600 dark:text-slate-400">
                  {evBC.top5
                    .slice(0, 2)
                    .map(([s, c]) => `${s} ${(c as number).toFixed(2)}`)
                    .join(", ")}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* SpeciesNet ranked list */}
        {!isEmpty && (
          <>
            {evSN?.skipped ? (
              <div className="flex items-center gap-1.5">
                <ModelTag name="SpeciesNet" />
                <span className="text-slate-400 dark:text-slate-500 italic">not loaded</span>
              </div>
            ) : evSN?.top5?.length ? (
              <div className="space-y-0.5">
                <div className="flex items-center gap-1.5 mb-1">
                  <ModelTag name="SpeciesNet" />
                </div>
                {evSN.top5.slice(0, 3).map(([s, c], i) => (
                  <SpeciesRow key={i} label={s} conf={c as number} rank={i + 1} />
                ))}
              </div>
            ) : null}
          </>
        )}

        {/* Final result */}
        <div
          className={`flex items-center gap-2 pt-1.5 border-t border-slate-100 dark:border-slate-800 ${
            isEmpty
              ? "text-slate-400 dark:text-slate-500"
              : isLowConf
              ? "text-amber-700 dark:text-amber-400"
              : "text-slate-800 dark:text-slate-200"
          }`}
        >
          <span className="material-symbols-outlined text-sm select-none">
            {isEmpty ? "help_outline" : isLowConf ? "warning" : "check_circle"}
          </span>
          {isEmpty ? (
            <span className="italic">No animal detected</span>
          ) : (
            <span className="font-bold">
              {evResult?.species}{" "}
              <span
                className={`font-normal ${
                  isLowConf ? "text-amber-500" : "text-slate-500 dark:text-slate-400"
                }`}
              >
                {((evResult?.confidence ?? 0) * 100).toFixed(0)}%
              </span>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function Upload() {
  const reviewerId = useConfigStore((s) => s.config?.reviewer_id ?? "anonymous");

  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);

  type UploadPhase = "idle" | "uploading" | "ready" | "upload_error";
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [uploadedJobId, setUploadedJobId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [job, setJob] = useState<JobState | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus>(null);
  const [imageRows, setImageRows] = useState<ImageRow[]>([]);
  const [flaggedImages, setFlaggedImages] = useState<Set<string>>(new Set());

  // Timer for elapsed/ETA display
  const startTimeRef = useRef<number | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const inputRef = useRef<HTMLInputElement>(null);
  const sseRef = useRef<EventSource | null>(null);
  const uploadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resultsScrollRef = useRef<HTMLDivElement>(null);
  // Mirror of flaggedImages kept in a ref so SSE closure always reads the latest value
  const flaggedImagesRef = useRef<Set<string>>(new Set());
  const navigate = useNavigate();

  // Poll model status until ready
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const s = await getModelStatus();
        if (!cancelled) {
          setModelStatus(s);
          if (!s.models_loaded && !s.error) setTimeout(check, 3000);
        }
      } catch {
        if (!cancelled) setTimeout(check, 5000);
      }
    };
    check();
    return () => {
      cancelled = true;
    };
  }, []);

  // Elapsed-time ticker while running
  useEffect(() => {
    if (job?.status === "running") {
      startTimeRef.current = startTimeRef.current ?? Date.now();
      const interval = setInterval(() => {
        setElapsed(Math.floor((Date.now() - (startTimeRef.current ?? Date.now())) / 1000));
      }, 1000);
      return () => clearInterval(interval);
    } else {
      if (job?.status !== "running") startTimeRef.current = null;
      setElapsed(0);
    }
  }, [job?.status]);

  useEffect(() => {
    return () => sseRef.current?.close();
  }, []);

  // ── Auto-upload whenever files change (debounced 400 ms) ─────────────────

  const doUpload = useCallback(async (toUpload: File[]) => {
    if (!toUpload.length) return;
    setUploadPhase("uploading");
    setUploadError(null);
    setUploadedJobId(null);
    try {
      const { job_id } = await uploadImages(toUpload);
      setUploadedJobId(job_id);
      setUploadPhase("ready");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setUploadError(msg);
      setUploadPhase("upload_error");
    }
  }, []);

  const scheduleUpload = useCallback(
    (updatedFiles: File[]) => {
      if (uploadTimerRef.current) clearTimeout(uploadTimerRef.current);
      uploadTimerRef.current = setTimeout(() => doUpload(updatedFiles), 400);
    },
    [doUpload]
  );

  const addFiles = useCallback(
    (incoming: File[]) => {
      const valid = incoming.filter((f) => /\.(jpe?g|png)$/i.test(f.name));
      if (!valid.length) return;
      setFiles((prev) => {
        const merged = [...prev, ...valid];
        scheduleUpload(merged);
        return merged;
      });
    },
    [scheduleUpload]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      addFiles(Array.from(e.dataTransfer.files));
    },
    [addFiles]
  );

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(Array.from(e.target.files));
  };

  const removeFile = (i: number) => {
    setFiles((prev) => {
      const next = prev.filter((_, idx) => idx !== i);
      scheduleUpload(next);
      return next;
    });
  };

  const clearAll = () => {
    setFiles([]);
    setUploadPhase("idle");
    setUploadedJobId(null);
    setUploadError(null);
    setJob(null);
    setImageRows([]);
    setFlaggedImages(new Set());
    flaggedImagesRef.current = new Set();
    startTimeRef.current = null;
    if (uploadTimerRef.current) clearTimeout(uploadTimerRef.current);
  };

  const toggleFlag = useCallback((name: string) => {
    setFlaggedImages((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      flaggedImagesRef.current = next;
      return next;
    });
  }, []);

  // ── Start processing ──────────────────────────────────────────────────────

  const handleProcess = async () => {
    if (!uploadedJobId) return;

    sseRef.current?.close();
    startTimeRef.current = Date.now();
    setJob({ jobId: uploadedJobId, status: "running", total: files.length, completed: 0 });
    setImageRows([]);
    setFlaggedImages(new Set());
    flaggedImagesRef.current = new Set();

    try {
      await startProcessing(uploadedJobId);

      const sse = new EventSource(`/api/images/job/${uploadedJobId}/stream`);
      sseRef.current = sse;

      sse.onmessage = (e) => {
        const data = JSON.parse(e.data) as { type: string } & Record<string, unknown>;

        if (data.type === "progress") {
          const prog = data as unknown as {
            status: string;
            total: number;
            completed: number;
            error?: string;
          };
          setJob({ jobId: uploadedJobId, ...prog });

          if (prog.status === "done") {
            sse.close();
            getJobResults(uploadedJobId);
            // Persist any upload-page flags to the review log (non-fatal)
            const toFlag = Array.from(flaggedImagesRef.current);
            if (toFlag.length > 0) {
              flagByFilenames(toFlag, reviewerId).catch(() => {});
            }
          } else if (prog.status === "error") {
            sse.close();
          }
        } else if (data.type === "model_event") {
          const ev = data as unknown as ModelEvent;
          setImageRows((prev) => {
            const idx = prev.findIndex((r) => r.name === ev.image);
            if (idx === -1) {
              return [...prev, { name: ev.image, index: ev.image_index, events: [ev] }];
            }
            const updated = [...prev];
            updated[idx] = { ...updated[idx], events: [...updated[idx].events, ev] };
            return updated;
          });
        }
      };

      sse.onerror = () => {
        sse.close();
        setJob((prev) =>
          prev ? { ...prev, status: "error", error: "Stream connection lost." } : prev
        );
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setJob((prev) => (prev ? { ...prev, status: "error", error: msg } : prev));
    }
  };

  // ── Derived state ─────────────────────────────────────────────────────────

  const processing = job?.status === "running";
  const pct = job && job.total > 0 ? Math.round((job.completed / job.total) * 100) : 0;
  const etaSecs =
    processing && pct > 5 && elapsed > 0
      ? Math.round((elapsed / pct) * (100 - pct))
      : null;

  // Per-file status for the file list
  const fileStatuses = new Map<string, FileStatus>();
  for (const row of imageRows) {
    const hasDone = row.events.some((e) => e.model === "Result");
    fileStatuses.set(row.name, hasDone ? "done" : "processing");
  }

  // Auto-scroll live results to bottom when new cards appear
  useEffect(() => {
    if (resultsScrollRef.current) {
      resultsScrollRef.current.scrollTop = resultsScrollRef.current.scrollHeight;
    }
  }, [imageRows.length]);

  const uploadStatusLabel = () => {
    if (uploadPhase === "uploading") return "Uploading…";
    if (uploadPhase === "ready") return `${files.length} file(s) ready`;
    if (uploadPhase === "upload_error") return `Upload failed: ${uploadError}`;
    return null;
  };

  const canStart =
    uploadPhase === "ready" &&
    !!uploadedJobId &&
    !processing &&
    !!modelStatus?.models_loaded;

  // Completion stats
  const resultRows = imageRows.filter((r) => {
    const res = r.events.find((e) => e.model === "Result");
    return res && (res.confidence ?? 0) > 0;
  });
  const speciesSet = new Set(
    resultRows.map((r) => r.events.find((e) => e.model === "Result")?.species)
  );
  const lowConfRows = resultRows.filter(
    (r) => (r.events.find((e) => e.model === "Result")?.confidence ?? 1) < 0.4
  );
  const needsReviewCount = new Set([
    ...lowConfRows.map((r) => r.name),
    ...flaggedImages,
  ]).size;

  // Pipeline step detection
  const allEvents = imageRows.flatMap((r) => r.events);
  const latestModel = allEvents.length > 0 ? allEvents[allEvents.length - 1].model : null;
  const atDetect =
    latestModel === "MDv5a" ||
    latestModel === "MDv1000" ||
    latestModel === "Detection";
  const atClassify = latestModel === "BioClip" || latestModel === "SpeciesNet";
  const atFusion = latestModel === "Result";

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
            Upload & Process
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1.5 text-sm">
            Images upload automatically on selection. Click Start when ready.
          </p>
        </div>
        <PipelineBadge status={modelStatus} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* ── Left: drop zone + file list + button ── */}
        <div className="lg:col-span-1 space-y-5">
          {/* Drop zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 group ${
              dragging
                ? "border-emerald-500 bg-emerald-50/60 dark:bg-emerald-950/20 shadow-inner scale-[0.99] ring-2 ring-emerald-500/20"
                : "border-slate-300 dark:border-slate-800 bg-white dark:bg-slate-900/40 hover:border-emerald-400 dark:hover:border-emerald-500 hover:bg-emerald-50/20 dark:hover:bg-emerald-950/10 hover:shadow-md"
            }`}
          >
            <span
              className={`material-symbols-outlined text-4xl mb-2 block select-none transition-transform duration-300 ${
                dragging
                  ? "scale-110 text-emerald-600 animate-bounce"
                  : "text-slate-400 dark:text-slate-500 group-hover:text-emerald-500"
              }`}
            >
              cloud_upload
            </span>
            <p className="text-slate-700 dark:text-slate-300 font-semibold text-sm">
              Drop camera images here
            </p>
            <p className="text-slate-400 dark:text-slate-500 text-xs mt-1">JPG / JPEG / PNG</p>
            <button
              type="button"
              className="mt-4 px-4 py-1.5 bg-slate-900 dark:bg-slate-800 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 dark:hover:bg-slate-700 transition-colors shadow-sm cursor-pointer"
            >
              Browse Files
            </button>
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png"
              multiple
              className="hidden"
              onChange={onFileInput}
            />
          </div>

          {/* Upload status badge */}
          {uploadPhase !== "idle" && (
            <div
              className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold border ${
                uploadPhase === "uploading"
                  ? "bg-amber-50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-900/50 text-amber-700 dark:text-amber-400"
                  : uploadPhase === "ready"
                  ? "bg-emerald-50 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-900/50 text-emerald-700 dark:text-emerald-400"
                  : "bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-900/50 text-red-700 dark:text-red-400"
              }`}
            >
              {uploadPhase === "uploading" && (
                <span className="material-symbols-outlined text-base animate-spin select-none">
                  sync
                </span>
              )}
              {uploadPhase === "ready" && (
                <span className="material-symbols-outlined text-base select-none">
                  check_circle
                </span>
              )}
              {uploadPhase === "upload_error" && (
                <span className="material-symbols-outlined text-base select-none">error</span>
              )}
              {uploadStatusLabel()}
            </div>
          )}

          {/* File list */}
          {files.length > 0 && (
            <div className="bg-white dark:bg-slate-900/60 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
              <div className="px-4 py-2.5 bg-slate-50 dark:bg-slate-950/80 border-b border-slate-105 flex justify-between items-center">
                <span className="font-semibold text-slate-700 dark:text-slate-300 text-xs flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-base text-slate-500 dark:text-slate-400 select-none">
                    photo_library
                  </span>
                  {files.length} image(s)
                </span>
                <button
                  onClick={clearAll}
                  disabled={processing}
                  className="text-xs font-semibold text-slate-400 hover:text-red-500 dark:hover:text-red-400 transition-colors disabled:opacity-50 cursor-pointer"
                >
                  Clear All
                </button>
              </div>
              <div className="p-3 max-h-48 overflow-y-auto custom-scrollbar space-y-1.5">
                {files.map((f, i) => (
                  <ThumbnailItem
                    key={i}
                    file={f}
                    onRemove={() => removeFile(i)}
                    disabled={processing}
                    status={fileStatuses.get(f.name) ?? "pending"}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Start button */}
          <button
            onClick={handleProcess}
            disabled={!canStart}
            className="w-full py-3 px-6 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-200 dark:disabled:bg-slate-800 disabled:text-slate-400 dark:disabled:text-slate-600 text-white font-semibold rounded-2xl transition-all duration-150 shadow-md hover:shadow-lg flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
          >
            {processing ? (
              <>
                <span className="material-symbols-outlined text-lg select-none animate-spin">
                  sync
                </span>
                Analysing…
              </>
            ) : !modelStatus?.models_loaded ? (
              <>
                <span className="material-symbols-outlined text-lg select-none">
                  hourglass_empty
                </span>
                {modelStatus?.error ? "Models failed to load" : "Waiting for models…"}
              </>
            ) : uploadPhase === "idle" || !files.length ? (
              <>
                <span className="material-symbols-outlined text-lg select-none">upload_file</span>
                Select files to begin
              </>
            ) : uploadPhase === "uploading" ? (
              <>
                <span className="material-symbols-outlined text-lg select-none animate-spin">
                  sync
                </span>
                Uploading…
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-lg select-none">play_circle</span>
                Start AI Analysis
              </>
            )}
          </button>

          {/* Progress widget */}
          {job && (
            <div className="bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 p-4 space-y-4 shadow-sm rounded-2xl">
              <div className="flex justify-between items-center text-xs">
                <span className="font-bold text-slate-700 dark:text-slate-300">
                  {job.status === "done"
                    ? "Complete"
                    : job.status === "error"
                    ? "Failed"
                    : "Running…"}
                </span>
                <div className="flex items-center gap-2">
                  {processing && elapsed > 0 && (
                    <span className="font-mono text-slate-400">
                      {fmtTime(elapsed)}
                      {etaSecs != null && (
                        <span className="text-slate-400 dark:text-slate-400">
                          {" "}/ ~{fmtTime(etaSecs)} left
                        </span>
                      )}
                    </span>
                  )}
                  <span className="font-mono font-bold text-emerald-600 dark:text-emerald-450">
                    {pct}%
                  </span>
                </div>
              </div>

              <div className="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div
                  className={`h-1.5 rounded-full transition-all duration-300 ${
                    job.status === "error" ? "bg-red-500" : "bg-emerald-500"
                  }`}
                  style={{ width: `${job.status === "done" ? 100 : pct}%` }}
                />
              </div>

              {/* Pipeline Stepper */}
              {job.status === "running" && (
                <div className="pt-2 border-t border-slate-100 dark:border-slate-800 space-y-2.5">
                  <p className="text-[10px] uppercase font-bold text-slate-400 dark:text-slate-500 tracking-wider">
                    Pipeline Progress
                  </p>
                  <div className="space-y-2">
                    {[
                      { id: "ocr", label: "OCR Date/Time Extraction" },
                      { id: "detect", label: "MegaDetector Bounding Boxes" },
                      { id: "classify", label: "BioClip & SpeciesNet Classifiers" },
                      { id: "fusion", label: "Ensemble Fusion & Database Write" },
                    ].map((step, sIdx) => {
                      let active = false;
                      let done = false;

                      if (job.status === "done") {
                        done = true;
                      } else if (step.id === "ocr") {
                        active = !latestModel;
                        done = atDetect || atClassify || atFusion;
                      } else if (step.id === "detect") {
                        active = atDetect;
                        done = atClassify || atFusion;
                      } else if (step.id === "classify") {
                        active = atClassify;
                        done = atFusion;
                      } else if (step.id === "fusion") {
                        active = atFusion;
                      }

                      return (
                        <div key={step.id} className="flex items-center gap-2 text-xs">
                          <div
                            className={`w-4 h-4 rounded-full flex items-center justify-center shrink-0 border transition ${
                              done
                                ? "bg-emerald-100 dark:bg-emerald-950/40 border-emerald-300 dark:border-emerald-800 text-emerald-600 dark:text-emerald-450"
                                : active
                                ? "bg-indigo-50 dark:bg-indigo-950/40 border-indigo-300 dark:border-indigo-800 text-indigo-600 dark:text-indigo-400 animate-pulse"
                                : "bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-400"
                            }`}
                          >
                            {done ? (
                              <span className="material-symbols-outlined text-[10px] leading-none font-bold">
                                check
                              </span>
                            ) : (
                              <span className="text-[9px] font-bold">{sIdx + 1}</span>
                            )}
                          </div>
                          <span
                            className={`font-semibold ${
                              done
                                ? "text-slate-400 dark:text-slate-500 line-through decoration-slate-300/40"
                                : active
                                ? "text-indigo-600 dark:text-indigo-400 font-bold"
                                : "text-slate-400 dark:text-slate-500"
                            }`}
                          >
                            {step.label}
                          </span>
                          {active && (
                            <span className="ml-auto shrink-0">
                              <span className="inline-flex gap-0.5">
                                {[0, 1, 2].map((i) => (
                                  <span
                                    key={i}
                                    className="w-1 h-1 rounded-full bg-indigo-400 animate-bounce"
                                    style={{ animationDelay: `${i * 0.15}s` }}
                                  />
                                ))}
                              </span>
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <div className="flex justify-between text-[10px] text-slate-400 dark:text-slate-500 font-medium pt-1">
                <span>Total: {job.total}</span>
                <span>Done: {job.completed}</span>
              </div>
              {job.error && (
                <p className="text-[10px] text-red-600 font-medium break-all">{job.error}</p>
              )}
            </div>
          )}

          {/* Completion summary */}
          {job?.status === "done" && imageRows.length > 0 && (
            <div className="bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900/50 rounded-2xl p-4 space-y-3">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-emerald-600 text-lg select-none">
                  check_circle
                </span>
                <span className="font-bold text-emerald-700 dark:text-emerald-400 text-sm">
                  Analysis complete
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="bg-white/60 dark:bg-slate-900/40 rounded-xl py-2">
                  <p className="text-lg font-extrabold text-slate-800 dark:text-slate-100">
                    {imageRows.length}
                  </p>
                  <p className="text-[9px] text-slate-500 font-semibold uppercase tracking-wide">
                    Images
                  </p>
                </div>
                <div className="bg-white/60 dark:bg-slate-900/40 rounded-xl py-2">
                  <p className="text-lg font-extrabold text-slate-800 dark:text-slate-100">
                    {speciesSet.size}
                  </p>
                  <p className="text-[9px] text-slate-500 font-semibold uppercase tracking-wide">
                    Species
                  </p>
                </div>
                <div
                  className={`rounded-xl py-2 ${
                    needsReviewCount > 0
                      ? "bg-amber-100/80 dark:bg-amber-950/40"
                      : "bg-white/60 dark:bg-slate-900/40"
                  }`}
                >
                  <p
                    className={`text-lg font-extrabold ${
                      needsReviewCount > 0
                        ? "text-amber-600 dark:text-amber-400"
                        : "text-slate-800 dark:text-slate-100"
                    }`}
                  >
                    {needsReviewCount}
                  </p>
                  <p className="text-[9px] text-slate-500 font-semibold uppercase tracking-wide">
                    Review
                  </p>
                </div>
              </div>
              <button
                onClick={() => navigate("/results")}
                className="w-full py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-xl transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
              >
                View Full Results
                <span className="material-symbols-outlined text-sm select-none">
                  arrow_forward
                </span>
              </button>
            </div>
          )}
        </div>

        {/* ── Right: live model output panel ── */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-slate-900/60 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden flex flex-col h-full min-h-[520px]">
            {/* Panel header */}
            <div className="px-4 py-3 bg-slate-50 dark:bg-slate-950/80 border-b border-slate-105 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-slate-500 dark:text-slate-400 select-none text-lg">
                  model_training
                </span>
                <span className="font-bold text-slate-700 dark:text-slate-300 text-sm">
                  Live Model Output
                </span>
                {flaggedImages.size > 0 && (
                  <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-amber-100 dark:bg-amber-950/50 text-amber-600 dark:text-amber-400 border border-amber-200 dark:border-amber-900/30">
                    {flaggedImages.size} flagged
                  </span>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-3 text-[10px] font-semibold">
                <span className="flex items-center gap-1 text-blue-600 dark:text-blue-400">
                  <span className="w-2 h-2 rounded bg-blue-200 dark:bg-blue-900/50" /> Detectors
                </span>
                <span className="flex items-center gap-1 text-violet-600 dark:text-violet-400">
                  <span className="w-2 h-2 rounded bg-violet-200 dark:bg-violet-900/50" /> BioClip
                </span>
                <span className="flex items-center gap-1 text-teal-600 dark:text-teal-400">
                  <span className="w-2 h-2 rounded bg-teal-200 dark:bg-teal-900/50" /> SpeciesNet
                </span>
                <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-450">
                  <span className="w-2 h-2 rounded bg-emerald-200 dark:bg-emerald-900/50" /> Result
                </span>
              </div>
            </div>

            {/* Scrollable card grid */}
            <div ref={resultsScrollRef} className="flex-1 overflow-y-auto custom-scrollbar p-4">
              {imageRows.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-400 dark:text-slate-500 gap-3">
                  <span className="material-symbols-outlined text-5xl select-none">analytics</span>
                  <p className="text-sm font-medium">
                    {processing
                      ? "Processing… results will appear here"
                      : "Model results will stream here as images are analysed"}
                  </p>
                  {processing && (
                    <div className="flex gap-1">
                      {[0, 1, 2].map((i) => (
                        <span
                          key={i}
                          className="w-2 h-2 rounded-full bg-emerald-400 dark:bg-emerald-500 animate-bounce"
                          style={{ animationDelay: `${i * 0.15}s` }}
                        />
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {imageRows.map((row) => {
                    const file = files.find((f) => f.name === row.name);
                    return (
                      <ImageResultCard
                        key={row.name}
                        row={row}
                        file={file}
                        flagged={flaggedImages.has(row.name)}
                        onFlag={() => toggleFlag(row.name)}
                      />
                    );
                  })}
                </div>
              )}
            </div>

            {/* Summary footer */}
            {imageRows.length > 0 && (
              <div className="px-4 py-2.5 bg-slate-50 dark:bg-slate-950/80 border-t border-slate-100 dark:border-slate-800 text-[10px] text-slate-500 dark:text-slate-400 font-medium flex items-center justify-between">
                <span>{imageRows.length} image(s) processed</span>
                <div className="flex items-center gap-3">
                  {lowConfRows.length > 0 && (
                    <span className="flex items-center gap-1 text-amber-500 dark:text-amber-400">
                      <span className="material-symbols-outlined text-xs select-none">warning</span>
                      {lowConfRows.length} low confidence
                    </span>
                  )}
                  {flaggedImages.size > 0 && (
                    <span className="flex items-center gap-1 text-amber-500 dark:text-amber-400">
                      <span className="material-symbols-outlined text-xs select-none">flag</span>
                      {flaggedImages.size} flagged
                    </span>
                  )}
                  <span>
                    {resultRows.length} animal(s) found
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
