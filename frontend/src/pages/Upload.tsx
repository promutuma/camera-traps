import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadImages, startProcessing, getJobResults, getModelStatus } from "../api/client";

// ── Types ─────────────────────────────────────────────────────────────────────

type JobState = {
  jobId: string;
  status: string;
  total: number;
  completed: number;
  error?: string;
};

type ModelStatus = { models_loaded: boolean; error: string | null } | null;

type Detection = { label: string; conf: number };

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

// Group events by image name for display
type ImageRow = {
  name: string;
  index: number;
  events: ModelEvent[];
};

// ── Small components ──────────────────────────────────────────────────────────

function ThumbnailItem({ file, onRemove, disabled }: { file: File; onRemove: () => void; disabled: boolean }) {
  const [preview, setPreview] = useState<string>("");

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div className="flex items-center justify-between p-2 rounded-xl border border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-900/40 hover:bg-slate-100/50 dark:hover:bg-slate-900/70 transition-colors">
      <div className="flex items-center gap-3 overflow-hidden">
        {preview ? (
          <img
            src={preview}
            alt={file.name}
            className="w-10 h-10 rounded-lg object-cover border border-slate-200 dark:border-slate-800 shrink-0"
          />
        ) : (
          <div className="w-10 h-10 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center shrink-0">
            <span className="material-symbols-outlined text-slate-400 select-none text-lg">image</span>
          </div>
        )}
        <div className="overflow-hidden">
          <p className="text-[10px] font-semibold text-slate-700 dark:text-slate-300 truncate max-w-[140px]" title={file.name}>
            {file.name}
          </p>
          <p className="text-[9px] text-slate-400 dark:text-slate-500">{(file.size / 1024).toFixed(0)} KB</p>
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
      <div className="flex items-center gap-1.5 px-3 py-1 bg-red-50 dark:bg-red-950/30 rounded-full border border-red-200 dark:border-red-900/50" title={status.error}>
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

/** One processed-image card in the live panel */
function ImageResultCard({ row }: { row: ImageRow }) {
  const detMDv5a = row.events.find((e) => e.model === "MDv5a");
  const detMDv1000 = row.events.find((e) => e.model === "MDv1000");
  const detFusion = row.events.find((e) => e.model === "Detection");
  const evBC = row.events.find((e) => e.model === "BioClip");
  const evSN = row.events.find((e) => e.model === "SpeciesNet");
  const evResult = row.events.find((e) => e.model === "Result");

  const isEmpty = !evResult || evResult.confidence === 0;

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60 overflow-hidden shadow-sm hover:shadow-md transition-shadow duration-200">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-slate-50 dark:bg-slate-950/80 border-b border-slate-100 dark:border-slate-850">
        <p className="text-[10px] font-bold text-slate-700 dark:text-slate-300 truncate max-w-[180px]">{row.name}</p>
        {evResult && !isEmpty && (
          <AgreementBadge level={evResult.agreement} />
        )}
      </div>

      {/* Body */}
      <div className="p-3 space-y-2 text-[10px]">
        {/* Detection row */}
        <div className="flex flex-wrap items-center gap-1.5">
          <ModelTag name="MDv5a" />
          <span className="text-slate-600 dark:text-slate-400">
            {detMDv5a?.detections?.length
              ? detMDv5a.detections.map((d) => `${d.label} ${d.conf.toFixed(2)}`).join(", ")
              : "—"}
          </span>
          {detMDv1000 && (
            <>
              <span className="text-slate-300 dark:text-slate-700">|</span>
              <ModelTag name="MDv1000" />
              <span className="text-slate-600 dark:text-slate-400">
                {detMDv1000.detections?.length
                  ? detMDv1000.detections.map((d) => `${d.label} ${d.conf.toFixed(2)}`).join(", ")
                  : "—"}
              </span>
            </>
          )}
          {detFusion && (
            <span className="text-slate-400 dark:text-slate-550 italic">
              → {detFusion.merged_count} merged
            </span>
          )}
        </div>

        {/* Classification row */}
        {!isEmpty && (
          <>
            <div className="flex flex-wrap items-center gap-1.5">
              <ModelTag name="BioClip" />
              <span className="text-slate-600 dark:text-slate-400">
                {evBC?.top5?.length
                  ? evBC.top5.slice(0, 3).map(([s, c]) => `${s} ${c.toFixed(2)}`).join(", ")
                  : "—"}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <ModelTag name="SpeciesNet" />
              {evSN?.skipped ? (
                <span className="text-slate-400 dark:text-slate-550 italic">not loaded</span>
              ) : (
                <span className="text-slate-600 dark:text-slate-400">
                  {evSN?.top5?.length
                    ? evSN.top5.slice(0, 3).map(([s, c]) => `${s} ${c.toFixed(2)}`).join(", ")
                    : "—"}
                </span>
              )}
            </div>
          </>
        )}

        {/* Final result */}
        <div className={`flex items-center gap-2 pt-1 border-t border-slate-100 dark:border-slate-800 ${isEmpty ? "text-slate-400 dark:text-slate-500" : "text-slate-800 dark:text-slate-200"}`}>
          <span className="material-symbols-outlined text-sm select-none">
            {isEmpty ? "help_outline" : "check_circle"}
          </span>
          {isEmpty ? (
            <span className="italic">No animal detected</span>
          ) : (
            <span className="font-bold">
              {evResult?.species}{" "}
              <span className="font-normal text-slate-500 dark:text-slate-400">
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
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);

  // Upload phase: idle → uploading → ready (files on server, job_id known)
  type UploadPhase = "idle" | "uploading" | "ready" | "upload_error";
  const [uploadPhase, setUploadPhase] = useState<UploadPhase>("idle");
  const [uploadedJobId, setUploadedJobId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Processing phase
  const [job, setJob] = useState<JobState | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus>(null);

  // Live model output
  const [imageRows, setImageRows] = useState<ImageRow[]>([]);

  const inputRef = useRef<HTMLInputElement>(null);
  const sseRef = useRef<EventSource | null>(null);
  const uploadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
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
    return () => { cancelled = true; };
  }, []);

  // Cleanup SSE on unmount
  useEffect(() => { return () => sseRef.current?.close(); }, []);

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

  const scheduleUpload = useCallback((updatedFiles: File[]) => {
    if (uploadTimerRef.current) clearTimeout(uploadTimerRef.current);
    uploadTimerRef.current = setTimeout(() => doUpload(updatedFiles), 400);
  }, [doUpload]);

  const addFiles = useCallback((incoming: File[]) => {
    const valid = incoming.filter((f) => /\.(jpe?g|png)$/i.test(f.name));
    if (!valid.length) return;
    setFiles((prev) => {
      const merged = [...prev, ...valid];
      scheduleUpload(merged);
      return merged;
    });
  }, [scheduleUpload]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    addFiles(Array.from(e.dataTransfer.files));
  }, [addFiles]);

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
    if (uploadTimerRef.current) clearTimeout(uploadTimerRef.current);
  };

  // ── Start processing ──────────────────────────────────────────────────────

  const handleProcess = async () => {
    if (!uploadedJobId) return;

    sseRef.current?.close();
    setJob({ jobId: uploadedJobId, status: "running", total: files.length, completed: 0 });
    setImageRows([]);

    try {
      await startProcessing(uploadedJobId);

      const sse = new EventSource(`/api/images/job/${uploadedJobId}/stream`);
      sseRef.current = sse;

      sse.onmessage = (e) => {
        const data = JSON.parse(e.data) as { type: string } & Record<string, unknown>;

        if (data.type === "progress") {
          const prog = data as unknown as { status: string; total: number; completed: number; error?: string };
          setJob({ jobId: uploadedJobId, ...prog });

          if (prog.status === "done") {
            sse.close();
            getJobResults(uploadedJobId).then(() => {
              setTimeout(() => navigate("/results"), 1200);
            });
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
        setJob((prev) => prev ? { ...prev, status: "error", error: "Stream connection lost." } : prev);
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setJob((prev) => prev ? { ...prev, status: "error", error: msg } : prev);
    }
  };

  // ── Derived state ─────────────────────────────────────────────────────────

  const processing = job?.status === "running";
  const pct = job && job.total > 0 ? Math.round((job.completed / job.total) * 100) : 0;

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

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">Upload & Process</h1>
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
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
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
                dragging ? "scale-110 text-emerald-600 animate-bounce" : "text-slate-400 dark:text-slate-500 group-hover:text-emerald-500"
              }`}
            >
              cloud_upload
            </span>
            <p className="text-slate-700 dark:text-slate-300 font-semibold text-sm">Drop camera images here</p>
            <p className="text-slate-400 dark:text-slate-500 text-xs mt-1">JPG / JPEG / PNG</p>
            <button
              type="button"
              className="mt-4 px-4 py-1.5 bg-slate-900 dark:bg-slate-800 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 dark:hover:bg-slate-700 transition-colors shadow-sm cursor-pointer"
            >
              Browse Files
            </button>
            <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png" multiple className="hidden" onChange={onFileInput} />
          </div>

          {/* Upload status badge */}
          {uploadPhase !== "idle" && (
            <div className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold border ${
              uploadPhase === "uploading"
                ? "bg-amber-50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-900/50 text-amber-700 dark:text-amber-400"
                : uploadPhase === "ready"
                ? "bg-emerald-50 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-900/50 text-emerald-700 dark:text-emerald-400"
                : "bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-900/50 text-red-700 dark:text-red-400"
            }`}>
              {uploadPhase === "uploading" && (
                <span className="material-symbols-outlined text-base animate-spin select-none">sync</span>
              )}
              {uploadPhase === "ready" && (
                <span className="material-symbols-outlined text-base select-none">check_circle</span>
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
                  <span className="material-symbols-outlined text-base text-slate-500 dark:text-slate-400 select-none">photo_library</span>
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
                <span className="material-symbols-outlined text-lg select-none animate-spin">sync</span>
                Analysing…
              </>
            ) : !modelStatus?.models_loaded ? (
              <>
                <span className="material-symbols-outlined text-lg select-none">hourglass_empty</span>
                {modelStatus?.error ? "Models failed to load" : "Waiting for models…"}
              </>
            ) : uploadPhase === "idle" || !files.length ? (
              <>
                <span className="material-symbols-outlined text-lg select-none">upload_file</span>
                Select files to begin
              </>
            ) : uploadPhase === "uploading" ? (
              <>
                <span className="material-symbols-outlined text-lg select-none animate-spin">sync</span>
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
            <div className="bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 p-4 space-y-3 shadow-sm rounded-2xl">
              <div className="flex justify-between items-center text-xs">
                <span className="font-bold text-slate-700 dark:text-slate-300">
                  {job.status === "done" ? "Complete" : job.status === "error" ? "Failed" : "Running…"}
                </span>
                <span className="font-mono font-bold text-emerald-600 dark:text-emerald-450">{pct}%</span>
              </div>
              <div className="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div
                  className={`h-1.5 rounded-full transition-all duration-300 ${
                    job.status === "error" ? "bg-red-500" : "bg-emerald-500"
                  }`}
                  style={{ width: `${job.status === "done" ? 100 : pct}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-slate-400 dark:text-slate-500 font-medium">
                <span>Total: {job.total}</span>
                <span>Done: {job.completed}</span>
              </div>
              {job.error && (
                <p className="text-[10px] text-red-600 font-medium break-all">{job.error}</p>
              )}
            </div>
          )}
        </div>

        {/* ── Right: live model output panel ── */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-slate-900/60 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden flex flex-col h-full min-h-[520px]">
            {/* Panel header */}
            <div className="px-4 py-3 bg-slate-50 dark:bg-slate-950/80 border-b border-slate-105 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-slate-500 dark:text-slate-400 select-none text-lg">model_training</span>
                <span className="font-bold text-slate-700 dark:text-slate-300 text-sm">Live Model Output</span>
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
            <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
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
                  {imageRows.map((row) => (
                    <ImageResultCard key={row.name} row={row} />
                  ))}
                </div>
              )}
            </div>

            {/* Summary footer */}
            {imageRows.length > 0 && (
              <div className="px-4 py-2.5 bg-slate-50 dark:bg-slate-950/80 border-t border-slate-100 dark:border-slate-800 text-[10px] text-slate-500 dark:text-slate-400 font-medium flex items-center justify-between">
                <span>{imageRows.length} image(s) processed</span>
                <span>
                  {imageRows.filter((r) => r.events.find((e) => e.model === "Result" && e.confidence && e.confidence > 0)).length} animal(s) found
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
