import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadImages, startProcessing, pollJob, getJobResults, getModelStatus, flagByFilenames, getStations } from "../api/client";
import { useConfigStore } from "../store/configStore";

// ── Constants ─────────────────────────────────────────────────────────────────

const CHUNK_SIZE   = 50;                        // files per upload+process batch
const MAX_LIST     = 60;                        // max rows in compact list view
const MAX_CARDS    = 20;                        // max cards in card grid view
const SESSION_KEY  = "cameraTrapsUploadSession"; // localStorage key for crash recovery

// ── Types ─────────────────────────────────────────────────────────────────────

type ModelStatus = { models_loaded: boolean; error: string | null } | null;

type Detection = { label: string; conf: number; bbox?: number[] };

type ModelEvent = {
  type: "model_event";
  image: string;
  image_index: number;
  model: string;
  detections?: Detection[];
  merged_count?: number;
  sources_used?: string[];
  top5?: [string, number][];
  skipped?: boolean;
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

type ResultFilter = "all" | "wildlife" | "empty" | "low_conf";

type SavedSession = {
  activeJobId: string;
  currentChunk: number;
  totalChunks: number;
  overallTotal: number;
  completedOffset: number;
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtTime(secs: number) {
  return `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
}

function chunkArray<T>(arr: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
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
      ].filter(Boolean).join("\n");
      return { display, tooltip };
    } catch {}
  }
  return { display: raw.trim(), tooltip: "" };
}

type FileStatus = "pending" | "processing" | "done";

// ── FileListRow — no image preview, used when many files selected ─────────────

function FileListRow({ file, onRemove, disabled, status = "pending" }: {
  file: File; onRemove: () => void; disabled: boolean; status?: FileStatus;
}) {
  return (
    <div className={`flex items-center gap-2 px-2 py-1.5 rounded-lg border transition-colors ${
      status === "done"    ? "border-emerald-100 dark:border-emerald-900/40 bg-emerald-50/20 dark:bg-emerald-950/10"
      : status === "processing" ? "border-indigo-100 dark:border-indigo-900/40 bg-indigo-50/10 dark:bg-indigo-950/10"
      : "border-transparent"
    }`}>
      <span className={`material-symbols-outlined text-sm select-none shrink-0 ${
        status === "done" ? "text-emerald-500" : status === "processing" ? "text-indigo-400 animate-spin" : "text-slate-300 dark:text-slate-600"
      }`}>
        {status === "done" ? "check_circle" : status === "processing" ? "sync" : "image"}
      </span>
      <span className="text-[10px] text-slate-600 dark:text-slate-300 truncate flex-1 font-medium" title={file.name}>{file.name}</span>
      <span className="text-[10px] text-slate-400 shrink-0">{(file.size / 1024).toFixed(0)} KB</span>
      <button onClick={onRemove} disabled={disabled}
        className="text-slate-300 dark:text-slate-600 hover:text-red-400 p-0.5 rounded transition-colors shrink-0 disabled:opacity-40 cursor-pointer">
        <span className="material-symbols-outlined text-xs select-none leading-none block">close</span>
      </button>
    </div>
  );
}

// ── ThumbnailItem — image preview, used only for small batches (≤ 10) ─────────

function ThumbnailItem({ file, onRemove, disabled, status = "pending" }: {
  file: File; onRemove: () => void; disabled: boolean; status?: FileStatus;
}) {
  const [preview, setPreview] = useState("");
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div className={`flex items-center justify-between p-2 rounded-xl border transition-colors ${
      status === "done"    ? "border-emerald-200 dark:border-emerald-900/50 bg-emerald-50/30 dark:bg-emerald-950/10"
      : status === "processing" ? "border-indigo-200 dark:border-indigo-900/50 bg-indigo-50/20 dark:bg-indigo-950/10"
      : "border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-900/40"
    }`}>
      <div className="flex items-center gap-3 overflow-hidden">
        <div className="relative shrink-0">
          {preview
            ? <img src={preview} alt={file.name} className="w-10 h-10 rounded-lg object-cover border border-slate-200 dark:border-slate-800" />
            : <div className="w-10 h-10 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
                <span className="material-symbols-outlined text-slate-400 text-lg select-none">image</span>
              </div>}
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
          <p className="text-[10px] font-semibold text-slate-700 dark:text-slate-300 truncate max-w-[130px]" title={file.name}>{file.name}</p>
          <p className="text-[10px] text-slate-400 dark:text-slate-500">
            {(file.size / 1024).toFixed(0)} KB
            {status === "done" && <span className="ml-1 text-emerald-500 font-semibold">· done</span>}
            {status === "processing" && <span className="ml-1 text-indigo-400 font-semibold">· analysing…</span>}
          </p>
        </div>
      </div>
      <button onClick={onRemove} disabled={disabled}
        className="text-slate-300 dark:text-slate-600 hover:text-red-500 p-1 rounded-full hover:bg-red-50 dark:hover:bg-red-950/30 transition-all shrink-0 disabled:opacity-50 cursor-pointer">
        <span className="material-symbols-outlined text-sm select-none leading-none block">close</span>
      </button>
    </div>
  );
}

// ── PipelineBadge ─────────────────────────────────────────────────────────────

function PipelineBadge({ status }: { status: ModelStatus }) {
  if (!status) return (
    <div className="flex items-center gap-1.5 px-3 py-1 bg-slate-50 dark:bg-slate-900 rounded-full border border-slate-200 dark:border-slate-800">
      <span className="w-2 h-2 rounded-full bg-slate-300 dark:bg-slate-600 animate-pulse" />
      <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">Checking…</span>
    </div>
  );
  if (status.error) return (
    <div className="flex items-center gap-1.5 px-3 py-1 bg-red-50 dark:bg-red-950/30 rounded-full border border-red-200 dark:border-red-900/50" title={status.error}>
      <span className="w-2 h-2 rounded-full bg-red-500" />
      <span className="text-xs font-semibold text-red-700 dark:text-red-400">Load Error</span>
    </div>
  );
  if (!status.models_loaded) return (
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

// ── DetectionCrop ─────────────────────────────────────────────────────────────

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
      const [bx, by, bw, bh] = bbox && bbox.length >= 4 ? bbox : [0, 0, 1, 1];
      const pad = 0.06;
      const sx = Math.max(0, (bx - pad)) * img.naturalWidth;
      const sy = Math.max(0, (by - pad)) * img.naturalHeight;
      const sw = Math.min(img.naturalWidth - sx, (bw + pad * 2) * img.naturalWidth);
      const sh = Math.min(img.naturalHeight - sy, (bh + pad * 2) * img.naturalHeight);
      canvas.width = 120; canvas.height = 90;
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, 120, 90);
      if (alive) setReady(true);
    };
    img.src = objUrl;
    return () => { alive = false; URL.revokeObjectURL(objUrl); };
  }, [file, bbox]);

  if (!file) return null;
  return (
    <div className="shrink-0 relative rounded-xl overflow-hidden bg-slate-100 dark:bg-slate-800" style={{ width: 120, height: 90 }}>
      {!ready && <div className="absolute inset-0 animate-pulse bg-slate-200 dark:bg-slate-700 rounded-xl" />}
      <canvas ref={canvasRef} className={`rounded-xl transition-opacity duration-300 ${ready ? "opacity-100" : "opacity-0"}`}
        style={{ width: 120, height: 90 }} />
    </div>
  );
}

// ── ResultListRow — compact one-liner per image ───────────────────────────────

function ResultListRow({ row, flagged, onFlag }: {
  row: ImageRow; flagged: boolean; onFlag: () => void;
}) {
  const evResult = row.events.find((e) => e.model === "Result");
  const isEmpty  = !evResult || evResult.confidence === 0;
  const isLow    = !isEmpty && (evResult?.confidence ?? 0) < 0.4;
  const confPct  = Math.round((evResult?.confidence ?? 0) * 100);
  const { display } = parseSnetLabel(evResult?.species ?? "");

  const detMDv5a = row.events.find((e) => e.model === "MDv5a");
  const isPerson = (detMDv5a?.detections ?? []).some((d) => d.label.toLowerCase() !== "animal");

  return (
    <div className="flex items-center gap-2.5 px-3 py-2 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors group">
      {/* Status dot */}
      <span className={`w-2 h-2 rounded-full shrink-0 ${
        isEmpty && isPerson ? "bg-blue-400"
        : isEmpty           ? "bg-slate-300 dark:bg-slate-600"
        : isLow             ? "bg-amber-400"
        : confPct >= 80     ? "bg-emerald-500"
        :                     "bg-blue-400"
      }`} />

      {/* Filename */}
      <span className="text-[11px] font-mono text-slate-400 dark:text-slate-500 truncate w-36 shrink-0" title={row.name}>
        {row.name}
      </span>

      {/* Species or label */}
      <span className={`text-xs font-semibold truncate flex-1 min-w-0 ${
        isEmpty   ? "text-slate-400 dark:text-slate-500 italic font-normal"
        : isLow   ? "text-amber-600 dark:text-amber-400"
        :           "text-slate-700 dark:text-slate-200"
      }`}>
        {isEmpty ? (isPerson ? "person / vehicle" : "blank") : display}
      </span>

      {/* Confidence pill */}
      {!isEmpty && (
        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full shrink-0 ${
          isLow       ? "bg-amber-100 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400"
          : confPct >= 80 ? "bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400"
          :                 "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
        }`}>{confPct}%</span>
      )}

      {/* Flag */}
      <button onClick={onFlag}
        className={`p-0.5 rounded transition-colors cursor-pointer opacity-0 group-hover:opacity-100 ${flagged ? "opacity-100 text-amber-500" : "text-slate-300 dark:text-slate-600 hover:text-amber-400"}`}>
        <span className="material-symbols-outlined text-sm select-none leading-none block">{flagged ? "flag" : "outlined_flag"}</span>
      </button>
    </div>
  );
}

// ── ImageResultCard ───────────────────────────────────────────────────────────

function ImageResultCard({ row, file, flagged, onFlag }: {
  row: ImageRow; file?: File; flagged: boolean; onFlag: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const detMDv5a  = row.events.find((e) => e.model === "MDv5a");
  const detFusion = row.events.find((e) => e.model === "Detection");
  const evSN      = row.events.find((e) => e.model === "SpeciesNet");
  const evResult  = row.events.find((e) => e.model === "Result");

  const isEmpty   = !evResult || evResult.confidence === 0;
  const isLowConf = !isEmpty && (evResult?.confidence ?? 0) < 0.4;
  const confPct   = Math.round((evResult?.confidence ?? 0) * 100);

  const firstBbox = detFusion?.detections?.[0]?.bbox ?? detMDv5a?.detections?.[0]?.bbox;
  const personLabels = (detMDv5a?.detections ?? [])
    .filter((d) => d.label.toLowerCase() !== "animal")
    .map((d) => d.label);
  const isPerson = personLabels.length > 0;

  if (isEmpty) {
    return (
      <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl border border-slate-100 dark:border-slate-800/60 bg-slate-50/50 dark:bg-slate-900/30">
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${isPerson ? "bg-blue-50 dark:bg-blue-950/30" : "bg-slate-100 dark:bg-slate-800"}`}>
          <span className={`material-symbols-outlined text-base select-none ${isPerson ? "text-blue-400" : "text-slate-300 dark:text-slate-600"}`}>
            {isPerson ? "person" : "hide_image"}
          </span>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-slate-600 dark:text-slate-400 truncate" title={row.name}>{row.name}</p>
          <p className="text-[10px] text-slate-400 italic mt-0.5">{isPerson ? personLabels.join(", ") : "No animal detected"}</p>
        </div>
        <button onClick={onFlag} title={flagged ? "Remove flag" : "Flag for review"}
          className={`p-1 rounded-full shrink-0 transition-colors cursor-pointer ${flagged ? "text-amber-500" : "text-slate-300 dark:text-slate-600 hover:text-amber-500"}`}>
          <span className="material-symbols-outlined text-base select-none leading-none block">{flagged ? "flag" : "outlined_flag"}</span>
        </button>
      </div>
    );
  }

  const { display: speciesDisplay, tooltip: speciesTooltip } = parseSnetLabel(evResult?.species ?? "");

  return (
    <div className={`rounded-xl border overflow-hidden shadow-sm transition-shadow hover:shadow-md ${
      isLowConf ? "border-amber-200 dark:border-amber-900/50 bg-amber-50/20 dark:bg-amber-950/10"
        : "border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/60"
    }`}>
      <div className="flex items-center gap-3 p-3">
        <DetectionCrop file={file} bbox={firstBbox} />
        <div className="flex-1 min-w-0 space-y-1.5">
          <p className="text-[10px] text-slate-400 truncate font-medium" title={row.name}>{row.name}</p>
          <div className="flex items-baseline gap-2 flex-wrap">
            <span className={`text-sm font-bold capitalize ${isLowConf ? "text-amber-700 dark:text-amber-400" : "text-slate-800 dark:text-slate-100"}`}
              title={speciesTooltip || undefined}>{speciesDisplay}</span>
            <span className={`text-xs font-semibold px-1.5 py-0.5 rounded-full ${
              isLowConf ? "bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400"
                : confPct >= 80 ? "bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400"
                : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300"
            }`}>{confPct}%</span>
            {isLowConf && (
              <span className="text-[10px] text-amber-500 flex items-center gap-0.5">
                <span className="material-symbols-outlined text-xs select-none">warning</span>Low confidence
              </span>
            )}
          </div>
          <div className="w-full h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
            <div className={`h-1.5 rounded-full transition-all duration-500 ${isLowConf ? "bg-amber-400" : confPct >= 80 ? "bg-emerald-500" : "bg-blue-400"}`}
              style={{ width: `${confPct}%` }} />
          </div>
        </div>
        <div className="flex flex-col items-center gap-1.5 shrink-0">
          <button onClick={onFlag} title={flagged ? "Remove review flag" : "Flag for review"}
            className={`p-1 rounded-full transition-colors cursor-pointer ${flagged ? "text-amber-500" : "text-slate-300 dark:text-slate-600 hover:text-amber-500"}`}>
            <span className="material-symbols-outlined text-base select-none leading-none block">{flagged ? "flag" : "outlined_flag"}</span>
          </button>
          <button onClick={() => setExpanded((v) => !v)} title="Toggle model details"
            className="p-1 rounded-full text-slate-300 dark:text-slate-600 hover:text-slate-500 dark:hover:text-slate-400 transition-colors cursor-pointer">
            <span className="material-symbols-outlined text-base select-none leading-none block">{expanded ? "expand_less" : "expand_more"}</span>
          </button>
        </div>
      </div>

      {expanded && (
        <div className="px-3 pb-3 pt-0 border-t border-slate-100 dark:border-slate-800 space-y-2 text-[10px]">
          <p className="text-[10px] uppercase font-bold text-slate-400 dark:text-slate-500 tracking-wider pt-2">Model Details</p>
          <div className="flex items-start gap-1.5">
            <span className="font-bold text-blue-600 dark:text-blue-400 shrink-0">MDv5a</span>
            <span className="text-slate-500 dark:text-slate-400">
              {detMDv5a?.detections?.length
                ? detMDv5a.detections.map((d) => `${d.label} ${(d.conf * 100).toFixed(0)}%`).join(", ")
                : "—"}
            </span>
          </div>
          {evSN && !evSN.skipped && evSN.top5?.length ? (
            <div className="space-y-0.5">
              <span className="font-bold text-teal-600 dark:text-teal-400">SpeciesNet top 3</span>
              {evSN.top5.slice(0, 3).map(([s, c], i) => {
                const { display } = parseSnetLabel(s);
                return (
                  <div key={i} className="flex items-center gap-1.5 ml-2">
                    <span className="text-slate-400 w-3 text-right shrink-0">{i + 1}.</span>
                    <span className="text-slate-600 dark:text-slate-300 truncate flex-1">{display}</span>
                    <div className="w-10 h-1 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden shrink-0">
                      <div className="h-1 rounded-full bg-teal-400" style={{ width: `${Math.round(c * 100)}%` }} />
                    </div>
                    <span className="font-mono text-slate-400 w-6 text-right shrink-0">{Math.round(c * 100)}%</span>
                  </div>
                );
              })}
            </div>
          ) : evSN?.skipped ? (
            <div className="flex gap-1.5">
              <span className="font-bold text-teal-600 dark:text-teal-400">SpeciesNet</span>
              <span className="text-slate-400 italic">not loaded</span>
            </div>
          ) : null}
          {evResult?.agreement && (
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-slate-500">Agreement</span>
              <span className={`px-1.5 py-0.5 rounded-full font-bold border ${
                evResult.agreement === "High"
                  ? "bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/30"
                  : evResult.agreement === "Medium"
                  ? "bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-900/30"
                  : "bg-red-100 dark:bg-red-950/60 text-red-700 dark:text-red-400 border-red-200 dark:border-red-900/30"
              }`}>{evResult.agreement}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function Upload() {
  const reviewerId     = useConfigStore((s) => s.config?.reviewer_id ?? "anonymous");
  const defaultStation = useConfigStore((s) => s.config?.default_station_id ?? "");

  const [files, setFiles]               = useState<File[]>([]);
  const [dragging, setDragging]         = useState(false);
  const [skippedDupes, setSkippedDupes] = useState<string[]>([]);
  const [rejectedFiles, setRejectedFiles] = useState<string[]>([]);
  const [stations, setStations]         = useState<{ station_id: string }[]>([]);
  const [selectedStation, setSelectedStation] = useState("");

  // Multi-batch progress
  const [processing, setProcessing]               = useState(false);
  const [overallTotal, setOverallTotal]           = useState(0);
  const [overallCompleted, setOverallCompleted]   = useState(0);
  const [totalChunks, setTotalChunks]             = useState(0);
  const [currentChunk, setCurrentChunk]           = useState(0);
  const [jobError, setJobError]                   = useState<string | null>(null);
  const [allDone, setAllDone]                     = useState(false);
  const [latestModel, setLatestModel]             = useState<string | null>(null);

  const [modelStatus, setModelStatus]             = useState<ModelStatus>(null);
  const [imageRows, setImageRows]                 = useState<ImageRow[]>([]);
  const [flaggedImages, setFlaggedImages]         = useState<Set<string>>(new Set());
  const [resultFilter, setResultFilter]           = useState<ResultFilter>("all");
  const [viewMode, setViewMode]                   = useState<"list" | "cards">("list");

  // Crash / tab-close recovery
  const [recoveredSession, setRecoveredSession]   = useState<SavedSession | null>(null);
  const [recoveryJobDone, setRecoveryJobDone]     = useState(false);

  const startTimeRef     = useRef<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const inputRef         = useRef<HTMLInputElement>(null);
  const sseRef           = useRef<EventSource | null>(null);
  const flaggedImagesRef = useRef<Set<string>>(new Set());
  const navigate         = useNavigate();

  // Poll model status
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const s = await getModelStatus();
        if (!cancelled) {
          setModelStatus(s);
          if (!s.models_loaded && !s.error) setTimeout(check, 3000);
        }
      } catch { if (!cancelled) setTimeout(check, 5000); }
    };
    check();
    return () => { cancelled = true; };
  }, []);

  // Load stations list once on mount
  useEffect(() => {
    getStations().then((data: any) => {
      const list = Array.isArray(data) ? data : (data?.items ?? []);
      setStations(list);
    }).catch(() => {});
  }, []);

  // Sync selectedStation when defaultStation loads from config
  useEffect(() => {
    if (defaultStation && !selectedStation) setSelectedStation(defaultStation);
  }, [defaultStation]); // eslint-disable-line react-hooks/exhaustive-deps

  // On mount: check whether a previous session was interrupted
  useEffect(() => {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return;
    let session: SavedSession;
    try { session = JSON.parse(raw); } catch { localStorage.removeItem(SESSION_KEY); return; }
    pollJob(session.activeJobId)
      .then((status) => {
        if (status.status === "running" || status.status === "queued") {
          setRecoveredSession(session);
          setRecoveryJobDone(false);
        } else if (status.status === "done") {
          setRecoveredSession(session);
          setRecoveryJobDone(true);
        } else {
          localStorage.removeItem(SESSION_KEY);
        }
      })
      .catch(() => localStorage.removeItem(SESSION_KEY));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Elapsed ticker
  useEffect(() => {
    if (processing) {
      startTimeRef.current = startTimeRef.current ?? Date.now();
      const iv = setInterval(() => {
        setElapsed(Math.floor((Date.now() - (startTimeRef.current ?? Date.now())) / 1000));
      }, 1000);
      return () => clearInterval(iv);
    } else {
      startTimeRef.current = null;
      setElapsed(0);
    }
  }, [processing]);

  useEffect(() => { return () => sseRef.current?.close(); }, []);

  const addFiles = useCallback((incoming: File[]) => {
    const valid = incoming.filter((f) => /\.(jpe?g|png|tiff?|bmp|webp)$/i.test(f.name));
    if (valid.length) setFiles((prev) => [...prev, ...valid]);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    addFiles(Array.from(e.dataTransfer.files));
  }, [addFiles]);

  const removeFile = (i: number) => setFiles((prev) => prev.filter((_, idx) => idx !== i));

  const clearAll = () => {
    sseRef.current?.close();
    localStorage.removeItem(SESSION_KEY);
    setRecoveredSession(null);
    setRecoveryJobDone(false);
    setFiles([]);
    setSkippedDupes([]);
    setRejectedFiles([]);
    setProcessing(false);
    setOverallTotal(0); setOverallCompleted(0);
    setTotalChunks(0); setCurrentChunk(0);
    setJobError(null); setAllDone(false);
    setLatestModel(null);
    setImageRows([]);
    setFlaggedImages(new Set());
    flaggedImagesRef.current = new Set();
    startTimeRef.current = null;
  };

  const toggleFlag = useCallback((name: string) => {
    setFlaggedImages((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      flaggedImagesRef.current = next;
      return next;
    });
  }, []);

  // ── SSE for one job chunk ─────────────────────────────────────────────────

  const runJobSSE = useCallback((
    jobId: string,
    chunkOffset: number,
    reconnectsLeft: number,
    cursor: number,
  ): Promise<void> => {
    return new Promise<void>((resolve, reject) => {
      const sse = new EventSource(`/api/images/job/${jobId}/stream?cursor=${cursor}`);
      sseRef.current = sse;
      let localCursor = cursor;

      sse.onmessage = (e) => {
        const data = JSON.parse(e.data) as { type: string } & Record<string, unknown>;

        if (data.type === "progress") {
          const prog = data as unknown as { status: string; total: number; completed: number; error?: string };
          setOverallCompleted(chunkOffset + prog.completed);
          if (prog.status === "done") { sse.close(); resolve(); }
          else if (prog.status === "error") { sse.close(); reject(new Error(prog.error ?? "Job failed")); }

        } else if (data.type === "model_event") {
          localCursor++;
          const ev = data as unknown as ModelEvent;
          setLatestModel(ev.model);
          setImageRows((prev) => {
            const idx = prev.findIndex((r) => r.name === ev.image);
            if (idx === -1) return [...prev, { name: ev.image, index: ev.image_index, events: [ev] }];
            const updated = [...prev];
            updated[idx] = { ...updated[idx], events: [...updated[idx].events, ev] };
            return updated;
          });
        }
      };

      sse.onerror = async () => {
        sse.close();
        try {
          const status = await pollJob(jobId);
          if (status.status === "done") { resolve(); return; }
          if (status.status === "error") { reject(new Error(status.error ?? "Job failed")); return; }
          if (reconnectsLeft > 0) {
            setTimeout(() => {
              runJobSSE(jobId, chunkOffset, reconnectsLeft - 1, localCursor).then(resolve).catch(reject);
            }, 2000);
          } else {
            reject(new Error("Stream disconnected — max reconnects reached."));
          }
        } catch { reject(new Error("Stream connection lost.")); }
      };
    });
  }, []);

  // ── Re-attach to a job that kept running after the tab was closed ────────

  const handleRecoveryReconnect = async (session: SavedSession) => {
    sseRef.current?.close();
    setRecoveredSession(null);
    setProcessing(true);
    setAllDone(false);
    setJobError(null);
    setOverallTotal(session.overallTotal);
    setTotalChunks(session.totalChunks);
    setCurrentChunk(session.currentChunk);
    setOverallCompleted(session.completedOffset);
    setImageRows([]);
    setFlaggedImages(new Set());
    flaggedImagesRef.current = new Set();
    setLatestModel(null);
    startTimeRef.current = Date.now();

    try {
      // cursor=0 — replay all model events accumulated while the tab was closed
      await runJobSSE(session.activeJobId, session.completedOffset, 3, 0);
      const remainingBatches = session.totalChunks - session.currentChunk;
      if (remainingBatches > 0) {
        setJobError(
          `Batch ${session.currentChunk} of ${session.totalChunks} complete. ` +
          `${remainingBatches} batch${remainingBatches !== 1 ? "es" : ""} were not started — ` +
          `re-upload the remaining files to continue.`
        );
      } else {
        setAllDone(true);
      }
    } catch (err) {
      setJobError(err instanceof Error ? err.message : String(err));
    } finally {
      setProcessing(false);
      localStorage.removeItem(SESSION_KEY);
    }
  };

  // ── Chunked upload + process ──────────────────────────────────────────────

  const handleProcess = async () => {
    sseRef.current?.close();

    const chunks = chunkArray(files, CHUNK_SIZE);
    const total  = files.length;

    setProcessing(true);
    setAllDone(false);
    setJobError(null);
    setOverallTotal(total);
    setOverallCompleted(0);
    setTotalChunks(chunks.length);
    setCurrentChunk(0);
    setSkippedDupes([]);
    setImageRows([]);
    setFlaggedImages(new Set());
    flaggedImagesRef.current = new Set();
    setResultFilter("all");
    setLatestModel(null);
    startTimeRef.current = Date.now();

    let allSkipped: string[] = [];
    let completedOffset = 0;

    for (let ci = 0; ci < chunks.length; ci++) {
      const ch = chunks[ci];
      setCurrentChunk(ci + 1);

      // 1 — Upload this chunk
      let jobId: string;
      try {
        const res = await uploadImages(ch);
        const newSkipped: string[] = res.duplicates_skipped ?? [];
        allSkipped = [...allSkipped, ...newSkipped];
        setSkippedDupes([...allSkipped]);
        if ((res.rejected ?? []).length > 0) {
          setRejectedFiles((prev) => [...prev, ...(res.rejected as string[])]);
        }
        if (!res.job_id || res.file_count === 0) {
          // All files in this chunk were duplicates or unsupported — skip processing
          completedOffset += ch.length;
          setOverallCompleted(completedOffset);
          continue;
        }
        jobId = res.job_id;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setJobError(`Batch ${ci + 1}/${chunks.length} upload failed: ${msg}`);
        setProcessing(false);
        return;
      }

      // 2 — Start processing
      try {
        await startProcessing(jobId, selectedStation || undefined);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setJobError(`Batch ${ci + 1}/${chunks.length} failed to start: ${msg}`);
        setProcessing(false);
        return;
      }

      // Persist session so a tab-close can be recovered
      localStorage.setItem(SESSION_KEY, JSON.stringify({
        activeJobId: jobId,
        currentChunk: ci + 1,
        totalChunks: chunks.length,
        overallTotal: total,
        completedOffset,
      } satisfies SavedSession));

      // 3 — Stream results until done
      try {
        await runJobSSE(jobId, completedOffset, 3, 0);
        completedOffset += ch.length;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setJobError(`Batch ${ci + 1}/${chunks.length} error: ${msg}`);
        setProcessing(false);
        return;
      }

      // Flag images from this batch before moving to next
      const toFlag = Array.from(flaggedImagesRef.current);
      if (toFlag.length > 0) flagByFilenames(toFlag, reviewerId).catch(() => {});

      // Populate in-app results cache (fire-and-forget)
      getJobResults(jobId).catch(() => {});
    }

    setProcessing(false);
    setAllDone(true);
    localStorage.removeItem(SESSION_KEY);
  };

  // ── Derived state ─────────────────────────────────────────────────────────

  const pct = overallTotal > 0 ? Math.round((overallCompleted / overallTotal) * 100) : 0;
  const etaSecs = processing && pct > 5 && elapsed > 0
    ? Math.round((elapsed / pct) * (100 - pct)) : null;

  const fileStatuses = new Map<string, FileStatus>();
  for (const row of imageRows) {
    const hasDone = row.events.some((e) => e.model === "Result");
    fileStatuses.set(row.name, hasDone ? "done" : "processing");
  }

  const atDetect   = latestModel === "MDv5a" || latestModel === "Detection";
  const atClassify = latestModel === "SpeciesNet";
  const atFusion   = latestModel === "Result";

  const resultRows  = imageRows.filter((r) => {
    const res = r.events.find((e) => e.model === "Result");
    return res && (res.confidence ?? 0) > 0;
  });
  const emptyRows   = imageRows.filter((r) => !resultRows.find((rr) => rr.name === r.name));
  const lowConfRows = resultRows.filter((r) => (r.events.find((e) => e.model === "Result")?.confidence ?? 1) < 0.4);
  const speciesSet  = new Set(resultRows.map((r) => r.events.find((e) => e.model === "Result")?.species));
  const needsReviewCount = new Set([...lowConfRows.map((r) => r.name), ...flaggedImages]).size;

  const allFilteredRows = imageRows.filter((row) => {
    if (resultFilter === "all")      return true;
    if (resultFilter === "wildlife") return resultRows.some((r) => r.name === row.name);
    if (resultFilter === "empty")    return emptyRows.some((r) => r.name === row.name);
    if (resultFilter === "low_conf") return lowConfRows.some((r) => r.name === row.name);
    return true;
  });
  const maxDisplay   = viewMode === "list" ? MAX_LIST : MAX_CARDS;
  const filteredRows = allFilteredRows.slice(-maxDisplay);
  const hiddenCount  = allFilteredRows.length - filteredRows.length;

  const canStart   = files.length > 0 && !processing && !!modelStatus?.models_loaded;
  const numBatches = Math.ceil(files.length / CHUNK_SIZE);

  const STEPS = [
    { id: "ocr",      label: "OCR — Date & Time",    active: !latestModel,              done: atDetect || atClassify || atFusion },
    { id: "detect",   label: "MegaDetector",          active: atDetect,                  done: atClassify || atFusion },
    { id: "classify", label: "SpeciesNet Classifier", active: atClassify,                done: atFusion },
    { id: "fusion",   label: "Ensemble & Save",       active: atFusion,                  done: false },
  ];

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-slate-900 dark:text-white">Upload &amp; Process</h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1 text-sm">
            Drop images, then click <span className="font-semibold text-slate-700 dark:text-slate-300">Start AI Analysis</span>.
            {files.length > CHUNK_SIZE && (
              <span className="ml-1 text-indigo-600 dark:text-indigo-400 font-semibold">
                {files.length} files → {numBatches} batches of {CHUNK_SIZE}
              </span>
            )}
          </p>
        </div>
        <PipelineBadge status={modelStatus} />
      </div>

      {/* ── Recovery banner ── */}
      {recoveredSession && !processing && (
        <div className={`rounded-2xl border p-4 flex items-start gap-3 ${
          recoveryJobDone
            ? "bg-emerald-50 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-900/50"
            : "bg-amber-50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-900/50"
        }`}>
          <span className={`material-symbols-outlined text-xl select-none shrink-0 mt-0.5 ${
            recoveryJobDone ? "text-emerald-500" : "text-amber-500"
          }`}>
            {recoveryJobDone ? "check_circle" : "wifi_tethering"}
          </span>
          <div className="flex-1 min-w-0">
            <p className={`font-bold text-sm ${recoveryJobDone ? "text-emerald-800 dark:text-emerald-300" : "text-amber-800 dark:text-amber-300"}`}>
              {recoveryJobDone
                ? `Batch ${recoveredSession.currentChunk} of ${recoveredSession.totalChunks} finished while you were away`
                : `Batch ${recoveredSession.currentChunk} of ${recoveredSession.totalChunks} is still running on the server`}
            </p>
            <p className={`text-xs mt-0.5 ${recoveryJobDone ? "text-emerald-700 dark:text-emerald-400" : "text-amber-700 dark:text-amber-400"}`}>
              {recoveryJobDone
                ? recoveredSession.currentChunk < recoveredSession.totalChunks
                  ? `${recoveredSession.totalChunks - recoveredSession.currentChunk} batch${recoveredSession.totalChunks - recoveredSession.currentChunk !== 1 ? "es" : ""} not started — re-upload the remaining files to continue.`
                  : "All batches complete. Results are saved to the database."
                : `${recoveredSession.overallTotal} total images · click Reconnect to resume the live view.`}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {!recoveryJobDone && (
              <button onClick={() => handleRecoveryReconnect(recoveredSession)}
                className="px-3 py-1.5 bg-amber-600 hover:bg-amber-700 text-white text-xs font-bold rounded-lg cursor-pointer transition-colors">
                Reconnect
              </button>
            )}
            <button onClick={() => { setRecoveredSession(null); localStorage.removeItem(SESSION_KEY); }}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg cursor-pointer transition-colors border ${
                recoveryJobDone
                  ? "bg-white dark:bg-slate-800 border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400 hover:bg-emerald-50"
                  : "bg-white dark:bg-slate-800 border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-400 hover:bg-amber-50"
              }`}>
              Dismiss
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ── Left ── */}
        <div className="lg:col-span-1 space-y-4">

          {/* Step 1 */}
          <div className="space-y-2">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500 flex items-center gap-1.5">
              <span className="w-4 h-4 rounded-full bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 text-[9px] font-black flex items-center justify-center">1</span>
              Select Images
            </p>

            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inputRef.current?.click(); }}}
              tabIndex={0} role="button" aria-label="Drop images here or click to browse"
              className={`border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
                dragging
                  ? "border-emerald-500 bg-emerald-50/60 dark:bg-emerald-950/20 scale-[0.99] ring-2 ring-emerald-500/20"
                  : "border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900/40 hover:border-emerald-400 hover:bg-emerald-50/20 dark:hover:bg-emerald-950/10"
              }`}
            >
              <span className={`material-symbols-outlined text-3xl mb-1.5 block select-none transition-all duration-200 ${dragging ? "scale-110 text-emerald-600 animate-bounce" : "text-slate-400 dark:text-slate-500"}`}>
                cloud_upload
              </span>
              <p className="text-sm font-semibold text-slate-700 dark:text-slate-300">Drop images here</p>
              <p className="text-[11px] text-slate-400 mt-0.5">JPG · PNG · TIFF · BMP · WebP</p>
              <button type="button" onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
                className="mt-3 px-4 py-1.5 bg-slate-900 dark:bg-slate-700 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 dark:hover:bg-slate-600 transition-colors cursor-pointer">
                Browse Files
              </button>
              <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp,.webp"
                multiple className="hidden" onChange={(e) => { if (e.target.files) addFiles(Array.from(e.target.files)); }} />
            </div>

            {/* Batch plan notice */}
            {files.length > CHUNK_SIZE && !processing && (
              <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-indigo-50 dark:bg-indigo-950/30 border border-indigo-200 dark:border-indigo-900/50 text-xs">
                <span className="material-symbols-outlined text-indigo-500 text-base select-none shrink-0">layers</span>
                <span className="text-indigo-700 dark:text-indigo-300 font-medium">
                  <span className="font-bold">{files.length} files</span> → <span className="font-bold">{numBatches} batches</span> of up to {CHUNK_SIZE}
                </span>
              </div>
            )}

            {/* Duplicates notice */}
            {skippedDupes.length > 0 && (
              <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 text-[11px] text-slate-500 dark:text-slate-400">
                <span className="material-symbols-outlined text-base text-slate-400 select-none shrink-0 mt-0.5">info</span>
                <span>
                  <span className="font-semibold text-slate-600 dark:text-slate-300">
                    {skippedDupes.length} duplicate{skippedDupes.length !== 1 ? "s" : ""} skipped
                  </span>{" "}— already in the database.
                </span>
              </div>
            )}

            {/* Rejected files notice */}
            {rejectedFiles.length > 0 && (
              <div className="flex items-start gap-2 px-3 py-2 rounded-xl bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/50 text-[11px] text-amber-700 dark:text-amber-400">
                <span className="material-symbols-outlined text-base text-amber-500 select-none shrink-0 mt-0.5">warning</span>
                <div>
                  <span className="font-semibold">
                    {rejectedFiles.length} file{rejectedFiles.length !== 1 ? "s" : ""} skipped
                  </span>{" "}— unsupported format or too large.
                  <ul className="mt-1 space-y-0.5 text-[10px] text-amber-600 dark:text-amber-500 max-h-20 overflow-y-auto">
                    {rejectedFiles.map((name, i) => (
                      <li key={i} className="truncate font-mono">{name}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {/* File list */}
            {files.length > 0 && (
              <div className="bg-white dark:bg-slate-900/60 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
                <div className="px-3 py-2 bg-slate-50 dark:bg-slate-950/80 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center">
                  <span className="text-xs font-semibold text-slate-600 dark:text-slate-300">
                    {files.length} file{files.length !== 1 ? "s" : ""} selected
                  </span>
                  <button onClick={clearAll} disabled={processing}
                    className="text-xs font-semibold text-slate-400 hover:text-red-500 disabled:opacity-50 cursor-pointer transition-colors">
                    Clear all
                  </button>
                </div>
                <div className="p-2 max-h-40 overflow-y-auto space-y-0.5 custom-scrollbar">
                  {files.length <= 10
                    ? files.map((f, i) => (
                        <ThumbnailItem key={i} file={f} onRemove={() => removeFile(i)}
                          disabled={processing} status={fileStatuses.get(f.name) ?? "pending"} />
                      ))
                    : <>
                        {files.slice(0, 8).map((f, i) => (
                          <FileListRow key={i} file={f} onRemove={() => removeFile(i)}
                            disabled={processing} status={fileStatuses.get(f.name) ?? "pending"} />
                        ))}
                        <p className="text-center py-1.5 text-[10px] text-slate-400 dark:text-slate-500 font-medium">
                          +{files.length - 8} more
                        </p>
                      </>
                  }
                </div>
              </div>
            )}
          </div>

          {/* Step 2 — Station */}
          <div className="space-y-1.5">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500 flex items-center gap-1.5">
              <span className="w-4 h-4 rounded-full bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 text-[9px] font-black flex items-center justify-center">2</span>
              Assign to Station
            </p>
            <div className="flex items-center gap-2 px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/60 shadow-sm">
              <span className="material-symbols-outlined text-base text-slate-400 select-none shrink-0">location_on</span>
              {stations.length > 0 ? (
                <select
                  value={selectedStation}
                  onChange={(e) => setSelectedStation(e.target.value)}
                  disabled={processing}
                  className="flex-1 min-w-0 text-sm font-semibold text-slate-700 dark:text-slate-200 bg-transparent border-none outline-none cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed py-0.5"
                >
                  {stations.map((s) => (
                    <option key={s.station_id} value={s.station_id}>{s.station_id}</option>
                  ))}
                </select>
              ) : (
                <span className="flex-1 text-sm font-semibold text-slate-700 dark:text-slate-200 truncate">
                  {selectedStation || defaultStation || "No stations configured"}
                </span>
              )}
            </div>
            {stations.length === 0 && (
              <p className="text-[10px] text-slate-400 dark:text-slate-500">
                No stations found. <a href="/stations" className="text-emerald-600 dark:text-emerald-400 font-semibold hover:underline">Add a station</a> or images will use the default.
              </p>
            )}
          </div>

          {/* Step 3 — Start */}
          <div className="space-y-1.5">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 dark:text-slate-500 flex items-center gap-1.5">
              <span className="w-4 h-4 rounded-full bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300 text-[9px] font-black flex items-center justify-center">3</span>
              Run AI Analysis
            </p>
            <button onClick={handleProcess} disabled={!canStart}
              className="w-full py-3 px-6 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-200 dark:disabled:bg-slate-800 disabled:text-slate-400 dark:disabled:text-slate-600 text-white font-bold rounded-2xl transition-all shadow-md hover:shadow-lg flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed text-sm">
              {processing ? (
                <><span className="material-symbols-outlined text-lg select-none animate-spin">sync</span>
                  {totalChunks > 1 ? `Batch ${currentChunk} of ${totalChunks}…` : "Analysing…"}</>
              ) : !modelStatus?.models_loaded ? (
                <><span className="material-symbols-outlined text-lg select-none">hourglass_empty</span>
                  {modelStatus?.error ? "Models failed to load" : "Waiting for models…"}</>
              ) : !files.length ? (
                <><span className="material-symbols-outlined text-lg select-none">upload_file</span>Select files first</>
              ) : (
                <><span className="material-symbols-outlined text-lg select-none">play_circle</span>
                  Start AI Analysis{numBatches > 1 ? ` (${numBatches} batches)` : ""}</>
              )}
            </button>
          </div>

          {/* Progress */}
          {(processing || allDone || jobError) && overallTotal > 0 && (
            <div className="bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 p-4 space-y-3 shadow-sm rounded-2xl">
              <div className="flex justify-between items-center text-xs">
                <span className={`font-bold ${
                  allDone ? "text-emerald-600 dark:text-emerald-400"
                  : jobError ? "text-red-600 dark:text-red-400"
                  : "text-slate-700 dark:text-slate-300"
                }`}>
                  {allDone ? "Complete ✓" : jobError ? "Failed" : totalChunks > 1 ? `Batch ${currentChunk} / ${totalChunks}` : "Running…"}
                </span>
                <div className="flex items-center gap-2 font-mono text-slate-500">
                  {processing && elapsed > 0 && (
                    <span>{fmtTime(elapsed)}{etaSecs != null && ` / ~${fmtTime(etaSecs)} left`}</span>
                  )}
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">{pct}%</span>
                </div>
              </div>

              <div className="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-2 overflow-hidden">
                <div className={`h-2 rounded-full transition-all duration-300 ${jobError ? "bg-red-500" : "bg-emerald-500"}`}
                  style={{ width: `${allDone ? 100 : pct}%` }} />
              </div>

              <div className="text-[10px] text-slate-400 flex justify-between">
                <span>{overallCompleted} of {overallTotal} images</span>
                <span>{overallTotal - overallCompleted} remaining</span>
              </div>

              {/* Batch dots for multi-chunk jobs */}
              {totalChunks > 1 && (
                <div className="flex flex-wrap gap-1">
                  {Array.from({ length: totalChunks }, (_, i) => (
                    <div key={i} title={`Batch ${i + 1}`}
                      className={`h-1.5 rounded-full flex-1 min-w-[6px] transition-all duration-300 ${
                        i + 1 < currentChunk ? "bg-emerald-500"
                        : i + 1 === currentChunk ? "bg-indigo-400 animate-pulse"
                        : "bg-slate-200 dark:bg-slate-700"
                      }`}
                    />
                  ))}
                </div>
              )}

              {/* Pipeline steps (current chunk) */}
              {processing && (
                <div className="pt-2 border-t border-slate-100 dark:border-slate-800 space-y-1.5">
                  {STEPS.map((step, si) => (
                    <div key={step.id} className="flex items-center gap-2 text-[11px]">
                      <div className={`w-4 h-4 rounded-full flex items-center justify-center shrink-0 border transition ${
                        step.done ? "bg-emerald-100 dark:bg-emerald-950/40 border-emerald-300 dark:border-emerald-800 text-emerald-600"
                        : step.active ? "bg-indigo-50 dark:bg-indigo-950/40 border-indigo-300 dark:border-indigo-700 text-indigo-600 animate-pulse"
                        : "bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-400"
                      }`}>
                        {step.done
                          ? <span className="material-symbols-outlined text-[10px] font-bold leading-none">check</span>
                          : <span className="text-[9px] font-bold">{si + 1}</span>}
                      </div>
                      <span className={`font-semibold ${
                        step.done ? "text-slate-400 dark:text-slate-500 line-through"
                        : step.active ? "text-indigo-600 dark:text-indigo-400"
                        : "text-slate-400 dark:text-slate-500"
                      }`}>{step.label}</span>
                      {step.active && (
                        <span className="ml-auto flex gap-0.5">
                          {[0, 1, 2].map((i) => (
                            <span key={i} className="w-1 h-1 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                          ))}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {jobError && <p className="text-[10px] text-red-600 break-all">{jobError}</p>}
            </div>
          )}

          {/* Completion summary */}
          {allDone && imageRows.length > 0 && (
            <div className="bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900/50 rounded-2xl p-4 space-y-3">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-emerald-600 text-xl select-none">check_circle</span>
                <span className="font-bold text-emerald-700 dark:text-emerald-400">
                  {totalChunks > 1 ? `All ${totalChunks} batches complete` : "Analysis complete"}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                {[
                  { label: "Images",  value: imageRows.length },
                  { label: "Species", value: speciesSet.size },
                  { label: "Review",  value: needsReviewCount },
                ].map(({ label, value }) => (
                  <div key={label} className={`rounded-xl py-2.5 ${label === "Review" && value > 0 ? "bg-amber-100/80 dark:bg-amber-950/40" : "bg-white/60 dark:bg-slate-900/40"}`}>
                    <p className={`text-xl font-extrabold ${label === "Review" && value > 0 ? "text-amber-600 dark:text-amber-400" : "text-slate-800 dark:text-slate-100"}`}>{value}</p>
                    <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide">{label}</p>
                  </div>
                ))}
              </div>
              <button onClick={() => navigate("/results")}
                className="w-full py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold rounded-xl transition-colors flex items-center justify-center gap-1.5 cursor-pointer">
                View Full Results
                <span className="material-symbols-outlined text-base select-none">arrow_forward</span>
              </button>
            </div>
          )}
        </div>

        {/* ── Right: live results ── */}
        <div className="lg:col-span-2 flex flex-col">
          <div className="bg-white dark:bg-slate-900/60 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden flex flex-col" style={{ minHeight: 520 }}>

            <div className="px-4 py-3 bg-slate-50 dark:bg-slate-950/80 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between gap-3 flex-wrap">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-slate-500 dark:text-slate-400 select-none">model_training</span>
                <span className="font-bold text-slate-700 dark:text-slate-300 text-sm">Live Results</span>
                {imageRows.length > 0 && (
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-700">
                    {imageRows.length}
                  </span>
                )}
                {flaggedImages.size > 0 && (
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-amber-100 dark:bg-amber-950/50 text-amber-600 dark:text-amber-400 border border-amber-200 dark:border-amber-900/30">
                    {flaggedImages.size} flagged
                  </span>
                )}
              </div>

              {imageRows.length > 0 && (
                <div className="flex items-center gap-2 flex-wrap">
                  {/* Filter tabs */}
                  <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 rounded-lg p-0.5 text-[10px] font-semibold">
                    {([
                      ["all",      `All (${imageRows.length})`],
                      ["wildlife", `Wildlife (${resultRows.length})`],
                      ["empty",    `Blank (${emptyRows.length})`],
                      ["low_conf", `Low (${lowConfRows.length})`],
                    ] as [ResultFilter, string][]).map(([key, label]) => (
                      <button key={key} onClick={() => setResultFilter(key)}
                        className={`px-2 py-1 rounded-md transition-colors cursor-pointer ${
                          resultFilter === key
                            ? "bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-200 shadow-sm"
                            : "text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
                        }`}>{label}</button>
                    ))}
                  </div>
                  {/* View mode toggle */}
                  <div className="flex items-center bg-slate-100 dark:bg-slate-800 rounded-lg p-0.5">
                    {(["list", "cards"] as const).map((mode) => (
                      <button key={mode} onClick={() => setViewMode(mode)} title={mode === "list" ? "Compact list" : "Card grid"}
                        className={`p-1 rounded-md transition-colors cursor-pointer ${
                          viewMode === mode
                            ? "bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 shadow-sm"
                            : "text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300"
                        }`}>
                        <span className="material-symbols-outlined text-sm select-none leading-none block">
                          {mode === "list" ? "view_list" : "grid_view"}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
              {hiddenCount > 0 && (
                <div className="mb-3 px-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 text-[10px] text-slate-500 dark:text-slate-400 font-medium flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-sm select-none">info</span>
                  Showing latest {maxDisplay} of {allFilteredRows.length} —{" "}
                  <button onClick={() => navigate("/results")} className="font-bold underline cursor-pointer hover:text-slate-600 mx-0.5">View Full Results</button>
                  or switch to list view for more rows.
                </div>
              )}

              {imageRows.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-400 dark:text-slate-500 gap-3 py-16">
                  <span className="material-symbols-outlined text-5xl select-none opacity-40">analytics</span>
                  <div className="text-center">
                    <p className="text-sm font-semibold">
                      {processing ? "Processing… results appear here as each image completes" : "Results stream here as images are analysed"}
                    </p>
                    {!processing && <p className="text-xs mt-1 opacity-75">Select images and click Start AI Analysis</p>}
                  </div>
                  {processing && (
                    <div className="flex gap-1">
                      {[0, 1, 2].map((i) => (
                        <span key={i} className="w-2 h-2 rounded-full bg-emerald-400 dark:bg-emerald-500 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                      ))}
                    </div>
                  )}
                </div>
              ) : filteredRows.length === 0 ? (
                <div className="h-full flex items-center justify-center text-slate-400 text-sm py-16">No results match this filter.</div>
              ) : viewMode === "list" ? (
                <div className="space-y-0.5 divide-y divide-slate-100 dark:divide-slate-800/60">
                  {filteredRows.map((row) => (
                    <ResultListRow key={row.name} row={row}
                      flagged={flaggedImages.has(row.name)}
                      onFlag={() => toggleFlag(row.name)} />
                  ))}
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {filteredRows.map((row) => (
                    <ImageResultCard key={row.name} row={row}
                      file={files.find((f) => f.name === row.name)}
                      flagged={flaggedImages.has(row.name)}
                      onFlag={() => toggleFlag(row.name)} />
                  ))}
                </div>
              )}
            </div>

            {imageRows.length > 0 && (
              <div className="px-4 py-2.5 bg-slate-50 dark:bg-slate-950/80 border-t border-slate-100 dark:border-slate-800 text-[10px] text-slate-500 dark:text-slate-400 font-medium flex items-center justify-between gap-2 flex-wrap">
                <span>{resultRows.length} wildlife · {emptyRows.length} blank</span>
                <div className="flex items-center gap-3">
                  {lowConfRows.length > 0 && (
                    <button onClick={() => setResultFilter("low_conf")}
                      className="flex items-center gap-1 text-amber-500 hover:text-amber-600 cursor-pointer transition-colors">
                      <span className="material-symbols-outlined text-xs select-none">warning</span>
                      {lowConfRows.length} low confidence
                    </button>
                  )}
                  {flaggedImages.size > 0 && (
                    <span className="flex items-center gap-1 text-amber-500">
                      <span className="material-symbols-outlined text-xs select-none">flag</span>
                      {flaggedImages.size} flagged
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
