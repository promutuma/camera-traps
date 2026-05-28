import { useCallback, useEffect, useRef, useState } from "react";
import { getReviewQueue, confirmDetection, correctDetection, flagDetection, getReviewLog, getPrivacyAudit, storedImageUrl } from "../api/client";
import { useConfigStore } from "../store/configStore";

type Row = Record<string, unknown>;
type Tab = "queue" | "log" | "privacy";

function parseBbox(raw: unknown): [number, number, number, number] | null {
  if (!raw) return null;
  try {
    const arr = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (Array.isArray(arr) && arr.length === 4) return arr as [number, number, number, number];
  } catch {}
  return null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Extract ranked species candidates from the model_breakdown JSON for the correction dropdown. */
function getCandidates(row: Row): { label: string; conf: number; source: "BioClip" | "SpeciesNet" }[] {
  let bd: Record<string, any> | null = null;
  try {
    if (typeof row.model_breakdown === "string") bd = JSON.parse(row.model_breakdown);
    else if (row.model_breakdown && typeof row.model_breakdown === "object") bd = row.model_breakdown as Record<string, any>;
  } catch {}
  if (!bd) return [];

  const out: { label: string; conf: number; source: "BioClip" | "SpeciesNet" }[] = [];
  const seen = new Set<string>();

  for (const [s, c] of ((bd.SpeciesNet ?? []) as [string, number][]).slice(0, 3)) {
    let label = String(s);
    if (label.startsWith("{")) { try { label = JSON.parse(label).common_name || label; } catch {} }
    const key = label.toLowerCase();
    if (!seen.has(key)) { seen.add(key); out.push({ label, conf: c, source: "SpeciesNet" }); }
  }
  for (const [s, c] of ((bd.BioClip ?? []) as [string, number][]).slice(0, 3)) {
    const key = String(s).toLowerCase();
    if (!seen.has(key)) { seen.add(key); out.push({ label: String(s), conf: c, source: "BioClip" }); }
  }
  return out;
}

// ── Model breakdown components ────────────────────────────────────────────────

function ModelPill({ name, conf }: { name: string; conf?: number }) {
  const base =
    name === "MDv5a"
      ? "bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-900/30"
      : name === "BioClip"
      ? "bg-violet-100 dark:bg-violet-950/60 text-violet-700 dark:text-violet-400 border-violet-200 dark:border-violet-900/30"
      : name === "SpeciesNet"
      ? "bg-teal-100 dark:bg-teal-950/60 text-teal-700 dark:text-teal-400 border-teal-200 dark:border-teal-900/30"
      : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";
  return (
    <span className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-bold border ${base}`}>
      {name}
      {conf !== undefined && conf > 0 && (
        <span className="font-normal opacity-80 ml-0.5">{Math.round(conf * 100)}%</span>
      )}
    </span>
  );
}

function AgreementBadge({ level }: { level?: string | null }) {
  if (!level) return null;
  const styles =
    level === "High"
      ? "bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/30"
      : level === "Medium"
      ? "bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-900/30"
      : "bg-red-100 dark:bg-red-950/60 text-red-700 dark:text-red-400 border-red-200 dark:border-red-900/30";
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold border ${styles}`}>
      {level}
    </span>
  );
}

function FullModelBreakdown({ row, detected }: { row: Row; detected: string }) {
  let bd: Record<string, any> | null = null;
  try {
    if (typeof row.model_breakdown === "string") bd = JSON.parse(row.model_breakdown as string);
    else if (row.model_breakdown && typeof row.model_breakdown === "object") bd = row.model_breakdown as Record<string, any>;
  } catch {}

  const method = String(row.detection_method ?? "");
  const bioclipConf = typeof row.bioclip_confidence === "number" ? row.bioclip_confidence as number : undefined;
  const speciesnetConf = typeof row.speciesnet_confidence === "number" ? row.speciesnet_confidence as number : undefined;
  const agreement = row.agreement as string | null;
  const isAnimal = !!detected && detected !== "Empty" && detected !== "Unidentified" &&
    detected !== "Person" && detected !== "Vehicle" && detected !== "Error";

  const detectors = method.split(" + ").filter((m) => m.startsWith("MDv") || m === "MegaDetector");
  const mdv5a: { label: string; conf: number }[] = bd?.MDv5a ?? [];
  const bcResults: [string, number][] = (bd?.BioClip ?? []).slice(0, 3);
  const snResults: [string, number][] = (bd?.SpeciesNet ?? []).slice(0, 3);

  const hasAny = detectors.length > 0 || bcResults.length > 0 || snResults.length > 0;
  if (!hasAny) return null;

  return (
    <div className="space-y-1.5">
      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Model Predictions</span>
      <div className="border border-slate-100 rounded-xl overflow-hidden divide-y divide-slate-100">

        {/* MegaDetector detections */}
        {detectors.length > 0 && (
          <div className="p-2.5 space-y-1.5 bg-slate-50/50">
            <span className="text-[10px] font-bold uppercase tracking-wider text-blue-500">Object Detector</span>
            {mdv5a.length > 0 ? mdv5a.map((d, i) => (
              <div key={`md5-${i}`} className="flex items-center gap-1.5 text-[10px]">
                <ModelPill name="MDv5a" />
                <span className="flex-1 font-medium text-slate-600 truncate">{d.label}</span>
                <div className="w-14 h-1.5 bg-slate-100 rounded-full overflow-hidden shrink-0">
                  <div className="h-1.5 bg-blue-400 rounded-full" style={{ width: `${Math.round((d.conf ?? 0) * 100)}%` }} />
                </div>
                <span className="font-mono text-slate-400 w-7 text-right shrink-0">{Math.round((d.conf ?? 0) * 100)}%</span>
              </div>
            )) : <ModelPill name="MDv5a" />}
          </div>
        )}

        {/* SpeciesNet ranked predictions (primary classifier) */}
        {isAnimal && (snResults.length > 0 || (speciesnetConf !== undefined && speciesnetConf > 0)) && (
          <div className="p-2.5 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-teal-500">SpeciesNet</span>
              {speciesnetConf !== undefined && speciesnetConf > 0 && (
                <span className="text-[10px] font-mono text-teal-500">{Math.round(speciesnetConf * 100)}% top</span>
              )}
            </div>
            {snResults.length > 0 ? snResults.map(([s, c], i) => {
              let label = String(s);
              let scientific = "";
              if (label.startsWith("{")) {
                try {
                  const p = JSON.parse(label);
                  label = p.common_name || label;
                  scientific = p.scientific_name || "";
                } catch {}
              }
              return (
                <div key={`sn-${i}`} className="flex items-start gap-1.5 text-[10px]">
                  <span className="text-slate-400 w-3 shrink-0 text-center mt-0.5">{i + 1}.</span>
                  <div className="flex-1 min-w-0">
                    <span className="font-medium text-slate-700 block truncate">{label}</span>
                    {scientific && <span className="text-[10px] italic text-slate-400 block truncate">{scientific}</span>}
                  </div>
                  <div className="w-14 h-1.5 bg-slate-100 rounded-full overflow-hidden shrink-0 mt-1.5">
                    <div className="h-1.5 bg-teal-400 rounded-full" style={{ width: `${Math.round(c * 100)}%` }} />
                  </div>
                  <span className="font-mono text-slate-400 w-7 text-right shrink-0">{Math.round(c * 100)}%</span>
                </div>
              );
            }) : (
              <div className="flex items-center gap-1.5 text-[10px]">
                <span className="text-slate-400 w-3 shrink-0 text-center">1.</span>
                <span className="flex-1 font-medium text-slate-700 truncate">{detected}</span>
                <div className="w-14 h-1.5 bg-slate-100 rounded-full overflow-hidden shrink-0">
                  <div className="h-1.5 bg-teal-400 rounded-full" style={{ width: `${Math.round((speciesnetConf ?? 0) * 100)}%` }} />
                </div>
                <span className="font-mono text-slate-400 w-7 text-right shrink-0">{Math.round((speciesnetConf ?? 0) * 100)}%</span>
              </div>
            )}
          </div>
        )}

        {/* BioClip supplement (only ran when SpeciesNet was uncertain) */}
        {isAnimal && (bcResults.length > 0 || (bioclipConf !== undefined && bioclipConf > 0)) && (
          <div className="p-2.5 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-violet-500">BioClip <span className="font-normal normal-case text-slate-400">(supplement)</span></span>
              {bioclipConf !== undefined && bioclipConf > 0 && (
                <span className="text-[10px] font-mono text-violet-500">{Math.round(bioclipConf * 100)}% top</span>
              )}
            </div>
            {bcResults.length > 0 ? bcResults.map(([s, c], i) => (
              <div key={`bc-${i}`} className="flex items-center gap-1.5 text-[10px]">
                <span className="text-slate-400 w-3 shrink-0 text-center">{i + 1}.</span>
                <span className="flex-1 font-medium text-slate-700 truncate">{s}</span>
                <div className="w-14 h-1.5 bg-slate-100 rounded-full overflow-hidden shrink-0">
                  <div className="h-1.5 bg-violet-400 rounded-full" style={{ width: `${Math.round(c * 100)}%` }} />
                </div>
                <span className="font-mono text-slate-400 w-7 text-right shrink-0">{Math.round(c * 100)}%</span>
              </div>
            )) : (
              <div className="flex items-center gap-1.5 text-[10px]">
                <span className="text-slate-400 w-3 shrink-0 text-center">1.</span>
                <span className="flex-1 font-medium text-slate-700 truncate">{detected}</span>
                <div className="w-14 h-1.5 bg-slate-100 rounded-full overflow-hidden shrink-0">
                  <div className="h-1.5 bg-violet-400 rounded-full" style={{ width: `${Math.round((bioclipConf ?? 0) * 100)}%` }} />
                </div>
                <span className="font-mono text-slate-400 w-7 text-right shrink-0">{Math.round((bioclipConf ?? 0) * 100)}%</span>
              </div>
            )}
          </div>
        )}

        {/* Fusion final result */}
        {bd?.Fusion && (
          <div className="p-2.5 bg-slate-50/60 flex items-center justify-between gap-2">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-0.5">Final Fusion</span>
              <span className="text-xs font-bold text-slate-800">{bd.Fusion.species}</span>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="text-[10px] font-mono text-slate-600">{Math.round((bd.Fusion.confidence ?? 0) * 100)}%</span>
              <AgreementBadge level={agreement} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


function DayNightBadge({ value }: { value: unknown }) {
  const v = String(value ?? "");
  if (!v) return null;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
      v === "Day"
        ? "bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-400"
        : "bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-400"
    }`}>
      <span className="material-symbols-outlined text-[13px] leading-none select-none">
        {v === "Day" ? "wb_sunny" : "bedtime"}
      </span>
      {v}
    </span>
  );
}

function CountChip({ n }: { n: number }) {
  if (!n) return null;
  return (
    <span className="ml-1.5 inline-flex items-center justify-center bg-red-500 text-white text-[10px] font-bold rounded-full w-4 h-4 leading-none">
      {n > 99 ? "99+" : n}
    </span>
  );
}

function EmptyState({ icon, title, sub }: { icon: string; title: string; sub: string }) {
  return (
    <div className="text-center py-16 space-y-2 text-slate-400">
      <div className="text-5xl">{icon}</div>
      <p className="font-semibold text-slate-500">{title}</p>
      <p className="text-sm">{sub}</p>
    </div>
  );
}

function DetailViewport({
  row,
  isDrawing,
  editedBbox,
  setEditedBbox,
  dragStart,
  setDragStart,
  imgNatural,
  setImgNatural,
}: {
  row: Row;
  isDrawing: boolean;
  editedBbox: [number, number, number, number] | null;
  setEditedBbox: (b: [number, number, number, number] | null) => void;
  dragStart: { x: number; y: number } | null;
  setDragStart: (d: { x: number; y: number } | null) => void;
  imgNatural: { w: number; h: number } | null;
  setImgNatural: (n: { w: number; h: number } | null) => void;
}) {
  const filename = String(row.filename ?? "");
  const species = String(row.detected_animal ?? "Unknown");
  const [imgError, setImgError] = useState(false);
  const imageContainerRef = useRef<HTMLDivElement>(null);

  const bbox = parseBbox(row.bbox);
  const activeBbox = editedBbox || bbox;

  const renderBbox = () => {
    if (!activeBbox || !imgNatural) return null;
    const rx = activeBbox[0] * imgNatural.w;
    const ry = activeBbox[1] * imgNatural.h;
    const rw = activeBbox[2] * imgNatural.w;
    const rh = activeBbox[3] * imgNatural.h;

    return (
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none"
        viewBox={`0 0 ${imgNatural.w} ${imgNatural.h}`}
        preserveAspectRatio="none"
      >
        <rect
          x={rx}
          y={ry}
          width={rw}
          height={rh}
          fill="none"
          stroke={editedBbox ? "#fb923c" : "#22c55e"}
          strokeWidth={Math.max(4, imgNatural.w / 250)}
          strokeDasharray={editedBbox ? "8 6" : undefined}
        />
        <text
          x={rx + 8}
          y={ry - 10}
          fill={editedBbox ? "#fb923c" : "#22c55e"}
          fontWeight="bold"
          fontSize={Math.max(16, imgNatural.w / 45)}
          style={{ filter: "drop-shadow(0 2px 4px #000)" }}
        >
          {editedBbox ? "New Box" : species}
        </text>
      </svg>
    );
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawing || !imageContainerRef.current) return;
    const rect = imageContainerRef.current.getBoundingClientRect();
    const startX = (e.clientX - rect.left) / rect.width;
    const startY = (e.clientY - rect.top) / rect.height;
    setDragStart({ x: startX, y: startY });
    setEditedBbox([startX, startY, 0, 0]);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawing || !dragStart || !imageContainerRef.current) return;
    const rect = imageContainerRef.current.getBoundingClientRect();
    const currX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const currY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

    const x = Math.min(dragStart.x, currX);
    const y = Math.min(dragStart.y, currY);
    const w = Math.abs(dragStart.x - currX);
    const h = Math.abs(dragStart.y - currY);

    setEditedBbox([x, y, w, h]);
  };

  const handleMouseUp = () => {
    setDragStart(null);
  };

  useEffect(() => {
    setImgError(false);
    setImgNatural(null);
  }, [filename, setImgNatural]);

  return (
    <div className="relative w-full h-full flex items-center justify-center bg-slate-900 rounded-xl overflow-hidden shadow-2xl border border-slate-800">
      <div
        ref={imageContainerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        className={`relative ${isDrawing ? "cursor-crosshair" : ""}`}
        style={imgNatural
          ? { aspectRatio: `${imgNatural.w} / ${imgNatural.h}`, maxHeight: "100%", maxWidth: "100%" }
          : { width: "100%", height: "100%" }
        }
      >
        {imgError ? (
          <div className="w-full h-full flex items-center justify-center text-slate-600 bg-slate-950">
            <span className="material-symbols-outlined text-5xl">broken_image</span>
          </div>
        ) : (
          <>
            <img
              src={storedImageUrl(filename)}
              alt={filename}
              className="w-full h-full block select-none"
              onLoad={(e) => {
                const img = e.currentTarget as HTMLImageElement;
                setImgNatural({ w: img.naturalWidth, h: img.naturalHeight });
              }}
              onError={() => setImgError(true)}
            />
            {renderBbox()}
          </>
        )}

        <div className="absolute top-3 left-3 flex gap-2">
          <DayNightBadge value={row.day_night} />
          {row.station_id != null && String(row.station_id) && (
            <span className="bg-black/60 backdrop-blur-md text-white text-[10px] font-semibold px-2.5 py-1 rounded-full border border-white/10">
              Station: {String(row.station_id)}
            </span>
          )}
        </div>

        {isDrawing && (
          <div className="absolute bottom-3 left-3 bg-orange-600 text-white text-[10px] font-bold px-3 py-1 rounded-full shadow-lg animate-pulse border border-orange-500">
            ✏️ Drag on image to draw bounding box
          </div>
        )}
      </div>
    </div>
  );
}

function DetailSidebar({
  row,
  reviewerId,
  onAction,
  mode,
  setMode,
  correctLabel,
  setCorrectLabel,
  notes,
  setNotes,
  busy,
  setBusy,
  isDrawing,
  setIsDrawing,
  editedBbox,
  setEditedBbox,
}: {
  row: Row;
  reviewerId: string;
  onAction: () => void;
  mode: "idle" | "correct" | "confirm-note" | "flag-note";
  setMode: (m: "idle" | "correct" | "confirm-note" | "flag-note") => void;
  correctLabel: string;
  setCorrectLabel: (l: string) => void;
  notes: string;
  setNotes: (n: string) => void;
  busy: boolean;
  setBusy: (b: boolean) => void;
  isDrawing: boolean;
  setIsDrawing: (d: boolean) => void;
  editedBbox: [number, number, number, number] | null;
  setEditedBbox: (b: [number, number, number, number] | null) => void;
}) {
  const id = Number(row.id ?? row.image_id ?? 0);
  const species = String(row.detected_animal ?? "Unknown");
  const filename = String(row.filename ?? "");
  const conf = typeof row.detection_confidence === "number" ? row.detection_confidence : parseFloat(String(row.detection_confidence ?? "0"));

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try {
      await fn();
      onAction();
    } finally {
      setBusy(false);
      setMode("idle");
      setNotes("");
      setCorrectLabel("");
      setIsDrawing(false);
      setEditedBbox(null);
    }
  };

  useEffect(() => {
    const active = document.activeElement?.tagName;
    if (active === "INPUT" || active === "TEXTAREA") return;

    const handler = (e: KeyboardEvent) => {
      if (mode === "idle" && !busy) {
        if (e.key === "a") { e.preventDefault(); setMode("confirm-note"); }
        if (e.key === "c") { e.preventDefault(); setMode("correct"); }
        if (e.key === "f") { e.preventDefault(); setMode("flag-note"); }
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setMode("idle");
        setIsDrawing(false);
        setEditedBbox(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [mode, busy, setMode, setIsDrawing, setEditedBbox]);

  return (
    <div className="w-80 border-l border-slate-200 bg-white flex flex-col min-h-0 select-none">
      <div className="p-4 border-b border-slate-100 bg-slate-50/50 shrink-0">
        <h3 className="font-bold text-slate-800 text-sm">Metadata & Diagnosis</h3>
        <p className="text-[10px] text-slate-400 mt-0.5 truncate">{filename}</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
        {/* Metadata */}
        <div className="bg-slate-50 border border-slate-100 rounded-xl p-3 space-y-2.5">
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-wider font-bold mb-0.5">Classified Species</p>
            <p className="font-bold text-slate-800 text-sm">{species}</p>
          </div>
          <div>
            <p className="text-[10px] text-slate-400 uppercase tracking-wider font-bold mb-1">Confidence</p>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-slate-200 rounded-full h-1.5 overflow-hidden">
                <div
                  className={`h-1.5 rounded-full transition-all ${conf >= 0.7 ? "bg-emerald-500" : conf >= 0.4 ? "bg-amber-400" : "bg-red-400"}`}
                  style={{ width: `${Math.round(conf * 100)}%` }}
                />
              </div>
              <span className={`font-bold text-xs w-8 text-right shrink-0 ${conf >= 0.7 ? "text-emerald-600" : conf >= 0.4 ? "text-amber-500" : "text-red-500"}`}>
                {Math.round(conf * 100)}%
              </span>
            </div>
          </div>
          {row.capture_date != null && (
            <div>
              <p className="text-[10px] text-slate-400 uppercase tracking-wider font-bold mb-0.5">Date / Time</p>
              <p className="font-medium text-slate-600 text-[11px]">
                {String(row.capture_date)} {String(row.capture_time ?? "")}
              </p>
            </div>
          )}
          {row.station_id != null && String(row.station_id) && (
            <div>
              <p className="text-[10px] text-slate-400 uppercase tracking-wider font-bold mb-0.5">Station</p>
              <p className="font-medium text-slate-600 text-[11px]">{String(row.station_id)}</p>
            </div>
          )}
        </div>

        <FullModelBreakdown row={row} detected={species} />

        <div className="pt-1 border-t border-slate-100">
          {mode === "idle" ? (
            <div className="space-y-2">
              <button
                onClick={() => setMode("confirm-note")}
                disabled={busy}
                className="w-full flex items-center gap-1.5 py-2.5 px-3 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-lg shadow-sm hover:shadow transition disabled:opacity-50"
              >
                <span className="material-symbols-outlined text-sm">check_circle</span>
                <span className="flex-1">Confirm Detection</span>
                <kbd className="text-[9px] bg-white/20 px-1.5 py-0.5 rounded font-mono border border-white/20">A</kbd>
              </button>
              <button
                onClick={() => setMode("correct")}
                disabled={busy}
                className="w-full flex items-center gap-1.5 py-2.5 px-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-sm hover:shadow transition disabled:opacity-50"
              >
                <span className="material-symbols-outlined text-sm">edit</span>
                <span className="flex-1">Correct Label / Box</span>
                <kbd className="text-[9px] bg-white/20 px-1.5 py-0.5 rounded font-mono border border-white/20">C</kbd>
              </button>
              <button
                onClick={() => setMode("flag-note")}
                disabled={busy}
                className="w-full flex items-center gap-1.5 py-2.5 px-3 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg shadow-sm hover:shadow transition disabled:opacity-50"
              >
                <span className="material-symbols-outlined text-sm">flag</span>
                <span className="flex-1">Flag For Review</span>
                <kbd className="text-[9px] bg-white/20 px-1.5 py-0.5 rounded font-mono border border-white/20">F</kbd>
              </button>
            </div>
          ) : (
            <div className="space-y-3 bg-slate-50 border border-slate-200/80 rounded-xl p-3.5">
              {mode === "confirm-note" && (
                <>
                  <p className="font-bold text-emerald-800 text-[11px] uppercase tracking-wider">Confirm Species</p>
                  <p className="text-[11px] text-slate-500">Confirming detection as <strong className="text-slate-700">{species}</strong>.</p>
                  <input
                    autoFocus
                    placeholder="Notes (optional)"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") run(() => confirmDetection(id, { reviewer_id: reviewerId, notes, bbox: editedBbox || undefined }));
                      if (e.key === "Escape") { setMode("idle"); setIsDrawing(false); setEditedBbox(null); }
                    }}
                    className="w-full border border-slate-200 rounded-lg px-2.5 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-emerald-400 bg-white"
                  />
                  <div className="flex gap-2">
                    <button
                      disabled={busy}
                      onClick={() => run(() => confirmDetection(id, { reviewer_id: reviewerId, notes, bbox: editedBbox || undefined }))}
                      className="flex-1 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg transition"
                    >
                      {busy ? "Confirming…" : "Confirm"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setIsDrawing(!isDrawing)}
                      className={`px-2.5 py-2 rounded-lg text-xs font-semibold border transition ${
                        isDrawing
                          ? "bg-orange-100 text-orange-700 border-orange-200"
                          : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      {isDrawing ? "Done Box" : "✏️ Box"}
                    </button>
                  </div>
                </>
              )}

              {mode === "correct" && (() => {
                const candidates = getCandidates(row);
                return (
                  <>
                    <p className="font-bold text-blue-800 text-[11px] uppercase tracking-wider">Correct Species</p>

                    {/* Model candidates */}
                    {candidates.length > 0 && (
                      <div className="space-y-1">
                        <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Model candidates — click to select</p>
                        <div className="space-y-0.5 max-h-40 overflow-y-auto">
                          {candidates.map((c, ci) => (
                            <button
                              key={ci}
                              type="button"
                              onClick={() => setCorrectLabel(c.label)}
                              className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs text-left transition cursor-pointer ${
                                correctLabel === c.label
                                  ? "bg-blue-100 text-blue-700 font-semibold border border-blue-200"
                                  : "bg-white hover:bg-blue-50 text-slate-700 border border-slate-100"
                              }`}
                            >
                              <span className="truncate">{c.label}</span>
                              <div className="flex items-center gap-1 shrink-0 ml-2">
                                <span className="text-[9px] font-mono text-slate-400">{Math.round(c.conf * 100)}%</span>
                                <span className={`text-[10px] font-bold px-1 py-0.5 rounded ${
                                  c.source === "BioClip"
                                    ? "bg-violet-100 dark:bg-violet-950/60 text-violet-600 dark:text-violet-400"
                                    : "bg-teal-100 dark:bg-teal-950/60 text-teal-600 dark:text-teal-400"
                                }`}>{c.source === "BioClip" ? "BC" : "SN"}</span>
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    <div>
                      <p className="text-[9px] font-bold text-slate-400 uppercase tracking-wider mb-1">
                        {candidates.length > 0 ? "Or type manually" : "Enter correct label"}
                      </p>
                      <input
                        autoFocus={candidates.length === 0}
                        placeholder="e.g. Lion"
                        value={correctLabel}
                        onChange={(e) => setCorrectLabel(e.target.value)}
                        onKeyDown={(e) => e.key === "Escape" && setMode("idle")}
                        className="w-full border border-slate-200 rounded-lg px-2.5 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white"
                      />
                    </div>
                    <input
                      placeholder="Notes (optional)"
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") run(() => correctDetection(id, { reviewer_id: reviewerId, corrected_label: correctLabel, notes, bbox: editedBbox || undefined }));
                        if (e.key === "Escape") { setMode("idle"); setIsDrawing(false); setEditedBbox(null); }
                      }}
                      className="w-full border border-slate-200 rounded-lg px-2.5 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white"
                    />
                    <div className="flex gap-2">
                      <button
                        disabled={busy || !correctLabel.trim()}
                        onClick={() => run(() => correctDetection(id, { reviewer_id: reviewerId, corrected_label: correctLabel, notes, bbox: editedBbox || undefined }))}
                        className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg disabled:opacity-50 transition"
                      >
                        {busy ? "Saving…" : "Save Correction"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setIsDrawing(!isDrawing)}
                        className={`px-2.5 py-2 rounded-lg text-xs font-semibold border transition ${
                          isDrawing
                            ? "bg-orange-100 text-orange-700 border-orange-200"
                            : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                        }`}
                      >
                        {isDrawing ? "Done Box" : "✏️ Box"}
                      </button>
                    </div>
                  </>
                );
              })()}

              {mode === "flag-note" && (
                <>
                  <p className="font-bold text-red-800 text-[11px] uppercase tracking-wider">Flag Detection</p>
                  <input
                    autoFocus
                    placeholder="Reason for flagging (optional)"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") run(() => flagDetection(id, { reviewer_id: reviewerId, notes }));
                      if (e.key === "Escape") setMode("idle");
                    }}
                    className="w-full border border-slate-200 rounded-lg px-2.5 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-red-400 bg-white"
                  />
                  <button
                    disabled={busy}
                    onClick={() => run(() => flagDetection(id, { reviewer_id: reviewerId, notes }))}
                    className="w-full py-2 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded-lg transition"
                  >
                    {busy ? "Flagging…" : "Flag"}
                  </button>
                </>
              )}

              <button
                onClick={() => {
                  setMode("idle");
                  setIsDrawing(false);
                  setEditedBbox(null);
                }}
                className="w-full py-1.5 border border-slate-200 text-slate-500 rounded-lg hover:bg-slate-100 transition text-[11px] font-medium"
              >
                Cancel Form
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Log / Audit tables ────────────────────────────────────────────────────────

const LOG_COLS: { key: string; label: string }[] = [
  { key: "image_id",        label: "Image ID" },
  { key: "action",          label: "Action" },
  { key: "corrected_label", label: "Corrected Label" },
  { key: "reviewer_id",     label: "Reviewer" },
  { key: "timestamp",       label: "Timestamp" },
  { key: "notes",           label: "Notes" },
];

const AUDIT_COLS: { key: string; label: string }[] = [
  { key: "filename", label: "Filename" },
  { key: "scrubbed", label: "Scrubbed" },
  { key: "skipped",  label: "Skipped" },
  { key: "reason",   label: "Reason" },
];

function DataTable({ rows, cols }: { rows: Row[]; cols: { key: string; label: string }[] }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-x-auto shadow-sm">
      <table className="w-full text-sm">
        <thead className="bg-slate-50 border-b border-slate-200">
          <tr>
            {cols.map((c) => (
              <th key={c.key} className="text-left px-4 py-3 font-semibold text-slate-500 whitespace-nowrap text-xs uppercase tracking-wide">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map((row, i) => (
            <tr key={i} className="hover:bg-slate-50">
              {cols.map((c) => (
                <td key={c.key} className="px-4 py-2.5 text-slate-700 max-w-xs truncate">
                  {c.key === "action" ? (
                    <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-semibold ${
                      String(row[c.key]) === "accept" ? "bg-green-100 text-green-700" :
                      String(row[c.key]) === "correct" ? "bg-blue-100 text-blue-700" :
                      "bg-red-100 text-red-700"
                    }`}>
                      {String(row[c.key] ?? "")}
                    </span>
                  ) : (
                    String(row[c.key] ?? "—")
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="px-4 py-2 border-t border-slate-100 text-xs text-slate-400">
        {rows.length} record(s)
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function ReviewQueue() {
  const config = useConfigStore((s) => s.config);
  const reviewerId = config?.reviewer_id ?? "anonymous";
  const threshold = config?.review_confidence_threshold ?? 0.9;

  const [queue, setQueue] = useState<Row[]>([]);
  const [log, setLog] = useState<Row[]>([]);
  const [audit, setAudit] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("queue");
  const [focusedIdx, setFocusedIdx] = useState(0);
  const [showHotkeys, setShowHotkeys] = useState(false);
  const [filterDisagreement, setFilterDisagreement] = useState(false);
  const [sortConf, setSortConf] = useState<"asc" | "desc">("asc"); // asc = worst first

  // Session progress tracking
  const initialQueueSize = useRef<number | null>(null);
  const [sessionReviewed, setSessionReviewed] = useState(0);

  const [mode, setMode] = useState<"idle" | "correct" | "confirm-note" | "flag-note">("idle");
  const [correctLabel, setCorrectLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [imgNatural, setImgNatural] = useState<{ w: number; h: number } | null>(null);
  
  const [isDrawing, setIsDrawing] = useState(false);
  const [editedBbox, setEditedBbox] = useState<[number, number, number, number] | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);

  const resetItemActionState = useCallback(() => {
    setMode("idle");
    setCorrectLabel("");
    setNotes("");
    setIsDrawing(false);
    setEditedBbox(null);
    setDragStart(null);
    setImgNatural(null);
  }, []);

  const sortedQueue = [...queue].sort((a, b) => {
    const ca = typeof a.detection_confidence === "number" ? a.detection_confidence as number : 0;
    const cb = typeof b.detection_confidence === "number" ? b.detection_confidence as number : 0;
    return sortConf === "asc" ? ca - cb : cb - ca;
  });

  const displayedQueue = filterDisagreement
    ? sortedQueue.filter((row) => row.agreement === "Low" || row.agreement === "Medium")
    : sortedQueue;

  const load = async (isAction = false) => {
    setLoading(true);
    const [q, l, a] = await Promise.all([getReviewQueue(), getReviewLog(), getPrivacyAudit()]);
    const newQueue = Array.isArray(q) ? q : [];
    if (initialQueueSize.current === null) initialQueueSize.current = newQueue.length;
    if (isAction) {
      setSessionReviewed((n) => n + 1);
      // Clamp so the last item confirming doesn't leave focusedIdx out of bounds
      setFocusedIdx((i) => Math.min(i, Math.max(0, newQueue.length - 1)));
    }
    setQueue(newQueue);
    setLog(Array.isArray(l) ? l : []);
    setAudit(Array.isArray(a) ? a : []);
    setLoading(false);
  };
  useEffect(() => { load(); }, []);

  useEffect(() => {
    if (tab !== "queue" || displayedQueue.length === 0) return;
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;
      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        setFocusedIdx((i) => Math.min(i + 1, displayedQueue.length - 1));
        resetItemActionState();
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        setFocusedIdx((i) => Math.max(i - 1, 0));
        resetItemActionState();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [tab, displayedQueue.length, resetItemActionState]);

  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: "queue",   label: "Review Queue",   count: queue.length },
    { id: "log",     label: "Correction Log", count: log.length },
    { id: "privacy", label: "Privacy Audit",  count: audit.length },
  ];

  return (
    <div className="flex flex-col h-[calc(100vh-6.5rem)] -mt-4 -mx-6 overflow-hidden">
      <div className="px-6 pt-4 pb-2 bg-white border-b border-slate-200 space-y-4 shrink-0">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <h1 className="text-xl font-bold text-slate-800">Review Queue</h1>
            <p className="text-slate-500 text-xs mt-0.5">
              Detections with confidence below <span className="font-mono font-semibold text-slate-700">{threshold}</span> need manual review.
              Reviewer: <span className="font-semibold text-slate-700">{reviewerId}</span>.
            </p>
            {/* Session progress bar */}
            {tab === "queue" && (initialQueueSize.current ?? 0) > 0 && (
              <div className="mt-2 flex items-center gap-3">
                <div className="flex-1 bg-slate-100 rounded-full h-1.5 overflow-hidden max-w-[200px]">
                  <div
                    className="h-1.5 bg-emerald-500 rounded-full transition-all duration-500"
                    style={{ width: `${Math.round((sessionReviewed / (initialQueueSize.current ?? 1)) * 100)}%` }}
                  />
                </div>
                <span className="text-[10px] font-semibold text-slate-500">
                  {sessionReviewed} / {initialQueueSize.current} reviewed this session
                  {queue.length > 0 && <span className="text-slate-400"> · {queue.length} remaining</span>}
                </span>
              </div>
            )}
          </div>
          {tab === "queue" && queue.length > 0 && (
            <p className="text-[10px] text-slate-400 font-mono bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 shrink-0">
              J/K or ↑/↓ navigate · A confirm · C correct · F flag · Esc cancel
            </p>
          )}
        </div>

        <div className="flex items-center justify-between">
          <div className="flex gap-1">
            {tabs.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex items-center px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
                  tab === t.id
                    ? "bg-slate-100 text-slate-800"
                    : "text-slate-500 hover:text-slate-700"
                }`}
              >
                {t.label}
                {t.count !== undefined && t.count > 0 && <CountChip n={t.count} />}
              </button>
            ))}
          </div>

          {tab === "queue" && queue.length > 0 && (
            <div className="flex items-center gap-2">
              {/* Sort control */}
              <button
                onClick={() => { setSortConf(s => s === "asc" ? "desc" : "asc"); setFocusedIdx(0); resetItemActionState(); }}
                className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 hover:bg-slate-100 transition shadow-sm"
                title={sortConf === "asc" ? "Sorted: worst confidence first" : "Sorted: best confidence first"}
              >
                <span className="material-symbols-outlined text-xs">sort</span>
                Conf {sortConf === "asc" ? "↑ Worst first" : "↓ Best first"}
              </button>
              <button
                onClick={() => setShowHotkeys(true)}
                className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 hover:bg-slate-100 transition shadow-sm"
              >
                <span className="material-symbols-outlined text-xs">keyboard</span>
                Key Shortcuts
              </button>
              <label className="flex items-center gap-2 text-xs font-semibold text-slate-600 cursor-pointer select-none bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 hover:bg-slate-100 transition shadow-sm">
                <input
                  type="checkbox"
                  checked={filterDisagreement}
                  onChange={(e) => {
                    setFilterDisagreement(e.target.checked);
                    setFocusedIdx(0);
                    resetItemActionState();
                  }}
                  className="rounded border-slate-300 text-emerald-600 focus:ring-emerald-500 w-3.5 h-3.5"
                />
                Disagreements Only
              </label>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 flex bg-slate-100">
        {loading ? (
          <div className="flex-1 flex items-center justify-center text-slate-400 gap-2">
            <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Loading…
          </div>
        ) : (
          <>
            {tab === "queue" && (() => {
              if (queue.length === 0) {
                return (
                  <div className="flex-1 flex items-center justify-center">
                    <EmptyState icon="✅" title="Queue is clear" sub="All detections are above the confidence threshold." />
                  </div>
                );
              }
              if (displayedQueue.length === 0) {
                return (
                  <div className="flex-1 flex items-center justify-center">
                    <EmptyState icon="⚖️" title="No classifier disagreements found" sub="All current queue items have high classifier agreement." />
                  </div>
                );
              }

              const currentItem = displayedQueue[focusedIdx];
              return (
                <div className="flex-1 flex min-h-0">
                  <div className="w-80 border-r border-slate-200 bg-white flex flex-col min-h-0">
                    <div className="p-3 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center shrink-0">
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Pending Detections</span>
                      <span className="text-[10px] font-semibold text-slate-600 bg-slate-200/60 px-1.5 py-0.5 rounded">
                        {focusedIdx + 1} / {displayedQueue.length}
                      </span>
                    </div>
                    <div className="flex-1 overflow-y-auto divide-y divide-slate-100">
                      {displayedQueue.map((row, idx) => {
                        const isSelected = idx === focusedIdx;
                        const species = String(row.detected_animal ?? "Unknown");
                        const filename = String(row.filename ?? "");
                        const itemConf = typeof row.detection_confidence === "number" ? row.detection_confidence : parseFloat(String(row.detection_confidence ?? "0"));
                        // Agreement-keyed left border: red=Low, amber=Medium, emerald=High, or selected=bright emerald
                        const borderCls = isSelected
                          ? "border-l-4 border-emerald-500"
                          : row.agreement === "Low"
                          ? "border-l-4 border-red-400"
                          : row.agreement === "Medium"
                          ? "border-l-4 border-amber-400"
                          : row.agreement === "High"
                          ? "border-l-4 border-emerald-300"
                          : "border-l-4 border-slate-200";
                        return (
                          <button
                            key={Number(row.id ?? row.image_id ?? idx)}
                            onClick={() => {
                              setFocusedIdx(idx);
                              resetItemActionState();
                            }}
                            className={`w-full text-left p-3 flex gap-3 transition-colors outline-none ${
                              isSelected ? "bg-emerald-50/80" : "hover:bg-slate-50"
                            } ${borderCls}`}
                          >
                            <div className="w-14 h-14 bg-slate-100 rounded border border-slate-200 overflow-hidden shrink-0 relative">
                              <img
                                src={storedImageUrl(filename)}
                                alt=""
                                className="w-full h-full object-cover"
                                onError={(e) => {
                                  (e.target as HTMLElement).style.display = "none";
                                }}
                              />
                              {/* Confidence dot on thumbnail */}
                              <span className={`absolute bottom-0.5 right-0.5 w-2 h-2 rounded-full border border-white ${
                                itemConf >= 0.7 ? "bg-emerald-500" : itemConf >= 0.4 ? "bg-amber-400" : "bg-red-400"
                              }`} />
                            </div>
                            <div className="flex-1 min-w-0 flex flex-col justify-between py-0.5">
                              <div className="min-w-0">
                                <p className={`font-bold truncate ${isSelected ? "text-emerald-700" : "text-slate-800"} text-sm`}>
                                  {species}
                                </p>
                                <p className="text-[10px] text-slate-400 truncate mt-0.5">{filename}</p>
                              </div>
                              <div className="flex items-center justify-between mt-1.5">
                                <span className={`text-[10px] font-semibold ${
                                  itemConf < 0.4 ? "text-red-500" : itemConf < 0.7 ? "text-amber-500" : "text-emerald-600"
                                }`}>{Math.round(itemConf * 100)}%</span>
                                {!!row.agreement && (
                                  <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full border ${
                                    row.agreement === "High"
                                      ? "bg-emerald-100 text-emerald-700 border-emerald-200"
                                      : row.agreement === "Medium"
                                      ? "bg-amber-100 text-amber-700 border-amber-200"
                                      : "bg-red-100 text-red-700 border-red-200"
                                  }`}>
                                    {String(row.agreement)}
                                  </span>
                                )}
                              </div>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <div className="flex-1 bg-slate-950 flex flex-col relative justify-center items-center p-4 group select-none">
                    {currentItem && (
                      <DetailViewport
                        row={currentItem}
                        isDrawing={isDrawing}
                        editedBbox={editedBbox}
                        setEditedBbox={setEditedBbox}
                        dragStart={dragStart}
                        setDragStart={setDragStart}
                        imgNatural={imgNatural}
                        setImgNatural={setImgNatural}
                      />
                    )}
                    {/* Prev / Next navigation arrows */}
                    {focusedIdx > 0 && (
                      <button
                        onClick={() => { setFocusedIdx(i => i - 1); resetItemActionState(); }}
                        className="absolute left-6 top-1/2 -translate-y-1/2 bg-black/40 hover:bg-emerald-600/80 text-white rounded-full w-10 h-10 flex items-center justify-center text-xl transition opacity-0 group-hover:opacity-100 cursor-pointer shadow-lg border border-white/10 z-10"
                        title="Previous (K / ↑)"
                      >‹</button>
                    )}
                    {focusedIdx < displayedQueue.length - 1 && (
                      <button
                        onClick={() => { setFocusedIdx(i => i + 1); resetItemActionState(); }}
                        className="absolute right-6 top-1/2 -translate-y-1/2 bg-black/40 hover:bg-emerald-600/80 text-white rounded-full w-10 h-10 flex items-center justify-center text-xl transition opacity-0 group-hover:opacity-100 cursor-pointer shadow-lg border border-white/10 z-10"
                        title="Next (J / ↓)"
                      >›</button>
                    )}
                    {/* Item counter */}
                    <div className="absolute bottom-6 right-6 text-xs text-white/40 font-mono select-none opacity-0 group-hover:opacity-100 transition">
                      {focusedIdx + 1} / {displayedQueue.length}
                    </div>
                  </div>

                  {currentItem && (
                    <DetailSidebar
                      row={currentItem}
                      reviewerId={reviewerId}
                      onAction={() => load(true)}
                      mode={mode}
                      setMode={setMode}
                      correctLabel={correctLabel}
                      setCorrectLabel={setCorrectLabel}
                      notes={notes}
                      setNotes={setNotes}
                      busy={busy}
                      setBusy={setBusy}
                      isDrawing={isDrawing}
                      setIsDrawing={setIsDrawing}
                      editedBbox={editedBbox}
                      setEditedBbox={setEditedBbox}
                    />
                  )}
                </div>
              );
            })()}

            {tab === "log" && (
              <div className="flex-1 overflow-auto p-6">
                {log.length === 0
                  ? <EmptyState icon="📋" title="No corrections yet" sub="Reviewed detections will appear here." />
                  : <DataTable rows={log} cols={LOG_COLS} />}
              </div>
            )}

            {tab === "privacy" && (
              <div className="flex-1 overflow-auto p-6">
                {audit.length === 0
                  ? <EmptyState icon="🔒" title="No privacy audit entries" sub="Privacy scrub events will be logged here." />
                  : <DataTable rows={audit} cols={AUDIT_COLS} />}
              </div>
            )}
          </>
        )}
      </div>

      <div
        className={`fixed inset-y-0 right-0 w-80 bg-white/95 backdrop-blur-md border-l border-slate-200 shadow-2xl z-50 transform transition-transform duration-300 ease-out flex flex-col ${
          showHotkeys ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="p-4 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-emerald-600">keyboard</span>
            <h3 className="font-bold text-slate-800 text-sm">Keyboard Shortcuts</h3>
          </div>
          <button
            onClick={() => setShowHotkeys(false)}
            className="p-1 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          <div className="space-y-3">
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Navigation</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">Next Card</span>
                <div className="flex gap-1">
                  <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded font-mono shadow-sm">J</kbd>
                  <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded font-mono shadow-sm">↓</kbd>
                </div>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">Previous Card</span>
                <div className="flex gap-1">
                  <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded font-mono shadow-sm">K</kbd>
                  <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded font-mono shadow-sm">↑</kbd>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Card Actions</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">Confirm detection</span>
                <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded font-mono shadow-sm">A</kbd>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">Correct label</span>
                <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded font-mono shadow-sm">C</kbd>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">Flag for review</span>
                <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded font-mono shadow-sm">F</kbd>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600">Cancel / Close Form</span>
                <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded font-mono shadow-sm">Esc</kbd>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Bounding Box Adjustment</h4>
            <div className="space-y-2.5 text-xs text-slate-600 leading-relaxed bg-orange-50/50 border border-orange-100 rounded-xl p-3">
              <p className="flex items-start gap-1.5">
                <span className="text-orange-500 font-bold">1.</span>
                Click <strong className="font-semibold text-slate-700">✏️ Box</strong> inside any active card to toggle editing.
              </p>
              <p className="flex items-start gap-1.5">
                <span className="text-orange-500 font-bold">2.</span>
                Click and drag directly on the image to draw a new bounding box.
              </p>
              <p className="flex items-start gap-1.5">
                <span className="text-orange-500 font-bold">3.</span>
                The new box appears in orange. Press <strong className="font-semibold text-slate-700">Done Box</strong> or save to commit.
              </p>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-slate-100 bg-slate-50/50 text-[10px] text-slate-400 text-center">
          Shortcuts are disabled when typing in input fields.
        </div>
      </div>

      {showHotkeys && (
        <div
          onClick={() => setShowHotkeys(false)}
          className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40 transition-opacity"
        />
      )}
    </div>
  );
}
