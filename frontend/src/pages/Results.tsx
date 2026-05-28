import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { getResults, updateResult, exportExcel, exportCsv, storedImageUrl, confirmDetection, flagDetection } from "../api/client";

type Row = Record<string, unknown>;
type SortDir = "asc" | "desc" | null;
type ViewMode = "table" | "gallery";
type ImageGroup = { filename: string; rows: Row[] };

const EDITABLE = new Set(["detected_animal", "user_notes", "station_id"]);
const BOX_COLORS = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseBbox(raw: unknown): [number, number, number, number] | null {
  if (!raw) return null;
  try {
    const arr = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (Array.isArray(arr) && arr.length === 4) return arr as [number, number, number, number];
  } catch {}
  return null;
}

function confColor(v: number): string {
  if (v >= 0.7) return "bg-emerald-500";
  if (v >= 0.4) return "bg-amber-400";
  return "bg-red-400";
}

/** Hex color for SVG bounding boxes — mirrors confColor thresholds. */
function confBoxColor(conf: number): string {
  if (conf >= 0.7) return "#10b981"; // emerald-500
  if (conf >= 0.4) return "#f59e0b"; // amber-400
  return "#ef4444";                  // red-400
}

/** Extract ranked species candidates from the model_breakdown JSON field. */
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

function DayNightBadge({ value }: { value: unknown }) {
  const v = String(value ?? "");
  if (!v) return <span className="text-slate-300 dark:text-slate-600">—</span>;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
      v === "Day"
        ? "bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300"
        : "bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-300"
    }`}>
      <span className="material-symbols-outlined text-[12px] leading-none select-none">
        {v === "Day" ? "wb_sunny" : "bedtime"}
      </span>
      {v}
    </span>
  );
}

function ConfBar({ value }: { value: unknown }) {
  const v = typeof value === "number" ? value : parseFloat(String(value ?? "0"));
  const pct = isNaN(v) ? 0 : Math.round(v * 100);
  return (
    <div className="flex items-center gap-2 min-w-[80px]">
      <div className="flex-1 bg-slate-100 dark:bg-slate-700 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full rounded-full ${confColor(v)}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-slate-500 dark:text-slate-400 w-8 text-right">{pct}%</span>
    </div>
  );
}

// ── Model breakdown shared component ─────────────────────────────────────────

function ModelPill({ name, conf }: { name: string; conf?: number }) {
  const base =
    name === "MDv5a"
      ? "bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-900/50"
      : name === "BioClip"
      ? "bg-violet-100 dark:bg-violet-950/60 text-violet-700 dark:text-violet-300 border-violet-200 dark:border-violet-900/50"
      : name === "SpeciesNet"
      ? "bg-teal-100 dark:bg-teal-950/60 text-teal-700 dark:text-teal-300 border-teal-200 dark:border-teal-900/50"
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
      ? "bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/50"
      : level === "Medium"
      ? "bg-amber-100 dark:bg-amber-950/60 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-900/50"
      : "bg-red-100 dark:bg-red-950/60 text-red-700 dark:text-red-400 border-red-200 dark:border-red-900/50";
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold border ${styles}`}>
      {level}
    </span>
  );
}

function ModelBreakdown({
  method,
  bioclipConf,
  speciesnetConf,
  agreement,
  detected,
  modelBreakdown,
  onTaxonClick,
  compact,
}: {
  method: string;
  bioclipConf?: number;
  speciesnetConf?: number;
  agreement?: string | null;
  detected?: string;
  modelBreakdown?: unknown;
  onTaxonClick?: (taxon: string) => void;
  compact?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const models = method ? method.split(" + ").filter(Boolean) : [];
  const detectors = models.filter((m) => m.startsWith("MDv") || m === "MegaDetector");
  const isAnimal =
    detected && detected !== "Empty" && detected !== "Unidentified" &&
    detected !== "Person" && detected !== "Vehicle" && detected !== "Error";

  if (!detectors.length && !isAnimal) return null;

  // Safely parse modelBreakdown
  let parsedBreakdown: Record<string, any> | null = null;
  try {
    if (typeof modelBreakdown === "string") {
      parsedBreakdown = JSON.parse(modelBreakdown);
    } else if (modelBreakdown && typeof modelBreakdown === "object") {
      parsedBreakdown = modelBreakdown as Record<string, any>;
    }
  } catch {}

  const hasDetail = parsedBreakdown && Object.keys(parsedBreakdown).length > 0;

  return (
    <div className="space-y-2 mt-1 w-full">
      {/* Pills row */}
      <div className="flex flex-wrap items-center justify-between gap-1 w-full">
        <div className="flex flex-wrap items-center gap-1">
          {detectors.map((m) => <ModelPill key={m} name={m} />)}
          {isAnimal && (
            <>
              {(speciesnetConf ?? 0) > 0 && <ModelPill name="SpeciesNet" conf={speciesnetConf} />}
              {(bioclipConf ?? 0) > 0 && <ModelPill name="BioClip" conf={bioclipConf} />}
              <AgreementBadge level={agreement} />
            </>
          )}
        </div>
        {!compact && hasDetail && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-[10px] text-emerald-600 hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300 font-bold tracking-wider uppercase cursor-pointer hover:underline inline-flex items-center gap-0.5 select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-1 rounded"
          >
            {expanded ? "Hide Details" : "Show Details"}
            <span className="material-symbols-outlined text-[12px] leading-none">
              {expanded ? "keyboard_arrow_up" : "keyboard_arrow_down"}
            </span>
          </button>
        )}
      </div>

      {/* Expanded details accordion */}
      {!compact && expanded && parsedBreakdown && (
        <div className="bg-slate-100/50 dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800/80 rounded-xl p-3 text-[10px] space-y-2.5 shadow-inner w-full">
          {/* Detectors Candidates */}
          {parsedBreakdown.MDv5a?.length > 0 && (
            <div className="space-y-1">
              <span className="font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wide">Object Detection</span>
              <div className="space-y-0.5 font-medium pl-1">
                {parsedBreakdown.MDv5a.map((d: any, idx: number) => (
                  <div key={`md5-${idx}`} className="flex justify-between items-center text-slate-600 dark:text-slate-400">
                    <span>MDv5a: {d.label}</span>
                    <span className="font-mono">{(d.conf * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* SpeciesNet Top Detections (primary) */}
          {parsedBreakdown.SpeciesNet?.length > 0 && (
            <div className="space-y-1">
              <span className="font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wide">SpeciesNet Taxonomy</span>
              <div className="space-y-1 font-medium pl-1">
                {parsedBreakdown.SpeciesNet.slice(0, 3).map(([s, c]: [string, number], idx: number) => {
                  const isJson = s.startsWith("{") && s.endsWith("}");
                  if (isJson) {
                    try {
                      const parsed = JSON.parse(s);
                      return (
                        <div key={`sn-${idx}`} className="border-b border-slate-200/40 dark:border-slate-800/40 pb-1.5 last:border-b-0 last:pb-0 space-y-0.5 text-slate-600 dark:text-slate-400">
                          <div className="flex justify-between items-center">
                            <span className="font-bold text-slate-700 dark:text-slate-300">
                              {onTaxonClick ? (
                                <button
                                  onClick={() => onTaxonClick(parsed.common_name)}
                                  className="hover:text-emerald-600 dark:hover:text-emerald-400 hover:underline cursor-pointer font-bold text-left"
                                >
                                  {parsed.common_name}
                                </button>
                              ) : (
                                parsed.common_name
                              )}
                            </span>
                            <span className="font-mono">{(c * 100).toFixed(0)}%</span>
                          </div>
                          {parsed.scientific_name && (
                            <div className="text-[10px] italic text-slate-500 dark:text-slate-400">
                              {onTaxonClick ? (
                                <button
                                  onClick={() => onTaxonClick(parsed.scientific_name)}
                                  className="hover:text-emerald-600 dark:hover:text-emerald-400 hover:underline cursor-pointer italic"
                                >
                                  {parsed.scientific_name}
                                </button>
                              ) : (
                                parsed.scientific_name
                              )}
                            </div>
                          )}
                          {parsed.hierarchy && parsed.hierarchy.length > 0 && (
                            <div className="text-[10px] text-slate-400 dark:text-slate-500 flex flex-wrap gap-1 items-center mt-0.5 select-none">
                              <span className="material-symbols-outlined text-[10px] leading-none shrink-0">schema</span>
                              {parsed.hierarchy.map((h: string, hIdx: number) => (
                                <span key={hIdx} className="inline-flex items-center gap-1">
                                  {onTaxonClick ? (
                                    <button
                                      onClick={() => onTaxonClick(h)}
                                      className="hover:text-emerald-600 dark:hover:text-emerald-400 hover:underline cursor-pointer"
                                    >
                                      {h}
                                    </button>
                                  ) : (
                                    <span>{h}</span>
                                  )}
                                  {hIdx < parsed.hierarchy.length - 1 && <span className="text-[10px] text-slate-300 dark:text-slate-600">&gt;</span>}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    } catch {}
                  }
                  return (
                    <div key={`sn-${idx}`} className="flex justify-between items-center text-slate-600 dark:text-slate-400">
                      <span>{s}</span>
                      <span className="font-mono">{(c * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* BioClip supplement (only ran when SpeciesNet was uncertain) */}
          {parsedBreakdown.BioClip?.length > 0 && (
            <div className="space-y-1">
              <span className="font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wide">BioClip <span className="font-normal normal-case">supplement</span></span>
              <div className="space-y-0.5 font-medium pl-1">
                {parsedBreakdown.BioClip.slice(0, 3).map(([s, c]: [string, number], idx: number) => (
                  <div key={`bc-${idx}`} className="flex justify-between items-center text-slate-600 dark:text-slate-400">
                    <span>{s}</span>
                    <span className="font-mono">{(c * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Fusion agreement detail */}
          {parsedBreakdown.Fusion && (
            <div className="border-t border-slate-200/50 dark:border-slate-800/50 pt-1.5 flex justify-between items-center text-[10px] text-slate-500 dark:text-slate-400 font-semibold select-none">
              <span>Agreement: {agreement}</span>
              {agreement === "High" && <span className="text-emerald-500 font-bold">+0.08 Agreement Bonus</span>}
              {agreement === "Medium" && <span className="text-amber-500 font-bold">+0.04 Agreement Bonus</span>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Lightbox ──────────────────────────────────────────────────────────────────

function Lightbox({
  group,
  groups,
  onClose,
  onNavigate,
  onSave,
  onTaxonClick,
  onVerify,
  onFlag,
}: {
  group: ImageGroup;
  groups: ImageGroup[];
  onClose: () => void;
  onNavigate: (g: ImageGroup) => void;
  onSave: (id: number, field: string, value: string) => Promise<void>;
  onTaxonClick?: (taxon: string) => void;
  onVerify?: (detId: number) => Promise<void>;
  onFlag?: (detId: number) => Promise<void>;
}) {
  const { filename, rows } = group;
  const imgUrl = storedImageUrl(filename);
  const primary = rows[0];

  const [imgNatural, setImgNatural] = useState<{ w: number; h: number } | null>(null);
  const [editField, setEditField] = useState<string | null>(null);
  const [editVal, setEditVal] = useState("");
  const [saving, setSaving] = useState(false);
  const [detEdit, setDetEdit] = useState<{ idx: number; val: string } | null>(null);
  const [actionBusy, setActionBusy] = useState<"verify" | "flag" | null>(null);

  const filmstripRef = useRef<HTMLDivElement>(null);

  // Zoom / Pan / Opacity states
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [boxOpacity, setBoxOpacity] = useState(0.85);
  const [showCheatsheet, setShowCheatsheet] = useState(true);

  const idx = groups.findIndex((g) => g.filename === filename);

  // Scroll filmstrip to keep selected thumbnail in view
  useEffect(() => {
    if (!filmstripRef.current) return;
    const el = filmstripRef.current.querySelector("[data-active]") as HTMLElement | null;
    el?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  }, [idx]);
  const hasPrev = idx > 0;
  const hasNext = idx < groups.length - 1;

  const startEdit = (field: string) => {
    setEditField(field);
    setEditVal(String(primary[field] ?? ""));
  };

  const commitEdit = async () => {
    if (!editField) return;
    setSaving(true);
    const detId = Number(primary.detection_id ?? primary.id ?? 0);
    await onSave(detId, editField, editVal);
    setSaving(false);
    setEditField(null);
  };

  const commitDetEdit = async () => {
    if (!detEdit) return;
    setSaving(true);
    const row = rows[detEdit.idx];
    const detId = Number(row.detection_id ?? row.id ?? 0);
    await onSave(detId, "detected_animal", detEdit.val);
    setSaving(false);
    setDetEdit(null);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = 1.15;
    const nextScale = e.deltaY < 0 ? scale * zoomFactor : scale / zoomFactor;
    const boundedScale = Math.max(1, Math.min(8, nextScale));
    setScale(boundedScale);
    if (boundedScale <= 1) {
      setTranslate({ x: 0, y: 0 });
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (scale <= 1) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - translate.x, y: e.clientY - translate.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setTranslate({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const resetZoom = () => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (editField || detEdit) return;
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && hasPrev) onNavigate(groups[idx - 1]);
      if (e.key === "ArrowRight" && hasNext) onNavigate(groups[idx + 1]);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [idx, hasPrev, hasNext, editField, onClose, onNavigate, groups, detEdit]);

  useEffect(() => {
    setImgNatural(null);
    setEditField(null);
    setDetEdit(null);
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }, [imgUrl]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="relative bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-2xl w-[90vw] max-w-6xl overflow-hidden flex flex-col max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Main content row: image + details ── */}
        <div className="flex flex-row flex-1 min-h-0 overflow-hidden">

        {/* ── Image panel ── */}
        <div
          className="flex-1 min-w-0 bg-slate-950 flex items-center justify-center relative overflow-hidden group/image select-none"
          onWheel={handleWheel}
        >
          {/* Zoom controls overlay */}
          <div className="absolute top-3 right-3 flex items-center gap-1.5 z-10 bg-slate-900/80 backdrop-blur-md px-2 py-1.5 rounded-lg border border-white/10 opacity-0 group-hover/image:opacity-100 transition-opacity duration-200">
            <button 
              onClick={() => setScale(s => Math.min(8, s * 1.25))}
              className="text-white hover:text-emerald-400 p-1 text-xs font-bold transition flex items-center justify-center cursor-pointer"
              title="Zoom In"
            >
              <span className="material-symbols-outlined text-sm leading-none block">zoom_in</span>
            </button>
            <button 
              onClick={() => setScale(s => Math.max(1, s / 1.25))}
              className="text-white hover:text-emerald-400 p-1 text-xs font-bold transition flex items-center justify-center cursor-pointer"
              title="Zoom Out"
            >
              <span className="material-symbols-outlined text-sm leading-none block">zoom_out</span>
            </button>
            <button 
              onClick={resetZoom}
              className="text-white hover:text-emerald-400 px-1.5 py-0.5 text-xs font-semibold tracking-wide uppercase transition border border-white/20 rounded cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/50"
              title="Reset Zoom"
            >
              Reset
            </button>
          </div>

          {/* Keyboard Cheatsheet Overlay */}
          {showCheatsheet && (
            <div className="absolute bottom-3 left-3 z-10 bg-slate-900/90 backdrop-blur-md border border-white/10 rounded-xl p-3 text-[10px] text-slate-400 space-y-1 shadow-lg max-w-[200px] transition-all">
              <div className="flex items-center justify-between font-bold border-b border-white/10 pb-1 mb-1 text-white">
                <span>Shortcuts Cheatsheet</span>
                <button onClick={() => setShowCheatsheet(false)} className="text-slate-400 hover:text-white shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/50 rounded">
                  <span className="material-symbols-outlined text-sm leading-none select-none">close</span>
                </button>
              </div>
              <div className="flex justify-between gap-4"><span>Navigate Left</span><kbd className="bg-white/10 px-1 rounded text-white font-semibold">←</kbd></div>
              <div className="flex justify-between gap-4"><span>Navigate Right</span><kbd className="bg-white/10 px-1 rounded text-white font-semibold">→</kbd></div>
              <div className="flex justify-between gap-4"><span>Exit Lightbox</span><kbd className="bg-white/10 px-1.5 rounded text-white font-semibold">Esc</kbd></div>
              <div className="flex justify-between gap-4"><span>Zoom (Image)</span><kbd className="bg-white/10 px-1 rounded text-white font-semibold">Scroll</kbd></div>
              <div className="flex justify-between gap-4"><span>Pan (Zoomed)</span><kbd className="bg-white/10 px-1 rounded text-white font-semibold">Drag</kbd></div>
            </div>
          )}

          {/* Nav buttons */}
          {hasPrev && (
            <button
              onClick={() => onNavigate(groups[idx - 1])}
              className="absolute left-3 top-1/2 -translate-y-1/2 bg-black/40 hover:bg-emerald-600/80 text-white rounded-full w-9 h-9 flex items-center justify-center text-lg transition z-10 cursor-pointer shadow-md select-none border border-white/10"
            >‹</button>
          )}
          {hasNext && (
            <button
              onClick={() => onNavigate(groups[idx + 1])}
              className="absolute right-3 top-1/2 -translate-y-1/2 bg-black/40 hover:bg-emerald-600/80 text-white rounded-full w-9 h-9 flex items-center justify-center text-lg transition z-10 cursor-pointer shadow-md select-none border border-white/10"
            >›</button>
          )}

          {/* Interactive Transform Wrapper */}
          <div
            style={{
              transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
              cursor: scale > 1 ? (isDragging ? "grabbing" : "grab") : "default",
              transition: isDragging ? "none" : "transform 0.15s ease-out",
            }}
            className="relative flex items-center justify-center max-h-[85vh] w-full"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            <img
              src={imgUrl}
              alt={filename}
              className="max-h-[85vh] w-full object-contain pointer-events-none"
              onLoad={(e) => {
                const img = e.currentTarget as HTMLImageElement;
                setImgNatural({ w: img.naturalWidth, h: img.naturalHeight });
              }}
              onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
            />

            {imgNatural && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox={`0 0 ${imgNatural.w} ${imgNatural.h}`}
                preserveAspectRatio="xMidYMid meet"
                style={{ opacity: boxOpacity }}
              >
                {rows.map((row, i) => {
                  const bbox = parseBbox(row.bbox);
                  if (!bbox) return null;
                  const rowConf = typeof row.detection_confidence === "number" ? row.detection_confidence as number : 0;
                  // Multi-animal frames: use distinct per-index colours so boxes are distinguishable.
                  // Single-animal: confidence-coded colour (green/amber/red).
                  const color = rows.length > 1 ? BOX_COLORS[i % BOX_COLORS.length] : confBoxColor(rowConf);
                  const sw = Math.max(2, imgNatural.w / 400);
                  const fs = Math.max(14, imgNatural.w / 55);
                  const conf = rowConf > 0 ? ` (${rowConf.toFixed(2)})` : "";
                  return (
                    <g key={i}>
                      <rect
                        x={bbox[0] * imgNatural.w} y={bbox[1] * imgNatural.h}
                        width={bbox[2] * imgNatural.w} height={bbox[3] * imgNatural.h}
                        fill="none" stroke={color} strokeWidth={sw}
                      />
                      <text
                        x={bbox[0] * imgNatural.w + 4} y={bbox[1] * imgNatural.h - 8}
                        fill={color} fontWeight="bold" fontSize={fs}
                        style={{ filter: "drop-shadow(0 1px 2px #000)" }}
                      >
                        {String(row.detected_animal ?? "")}{conf}
                      </text>
                    </g>
                  );
                })}
              </svg>
            )}
          </div>

          <div className="absolute bottom-3 right-3 text-xs text-white/50 font-mono select-none">
            {idx + 1} / {groups.length}
          </div>
        </div>

        {/* ── Details panel ── */}
        <div className="w-80 shrink-0 flex flex-col gap-3 overflow-y-auto border-l border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 p-4">
          <div className="flex items-start justify-between gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
            <p className="font-bold text-slate-800 dark:text-slate-100 text-sm break-all leading-snug">{filename}</p>
            <button onClick={onClose} className="shrink-0 text-slate-400 hover:text-slate-700 dark:text-slate-500 dark:hover:text-slate-300 leading-none mt-0.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-1 rounded">
              <span className="material-symbols-outlined text-base select-none">close</span>
            </button>
          </div>

          {/* Box Opacity Slider */}
          <div className="flex items-center justify-between gap-2 px-1 py-1.5 bg-slate-50 dark:bg-slate-950/40 rounded-lg border border-slate-100 dark:border-slate-800">
            <label className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider pl-1.5">
              Box Opacity
            </label>
            <div className="flex items-center gap-2 pr-1.5">
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={boxOpacity}
                onChange={(e) => setBoxOpacity(parseFloat(e.target.value))}
                className="w-20 accent-emerald-500 bg-slate-200 dark:bg-slate-800 cursor-pointer h-1 rounded-lg appearance-none"
              />
              <span className="text-[10px] font-mono text-slate-400 dark:text-slate-500 w-8 text-right">{Math.round(boxOpacity * 100)}%</span>
            </div>
          </div>

          {/* Detection list */}
          <div className="space-y-2.5">
            <p className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wide">
              Detections ({rows.length})
            </p>
            {rows.map((row, i) => {
              const rawConf = typeof row.detection_confidence === "number" ? row.detection_confidence as number : 0;
              const conf = rawConf > 0 ? Math.round(rawConf * 100) : null;
              const boxColor = rows.length > 1 ? BOX_COLORS[i % BOX_COLORS.length] : confBoxColor(rawConf);
              const isEditingDet = detEdit?.idx === i;
              const candidates = getCandidates(row);

              return (
                <div key={i} className="rounded-xl border border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-900/20 p-3 space-y-2">
                  {isEditingDet ? (
                    /* ── Candidate dropdown + manual override ── */
                    <div className="space-y-2.5">
                      {candidates.length > 0 && (
                        <div className="space-y-1">
                          <p className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Model candidates</p>
                          <div className="space-y-0.5 max-h-40 overflow-y-auto">
                            {candidates.map((c, ci) => (
                              <button
                                key={ci}
                                onClick={() => setDetEdit({ idx: i, val: c.label })}
                                className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs text-left transition cursor-pointer ${
                                  detEdit?.val === c.label
                                    ? "bg-emerald-100 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-400 font-semibold border border-emerald-200 dark:border-emerald-800"
                                    : "bg-white dark:bg-slate-900 hover:bg-emerald-50 dark:hover:bg-emerald-950/20 text-slate-700 dark:text-slate-300 border border-slate-100 dark:border-slate-800"
                                }`}
                              >
                                <span className="truncate">{c.label}</span>
                                <div className="flex items-center gap-1 shrink-0 ml-2">
                                  <span className="text-[10px] font-mono text-slate-400 dark:text-slate-500">{Math.round(c.conf * 100)}%</span>
                                  <span className={`text-[10px] font-bold px-1 py-0.5 rounded ${
                                    c.source === "BioClip"
                                      ? "bg-violet-100 dark:bg-violet-950/50 text-violet-600 dark:text-violet-400"
                                      : "bg-teal-100 dark:bg-teal-950/50 text-teal-600 dark:text-teal-400"
                                  }`}>{c.source === "BioClip" ? "BC" : "SN"}</span>
                                </div>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      <div>
                        <p className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-1">Manual override</p>
                        <input
                          autoFocus={candidates.length === 0}
                          value={detEdit!.val}
                          onChange={(e) => setDetEdit({ idx: i, val: e.target.value })}
                          onKeyDown={(e) => { if (e.key === "Enter") commitDetEdit(); if (e.key === "Escape") setDetEdit(null); }}
                          placeholder="Type species name…"
                          className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500 dark:bg-slate-900 text-slate-800 dark:text-white placeholder-slate-400"
                        />
                      </div>
                      <div className="flex gap-1.5">
                        <button onClick={commitDetEdit} disabled={saving || !detEdit?.val.trim()}
                          className="flex-1 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-xs rounded-lg font-semibold disabled:opacity-50 cursor-pointer transition">
                          {saving ? "Saving…" : "Save"}
                        </button>
                        <button onClick={() => setDetEdit(null)}
                          className="px-3 py-1.5 bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 text-xs rounded-lg cursor-pointer hover:bg-slate-200 dark:hover:bg-slate-700 transition">
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      {/* Species + confidence */}
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-sm shrink-0 shadow-sm" style={{ background: boxColor }} />
                        <span
                          className="font-bold text-slate-800 dark:text-slate-200 text-sm flex-1 truncate cursor-pointer hover:text-emerald-500 dark:hover:text-emerald-400 hover:underline underline-offset-2"
                          title="Click to correct species"
                          onClick={() => setDetEdit({ idx: i, val: String(row.detected_animal ?? "") })}
                        >
                          {String(row.detected_animal ?? "Unknown")}
                        </span>
                        {conf !== null && (
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full text-white font-bold shrink-0"
                            style={{ background: boxColor }}>
                            {conf}%
                          </span>
                        )}
                      </div>

                      {/* Model breakdown */}
                      <ModelBreakdown
                        method={String(row.detection_method ?? "")}
                        bioclipConf={typeof row.bioclip_confidence === "number" ? row.bioclip_confidence as number : undefined}
                        speciesnetConf={typeof row.speciesnet_confidence === "number" ? row.speciesnet_confidence as number : undefined}
                        agreement={row.agreement as string | null}
                        detected={String(row.detected_animal ?? "")}
                        modelBreakdown={row.model_breakdown}
                        onTaxonClick={onTaxonClick}
                      />

                      {/* Verify / Flag quick actions */}
                      {(onVerify || onFlag) && (
                        <div className="flex gap-1.5 pt-1.5 border-t border-slate-100 dark:border-slate-800">
                          {onVerify && (
                            <button
                              onClick={async () => {
                                setActionBusy("verify");
                                try { await onVerify(Number(row.detection_id ?? row.id ?? 0)); }
                                finally { setActionBusy(null); }
                              }}
                              disabled={actionBusy !== null}
                              className="flex-1 flex items-center justify-center gap-1 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-[10px] font-semibold rounded-lg transition disabled:opacity-50 cursor-pointer"
                            >
                              <span className="material-symbols-outlined text-xs select-none">check_circle</span>
                              {actionBusy === "verify" ? "Verifying…" : "Verify"}
                            </button>
                          )}
                          {onFlag && (
                            <button
                              onClick={async () => {
                                setActionBusy("flag");
                                try { await onFlag(Number(row.detection_id ?? row.id ?? 0)); }
                                finally { setActionBusy(null); }
                              }}
                              disabled={actionBusy !== null}
                              className="flex-1 flex items-center justify-center gap-1 py-1.5 bg-amber-500 hover:bg-amber-600 text-white text-[10px] font-semibold rounded-lg transition disabled:opacity-50 cursor-pointer"
                            >
                              <span className="material-symbols-outlined text-xs select-none">flag</span>
                              {actionBusy === "flag" ? "Flagging…" : "Flag"}
                            </button>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
              );
            })}
          </div>

          {/* Shared image fields */}
          <div className="space-y-3.5 flex-1 border-t border-slate-100 dark:border-slate-800/80 pt-3">
            {([
              ["Station",      "station_id",   true ],
              ["Day / Night",  "day_night",     false],
              ["Capture date", "capture_date",  false],
              ["Temperature",  "temperature",   false],
              ["Notes",        "user_notes",    true ],
            ] as [string, string, boolean][]).map(([label, field, editable]) => {
              const isEditing = editField === field;
              const val = primary[field];

              // Notes: always-visible textarea, saves on blur
              if (field === "user_notes") {
                return (
                  <div key={field} className="px-1">
                    <p className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-1">{label}</p>
                    <textarea
                      value={isEditing ? editVal : String(val ?? "")}
                      onFocus={() => { setEditField(field); setEditVal(String(val ?? "")); }}
                      onChange={(e) => setEditVal(e.target.value)}
                      onBlur={() => { if (editField === "user_notes") commitEdit(); }}
                      rows={2}
                      placeholder="Add notes…"
                      className="w-full border border-slate-200 dark:border-slate-700 rounded-xl px-2.5 py-1.5 text-xs resize-none focus:outline-none focus:ring-1 focus:ring-emerald-500 dark:bg-slate-900/60 text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-600"
                    />
                  </div>
                );
              }

              return (
                <div key={field} className="px-1">
                  <p className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-1">{label}</p>
                  {isEditing ? (
                    <div className="flex gap-1.5">
                      <input autoFocus value={editVal} onChange={(e) => setEditVal(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && commitEdit()}
                        className="flex-1 border border-emerald-500 dark:border-emerald-700 rounded-xl px-2.5 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500 dark:bg-slate-900 text-white" />
                      <div className="flex flex-col gap-1.5">
                        <button onClick={commitEdit} disabled={saving}
                          className="px-2.5 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-xs rounded-lg font-semibold hover:shadow transition disabled:opacity-50 cursor-pointer">
                          {saving ? "…" : "✓"}
                        </button>
                        <button onClick={() => setEditField(null)}
                          className="px-2.5 py-1.5 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 text-xs rounded-lg transition cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500">
                          <span className="material-symbols-outlined text-sm leading-none select-none">close</span>
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div
                      className={`text-sm text-slate-800 dark:text-slate-200 font-semibold ${editable ? "cursor-pointer hover:text-emerald-500 dark:hover:text-emerald-400 hover:underline underline-offset-2" : ""}`}
                      onClick={() => editable && startEdit(field)}
                    >
                      {field === "day_night" ? <DayNightBadge value={val} /> : String(val ?? "—")}
                      {editable && !val && <span className="text-slate-300 dark:text-slate-600 font-normal italic">click to add…</span>}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <p className="text-xs text-slate-400 dark:text-slate-500 text-center pt-2 border-t border-slate-100 dark:border-slate-800/80 select-none">
            ← → navigate &nbsp;·&nbsp; Esc close
          </p>
        </div>
      </div>

      {/* ── Filmstrip ── */}
      {groups.length > 1 && (
        <div
          ref={filmstripRef}
          className="border-t border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/80 px-3 py-2 flex gap-1.5 overflow-x-auto shrink-0"
          style={{ scrollbarWidth: "thin" }}
        >
          {groups.map((g, gIdx) => {
            const isActive = g.filename === filename;
            const gPrimary = g.rows[0];
            const gConf = typeof gPrimary.detection_confidence === "number" ? gPrimary.detection_confidence as number : 0;
            return (
              <button
                key={g.filename}
                data-active={isActive ? "true" : undefined}
                onClick={() => onNavigate(g)}
                title={g.filename}
                className={`shrink-0 relative rounded-lg overflow-hidden transition-all cursor-pointer border-2 ${
                  isActive ? "border-emerald-500 opacity-100" : "border-transparent opacity-50 hover:opacity-90"
                }`}
                style={{ width: 64, height: 48 }}
              >
                <img
                  src={storedImageUrl(g.filename)}
                  alt=""
                  className="w-full h-full object-cover"
                  onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                />
                <div className={`absolute bottom-0 left-0 right-0 h-1 ${
                  gConf >= 0.7 ? "bg-emerald-500" : gConf >= 0.4 ? "bg-amber-400" : "bg-red-400"
                }`} />
                <span className="absolute top-0.5 right-0.5 text-[8px] font-mono text-white/80 bg-black/50 px-0.5 rounded leading-tight">
                  {gIdx + 1}
                </span>
              </button>
            );
          })}
        </div>
      )}
      </div>
    </div>
  );
}

// ── Gallery card ──────────────────────────────────────────────────────────────

function GalleryCard({ rows, onClick }: { rows: Row[]; onClick: () => void }) {
  const primary = rows[0];
  const filename = String(primary.filename ?? "");
  const [imgNatural, setImgNatural] = useState<{ w: number; h: number } | null>(null);

  const bestConf = rows.reduce((max, r) => {
    const v = typeof r.detection_confidence === "number"
      ? (r.detection_confidence as number)
      : parseFloat(String(r.detection_confidence ?? "0"));
    return isNaN(v) ? max : Math.max(max, v);
  }, 0);
  const bestPct = Math.round(bestConf * 100);

  // Pick the detection with highest confidence for model breakdown display
  const bestRow = rows.reduce((best, r) => {
    const v = typeof r.detection_confidence === "number" ? (r.detection_confidence as number) : 0;
    const bv = typeof best.detection_confidence === "number" ? (best.detection_confidence as number) : 0;
    return v > bv ? r : best;
  }, rows[0]);

  return (
    <div
      onClick={onClick}
      className="group bg-white rounded-xl border border-slate-200 overflow-hidden cursor-pointer hover:shadow-md hover:border-green-300 transition-all"
    >
      <div className="relative bg-slate-900 aspect-video overflow-hidden">
        <img
          src={storedImageUrl(filename)}
          alt={filename}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          onLoad={(e) => {
            const img = e.currentTarget as HTMLImageElement;
            setImgNatural({ w: img.naturalWidth, h: img.naturalHeight });
          }}
          onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
        />

        {imgNatural && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${imgNatural.w} ${imgNatural.h}`}
            preserveAspectRatio="xMidYMid slice"
          >
            {rows.map((row, i) => {
              const bbox = parseBbox(row.bbox);
              if (!bbox) return null;
              const color = BOX_COLORS[i % BOX_COLORS.length];
              const sw = Math.max(3, imgNatural.w / 300);
              const fs = Math.max(16, imgNatural.w / 50);
              return (
                <g key={i}>
                  <rect
                    x={bbox[0] * imgNatural.w} y={bbox[1] * imgNatural.h}
                    width={bbox[2] * imgNatural.w} height={bbox[3] * imgNatural.h}
                    fill="none" stroke={color} strokeWidth={sw}
                  />
                  <text
                    x={bbox[0] * imgNatural.w + 4} y={bbox[1] * imgNatural.h - 6}
                    fill={color} fontWeight="bold" fontSize={fs}
                    style={{ filter: "drop-shadow(0 1px 3px #000)" }}
                  >
                    {String(row.detected_animal ?? "")}
                  </text>
                </g>
              );
            })}
          </svg>
        )}

        <div className="absolute top-2 left-2">
          <DayNightBadge value={primary.day_night} />
        </div>
        {rows.length > 1 && (
          <div className="absolute bottom-2 left-2 bg-black/60 text-white text-xs font-semibold px-2 py-0.5 rounded-full">
            {rows.length} animals
          </div>
        )}
        <div className={`absolute top-2 right-2 text-xs font-mono text-white px-2 py-0.5 rounded-full ${confColor(bestConf)}`}>
          {bestPct}%
        </div>
      </div>

      <div className="p-3 space-y-1.5">
        <p className="font-semibold text-slate-800 text-sm truncate">
          {rows.map((r) => String(r.detected_animal ?? "Unknown")).join(", ")}
        </p>
        <p className="text-[10px] text-slate-400 truncate">{filename}</p>
        <p className="text-[10px] text-slate-500">
          {String(primary.station_id ?? "")} · {String(primary.capture_date ?? "")}
        </p>
        <ModelBreakdown
          method={String(bestRow.detection_method ?? "")}
          bioclipConf={typeof bestRow.bioclip_confidence === "number" ? bestRow.bioclip_confidence as number : undefined}
          speciesnetConf={typeof bestRow.speciesnet_confidence === "number" ? bestRow.speciesnet_confidence as number : undefined}
          agreement={bestRow.agreement as string | null}
          detected={String(bestRow.detected_animal ?? "")}
          modelBreakdown={bestRow.model_breakdown}
        />
      </div>
    </div>
  );
}

// ── Pagination control ────────────────────────────────────────────────────────

function Pagination({ page, totalPages, total, onPage }: {
  page: number; totalPages: number; total: number; onPage: (p: number) => void;
}) {
  if (totalPages <= 1) return null;
  return (
    <div className="flex items-center justify-between pt-1">
      <p className="text-xs text-slate-400">{total} total · page {page} of {totalPages}</p>
      <div className="flex gap-1">
        <button disabled={page === 1} onClick={() => onPage(1)}
          className="px-2 py-1 text-xs rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-50">«</button>
        <button disabled={page === 1} onClick={() => onPage(page - 1)}
          className="px-3 py-1 text-xs rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-50">‹ Prev</button>
        {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
          const start = Math.max(1, Math.min(page - 2, totalPages - 4));
          const p = start + i;
          return (
            <button key={p} onClick={() => onPage(p)}
              className={`px-3 py-1 text-xs rounded border transition ${
                p === page
                  ? "bg-green-600 text-white border-green-600"
                  : "border-slate-200 hover:bg-slate-50 text-slate-600"
              }`}>{p}</button>
          );
        })}
        <button disabled={page === totalPages} onClick={() => onPage(page + 1)}
          className="px-3 py-1 text-xs rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-50">Next ›</button>
        <button disabled={page === totalPages} onClick={() => onPage(totalPages)}
          className="px-2 py-1 text-xs rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-50">»</button>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function Results() {
  const PAGE_SIZE = 50;

  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("table");
  const [filter, setFilter] = useState({ species: "", day_night: "", min_conf: "", max_conf: "", station: "" });
  const [selectedTaxon, setSelectedTaxon] = useState<string | null>(null);
  const [editing, setEditing] = useState<{ id: number; field: string } | null>(null);
  const [editVal, setEditVal] = useState("");
  const [lightbox, setLightbox] = useState<ImageGroup | null>(null);
  const [sort, setSort] = useState<{ col: string; dir: SortDir }>({ col: "", dir: null });
  const [page, setPage] = useState(1);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [bulkBusy, setBulkBusy] = useState<"verify" | "flag" | null>(null);
  const speciesInputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setPage(1);
    setSelectedIds(new Set());
    try {
      const params: Record<string, string> = {};
      // Confidence range goes to server; text filters are applied client-side
      if (filter.min_conf) params.min_conf = filter.min_conf;
      if (filter.max_conf) params.max_conf = filter.max_conf;
      const data = await getResults(params);
      setRows(data.items as Row[]);
    } finally {
      setLoading(false);
    }
  }, [filter.min_conf, filter.max_conf]);

  useEffect(() => { load(); }, []);

  const filteredRows = useMemo(() => {
    return rows.filter((r) => {
      // Live client-side text filters
      if (filter.species) {
        const s = filter.species.toLowerCase();
        if (!String(r.detected_animal ?? "").toLowerCase().includes(s)) return false;
      }
      if (filter.station) {
        const s = filter.station.toLowerCase();
        if (!String(r.station_id ?? "").toLowerCase().includes(s)) return false;
      }
      if (filter.day_night && r.day_night !== filter.day_night) return false;

      // Taxon drill-down from ModelBreakdown clicks
      if (selectedTaxon) {
        const taxonLower = selectedTaxon.toLowerCase();
        let breakdown: any = null;
        try {
          if (typeof r.model_breakdown === "string") breakdown = JSON.parse(r.model_breakdown);
          else if (r.model_breakdown) breakdown = r.model_breakdown;
        } catch {}
        const top5: [string, number][] = breakdown?.SpeciesNet?.top5 ?? [];
        const hasTaxon = top5.some(([label]) => {
          if (label.startsWith("{") && label.endsWith("}")) {
            try {
              const p = JSON.parse(label);
              return (p.common_name ?? "").toLowerCase().includes(taxonLower)
                || (p.scientific_name ?? "").toLowerCase().includes(taxonLower)
                || (p.hierarchy ?? []).some((h: string) => h.toLowerCase() === taxonLower);
            } catch {}
          }
          return label.toLowerCase().includes(taxonLower);
        });
        const inDetected = String(r.detected_animal ?? "").toLowerCase().includes(taxonLower);
        const inLabel = String(r.species_label ?? "").toLowerCase().includes(taxonLower);
        if (!hasTaxon && !inDetected && !inLabel) return false;
      }
      return true;
    });
  }, [rows, filter.species, filter.station, filter.day_night, selectedTaxon]);

  const sorted = [...filteredRows].sort((a, b) => {
    if (!sort.col || !sort.dir) return 0;
    const va = a[sort.col] ?? "";
    const vb = b[sort.col] ?? "";
    const cmp = String(va).localeCompare(String(vb), undefined, { numeric: true });
    return sort.dir === "asc" ? cmp : -cmp;
  });

  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const paginated = sorted.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const galleryGroups = useMemo<ImageGroup[]>(() => {
    const map = new Map<string, Row[]>();
    for (const row of sorted) {
      const fn = String(row.filename ?? "");
      if (!map.has(fn)) map.set(fn, []);
      map.get(fn)!.push(row);
    }
    return Array.from(map.entries()).map(([filename, rows]) => ({ filename, rows }));
  }, [sorted]);

  const pagedGalleryGroups = useMemo<ImageGroup[]>(() => {
    const map = new Map<string, Row[]>();
    for (const row of paginated) {
      const fn = String(row.filename ?? "");
      if (!map.has(fn)) map.set(fn, []);
      map.get(fn)!.push(row);
    }
    return Array.from(map.entries()).map(([filename, rows]) => ({ filename, rows }));
  }, [paginated]);

  const toggleSort = (col: string) => {
    setSort((s) =>
      s.col === col
        ? { col, dir: s.dir === "asc" ? "desc" : s.dir === "desc" ? null : "asc" }
        : { col, dir: "asc" }
    );
  };

  const sortIcon = (col: string) => {
    if (sort.col !== col) return <span className="text-slate-300 ml-1">↕</span>;
    return <span className="text-green-600 ml-1">{sort.dir === "asc" ? "↑" : "↓"}</span>;
  };

  const saveEdit = async (id: number, field?: string, value?: string) => {
    const f = field ?? editing?.field;
    const v = value ?? editVal;
    if (!f) return;
    await updateResult(id, { [f]: v });
    setEditing(null);
    load();
  };

  const toggleSelect = (id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const allPageSelected = paginated.length > 0 && paginated.every((r) => {
    const id = Number(r.detection_id ?? r.id ?? 0);
    return selectedIds.has(id);
  });

  const toggleSelectAll = () => {
    if (allPageSelected) {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        paginated.forEach((r) => next.delete(Number(r.detection_id ?? r.id ?? 0)));
        return next;
      });
    } else {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        paginated.forEach((r) => next.add(Number(r.detection_id ?? r.id ?? 0)));
        return next;
      });
    }
  };

  const bulkVerify = async () => {
    setBulkBusy("verify");
    try {
      await Promise.all([...selectedIds].map((id) => confirmDetection(id, { reviewer_id: "viewer", action: "accept" })));
      setSelectedIds(new Set());
      load();
    } finally { setBulkBusy(null); }
  };

  const bulkFlag = async () => {
    setBulkBusy("flag");
    try {
      await Promise.all([...selectedIds].map((id) => flagDetection(id, { reviewer_id: "viewer", notes: "" })));
      setSelectedIds(new Set());
      load();
    } finally { setBulkBusy(null); }
  };

  const totalAnimals = rows.filter((r) => String(r.primary_label ?? r.detected_animal ?? "").toLowerCase() !== "empty").length;
  const dayCount = rows.filter((r) => r.day_night === "Day").length;
  const nightCount = rows.filter((r) => r.day_night === "Night").length;
  const uniqueSpecies = new Set(rows.map((r) => r.detected_animal)).size;
  const lowConfCount = rows.filter((r) => {
    const v = typeof r.detection_confidence === "number" ? r.detection_confidence as number : parseFloat(String(r.detection_confidence ?? "1"));
    return !isNaN(v) && v < 0.4;
  }).length;

  const COLS: { key: string; label: string; sortable: boolean }[] = [
    { key: "filename",             label: "Filename",   sortable: true  },
    { key: "station_id",           label: "Station",    sortable: true  },
    { key: "detected_animal",      label: "Species",    sortable: true  },
    { key: "detection_confidence", label: "Confidence", sortable: true  },
    { key: "day_night",            label: "Time",       sortable: true  },
    { key: "capture_date",         label: "Date",       sortable: true  },
    { key: "detection_method",     label: "Models",     sortable: false },
    { key: "user_notes",           label: "Notes",      sortable: false },
  ];

  return (
    <div className="space-y-4">
      {lightbox && (
        <Lightbox
          group={lightbox}
          groups={galleryGroups}
          onClose={() => setLightbox(null)}
          onNavigate={setLightbox}
          onSave={async (id, field, value) => { await saveEdit(id, field, value); }}
          onTaxonClick={setSelectedTaxon}
          onVerify={async (detId) => {
            await confirmDetection(detId, { reviewer_id: "viewer", action: "accept" });
            load();
          }}
          onFlag={async (detId) => {
            await flagDetection(detId, { reviewer_id: "viewer", notes: "" });
            load();
          }}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-slate-800 dark:text-white">Review Results</h1>
        <div className="flex gap-2 flex-wrap">
          <div className="flex rounded-lg border border-slate-200 dark:border-slate-800 overflow-hidden bg-white dark:bg-slate-900 shadow-sm">
            <button
              onClick={() => setViewMode("table")}
              className={`px-3 py-1.5 text-sm font-medium transition cursor-pointer ${viewMode === "table" ? "bg-green-600 text-white" : "text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"}`}
            >☰ Table</button>
            <button
              onClick={() => setViewMode("gallery")}
              className={`px-3 py-1.5 text-sm font-medium transition cursor-pointer ${viewMode === "gallery" ? "bg-green-600 text-white" : "text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"}`}
            >⊞ Gallery</button>
          </div>
          <a href={exportExcel()} className="px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 shadow-sm transition">Export Excel</a>
          <a href={exportCsv()} className="px-4 py-2 bg-slate-600 dark:bg-slate-800 text-white text-sm rounded-lg hover:bg-slate-700 dark:hover:bg-slate-700 shadow-sm transition">Export CSV</a>
        </div>
      </div>

      {/* Summary stats */}
      {rows.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {[
            ["Total Images", rows.length, "text-slate-700 dark:text-slate-200"],
            ["Animals", totalAnimals, "text-green-700 dark:text-emerald-400"],
            ["Unique Species", uniqueSpecies, "text-indigo-700 dark:text-indigo-400"],
            ["Day / Night", `${dayCount} / ${nightCount}`, "text-amber-700 dark:text-amber-400"],
            ["Needs Review", lowConfCount, lowConfCount > 0 ? "text-red-600 dark:text-red-400" : "text-slate-400 dark:text-slate-500"],
          ].map(([label, val, cls]) => (
            <div key={String(label)} className="bg-white dark:bg-slate-900/60 rounded-xl border border-slate-200 dark:border-slate-800/80 px-4 py-3 shadow-sm">
              <p className="text-xs text-slate-400 dark:text-slate-500 font-medium uppercase tracking-wide">{label}</p>
              <p className={`text-xl font-bold mt-0.5 ${cls}`}>{String(val)}</p>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="bg-white dark:bg-slate-900/60 rounded-xl border border-slate-200 dark:border-slate-800/80 p-3 flex flex-wrap gap-2 items-end shadow-sm">
        <input ref={speciesInputRef} placeholder="Species (live)…" value={filter.species}
          onChange={(e) => setFilter((f) => ({ ...f, species: e.target.value }))}
          className="border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300" />
        <input placeholder="Station (live)…" value={filter.station}
          onChange={(e) => setFilter((f) => ({ ...f, station: e.target.value }))}
          className="border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300" />
        <select value={filter.day_night}
          onChange={(e) => setFilter((f) => ({ ...f, day_night: e.target.value }))}
          className="border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300">
          <option value="">All times</option>
          <option value="Day">☀ Day</option>
          <option value="Night">🌙 Night</option>
        </select>
        <div className="flex items-center gap-1">
          <input placeholder="Min conf" value={filter.min_conf}
            onChange={(e) => setFilter((f) => ({ ...f, min_conf: e.target.value }))}
            onKeyDown={(e) => e.key === "Enter" && load()}
            className="border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm w-24 focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300" />
          <span className="text-slate-500 dark:text-slate-600 text-sm">–</span>
          <input placeholder="Max conf" value={filter.max_conf}
            onChange={(e) => setFilter((f) => ({ ...f, max_conf: e.target.value }))}
            onKeyDown={(e) => e.key === "Enter" && load()}
            className="border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm w-24 focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300" />
        </div>
        <button onClick={load} title="Apply confidence range filter" className="px-4 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 cursor-pointer shadow-sm">Apply Conf</button>
        {selectedTaxon && (
          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-900/50 rounded-lg text-xs font-bold leading-none h-[34px] tracking-wide uppercase select-none">
            <span>Taxon: {selectedTaxon}</span>
            <button onClick={() => setSelectedTaxon(null)} className="hover:text-emerald-900 dark:hover:text-emerald-300 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 rounded">
              <span className="material-symbols-outlined text-sm leading-none select-none">close</span>
            </button>
          </div>
        )}
        {(filter.species || filter.day_night || filter.min_conf || filter.max_conf || filter.station || selectedTaxon) && (
          <button
            onClick={() => {
              setFilter({ species: "", day_night: "", min_conf: "", max_conf: "", station: "" });
              setSelectedTaxon(null);
              setTimeout(load, 0);
            }}
            className="px-3 py-1.5 text-slate-500 dark:text-slate-400 text-sm rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
          >Clear</button>
        )}
      </div>

      {/* Bulk action bar */}
      {selectedIds.size > 0 && (
        <div className="flex items-center gap-3 px-4 py-2.5 bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900/50 rounded-xl shadow-sm">
          <span className="text-sm font-semibold text-emerald-700 dark:text-emerald-400">{selectedIds.size} selected</span>
          <button
            onClick={bulkVerify}
            disabled={bulkBusy !== null}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-semibold disabled:opacity-50 cursor-pointer transition shadow-sm"
          >
            <span className="material-symbols-outlined text-xs">check_circle</span>
            {bulkBusy === "verify" ? "Verifying…" : "Verify All"}
          </button>
          <button
            onClick={bulkFlag}
            disabled={bulkBusy !== null}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500 hover:bg-amber-600 text-white rounded-lg text-xs font-semibold disabled:opacity-50 cursor-pointer transition shadow-sm"
          >
            <span className="material-symbols-outlined text-xs">flag</span>
            {bulkBusy === "flag" ? "Flagging…" : "Flag All"}
          </button>
          <button
            onClick={() => setSelectedIds(new Set())}
            className="ml-auto px-3 py-1.5 text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg text-xs cursor-pointer transition"
          >Clear selection</button>
        </div>
      )}

      {/* Content */}
      {loading ? (
        <div className="flex items-center justify-center py-16 text-slate-400 dark:text-slate-500 gap-2">
          <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Loading…
        </div>
      ) : sorted.length === 0 ? (
        <div className="text-center py-16 text-slate-400 dark:text-slate-500 space-y-2">
          <div className="text-4xl">📷</div>
          <p className="font-semibold">No results yet</p>
          <p className="text-sm">Process images in the Upload tab first.</p>
        </div>
      ) : viewMode === "gallery" ? (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {pagedGalleryGroups.map((group) => (
              <GalleryCard key={group.filename} rows={group.rows} onClick={() => setLightbox(group)} />
            ))}
          </div>
          <Pagination page={page} totalPages={totalPages} total={sorted.length} onPage={setPage} />
        </>
      ) : (
        <>
          <div className="bg-white dark:bg-slate-900/60 rounded-xl border border-slate-200 dark:border-slate-800/80 overflow-x-auto shadow-sm">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 dark:bg-slate-950/80 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  <th className="px-3 py-3 w-8">
                    <input
                      type="checkbox"
                      checked={allPageSelected}
                      onChange={toggleSelectAll}
                      className="accent-emerald-600 cursor-pointer w-4 h-4 rounded"
                      title="Select all on this page"
                    />
                  </th>
                  <th className="text-left px-3 py-3 font-semibold text-slate-500 dark:text-slate-500 w-20">Image</th>
                  {COLS.map(({ key, label, sortable }) => (
                    <th
                      key={key}
                      className={`text-left px-3 py-3 font-semibold text-slate-500 dark:text-slate-500 whitespace-nowrap ${sortable ? "cursor-pointer select-none hover:text-slate-800 dark:hover:text-slate-200" : ""}`}
                      onClick={() => sortable && toggleSort(key)}
                    >
                      {label}{sortable && sortIcon(key)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {paginated.map((row, i) => {
                  const id = Number(row.detection_id ?? row.id ?? i);
                  const filename = String(row.filename ?? "");
                  const rowConf = typeof row.detection_confidence === "number"
                    ? row.detection_confidence as number
                    : parseFloat(String(row.detection_confidence ?? "1"));
                  const isLowConf = !isNaN(rowConf) && rowConf < 0.4;
                  const isSelected = selectedIds.has(id);

                  return (
                    <tr
                      key={id}
                      className={`group transition ${
                        isSelected
                          ? "bg-emerald-50/60 dark:bg-emerald-950/15"
                          : isLowConf
                          ? "bg-amber-50/50 dark:bg-amber-950/10 hover:bg-amber-50 dark:hover:bg-amber-950/20"
                          : "hover:bg-slate-50/50 dark:hover:bg-slate-900/20"
                      }`}
                    >
                      {/* Checkbox */}
                      <td className="px-3 py-2">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleSelect(id)}
                          className="accent-emerald-600 cursor-pointer w-4 h-4 rounded"
                        />
                      </td>

                      {/* Thumbnail */}
                      <td className="px-3 py-2">
                        <button
                          onClick={() => {
                            const g = galleryGroups.find((g) => g.filename === filename);
                            setLightbox(g ?? { filename, rows: [row] });
                          }}
                          className="relative block w-16 h-12 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-800 hover:ring-2 hover:ring-green-500 transition shrink-0 cursor-pointer"
                          title="View image"
                        >
                          <img
                            src={storedImageUrl(filename)}
                            alt={filename}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              const el = e.currentTarget as HTMLImageElement;
                              el.style.display = "none";
                              el.parentElement!.innerHTML = '<span class="text-[10px] text-slate-400 dark:text-slate-500 flex items-center justify-center h-full w-full">No img</span>';
                            }}
                          />
                          {isLowConf && (
                            <div className="absolute inset-0 flex items-center justify-center bg-amber-500/20">
                              <span className="material-symbols-outlined text-amber-500 text-sm drop-shadow">warning</span>
                            </div>
                          )}
                        </button>
                      </td>

                      {COLS.map(({ key }) => {
                        const editable = EDITABLE.has(key);
                        const isEditing = editing?.id === id && editing?.field === key;
                        const val = row[key];

                        return (
                          <td key={key} className="px-3 py-2 max-w-[220px] text-slate-700 dark:text-slate-300">
                            {isEditing && key === "detected_animal" ? (
                              /* ── Inline species correction with candidate dropdown ── */
                              <div className="min-w-[200px] space-y-1.5">
                                {getCandidates(row).length > 0 && (
                                  <div className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
                                    {getCandidates(row).slice(0, 4).map((c, ci) => (
                                      <button
                                        key={ci}
                                        onClick={() => setEditVal(c.label)}
                                        className={`w-full flex items-center justify-between px-2 py-1 text-xs text-left cursor-pointer border-b last:border-b-0 border-slate-100 dark:border-slate-800 transition ${
                                          editVal === c.label
                                            ? "bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400"
                                            : "bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300"
                                        }`}
                                      >
                                        <span className="truncate font-medium">{c.label}</span>
                                        <div className="flex items-center gap-1 shrink-0 ml-2">
                                          <span className="text-[9px] font-mono text-slate-400">{Math.round(c.conf * 100)}%</span>
                                          <span className={`text-[8px] font-bold px-1 rounded ${
                                            c.source === "BioClip"
                                              ? "bg-violet-100 dark:bg-violet-950/50 text-violet-600 dark:text-violet-400"
                                              : "bg-teal-100 dark:bg-teal-950/50 text-teal-600 dark:text-teal-400"
                                          }`}>{c.source === "BioClip" ? "BC" : "SN"}</span>
                                        </div>
                                      </button>
                                    ))}
                                  </div>
                                )}
                                <div className="flex gap-1 items-center">
                                  <input
                                    autoFocus={getCandidates(row).length === 0}
                                    value={editVal}
                                    onChange={(e) => setEditVal(e.target.value)}
                                    onKeyDown={(e) => { if (e.key === "Enter") saveEdit(id); if (e.key === "Escape") setEditing(null); }}
                                    placeholder="Type species name…"
                                    className="flex-1 min-w-0 border border-green-400 dark:border-emerald-500 rounded px-2 py-0.5 text-xs focus:outline-none bg-white dark:bg-slate-900 text-slate-800 dark:text-white"
                                  />
                                  <button onClick={() => saveEdit(id)} className="text-green-600 dark:text-emerald-500 font-bold shrink-0 cursor-pointer">✓</button>
                                  <button onClick={() => setEditing(null)} className="text-slate-400 shrink-0 cursor-pointer">✕</button>
                                </div>
                              </div>
                            ) : isEditing ? (
                              <div className="flex gap-1 items-center">
                                <input autoFocus value={editVal}
                                  onChange={(e) => setEditVal(e.target.value)}
                                  onKeyDown={(e) => e.key === "Enter" && saveEdit(id)}
                                  className="border border-green-400 dark:border-emerald-500 rounded px-2 py-0.5 text-sm w-full focus:outline-none bg-white dark:bg-slate-900 text-slate-800 dark:text-white" />
                                <button onClick={() => saveEdit(id)} className="text-green-600 dark:text-emerald-500 font-bold shrink-0 cursor-pointer">✓</button>
                                <button onClick={() => setEditing(null)} className="text-slate-300 dark:text-slate-600 hover:text-red-500 shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 rounded">
                                  <span className="material-symbols-outlined text-sm leading-none select-none">close</span>
                                </button>
                              </div>
                            ) : key === "detection_confidence" ? (
                              <ConfBar value={val} />
                            ) : key === "day_night" ? (
                              <DayNightBadge value={val} />
                            ) : key === "detection_method" ? (
                              <ModelBreakdown
                                method={String(val ?? "")}
                                bioclipConf={typeof row.bioclip_confidence === "number" ? row.bioclip_confidence as number : undefined}
                                speciesnetConf={typeof row.speciesnet_confidence === "number" ? row.speciesnet_confidence as number : undefined}
                                agreement={row.agreement as string | null}
                                detected={String(row.detected_animal ?? "")}
                                compact
                              />
                            ) : key === "detected_animal" ? (
                              <div className="flex items-center gap-1.5">
                                {isLowConf && (
                                  <span className="material-symbols-outlined text-amber-400 text-sm shrink-0" title="Low confidence">warning</span>
                                )}
                                <span
                                  className="block truncate font-semibold cursor-pointer group-hover:text-green-700 dark:group-hover:text-emerald-400 hover:underline underline-offset-2"
                                  title={`${String(val ?? "")} — click to correct`}
                                  onClick={() => { setEditing({ id, field: key }); setEditVal(String(val ?? "")); }}
                                >
                                  {String(val ?? "")}
                                </span>
                              </div>
                            ) : (
                              <span
                                className={`block truncate font-medium ${editable ? "cursor-pointer group-hover:text-green-700 dark:group-hover:text-emerald-400 hover:underline underline-offset-2" : "text-slate-700 dark:text-slate-300"}`}
                                title={String(val ?? "")}
                                onClick={() => editable && (() => { setEditing({ id, field: key }); setEditVal(String(val ?? "")); })()}
                              >
                                {String(val ?? "")}
                              </span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="px-4 py-2 text-xs text-slate-400 dark:text-slate-500 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between select-none">
              <span>{filteredRows.length} record(s){filteredRows.length !== rows.length ? ` (filtered from ${rows.length})` : ""} · Checkbox to select · Click species to correct · Click thumbnail to open</span>
              <span className="text-slate-400 dark:text-slate-700">Column headers to sort</span>
            </div>
          </div>
          <Pagination page={page} totalPages={totalPages} total={sorted.length} onPage={setPage} />
        </>
      )}
    </div>
  );
}
