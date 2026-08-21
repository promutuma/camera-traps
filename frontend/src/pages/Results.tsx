import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { getResults, updateResult, exportExcel, exportCsv, storedThumbUrl, storedImageUrl, confirmDetection, flagDetection, getStations } from "../api/client";

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

/** Extract ranked SpeciesNet candidates from the top_candidates JSON column. */
function getCandidates(row: Row): { label: string; conf: number; scientific?: string; source: "SpeciesNet" }[] {
  let cands: any[] | null = null;
  try {
    if (typeof row.top_candidates === "string") cands = JSON.parse(row.top_candidates);
    else if (Array.isArray(row.top_candidates)) cands = row.top_candidates as any[];
  } catch {}
  if (!cands) return [];

  const seen = new Set<string>();
  return cands.slice(0, 8).flatMap((c: any) => {
    const label = c.common_name || c.display || "";
    if (!label) return [];
    const key = label.toLowerCase();
    if (seen.has(key)) return [];
    seen.add(key);
    return [{ label, conf: c.confidence ?? 0, scientific: c.scientific_name || undefined, source: "SpeciesNet" as const }];
  });
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


function ModelBreakdown({
  method,
  speciesnetConf,
  detected,
  modelBreakdown,
  topCandidates,
  onTaxonClick,
  compact,
}: {
  method: string;
  speciesnetConf?: number;
  detected?: string;
  modelBreakdown?: unknown;
  topCandidates?: unknown;
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

  // Parse MDv5a detections from model_breakdown
  let mdv5a: any[] = [];
  try {
    let bd: any = typeof modelBreakdown === "string" ? JSON.parse(modelBreakdown) : modelBreakdown;
    if (bd?.MDv5a) mdv5a = bd.MDv5a;
  } catch {}

  // Parse SpeciesNet candidates from top_candidates column
  let snCandidates: any[] = [];
  try {
    let tc: any = typeof topCandidates === "string" ? JSON.parse(topCandidates) : topCandidates;
    if (Array.isArray(tc)) snCandidates = tc;
  } catch {}

  const hasDetail = mdv5a.length > 0 || (isAnimal && snCandidates.length > 0);

  return (
    <div className="space-y-2 mt-1 w-full">
      {/* Pills row */}
      <div className="flex flex-wrap items-center justify-between gap-1 w-full">
        <div className="flex flex-wrap items-center gap-1">
          {detectors.map((m) => <ModelPill key={m} name={m} />)}
          {isAnimal && (speciesnetConf ?? 0) > 0 && (
            <ModelPill name="SpeciesNet" conf={speciesnetConf} />
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
      {!compact && expanded && (
        <div className="bg-slate-100/50 dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800/80 rounded-xl p-3 text-[10px] space-y-2.5 shadow-inner w-full">
          {/* MegaDetector object detection candidates */}
          {mdv5a.length > 0 && (
            <div className="space-y-1">
              <span className="font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wide">Object Detection</span>
              <div className="space-y-0.5 font-medium pl-1">
                {mdv5a.map((d: any, idx: number) => (
                  <div key={`md5-${idx}`} className="flex justify-between items-center text-slate-600 dark:text-slate-400">
                    <span>MDv5a: {d.label}</span>
                    <span className="font-mono">{(d.conf * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* SpeciesNet top candidates */}
          {isAnimal && snCandidates.length > 0 && (
            <div className="space-y-1">
              <span className="font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wide">SpeciesNet Taxonomy</span>
              <div className="space-y-1 font-medium pl-1">
                {snCandidates.slice(0, 5).map((c: any, idx: number) => (
                  <div key={`sn-${idx}`} className="border-b border-slate-200/40 dark:border-slate-800/40 pb-1.5 last:border-b-0 last:pb-0 space-y-0.5 text-slate-600 dark:text-slate-400">
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-slate-700 dark:text-slate-300">
                        {onTaxonClick ? (
                          <button onClick={() => onTaxonClick(c.common_name)} className="hover:text-emerald-600 dark:hover:text-emerald-400 hover:underline cursor-pointer font-bold text-left">
                            {c.common_name}
                          </button>
                        ) : c.common_name}
                      </span>
                      <span className="font-mono">{Math.round((c.confidence ?? 0) * 100)}%</span>
                    </div>
                    {c.scientific_name && (
                      <div className="text-[10px] italic text-slate-500 dark:text-slate-400">
                        {onTaxonClick ? (
                          <button onClick={() => onTaxonClick(c.scientific_name)} className="hover:text-emerald-600 dark:hover:text-emerald-400 hover:underline cursor-pointer italic">
                            {c.scientific_name}
                          </button>
                        ) : c.scientific_name}
                      </div>
                    )}
                    {idx === 0 && Array.isArray(c.hierarchy) && c.hierarchy.length > 0 && (
                      <div className="text-[10px] text-slate-400 dark:text-slate-500 flex flex-wrap gap-1 items-center mt-0.5 select-none">
                        <span className="material-symbols-outlined text-[10px] leading-none shrink-0">schema</span>
                        {c.hierarchy.map((h: string, hIdx: number) => (
                          <span key={hIdx} className="inline-flex items-center gap-1">
                            {onTaxonClick ? (
                              <button onClick={() => onTaxonClick(h)} className="hover:text-emerald-600 dark:hover:text-emerald-400 hover:underline cursor-pointer">{h}</button>
                            ) : <span>{h}</span>}
                            {hIdx < c.hierarchy.length - 1 && <span className="text-[10px] text-slate-300 dark:text-slate-600">&gt;</span>}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
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
  const [hoveredDetIdx, setHoveredDetIdx] = useState<number | null>(null);

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
                  const isHovered = hoveredDetIdx === i;
                  return (
                    <g
                      key={i}
                      className="pointer-events-auto cursor-pointer"
                      onMouseEnter={() => setHoveredDetIdx(i)}
                      onMouseLeave={() => setHoveredDetIdx(null)}
                    >
                      <rect
                        x={bbox[0] * imgNatural.w} y={bbox[1] * imgNatural.h}
                        width={bbox[2] * imgNatural.w} height={bbox[3] * imgNatural.h}
                        fill={isHovered ? `${color}18` : "none"}
                        stroke={color}
                        strokeWidth={isHovered ? sw + 2 : sw}
                        className="transition-all duration-200"
                      />
                      <text
                        x={bbox[0] * imgNatural.w + 4} y={bbox[1] * imgNatural.h - 8}
                        fill={color} fontWeight="bold" fontSize={isHovered ? fs + 2 : fs}
                        className="pointer-events-none transition-all duration-200"
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
              const isHovered = hoveredDetIdx === i;

              return (
                <div
                  key={i}
                  onMouseEnter={() => setHoveredDetIdx(i)}
                  onMouseLeave={() => setHoveredDetIdx(null)}
                  className={`rounded-xl border transition-all duration-200 p-3 space-y-2 ${
                    isHovered
                      ? "border-emerald-500/80 bg-emerald-50/20 dark:bg-emerald-950/15 shadow-sm ring-1 ring-emerald-500/50 scale-[1.01]"
                      : "border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-900/20"
                  }`}
                >
                  {isEditingDet ? (
                    /* ── Candidate dropdown + manual override ── */
                    <div className="space-y-2.5">
                      {candidates.length > 0 && (
                        <div className="space-y-1">
                          <p className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Model candidates</p>
                          <div className="space-y-0.5 max-h-48 overflow-y-auto">
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
                                <div className="flex-1 min-w-0 mr-2">
                                  <span className="block truncate font-medium">{c.label}</span>
                                  {c.scientific && <span className="block truncate text-[9px] italic text-slate-400 dark:text-slate-500 leading-tight">{c.scientific}</span>}
                                </div>
                                <div className="flex items-center gap-1 shrink-0">
                                  <span className="text-[10px] font-mono text-slate-400 dark:text-slate-500">{Math.round(c.conf * 100)}%</span>
                                  <span className="text-[10px] font-bold px-1 py-0.5 rounded bg-teal-100 dark:bg-teal-950/50 text-teal-600 dark:text-teal-400">SN</span>
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
                      <div className="flex items-start gap-2">
                        <span className="w-2 h-2 rounded-sm shrink-0 shadow-sm mt-1" style={{ background: boxColor }} />
                        <div className="flex-1 min-w-0">
                          <span
                            className="font-bold text-slate-800 dark:text-slate-200 text-sm block truncate cursor-pointer hover:text-emerald-500 dark:hover:text-emerald-400 hover:underline underline-offset-2"
                            title="Click to correct species"
                            onClick={() => setDetEdit({ idx: i, val: String(row.detected_animal ?? "") })}
                          >
                            {String(row.detected_animal ?? "Unknown")}
                          </span>
                          {!!row.scientific_name && (
                            <span className="block text-[10px] italic text-slate-400 dark:text-slate-500 truncate leading-tight">
                              {String(row.scientific_name)}
                            </span>
                          )}
                        </div>
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
                        speciesnetConf={typeof row.speciesnet_confidence === "number" ? row.speciesnet_confidence as number : undefined}
                        detected={String(row.detected_animal ?? "")}
                        modelBreakdown={row.model_breakdown}
                        topCandidates={row.top_candidates}
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
                          {saving ? "…" : <span className="material-symbols-outlined text-xs leading-none select-none">check</span>}
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
                  src={storedThumbUrl(g.filename, 160)}
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
          src={storedThumbUrl(filename, 640)}
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
          speciesnetConf={typeof bestRow.speciesnet_confidence === "number" ? bestRow.speciesnet_confidence as number : undefined}
          detected={String(bestRow.detected_animal ?? "")}
          modelBreakdown={bestRow.model_breakdown}
          topCandidates={bestRow.top_candidates}
        />
      </div>
    </div>
  );
}

// ── Filter chip ───────────────────────────────────────────────────────────────

function Chip({ label, onRemove, color = "slate" }: { label: string; onRemove: () => void; color?: "slate" | "amber" | "emerald" }) {
  const cls = color === "amber"
    ? "bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-900/40"
    : color === "emerald"
    ? "bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/40"
    : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700";
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold border ${cls}`}>
      {label}
      <button onClick={onRemove} className="hover:opacity-70 cursor-pointer leading-none">
        <span className="material-symbols-outlined text-[11px] leading-none select-none">close</span>
      </button>
    </span>
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
  const [filter, setFilter] = useState({ species: "", day_night: "", min_conf: "", max_conf: "", station: "", hideEmpty: true, lowConfOnly: false });
  const [selectedTaxon, setSelectedTaxon] = useState<string | null>(null);
  const [stations, setStations] = useState<string[]>([]);
  const [editing, setEditing] = useState<{ id: number; field: string } | null>(null);
  const [editVal, setEditVal] = useState("");
  const [lightbox, setLightbox] = useState<ImageGroup | null>(null);
  const [sort, setSort] = useState<{ col: string; dir: SortDir }>({ col: "", dir: null });
  const [page, setPage] = useState(1);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [bulkBusy, setBulkBusy] = useState<"verify" | "flag" | null>(null);
  const speciesInputRef = useRef<HTMLSelectElement>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setPage(1);
    setSelectedIds(new Set());
    try {
      const data = await getResults({ limit: 50000 });
      const items = Array.isArray(data) ? data : (data?.items ?? []);
      setRows(items as Row[]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    getStations().then((data: any) => {
      const list: any[] = Array.isArray(data) ? data : (data?.items ?? []);
      setStations(list.map((s) => s.station_id).filter(Boolean));
    }).catch(() => {});
  }, []);

  // Reset to page 1 whenever any filter changes — without this, changing a filter
  // while on page 2+ leaves the user past the last page and the table appears empty.
  useEffect(() => {
    setPage(1);
  }, [filter.species, filter.station, filter.day_night, filter.hideEmpty, filter.min_conf, filter.max_conf, filter.lowConfOnly, selectedTaxon]);

  const filteredRows = useMemo(() => {
    const minConf = filter.min_conf !== "" ? parseFloat(filter.min_conf) : null;
    const maxConf = filter.max_conf !== "" ? parseFloat(filter.max_conf) : null;

    return rows.filter((r) => {
      if (filter.species && String(r.detected_animal ?? "") !== filter.species) return false;
      if (filter.station && r.station_id !== filter.station) return false;
      if (filter.day_night && r.day_night !== filter.day_night) return false;
      if (filter.hideEmpty && String(r.detected_animal ?? "").toLowerCase() === "empty") return false;

      const conf = typeof r.detection_confidence === "number"
        ? r.detection_confidence as number
        : parseFloat(String(r.detection_confidence ?? ""));
      const confNum = isNaN(conf) ? 0 : conf;

      if (minConf !== null && !isNaN(minConf) && confNum < minConf) return false;
      if (maxConf !== null && !isNaN(maxConf) && confNum > maxConf) return false;
      if (filter.lowConfOnly && confNum >= 0.4) return false;

      // Taxon drill-down from ModelBreakdown clicks
      if (selectedTaxon) {
        const taxonLower = selectedTaxon.toLowerCase();
        let candidates: any[] = [];
        try {
          const tc = typeof r.top_candidates === "string" ? JSON.parse(r.top_candidates) : r.top_candidates;
          if (Array.isArray(tc)) candidates = tc;
        } catch {}
        const hasTaxon = candidates.some((c: any) =>
          (c.common_name ?? "").toLowerCase().includes(taxonLower)
          || (c.scientific_name ?? "").toLowerCase().includes(taxonLower)
          || (c.hierarchy ?? []).some((h: string) => h.toLowerCase() === taxonLower)
        );
        const inDetected = String(r.detected_animal ?? "").toLowerCase().includes(taxonLower);
        const inScientific = String(r.scientific_name ?? "").toLowerCase().includes(taxonLower);
        if (!hasTaxon && !inDetected && !inScientific) return false;
      }
      return true;
    });
  }, [rows, filter.species, filter.station, filter.day_night, filter.hideEmpty, filter.min_conf, filter.max_conf, filter.lowConfOnly, selectedTaxon]);

  const sorted = [...filteredRows].sort((a, b) => {
    if (!sort.col || !sort.dir) return 0;
    const va = a[sort.col] ?? "";
    const vb = b[sort.col] ?? "";
    const cmp = String(va).localeCompare(String(vb), undefined, { numeric: true });
    return sort.dir === "asc" ? cmp : -cmp;
  });

  // Gallery groups by image — must paginate on unique images, not detection rows,
  // otherwise a single image with 3 detections splits across 3 page slots and
  // the same image card appears on multiple pages with only partial detections.
  const galleryGroups = useMemo<ImageGroup[]>(() => {
    const map = new Map<string, Row[]>();
    for (const row of sorted) {
      const fn = String(row.filename ?? "");
      if (!map.has(fn)) map.set(fn, []);
      map.get(fn)!.push(row);
    }
    return Array.from(map.entries()).map(([filename, rows]) => ({ filename, rows }));
  }, [sorted]);

  const GALLERY_PAGE_SIZE = 48;
  const totalPages = viewMode === "gallery"
    ? Math.max(1, Math.ceil(galleryGroups.length / GALLERY_PAGE_SIZE))
    : Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const paginated = sorted.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const pagedGalleryGroups = galleryGroups.slice((page - 1) * GALLERY_PAGE_SIZE, page * GALLERY_PAGE_SIZE);

  // Count empty rows hidden by the hideEmpty toggle
  const hiddenEmptyCount = filter.hideEmpty
    ? rows.filter((r) => String(r.detected_animal ?? "").toLowerCase() === "empty").length
    : 0;

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

  const speciesList = useMemo(() => {
    const s = new Set<string>();
    for (const r of rows) {
      const v = String(r.detected_animal ?? "").trim();
      if (v && v.toLowerCase() !== "empty") s.add(v);
    }
    return Array.from(s).sort((a, b) => a.localeCompare(b));
  }, [rows]);

  const uniqueImages = new Set(rows.map((r) => String(r.filename))).size;
  const totalAnimals = rows.filter((r) => String(r.primary_label ?? r.detected_animal ?? "").toLowerCase() !== "empty").length;
  const dayImages = new Set(rows.filter((r) => r.day_night === "Day").map((r) => String(r.filename))).size;
  const nightImages = new Set(rows.filter((r) => r.day_night === "Night").map((r) => String(r.filename))).size;
  const uniqueSpecies = new Set(rows.filter((r) => String(r.detected_animal ?? "").toLowerCase() !== "empty").map((r) => r.detected_animal)).size;
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
              className={`px-3 py-1.5 text-sm font-medium transition cursor-pointer flex items-center gap-1 ${viewMode === "table" ? "bg-green-600 text-white" : "text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"}`}
            >
              <span className="material-symbols-outlined text-sm select-none">view_list</span> Table
            </button>
            <button
              onClick={() => setViewMode("gallery")}
              className={`px-3 py-1.5 text-sm font-medium transition cursor-pointer flex items-center gap-1 ${viewMode === "gallery" ? "bg-green-600 text-white" : "text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"}`}
            >
              <span className="material-symbols-outlined text-sm select-none">grid_view</span> Gallery
            </button>
          </div>
          <a href={exportExcel()} className="px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 shadow-sm transition">Export Excel</a>
          <a href={exportCsv()} className="px-4 py-2 bg-slate-600 dark:bg-slate-800 text-white text-sm rounded-lg hover:bg-slate-700 dark:hover:bg-slate-700 shadow-sm transition">Export CSV</a>
        </div>
      </div>

      {/* Summary stats */}
      {rows.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {[
            ["Total Images", uniqueImages, "text-slate-700 dark:text-slate-200"],
            ["Animals", totalAnimals, "text-green-700 dark:text-emerald-400"],
            ["Unique Species", uniqueSpecies, "text-indigo-700 dark:text-indigo-400"],
            ["Day / Night", `${dayImages} / ${nightImages}`, "text-amber-700 dark:text-amber-400"],
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
      <div className="bg-white dark:bg-slate-900/60 rounded-xl border border-slate-200 dark:border-slate-800/80 shadow-sm overflow-hidden">
        <div className="px-3 py-2.5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {/* Species */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Species</label>
            <select ref={speciesInputRef} value={filter.species}
              onChange={(e) => setFilter((f) => ({ ...f, species: e.target.value }))}
              className="border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300 w-full">
              <option value="">All species</option>
              {speciesList.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          {/* Station */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Station</label>
            {stations.length > 0 ? (
              <select value={filter.station}
                onChange={(e) => setFilter((f) => ({ ...f, station: e.target.value }))}
                className="border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300 w-full">
                <option value="">All stations</option>
                {stations.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            ) : (
              <input placeholder="Search…" value={filter.station}
                onChange={(e) => setFilter((f) => ({ ...f, station: e.target.value }))}
                className="border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300 w-full" />
            )}
          </div>

          {/* Time of day */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Time of day</label>
            <select value={filter.day_night}
              onChange={(e) => setFilter((f) => ({ ...f, day_night: e.target.value }))}
              className="border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300 w-full">
              <option value="">All</option>
              <option value="Day">Day</option>
              <option value="Night">Night</option>
            </select>
          </div>

          {/* Min confidence */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Min confidence</label>
            <input type="number" placeholder="0.0" min="0" max="1" step="0.05" value={filter.min_conf}
              onChange={(e) => setFilter((f) => ({ ...f, min_conf: e.target.value, lowConfOnly: false }))}
              className="border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300 w-full" />
          </div>

          {/* Max confidence */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Max confidence</label>
            <input type="number" placeholder="1.0" min="0" max="1" step="0.05" value={filter.max_conf}
              onChange={(e) => setFilter((f) => ({ ...f, max_conf: e.target.value, lowConfOnly: false }))}
              className="border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-300 w-full" />
          </div>

          {/* Toggles */}
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Show</label>
            <div className="flex flex-col gap-1 pt-0.5">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input type="checkbox" checked={filter.hideEmpty}
                  onChange={(e) => setFilter((f) => ({ ...f, hideEmpty: e.target.checked }))}
                  className="rounded border-slate-300 dark:border-slate-700 text-green-600 focus:ring-green-400 bg-white dark:bg-slate-950 cursor-pointer" />
                <span className="text-xs text-slate-600 dark:text-slate-400">Hide empty</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input type="checkbox" checked={filter.lowConfOnly}
                  onChange={(e) => setFilter((f) => ({ ...f, lowConfOnly: e.target.checked, min_conf: "", max_conf: "" }))}
                  className="rounded border-slate-300 dark:border-slate-700 text-amber-500 focus:ring-amber-400 bg-white dark:bg-slate-950 cursor-pointer" />
                <span className="text-xs text-amber-600 dark:text-amber-400 font-medium">Low conf only</span>
              </label>
            </div>
          </div>
        </div>

        {/* Active filter chips + clear */}
        {(filter.species || filter.day_night || filter.min_conf || filter.max_conf || (filter.station && stations.length === 0) || filter.station || filter.lowConfOnly || selectedTaxon) && (
          <div className="px-3 py-2 border-t border-slate-100 dark:border-slate-800 flex flex-wrap items-center gap-2">
            <span className="text-[10px] font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider shrink-0">Active:</span>
            {filter.species && <Chip label={`Species: ${filter.species}`} onRemove={() => setFilter((f) => ({ ...f, species: "" }))} />}
            {filter.station && <Chip label={`Station: ${filter.station}`} onRemove={() => setFilter((f) => ({ ...f, station: "" }))} />}
            {filter.day_night && <Chip label={filter.day_night} onRemove={() => setFilter((f) => ({ ...f, day_night: "" }))} />}
            {filter.min_conf && <Chip label={`Conf ≥ ${filter.min_conf}`} onRemove={() => setFilter((f) => ({ ...f, min_conf: "" }))} />}
            {filter.max_conf && <Chip label={`Conf ≤ ${filter.max_conf}`} onRemove={() => setFilter((f) => ({ ...f, max_conf: "" }))} />}
            {filter.lowConfOnly && <Chip label="Low conf < 40%" onRemove={() => setFilter((f) => ({ ...f, lowConfOnly: false }))} color="amber" />}
            {selectedTaxon && <Chip label={`Taxon: ${selectedTaxon}`} onRemove={() => setSelectedTaxon(null)} color="emerald" />}
            <button onClick={() => {
              setFilter({ species: "", day_night: "", min_conf: "", max_conf: "", station: "", hideEmpty: true, lowConfOnly: false });
              setSelectedTaxon(null);
            }} className="ml-auto text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 cursor-pointer px-2 py-0.5 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition">
              Clear all
            </button>
          </div>
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

      {/* Hidden-empty notice */}
      {!loading && hiddenEmptyCount > 0 && sorted.length > 0 && (
        <div className="flex items-center gap-2.5 px-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 text-sm text-slate-500 dark:text-slate-400">
          <span className="material-symbols-outlined text-base select-none text-slate-400 shrink-0">hide_image</span>
          <span className="flex-1">
            <span className="font-semibold text-slate-600 dark:text-slate-300">{hiddenEmptyCount}</span> image{hiddenEmptyCount !== 1 ? "s" : ""} with no animal detected {hiddenEmptyCount !== 1 ? "are" : "is"} hidden.
          </span>
          <button
            onClick={() => setFilter((f) => ({ ...f, hideEmpty: false }))}
            className="text-xs font-semibold text-green-600 dark:text-green-400 hover:underline cursor-pointer shrink-0"
          >Show them</button>
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
        <div className="text-center py-16 text-slate-400 dark:text-slate-500 space-y-3">
          <span className="material-symbols-outlined text-5xl select-none opacity-40">photo_camera</span>
          {rows.length === 0 ? (
            <>
              <p className="font-semibold">No results yet</p>
              <p className="text-sm">Process images in the Upload tab first.</p>
            </>
          ) : (
            <>
              <p className="font-semibold text-slate-600 dark:text-slate-400">No images match the current filters</p>
              <p className="text-sm">{rows.length} detection{rows.length !== 1 ? "s" : ""} in database
                {hiddenEmptyCount > 0 && ` · ${hiddenEmptyCount} hidden (no detection)`}</p>
              <button
                onClick={() => {
                  setFilter({ species: "", day_night: "", min_conf: "", max_conf: "", station: "", hideEmpty: false, lowConfOnly: false });
                  setSelectedTaxon(null);
                }}
                className="px-4 py-2 bg-green-600 text-white text-sm font-semibold rounded-lg hover:bg-green-700 cursor-pointer transition"
              >Clear all filters</button>
            </>
          )}
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
                            src={storedThumbUrl(filename, 240)}
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
                                    {getCandidates(row).slice(0, 5).map((c, ci) => (
                                      <button
                                        key={ci}
                                        onClick={() => setEditVal(c.label)}
                                        className={`w-full flex items-center justify-between px-2 py-1.5 text-xs text-left cursor-pointer border-b last:border-b-0 border-slate-100 dark:border-slate-800 transition ${
                                          editVal === c.label
                                            ? "bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400"
                                            : "bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300"
                                        }`}
                                      >
                                        <div className="flex-1 min-w-0 mr-1">
                                          <span className="block truncate font-medium">{c.label}</span>
                                          {c.scientific && <span className="block truncate text-[9px] italic text-slate-400 dark:text-slate-500 leading-tight">{c.scientific}</span>}
                                        </div>
                                        <div className="flex items-center gap-1 shrink-0">
                                          <span className="text-[9px] font-mono text-slate-400">{Math.round(c.conf * 100)}%</span>
                                          <span className="text-[8px] font-bold px-1 rounded bg-teal-100 dark:bg-teal-950/50 text-teal-600 dark:text-teal-400">SN</span>
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
                                  <button onClick={() => saveEdit(id)} className="text-green-600 dark:text-emerald-500 font-bold shrink-0 cursor-pointer flex items-center justify-center" title="Save">
                                    <span className="material-symbols-outlined text-sm leading-none select-none">check</span>
                                  </button>
                                  <button onClick={() => setEditing(null)} className="text-slate-400 shrink-0 cursor-pointer flex items-center justify-center" title="Cancel">
                                    <span className="material-symbols-outlined text-sm leading-none select-none">close</span>
                                  </button>
                                </div>
                              </div>
                            ) : isEditing ? (
                              <div className="flex gap-1 items-center">
                                <input autoFocus value={editVal}
                                  onChange={(e) => setEditVal(e.target.value)}
                                  onKeyDown={(e) => e.key === "Enter" && saveEdit(id)}
                                  className="border border-green-400 dark:border-emerald-500 rounded px-2 py-0.5 text-sm w-full focus:outline-none bg-white dark:bg-slate-900 text-slate-800 dark:text-white" />
                                <button onClick={() => saveEdit(id)} className="text-green-600 dark:text-emerald-500 font-bold shrink-0 cursor-pointer flex items-center justify-center" title="Save">
                                  <span className="material-symbols-outlined text-sm leading-none select-none">check</span>
                                </button>
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
                                speciesnetConf={typeof row.speciesnet_confidence === "number" ? row.speciesnet_confidence as number : undefined}
                                detected={String(row.detected_animal ?? "")}
                                modelBreakdown={row.model_breakdown}
                                topCandidates={row.top_candidates}
                                compact
                              />
                            ) : key === "detected_animal" ? (
                              <div className="flex items-start gap-1.5">
                                {isLowConf && (
                                  <span className="material-symbols-outlined text-amber-400 text-sm shrink-0 mt-0.5" title="Low confidence">warning</span>
                                )}
                                <div className="min-w-0">
                                  <span
                                    className="block truncate font-semibold cursor-pointer group-hover:text-green-700 dark:group-hover:text-emerald-400 hover:underline underline-offset-2"
                                    title={`${String(val ?? "")} — click to correct`}
                                    onClick={() => { setEditing({ id, field: key }); setEditVal(String(val ?? "")); }}
                                  >
                                    {String(val ?? "")}
                                  </span>
                                  {!!row.scientific_name && (
                                    <span className="block truncate text-[10px] italic text-slate-400 dark:text-slate-500 leading-tight">
                                      {String(row.scientific_name)}
                                    </span>
                                  )}
                                </div>
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
