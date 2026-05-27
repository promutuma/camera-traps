import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { getResults, updateResult, exportExcel, exportCsv, storedImageUrl } from "../api/client";

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
  if (v >= 0.7) return "bg-green-500";
  if (v >= 0.4) return "bg-amber-400";
  return "bg-red-400";
}

function DayNightBadge({ value }: { value: unknown }) {
  const v = String(value ?? "");
  if (!v) return <span className="text-slate-300">—</span>;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
      v === "Day" ? "bg-amber-50 text-amber-700" : "bg-indigo-50 text-indigo-700"
    }`}>
      {v === "Day" ? "☀" : "🌙"} {v}
    </span>
  );
}

function ConfBar({ value }: { value: unknown }) {
  const v = typeof value === "number" ? value : parseFloat(String(value ?? "0"));
  const pct = isNaN(v) ? 0 : Math.round(v * 100);
  return (
    <div className="flex items-center gap-2 min-w-[80px]">
      <div className="flex-1 bg-slate-100 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full rounded-full ${confColor(v)}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-slate-500 w-8 text-right">{pct}%</span>
    </div>
  );
}

// ── Model breakdown shared component ─────────────────────────────────────────

function ModelPill({ name, conf }: { name: string; conf?: number }) {
  const base =
    name === "MDv5a" || name === "MDv1000"
      ? "bg-blue-100 text-blue-700 border-blue-200"
      : name === "BioClip"
      ? "bg-violet-100 text-violet-700 border-violet-200"
      : name === "SpeciesNet"
      ? "bg-teal-100 text-teal-700 border-teal-200"
      : "bg-slate-100 text-slate-600 border-slate-200";
  return (
    <span className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9px] font-bold border ${base}`}>
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
      ? "bg-emerald-100 text-emerald-700 border-emerald-200"
      : level === "Medium"
      ? "bg-amber-100 text-amber-700 border-amber-200"
      : "bg-red-100 text-red-700 border-red-200";
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold border ${styles}`}>
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
}: {
  method: string;
  bioclipConf?: number;
  speciesnetConf?: number;
  agreement?: string | null;
  detected?: string;
}) {
  const models = method ? method.split(" + ").filter(Boolean) : [];
  const detectors = models.filter((m) => m.startsWith("MDv") || m === "MegaDetector");
  const isAnimal =
    detected && detected !== "Empty" && detected !== "Unidentified" &&
    detected !== "Person" && detected !== "Vehicle" && detected !== "Error";

  if (!detectors.length && !isAnimal) return null;

  return (
    <div className="flex flex-wrap items-center gap-1">
      {detectors.map((m) => <ModelPill key={m} name={m} />)}
      {isAnimal && (
        <>
          {(bioclipConf ?? 0) > 0 && <ModelPill name="BioClip" conf={bioclipConf} />}
          {(speciesnetConf ?? 0) > 0 && <ModelPill name="SpeciesNet" conf={speciesnetConf} />}
          <AgreementBadge level={agreement} />
        </>
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
}: {
  group: ImageGroup;
  groups: ImageGroup[];
  onClose: () => void;
  onNavigate: (g: ImageGroup) => void;
  onSave: (id: number, field: string, value: string) => Promise<void>;
}) {
  const { filename, rows } = group;
  const imgUrl = storedImageUrl(filename);
  const primary = rows[0];

  const [imgNatural, setImgNatural] = useState<{ w: number; h: number } | null>(null);
  const [editField, setEditField] = useState<string | null>(null);
  const [editVal, setEditVal] = useState("");
  const [saving, setSaving] = useState(false);
  const [detEdit, setDetEdit] = useState<{ idx: number; val: string } | null>(null);

  const idx = groups.findIndex((g) => g.filename === filename);
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

  useEffect(() => { setImgNatural(null); setEditField(null); setDetEdit(null); }, [imgUrl]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative bg-white rounded-2xl shadow-2xl max-w-5xl w-full mx-4 overflow-hidden flex flex-col md:flex-row max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Image panel ── */}
        <div className="md:w-[60%] bg-slate-950 flex items-center justify-center relative min-h-64">
          <img
            src={imgUrl}
            alt={filename}
            className="max-h-[85vh] w-full object-contain"
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
            >
              {rows.map((row, i) => {
                const bbox = parseBbox(row.bbox);
                if (!bbox) return null;
                const color = BOX_COLORS[i % BOX_COLORS.length];
                const sw = Math.max(2, imgNatural.w / 400);
                const fs = Math.max(14, imgNatural.w / 55);
                const conf = typeof row.detection_confidence === "number"
                  ? ` (${(row.detection_confidence as number).toFixed(2)})`
                  : "";
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

          {hasPrev && (
            <button
              onClick={() => onNavigate(groups[idx - 1])}
              className="absolute left-2 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/80 text-white rounded-full w-10 h-10 flex items-center justify-center text-lg transition"
            >‹</button>
          )}
          {hasNext && (
            <button
              onClick={() => onNavigate(groups[idx + 1])}
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-black/50 hover:bg-black/80 text-white rounded-full w-10 h-10 flex items-center justify-center text-lg transition"
            >›</button>
          )}

          <div className="absolute bottom-2 right-3 text-xs text-white/60 font-mono">
            {idx + 1} / {groups.length}
          </div>
        </div>

        {/* ── Details panel ── */}
        <div className="md:w-[40%] p-5 flex flex-col gap-4 overflow-y-auto">
          <div className="flex items-start justify-between gap-2">
            <p className="font-semibold text-slate-800 text-sm break-all leading-snug">{filename}</p>
            <button onClick={onClose} className="shrink-0 text-slate-400 hover:text-slate-700 text-xl leading-none mt-0.5">✕</button>
          </div>

          {/* Detection list */}
          <div className="space-y-2.5">
            <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide">
              Detections ({rows.length})
            </p>
            {rows.map((row, i) => {
              const color = BOX_COLORS[i % BOX_COLORS.length];
              const conf = typeof row.detection_confidence === "number"
                ? Math.round((row.detection_confidence as number) * 100)
                : null;
              const isEditingDet = detEdit?.idx === i;
              return (
                <div key={i} className="rounded-lg border border-slate-100 bg-slate-50/60 p-2.5 space-y-1.5">
                  {/* Species row */}
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: color }} />
                    {isEditingDet ? (
                      <>
                        <input
                          autoFocus
                          value={detEdit.val}
                          onChange={(e) => setDetEdit({ idx: i, val: e.target.value })}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") commitDetEdit();
                            if (e.key === "Escape") setDetEdit(null);
                          }}
                          className="flex-1 border border-green-400 rounded px-2 py-0.5 text-xs focus:outline-none focus:ring-1 focus:ring-green-400"
                        />
                        <button onClick={commitDetEdit} disabled={saving}
                          className="px-2 py-0.5 bg-green-600 text-white text-[10px] rounded hover:bg-green-700 disabled:opacity-50 shrink-0">
                          {saving ? "…" : "✓"}
                        </button>
                        <button onClick={() => setDetEdit(null)}
                          className="text-slate-300 hover:text-slate-600 shrink-0 text-sm">✕</button>
                      </>
                    ) : (
                      <>
                        <span
                          className="font-semibold text-slate-800 text-sm flex-1 truncate cursor-pointer hover:text-green-700 hover:underline underline-offset-2"
                          title="Click to edit species"
                          onClick={() => setDetEdit({ idx: i, val: String(row.detected_animal ?? "") })}
                        >
                          {String(row.detected_animal ?? "Unknown")}
                        </span>
                        {conf !== null && (
                          <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded-full text-white shrink-0 ${confColor(conf / 100)}`}>
                            {conf}%
                          </span>
                        )}
                      </>
                    )}
                  </div>

                  {/* Model breakdown */}
                  {!isEditingDet && (
                    <ModelBreakdown
                      method={String(row.detection_method ?? "")}
                      bioclipConf={typeof row.bioclip_confidence === "number" ? row.bioclip_confidence as number : undefined}
                      speciesnetConf={typeof row.speciesnet_confidence === "number" ? row.speciesnet_confidence as number : undefined}
                      agreement={row.agreement as string | null}
                      detected={String(row.detected_animal ?? "")}
                    />
                  )}
                </div>
              );
            })}
          </div>

          {/* Shared image fields */}
          <div className="space-y-3 flex-1">
            {([
              ["Station",      "station_id",   true ],
              ["Day / Night",  "day_night",     false],
              ["Capture date", "capture_date",  false],
              ["Temperature",  "temperature",   false],
              ["Notes",        "user_notes",    true ],
            ] as [string, string, boolean][]).map(([label, field, editable]) => {
              const isEditing = editField === field;
              const val = primary[field];
              return (
                <div key={field}>
                  <p className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide mb-0.5">{label}</p>
                  {isEditing ? (
                    <div className="flex gap-1.5">
                      {field === "user_notes" ? (
                        <textarea autoFocus value={editVal} onChange={(e) => setEditVal(e.target.value)} rows={3}
                          className="flex-1 border border-green-400 rounded px-2 py-1 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-green-400" />
                      ) : (
                        <input autoFocus value={editVal} onChange={(e) => setEditVal(e.target.value)}
                          onKeyDown={(e) => e.key === "Enter" && commitEdit()}
                          className="flex-1 border border-green-400 rounded px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-green-400" />
                      )}
                      <div className="flex flex-col gap-1">
                        <button onClick={commitEdit} disabled={saving}
                          className="px-2 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700 disabled:opacity-50">
                          {saving ? "…" : "✓"}
                        </button>
                        <button onClick={() => setEditField(null)}
                          className="px-2 py-1 bg-slate-100 text-slate-600 text-xs rounded hover:bg-slate-200">✕</button>
                      </div>
                    </div>
                  ) : (
                    <div
                      className={`text-sm text-slate-800 font-medium ${editable ? "cursor-pointer hover:text-green-700 hover:underline underline-offset-2" : ""}`}
                      onClick={() => editable && startEdit(field)}
                    >
                      {field === "day_night" ? <DayNightBadge value={val} /> : String(val ?? "—")}
                      {editable && !val && <span className="text-slate-300 font-normal italic">click to add…</span>}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <p className="text-xs text-slate-300 text-center pt-2 border-t border-slate-100">
            ← → navigate &nbsp;·&nbsp; Esc close
          </p>
        </div>
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
  const [editing, setEditing] = useState<{ id: number; field: string } | null>(null);
  const [editVal, setEditVal] = useState("");
  const [lightbox, setLightbox] = useState<ImageGroup | null>(null);
  const [sort, setSort] = useState<{ col: string; dir: SortDir }>({ col: "", dir: null });
  const [page, setPage] = useState(1);
  const speciesInputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setPage(1);
    try {
      const params: Record<string, string> = {};
      if (filter.species)   params.species   = filter.species;
      if (filter.day_night) params.day_night = filter.day_night;
      if (filter.min_conf)  params.min_conf  = filter.min_conf;
      if (filter.max_conf)  params.max_conf  = filter.max_conf;
      if (filter.station)   params.station   = filter.station;
      setRows(await getResults(params));
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => { load(); }, []);

  const sorted = [...rows].sort((a, b) => {
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

  const totalAnimals = rows.filter((r) => String(r.primary_label ?? r.detected_animal ?? "").toLowerCase() !== "empty").length;
  const dayCount = rows.filter((r) => r.day_night === "Day").length;
  const nightCount = rows.filter((r) => r.day_night === "Night").length;
  const uniqueSpecies = new Set(rows.map((r) => r.detected_animal)).size;

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
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-slate-800">Review Results</h1>
        <div className="flex gap-2 flex-wrap">
          <div className="flex rounded-lg border border-slate-200 overflow-hidden bg-white">
            <button
              onClick={() => setViewMode("table")}
              className={`px-3 py-1.5 text-sm font-medium transition ${viewMode === "table" ? "bg-green-600 text-white" : "text-slate-500 hover:bg-slate-50"}`}
            >☰ Table</button>
            <button
              onClick={() => setViewMode("gallery")}
              className={`px-3 py-1.5 text-sm font-medium transition ${viewMode === "gallery" ? "bg-green-600 text-white" : "text-slate-500 hover:bg-slate-50"}`}
            >⊞ Gallery</button>
          </div>
          <a href={exportExcel()} className="px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700">Export Excel</a>
          <a href={exportCsv()} className="px-4 py-2 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-700">Export CSV</a>
        </div>
      </div>

      {/* Summary stats */}
      {rows.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            ["Total Images", rows.length, "text-slate-700"],
            ["Animals", totalAnimals, "text-green-700"],
            ["Unique Species", uniqueSpecies, "text-indigo-700"],
            ["Day / Night", `${dayCount} / ${nightCount}`, "text-amber-700"],
          ].map(([label, val, cls]) => (
            <div key={String(label)} className="bg-white rounded-xl border border-slate-200 px-4 py-3">
              <p className="text-xs text-slate-400 font-medium uppercase tracking-wide">{label}</p>
              <p className={`text-xl font-bold mt-0.5 ${cls}`}>{String(val)}</p>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="bg-white rounded-xl border border-slate-200 p-3 flex flex-wrap gap-2 items-end">
        <input ref={speciesInputRef} placeholder="Species…" value={filter.species}
          onChange={(e) => setFilter((f) => ({ ...f, species: e.target.value }))}
          onKeyDown={(e) => e.key === "Enter" && load()}
          className="border border-slate-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400" />
        <input placeholder="Station…" value={filter.station}
          onChange={(e) => setFilter((f) => ({ ...f, station: e.target.value }))}
          onKeyDown={(e) => e.key === "Enter" && load()}
          className="border border-slate-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400" />
        <select value={filter.day_night}
          onChange={(e) => setFilter((f) => ({ ...f, day_night: e.target.value }))}
          className="border border-slate-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-green-400">
          <option value="">All times</option>
          <option value="Day">☀ Day</option>
          <option value="Night">🌙 Night</option>
        </select>
        <div className="flex items-center gap-1">
          <input placeholder="Min conf" value={filter.min_conf}
            onChange={(e) => setFilter((f) => ({ ...f, min_conf: e.target.value }))}
            className="border border-slate-300 rounded-lg px-3 py-1.5 text-sm w-24 focus:outline-none focus:ring-1 focus:ring-green-400" />
          <span className="text-slate-400 text-sm">–</span>
          <input placeholder="Max conf" value={filter.max_conf}
            onChange={(e) => setFilter((f) => ({ ...f, max_conf: e.target.value }))}
            className="border border-slate-300 rounded-lg px-3 py-1.5 text-sm w-24 focus:outline-none focus:ring-1 focus:ring-green-400" />
        </div>
        <button onClick={load} className="px-4 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700">Apply</button>
        {(filter.species || filter.day_night || filter.min_conf || filter.max_conf || filter.station) && (
          <button
            onClick={() => { setFilter({ species: "", day_night: "", min_conf: "", max_conf: "", station: "" }); setTimeout(load, 0); }}
            className="px-3 py-1.5 text-slate-500 text-sm rounded-lg hover:bg-slate-100"
          >Clear</button>
        )}
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex items-center justify-center py-16 text-slate-400 gap-2">
          <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Loading…
        </div>
      ) : sorted.length === 0 ? (
        <div className="text-center py-16 text-slate-400 space-y-2">
          <div className="text-4xl">📷</div>
          <p className="font-medium">No results yet</p>
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
          <div className="bg-white rounded-xl border border-slate-200 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>
                  <th className="text-left px-3 py-3 font-medium text-slate-500 w-20">Image</th>
                  {COLS.map(({ key, label, sortable }) => (
                    <th
                      key={key}
                      className={`text-left px-3 py-3 font-medium text-slate-500 whitespace-nowrap ${sortable ? "cursor-pointer select-none hover:text-slate-800" : ""}`}
                      onClick={() => sortable && toggleSort(key)}
                    >
                      {label}{sortable && sortIcon(key)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {paginated.map((row, i) => {
                  const id = Number(row.detection_id ?? row.id ?? i);
                  const filename = String(row.filename ?? "");
                  return (
                    <tr key={id} className="hover:bg-slate-50 group">
                      {/* Thumbnail */}
                      <td className="px-3 py-2">
                        <button
                          onClick={() => {
                            const g = galleryGroups.find((g) => g.filename === filename);
                            setLightbox(g ?? { filename, rows: [row] });
                          }}
                          className="block w-16 h-12 rounded-lg overflow-hidden border border-slate-200 bg-slate-100 hover:ring-2 hover:ring-green-500 transition shrink-0"
                          title="View image"
                        >
                          <img
                            src={storedImageUrl(filename)}
                            alt={filename}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              const el = e.currentTarget as HTMLImageElement;
                              el.style.display = "none";
                              el.parentElement!.innerHTML = '<span class="text-[10px] text-slate-400 flex items-center justify-center h-full w-full">No img</span>';
                            }}
                          />
                        </button>
                      </td>

                      {COLS.map(({ key }) => {
                        const editable = EDITABLE.has(key);
                        const isEditing = editing?.id === id && editing?.field === key;
                        const val = row[key];

                        return (
                          <td key={key} className="px-3 py-2 max-w-[200px]">
                            {isEditing ? (
                              <div className="flex gap-1 items-center">
                                <input autoFocus value={editVal}
                                  onChange={(e) => setEditVal(e.target.value)}
                                  onKeyDown={(e) => e.key === "Enter" && saveEdit(id)}
                                  className="border border-green-400 rounded px-2 py-0.5 text-sm w-full focus:outline-none" />
                                <button onClick={() => saveEdit(id)} className="text-green-600 font-bold shrink-0">✓</button>
                                <button onClick={() => setEditing(null)} className="text-slate-300 shrink-0">✕</button>
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
                              />
                            ) : (
                              <span
                                className={`block truncate ${editable ? "cursor-pointer group-hover:text-green-700 hover:underline underline-offset-2" : "text-slate-700"}`}
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
            <div className="px-4 py-2 text-xs text-slate-400 border-t border-slate-100 flex items-center justify-between">
              <span>{sorted.length} record(s) · Click thumbnail to view · Click species/station/notes to edit</span>
              <span className="text-slate-300">Click column headers to sort</span>
            </div>
          </div>
          <Pagination page={page} totalPages={totalPages} total={sorted.length} onPage={setPage} />
        </>
      )}
    </div>
  );
}
