import { useEffect, useState } from "react";
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

function confColor(v: number) {
  if (v >= 0.7) return "bg-green-500";
  if (v >= 0.4) return "bg-amber-400";
  return "bg-red-400";
}

function ConfBar({ value }: { value: unknown }) {
  const v = typeof value === "number" ? value : parseFloat(String(value ?? "0"));
  const pct = isNaN(v) ? 0 : Math.round(v * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-slate-100 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full rounded-full ${confColor(v)}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-slate-400 w-8 text-right shrink-0">{pct}%</span>
    </div>
  );
}

function DayNightBadge({ value }: { value: unknown }) {
  const v = String(value ?? "");
  if (!v) return null;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
      v === "Day" ? "bg-amber-50 text-amber-700" : "bg-indigo-50 text-indigo-700"
    }`}>
      {v === "Day" ? "☀" : "🌙"} {v}
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

// ── Queue item card ───────────────────────────────────────────────────────────

function QueueCard({
  row,
  reviewerId,
  onAction,
}: {
  row: Row;
  reviewerId: string;
  onAction: () => void;
}) {
  const id = Number(row.id ?? row.image_id ?? 0);
  const filename = String(row.filename ?? "");
  const species = String(row.detected_animal ?? "Unknown");
  const conf = typeof row.detection_confidence === "number"
    ? (row.detection_confidence as number)
    : parseFloat(String(row.detection_confidence ?? "0"));

  const [mode, setMode] = useState<"idle" | "correct" | "confirm-note" | "flag-note">("idle");
  const [correctLabel, setCorrectLabel] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [imgError, setImgError] = useState(false);
  const [imgNatural, setImgNatural] = useState<{ w: number; h: number } | null>(null);
  const bbox = parseBbox(row.bbox);

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); onAction(); }
    finally { setBusy(false); setMode("idle"); setNotes(""); setCorrectLabel(""); }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
      {/* Image — full width top */}
      <div className="relative bg-slate-900 aspect-video overflow-hidden">
        {imgError ? (
          <div className="w-full h-full flex items-center justify-center">
            <span className="material-symbols-outlined text-slate-600 text-4xl select-none">broken_image</span>
          </div>
        ) : (
          <>
            <img
              src={storedImageUrl(filename)}
              alt={filename}
              className="w-full h-full object-cover"
              onLoad={(e) => {
                const img = e.currentTarget as HTMLImageElement;
                setImgNatural({ w: img.naturalWidth, h: img.naturalHeight });
              }}
              onError={() => setImgError(true)}
            />
            {bbox && imgNatural && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox={`0 0 ${imgNatural.w} ${imgNatural.h}`}
                preserveAspectRatio="xMidYMid slice"
              >
                <rect
                  x={bbox[0] * imgNatural.w} y={bbox[1] * imgNatural.h}
                  width={bbox[2] * imgNatural.w} height={bbox[3] * imgNatural.h}
                  fill="none" stroke="#22c55e"
                  strokeWidth={Math.max(3, imgNatural.w / 300)}
                />
                <text
                  x={bbox[0] * imgNatural.w + 4} y={bbox[1] * imgNatural.h - 6}
                  fill="#22c55e" fontWeight="bold"
                  fontSize={Math.max(16, imgNatural.w / 50)}
                  style={{ filter: "drop-shadow(0 1px 3px #000)" }}
                >
                  {species}
                </text>
              </svg>
            )}
          </>
        )}
        {/* Overlaid badges */}
        <div className="absolute top-2 left-2">
          <DayNightBadge value={row.day_night} />
        </div>
        {row.station_id != null && String(row.station_id) && (
          <div className="absolute top-2 right-2 bg-black/50 text-white text-[10px] font-semibold px-2 py-0.5 rounded-full">
            {String(row.station_id)}
          </div>
        )}
      </div>

      {/* Info */}
      <div className="p-3 space-y-2 flex-1">
        <div>
          <p className="font-semibold text-slate-800 text-sm truncate">{species}</p>
          <p className="text-[11px] text-slate-400 truncate mt-0.5">{filename}</p>
        </div>
        <ConfBar value={conf} />
        {row.capture_date != null && String(row.capture_date) && (
          <p className="text-[11px] text-slate-400">{String(row.capture_date)} {String(row.capture_time ?? "")}</p>
        )}
      </div>

      {/* Action area */}
      {mode === "idle" && (
        <div className="flex border-t border-slate-100">
          <button
            onClick={() => setMode("confirm-note")}
            disabled={busy}
            className="flex-1 flex items-center justify-center gap-1 py-2.5 text-xs font-semibold text-green-700 hover:bg-green-50 transition disabled:opacity-50 border-r border-slate-100"
          >
            <span className="material-symbols-outlined text-sm select-none">check_circle</span>
            Confirm
          </button>
          <button
            onClick={() => setMode("correct")}
            disabled={busy}
            className="flex-1 flex items-center justify-center gap-1 py-2.5 text-xs font-semibold text-blue-700 hover:bg-blue-50 transition disabled:opacity-50 border-r border-slate-100"
          >
            <span className="material-symbols-outlined text-sm select-none">edit</span>
            Correct
          </button>
          <button
            onClick={() => setMode("flag-note")}
            disabled={busy}
            className="flex-1 flex items-center justify-center gap-1 py-2.5 text-xs font-semibold text-red-700 hover:bg-red-50 transition disabled:opacity-50"
          >
            <span className="material-symbols-outlined text-sm select-none">flag</span>
            Flag
          </button>
        </div>
      )}

      {mode === "correct" && (
        <div className="border-t border-slate-100 bg-blue-50/50 px-3 py-3 space-y-2">
          <p className="text-xs font-semibold text-blue-700">Correct label</p>
          <input
            autoFocus
            placeholder={`Current: ${species}`}
            value={correctLabel}
            onChange={(e) => setCorrectLabel(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && setMode("idle")}
            className="w-full border border-blue-300 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <input
            placeholder="Notes (optional)"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") run(() => correctDetection(id, { reviewer_id: reviewerId, corrected_label: correctLabel, notes }));
              if (e.key === "Escape") setMode("idle");
            }}
            className="w-full border border-blue-200 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-blue-300"
          />
          <div className="flex gap-2">
            <button
              disabled={busy || !correctLabel.trim()}
              onClick={() => run(() => correctDetection(id, { reviewer_id: reviewerId, corrected_label: correctLabel, notes }))}
              className="flex-1 py-1.5 bg-blue-600 text-white text-xs font-semibold rounded-lg hover:bg-blue-700 disabled:opacity-50 transition"
            >
              {busy ? "Saving…" : "Save"}
            </button>
            <button onClick={() => setMode("idle")} className="px-3 py-1.5 text-slate-500 text-xs rounded-lg hover:bg-slate-100">✕</button>
          </div>
        </div>
      )}

      {mode === "confirm-note" && (
        <div className="border-t border-slate-100 bg-green-50/50 px-3 py-3 space-y-2">
          <p className="text-xs font-semibold text-green-700">Confirm as <em>{species}</em></p>
          <input
            autoFocus
            placeholder="Notes (optional)"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") run(() => confirmDetection(id, { reviewer_id: reviewerId, notes }));
              if (e.key === "Escape") setMode("idle");
            }}
            className="w-full border border-green-300 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-green-400"
          />
          <div className="flex gap-2">
            <button
              disabled={busy}
              onClick={() => run(() => confirmDetection(id, { reviewer_id: reviewerId, notes }))}
              className="flex-1 py-1.5 bg-green-600 text-white text-xs font-semibold rounded-lg hover:bg-green-700 disabled:opacity-50 transition"
            >
              {busy ? "Saving…" : "Confirm"}
            </button>
            <button onClick={() => setMode("idle")} className="px-3 py-1.5 text-slate-500 text-xs rounded-lg hover:bg-slate-100">✕</button>
          </div>
        </div>
      )}

      {mode === "flag-note" && (
        <div className="border-t border-slate-100 bg-red-50/50 px-3 py-3 space-y-2">
          <p className="text-xs font-semibold text-red-700">Flag for review</p>
          <input
            autoFocus
            placeholder="Reason (optional)"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") run(() => flagDetection(id, { reviewer_id: reviewerId, notes }));
              if (e.key === "Escape") setMode("idle");
            }}
            className="w-full border border-red-300 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-red-400"
          />
          <div className="flex gap-2">
            <button
              disabled={busy}
              onClick={() => run(() => flagDetection(id, { reviewer_id: reviewerId, notes }))}
              className="flex-1 py-1.5 bg-red-600 text-white text-xs font-semibold rounded-lg hover:bg-red-700 disabled:opacity-50 transition"
            >
              {busy ? "Flagging…" : "Flag"}
            </button>
            <button onClick={() => setMode("idle")} className="px-3 py-1.5 text-slate-500 text-xs rounded-lg hover:bg-slate-100">✕</button>
          </div>
        </div>
      )}
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

  const load = async () => {
    setLoading(true);
    const [q, l, a] = await Promise.all([getReviewQueue(), getReviewLog(), getPrivacyAudit()]);
    setQueue(Array.isArray(q) ? q : []);
    setLog(Array.isArray(l) ? l : []);
    setAudit(Array.isArray(a) ? a : []);
    setLoading(false);
  };
  useEffect(() => { load(); }, []);

  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: "queue",   label: "Review Queue",   count: queue.length },
    { id: "log",     label: "Correction Log", count: log.length },
    { id: "privacy", label: "Privacy Audit",  count: audit.length },
  ];

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Review Queue</h1>
        <p className="text-slate-500 text-sm mt-1">
          Detections with confidence below <span className="font-mono font-semibold text-slate-700">{threshold}</span> need manual review.
          Reviewing as <span className="font-semibold text-slate-700">{reviewerId}</span>.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-200 pb-0">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center px-4 py-2 text-sm font-semibold rounded-t-lg border border-b-0 transition -mb-px ${
              tab === t.id
                ? "bg-white border-slate-200 text-slate-800 shadow-sm"
                : "bg-transparent border-transparent text-slate-500 hover:text-slate-700"
            }`}
          >
            {t.label}
            {t.count !== undefined && t.count > 0 && <CountChip n={t.count} />}
          </button>
        ))}
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
      ) : (
        <>
          {tab === "queue" && (
            queue.length === 0
              ? <EmptyState icon="✅" title="Queue is clear" sub="All detections are above the confidence threshold." />
              : (
                <div className="space-y-3">
                  <p className="text-xs text-slate-400 font-medium">{queue.length} item(s) pending review</p>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {queue.map((row, i) => (
                      <QueueCard
                        key={Number(row.id ?? row.image_id ?? i)}
                        row={row}
                        reviewerId={reviewerId}
                        onAction={load}
                      />
                    ))}
                  </div>
                </div>
              )
          )}

          {tab === "log" && (
            log.length === 0
              ? <EmptyState icon="📋" title="No corrections yet" sub="Reviewed detections will appear here." />
              : <DataTable rows={log} cols={LOG_COLS} />
          )}

          {tab === "privacy" && (
            audit.length === 0
              ? <EmptyState icon="🔒" title="No privacy audit entries" sub="Privacy scrub events will be logged here." />
              : <DataTable rows={audit} cols={AUDIT_COLS} />
          )}
        </>
      )}
    </div>
  );
}
