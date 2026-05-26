import { useEffect, useState } from "react";
import { getReviewQueue, confirmDetection, correctDetection, flagDetection, getReviewLog, getPrivacyAudit } from "../api/client";
import { useConfigStore } from "../store/configStore";

type Row = Record<string, unknown>;

export default function ReviewQueue() {
  const config = useConfigStore((s) => s.config);
  const reviewerId = config?.reviewer_id ?? "anonymous";
  const [queue, setQueue] = useState<Row[]>([]);
  const [log, setLog] = useState<Row[]>([]);
  const [audit, setAudit] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [correcting, setCorrecting] = useState<number | null>(null);
  const [correctLabel, setCorrectLabel] = useState("");
  const [tab, setTab] = useState<"queue" | "log" | "privacy">("queue");

  const load = async () => {
    setLoading(true);
    const [q, l, a] = await Promise.all([getReviewQueue(), getReviewLog(), getPrivacyAudit()]);
    setQueue(Array.isArray(q) ? q : []);
    setLog(Array.isArray(l) ? l : []);
    setAudit(Array.isArray(a) ? a : []);
    setLoading(false);
  };
  useEffect(() => { load(); }, []);

  const action = async (fn: () => Promise<unknown>) => { await fn(); load(); };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-800">Review Queue</h1>
      <p className="text-slate-500 text-sm">
        Detections below the confidence threshold ({config?.review_confidence_threshold ?? 0.9}) for manual review.
      </p>

      <div className="flex gap-2">
        {(["queue", "log", "privacy"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium capitalize ${tab === t ? "bg-green-600 text-white" : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"}`}>
            {t === "privacy" ? "Privacy Audit" : t === "log" ? "Correction Log" : "Review Queue"}
          </button>
        ))}
      </div>

      {loading ? <div className="text-center py-12 text-slate-400">Loading…</div> : (
        <>
          {tab === "queue" && (
            queue.length === 0 ? <div className="text-center py-12 text-slate-400">No items in queue.</div> : (
              <div className="space-y-3">
                {queue.map((row, i) => {
                  const id = Number(row.id ?? row.image_id ?? i);
                  const isCorrectingThis = correcting === id;
                  return (
                    <div key={id} className="bg-white rounded-xl border border-slate-200 p-4 flex items-center justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-slate-800 truncate">{String(row.filename ?? "")}</p>
                        <p className="text-sm text-slate-500">
                          {String(row.detected_animal ?? "Unknown")} — conf: {typeof row.detection_confidence === "number" ? (row.detection_confidence as number).toFixed(2) : "?"} — {String(row.day_night ?? "")}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        {isCorrectingThis ? (
                          <>
                            <input
                              autoFocus
                              placeholder="Correct label"
                              value={correctLabel}
                              onChange={(e) => setCorrectLabel(e.target.value)}
                              className="border border-slate-300 rounded px-2 py-1 text-sm"
                            />
                            <button onClick={() => action(() => correctDetection(id, { reviewer_id: reviewerId, corrected_label: correctLabel }))} className="px-3 py-1 bg-green-600 text-white text-xs rounded">Save</button>
                            <button onClick={() => setCorrecting(null)} className="px-3 py-1 bg-slate-200 text-xs rounded">Cancel</button>
                          </>
                        ) : (
                          <>
                            <button onClick={() => action(() => confirmDetection(id, { reviewer_id: reviewerId }))} className="px-3 py-1.5 bg-green-100 text-green-700 text-xs rounded hover:bg-green-200">✓ Confirm</button>
                            <button onClick={() => { setCorrecting(id); setCorrectLabel(""); }} className="px-3 py-1.5 bg-blue-100 text-blue-700 text-xs rounded hover:bg-blue-200">✏ Correct</button>
                            <button onClick={() => action(() => flagDetection(id, { reviewer_id: reviewerId }))} className="px-3 py-1.5 bg-red-100 text-red-700 text-xs rounded hover:bg-red-200">⚑ Flag</button>
                          </>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )
          )}
          {tab === "log" && (
            log.length === 0 ? <div className="text-center py-12 text-slate-400">No corrections logged yet.</div> :
            <SimpleTable rows={log} cols={["image_id", "action", "corrected_label", "reviewer_id", "timestamp", "notes"]} />
          )}
          {tab === "privacy" && (
            audit.length === 0 ? <div className="text-center py-12 text-slate-400">No privacy scrub audit entries.</div> :
            <SimpleTable rows={audit} cols={["filename", "scrubbed", "skipped", "reason"]} />
          )}
        </>
      )}
    </div>
  );
}

function SimpleTable({ rows, cols }: { rows: Row[]; cols: string[] }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-slate-50 border-b border-slate-200">
          <tr>{cols.map((c) => <th key={c} className="text-left px-4 py-3 font-medium text-slate-600 capitalize">{c.replace(/_/g, " ")}</th>)}</tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map((row, i) => (
            <tr key={i} className="hover:bg-slate-50">
              {cols.map((c) => <td key={c} className="px-4 py-2 text-slate-700">{String(row[c] ?? "")}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
