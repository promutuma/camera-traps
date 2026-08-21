import { useEffect, useState } from "react";
import { getHistory, clearHistory } from "../api/client";

type Row = Record<string, unknown>;

export default function History() {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmClear, setConfirmClear] = useState(false);

  const load = () => getHistory().then(setRows).finally(() => setLoading(false));
  useEffect(() => { load(); }, []);

  const handleClear = async () => {
    await clearHistory();
    setConfirmClear(false);
    load();
  };

  const uniqueImages = new Set(rows.map((r) => r.filename)).size;
  const uniqueSpecies = new Set(
    rows.filter((r) => String(r.detected_animal ?? "").toLowerCase() !== "empty").map((r) => r.detected_animal)
  ).size;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Analysis History</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Review past detection exports, audit classification lists, and clear records logs.
          </p>
        </div>
        <div className="flex gap-2">
          <a
            href="/api/history/export/csv"
            className="px-4 py-2 bg-slate-700 hover:bg-slate-800 text-white text-sm font-semibold rounded-lg shadow-sm hover:shadow transition"
          >
            Export CSV
          </a>
          {!confirmClear ? (
            <button
              onClick={() => setConfirmClear(true)}
              className="px-4 py-2 bg-red-50 hover:bg-red-100 dark:bg-red-950/20 dark:hover:bg-red-950/40 text-red-650 dark:text-red-400 text-sm font-semibold rounded-lg border border-red-200 dark:border-red-900/35 transition cursor-pointer"
            >
              Clear Logs
            </button>
          ) : (
            <div className="flex gap-2.5 items-center bg-red-50 dark:bg-red-950/20 border border-red-150 dark:border-red-900/30 rounded-lg px-3 py-1.5">
              <span className="text-xs font-semibold text-red-700 dark:text-red-400">Are you sure?</span>
              <button
                onClick={handleClear}
                className="px-2.5 py-1 bg-red-600 hover:bg-red-700 text-white text-xs font-bold rounded cursor-pointer"
              >
                Yes, clear
              </button>
              <button
                onClick={() => setConfirmClear(false)}
                className="px-2.5 py-1 bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 text-xs font-bold rounded cursor-pointer"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm">
          <p className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Total Images</p>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white mt-1.5">{uniqueImages}</p>
        </div>
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm">
          <p className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Total Detections</p>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white mt-1.5">{rows.length}</p>
        </div>
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm">
          <p className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Unique Species</p>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white mt-1.5">{uniqueSpecies}</p>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12 text-slate-400 dark:text-slate-550 animate-pulse">
          Loading history log list…
        </div>
      ) : rows.length === 0 ? (
        <div className="text-center py-16 text-slate-400 dark:text-slate-550 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-850 rounded-xl">
          No analysis history found. Process and save images to build your catalog.
        </div>
      ) : (
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 dark:bg-slate-950/40 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  {["filename", "station_id", "detected_animal", "detection_confidence", "day_night", "processed_at", "user_notes"].map((c) => (
                    <th
                      key={c}
                      className="text-left px-4 py-3 font-semibold text-slate-600 dark:text-slate-450 uppercase tracking-wider text-[11px]"
                    >
                      {c.replace(/_/g, " ")}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {rows.map((row, i) => (
                  <tr key={i} className="hover:bg-slate-50/50 dark:hover:bg-slate-950/20 transition-colors">
                    {["filename", "station_id", "detected_animal", "detection_confidence", "day_night", "processed_at", "user_notes"].map((col) => (
                      <td key={col} className="px-4 py-3 text-slate-750 dark:text-slate-300">
                        {col === "detection_confidence"
                          ? typeof row[col] === "number"
                            ? (row[col] as number).toFixed(2)
                            : String(row[col] ?? "")
                          : String(row[col] ?? "")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

