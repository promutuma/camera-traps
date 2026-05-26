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

  const uniqueSpecies = new Set(rows.map((r) => r.detected_animal)).size;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-800">Analysis History</h1>
        <div className="flex gap-2">
          <a
            href="/api/history/export/csv"
            className="px-4 py-2 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-700"
          >
            Export CSV
          </a>
          {!confirmClear ? (
            <button
              onClick={() => setConfirmClear(true)}
              className="px-4 py-2 bg-red-50 text-red-600 text-sm rounded-lg hover:bg-red-100 border border-red-200"
            >
              Clear History
            </button>
          ) : (
            <div className="flex gap-2 items-center">
              <span className="text-sm text-red-600">Are you sure?</span>
              <button onClick={handleClear} className="px-3 py-1.5 bg-red-600 text-white text-sm rounded">Yes, clear</button>
              <button onClick={() => setConfirmClear(false)} className="px-3 py-1.5 bg-slate-200 text-sm rounded">Cancel</button>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-sm text-slate-500">Total Records</p>
          <p className="text-3xl font-bold text-slate-800 mt-1">{rows.length}</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-sm text-slate-500">Unique Species</p>
          <p className="text-3xl font-bold text-slate-800 mt-1">{uniqueSpecies}</p>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12 text-slate-400">Loading…</div>
      ) : rows.length === 0 ? (
        <div className="text-center py-12 text-slate-400">No history. Process and save images to build your history.</div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200 overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                {["filename", "station_id", "detected_animal", "detection_confidence", "day_night", "processed_at", "user_notes"].map((c) => (
                  <th key={c} className="text-left px-4 py-3 font-medium text-slate-600 capitalize">
                    {c.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rows.map((row, i) => (
                <tr key={i} className="hover:bg-slate-50">
                  {["filename", "station_id", "detected_animal", "detection_confidence", "day_night", "processed_at", "user_notes"].map((col) => (
                    <td key={col} className="px-4 py-2 text-slate-700">
                      {col === "detection_confidence"
                        ? typeof row[col] === "number" ? (row[col] as number).toFixed(2) : String(row[col] ?? "")
                        : String(row[col] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
