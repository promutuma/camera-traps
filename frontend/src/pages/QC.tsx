import { useEffect, useState } from "react";
import { getQCFlags, getQCSummary } from "../api/client";

type Row = Record<string, unknown>;

const SEVERITY_COLORS: Record<string, string> = {
  high: "bg-red-100 text-red-700",
  medium: "bg-amber-100 text-amber-700",
  low: "bg-blue-100 text-blue-700",
};

export default function QC() {
  const [flags, setFlags] = useState<Row[]>([]);
  const [summary, setSummary] = useState<{ total_flags: number; by_type: Row[] } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getQCFlags(), getQCSummary()])
      .then(([f, s]) => { setFlags(f); setSummary(s); })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-center py-12 text-slate-400">Loading…</div>;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-slate-800">QC Dashboard</h1>

      {summary && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <p className="text-sm text-slate-500">Total Flags</p>
            <p className="text-3xl font-bold text-slate-800 mt-1">{summary.total_flags}</p>
          </div>
          {summary.by_type.length > 0 && (
            <div className="bg-white rounded-xl border border-slate-200 p-4">
              <p className="text-sm font-medium text-slate-600 mb-2">By Type</p>
              <div className="space-y-1">
                {summary.by_type.map((t, i) => (
                  <div key={i} className="flex justify-between text-sm">
                    <span className="text-slate-600">{String(t.flag_type ?? t.type ?? "")}</span>
                    <span className="font-semibold">{String(t.count ?? "")}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {flags.length === 0 ? (
        <div className="text-center py-12 text-slate-400">No QC flags found. Process images first.</div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200 overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                {["filename", "flag_type", "severity", "message"].map((c) => (
                  <th key={c} className="text-left px-4 py-3 font-medium text-slate-600 capitalize">
                    {c.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {flags.map((row, i) => (
                <tr key={i} className="hover:bg-slate-50">
                  <td className="px-4 py-2 text-slate-700">{String(row.filename ?? "")}</td>
                  <td className="px-4 py-2 text-slate-700">{String(row.flag_type ?? "")}</td>
                  <td className="px-4 py-2">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${SEVERITY_COLORS[String(row.severity ?? "").toLowerCase()] ?? "bg-slate-100 text-slate-600"}`}>
                      {String(row.severity ?? "")}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-slate-600">{String(row.message ?? "")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
