import { useState } from "react";
import { getCorridorPairs, getMovements, getBottlenecks, getUtilisation } from "../api/client";

type Row = Record<string, unknown>;

export default function Corridor() {
  const [maxKm, setMaxKm] = useState(50);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [pairs, setPairs] = useState<Row[]>([]);
  const [movements, setMovements] = useState<Row[]>([]);
  const [bottlenecks, setBottlenecks] = useState<Row[]>([]);
  const [utilisation, setUtilisation] = useState<Row[]>([]);
  const [computed, setComputed] = useState(false);

  const compute = async () => {
    setLoading(true);
    setError("");
    try {
      const [p, m, b, u] = await Promise.all([
        getCorridorPairs(maxKm),
        getMovements(maxKm),
        getBottlenecks(maxKm),
        getUtilisation(maxKm),
      ]);
      setPairs(Array.isArray(p) ? p : []);
      setMovements(Array.isArray(m) ? m : []);
      setBottlenecks(Array.isArray(b) ? b : []);
      setUtilisation(Array.isArray(u) ? u : []);
      setComputed(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Corridor Movement Analysis</h1>
          <p className="text-slate-500 text-sm mt-1">Analyse animal movements between camera stations.</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm text-slate-600">
            Max corridor distance (km)
            <input
              type="number"
              value={maxKm}
              min={1}
              max={500}
              onChange={(e) => setMaxKm(Number(e.target.value))}
              className="ml-2 border border-slate-300 rounded px-2 py-1 text-sm w-20"
            />
          </label>
          <button onClick={compute} disabled={loading} className="px-6 py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-slate-300 text-white font-semibold rounded-lg">
            {loading ? "Computing…" : "Analyse"}
          </button>
        </div>
      </div>

      {error && <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 text-sm">{error}</div>}
      {!computed && !loading && <div className="text-center py-16 text-slate-400">Click "Analyse" to run corridor analysis. Requires stations with coordinates.</div>}

      {computed && (
        <div className="space-y-6">
          <Section title={`Station Pairs within ${maxKm} km`} rows={pairs} cols={["station_a", "station_b", "distance_km"]} />
          <Section title="Directional Movement Events" rows={movements} cols={["species", "from_station", "to_station", "datetime", "confidence"]} />
          <Section title="Bottleneck Pairs" rows={bottlenecks} cols={["station_a", "station_b", "species_count", "total_passages"]} />
          <Section title="Corridor Utilisation by Species" rows={utilisation} cols={["species", "corridor_pair", "passages"]} />
        </div>
      )}
    </div>
  );
}

function Section({ title, rows, cols }: { title: string; rows: Row[]; cols: string[] }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100">
        <h2 className="font-semibold text-slate-700">{title}</h2>
      </div>
      {rows.length === 0 ? (
        <div className="text-center py-8 text-slate-400 text-sm">No data.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50">
              <tr>{cols.map((c) => <th key={c} className="text-left px-4 py-2 font-medium text-slate-600 capitalize">{c.replace(/_/g, " ")}</th>)}</tr>
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
      )}
    </div>
  );
}
