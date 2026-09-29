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
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Corridor Movement Analysis</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Analyse spatial animal movements and structural corridors between camera stations.
          </p>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <label className="text-xs font-semibold text-slate-550 dark:text-slate-400 uppercase tracking-wider flex items-center">
            Max range (km)
            <input
              type="number"
              value={maxKm}
              min={1}
              max={500}
              onChange={(e) => setMaxKm(Number(e.target.value))}
              className="ml-2 border border-slate-300 dark:border-slate-800 rounded-lg px-2.5 py-1.5 text-sm w-20 focus:outline-none focus:ring-2 focus:ring-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-400 transition"
            />
          </label>
          <button
            onClick={compute}
            disabled={loading}
            className="px-5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold rounded-lg shadow-sm hover:shadow transition disabled:bg-slate-300 dark:disabled:bg-slate-800 dark:disabled:text-slate-500 cursor-pointer"
          >
            {loading ? "Computing…" : "Analyse Corridors"}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-955/20 border border-red-200 dark:border-red-900/35 text-red-700 dark:text-red-400 rounded-lg p-4 text-sm">
          {error}
        </div>
      )}

      {!computed && !loading && (
        <div className="text-center py-20 text-slate-400 dark:text-slate-550 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-850 rounded-xl shadow-sm">
          Click "Analyse Corridors" to run calculations. Requires stations with geographic coordinates loaded.
        </div>
      )}

      {loading && (
        <div className="text-center py-20 text-slate-400 dark:text-slate-550 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-850 rounded-xl shadow-sm animate-pulse">
          Computing directional movement event flows between stations…
        </div>
      )}

      {computed && !loading && (
        <div className="space-y-6">
          <Section
            title={`Station Pairs within ${maxKm} km`}
            rows={pairs}
            cols={["station_a", "station_b", "distance_m"]}
          />
          <Section
            title="Directional Movement Events (Flows)"
            rows={movements}
            cols={["species", "station_a", "station_b", "flow_a_to_b", "flow_b_to_a", "total_passages", "dominant_direction"]}
          />
          <Section
            title="Bottleneck Pairs"
            rows={bottlenecks}
            cols={["station_a", "station_b", "species", "total_passages"]}
          />
          <Section
            title="Corridor Utilisation by Species"
            rows={utilisation}
            cols={["species", "pairs_used", "total_pairs", "utilisation_pct"]}
          />
        </div>
      )}
    </div>
  );
}

function Section({ title, rows, cols }: { title: string; rows: Row[]; cols: string[] }) {
  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden shadow-sm">
      <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/20">
        <h2 className="font-bold text-sm text-slate-800 dark:text-slate-200">{title}</h2>
      </div>
      {rows.length === 0 ? (
        <div className="text-center py-8 text-slate-400 dark:text-slate-550 text-sm">No corridor links detected for this metric.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 dark:bg-slate-950/40 border-b border-slate-200 dark:border-slate-800">
              <tr>
                {cols.map((c) => (
                  <th
                    key={c}
                    className="text-left px-4 py-3 font-semibold text-slate-655 dark:text-slate-450 uppercase tracking-wider text-[11px]"
                  >
                    {c.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {rows.map((row, i) => (
                <tr key={i} className="hover:bg-slate-50/50 dark:hover:bg-slate-950/25 transition-colors">
                  {cols.map((c) => (
                    <td key={c} className="px-4 py-3 text-slate-700 dark:text-slate-300">
                      {typeof row[c] === "number"
                        ? c === "distance_m"
                          ? `${(row[c] as number).toFixed(0)} m`
                          : c.endsWith("_pct") || c.endsWith("_ratio")
                          ? `${(row[c] as number).toFixed(1)}%`
                          : (row[c] as number).toString()
                        : String(row[c] ?? "")}
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

