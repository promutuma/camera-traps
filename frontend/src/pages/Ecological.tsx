import { useState } from "react";
import { getIDE, getRAI, getTimeline, getRichness, getAccumulation, getGroupSize, getVisitation } from "../api/client";
import { useConfigStore } from "../store/configStore";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";

type Row = Record<string, unknown>;

export default function Ecological() {
  const config = useConfigStore((s) => s.config);
  const trapNights = config?.trap_nights_default ?? 30;

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [ideSummary, setIdeSummary] = useState<Row[]>([]);
  const [rai, setRai] = useState<Row[]>([]);
  const [timeline, setTimeline] = useState<Row[]>([]);
  const [richness, setRichness] = useState<Row[]>([]);
  const [accumulation, setAccumulation] = useState<Row[]>([]);
  const [groupSize, setGroupSize] = useState<Row[]>([]);
  const [visitation, setVisitation] = useState<{ visitation: Row[]; heatmap: Row[] } | null>(null);
  const [computed, setComputed] = useState(false);

  const compute = async () => {
    setLoading(true);
    setError("");
    try {
      const [ide, raiData, tl, rich, accum, gs, vis] = await Promise.all([
        getIDE(),
        getRAI(trapNights),
        getTimeline(),
        getRichness(),
        getAccumulation(),
        getGroupSize(),
        getVisitation(trapNights),
      ]);
      setIdeSummary(ide.summary ?? []);
      setRai(raiData);
      setTimeline(tl);
      setRichness(rich);
      setAccumulation(accum);
      setGroupSize(gs);
      setVisitation(vis);
      setComputed(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to compute");
    } finally {
      setLoading(false);
    }
  };

  const speciesCounts = ideSummary
    .filter((r) => String(r.species ?? "").toLowerCase() !== "empty")
    .reduce((acc: Record<string, number>, r) => {
      const s = String(r.species ?? "");
      acc[s] = (acc[s] ?? 0) + 1;
      return acc;
    }, {});
  const speciesChartData = Object.entries(speciesCounts).map(([species, count]) => ({ species, count }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Ecological Analytics</h1>
          <p className="text-slate-500 text-sm mt-1">All indicators derived from Independent Detection Events (IDEs)</p>
        </div>
        <button
          onClick={compute}
          disabled={loading}
          className="px-6 py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-slate-300 text-white font-semibold rounded-lg"
        >
          {loading ? "Computing…" : "Compute IDEs"}
        </button>
      </div>

      {error && <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 text-sm">{error}</div>}

      {!computed && !loading && (
        <div className="text-center py-16 text-slate-400">Click "Compute IDEs" to run the analysis.</div>
      )}

      {computed && (
        <div className="space-y-6">
          {/* Headline metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              ["Total IDEs", ideSummary.length],
              ["Animal IDEs", ideSummary.filter((r) => String(r.species ?? "").toLowerCase() !== "empty").length],
              ["Unique Species", new Set(ideSummary.map((r) => r.species)).size],
              ["Stations", new Set(ideSummary.map((r) => r.station_id)).size],
            ].map(([label, val]) => (
              <div key={String(label)} className="bg-white rounded-xl border border-slate-200 p-4">
                <p className="text-sm text-slate-500">{label}</p>
                <p className="text-3xl font-bold text-slate-800 mt-1">{String(val)}</p>
              </div>
            ))}
          </div>

          {/* IDEs per species */}
          {speciesChartData.length > 0 && (
            <Card title="IDEs per Species">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={speciesChartData}>
                  <XAxis dataKey="species" tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#16a34a" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* RAI table */}
          {rai.length > 0 && (
            <Card title="Relative Abundance Index (RAI)">
              <DataTable rows={rai} cols={["station_id", "species", "ide_count", "trap_nights", "rai"]} />
            </Card>
          )}

          {/* Timeline */}
          {timeline.length > 0 && (
            <Card title="Detection Timeline">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={timeline}>
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="count" stroke="#16a34a" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Species richness */}
          {richness.length > 0 && (
            <Card title="Species Richness per Station">
              <DataTable rows={richness} cols={["station_id", "species_count", "singleton_count", "species_list"]} />
            </Card>
          )}

          {/* Accumulation curve */}
          {accumulation.length > 0 && (
            <Card title="Species Accumulation Curve">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={accumulation}>
                  <XAxis dataKey="cumulative_ides" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="cumulative_species" stroke="#2563eb" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Group size */}
          {groupSize.length > 0 && (
            <Card title="Mean Group Size per Species">
              <DataTable rows={groupSize} cols={["station_id", "species", "mean_group_size", "min_group", "max_group", "ide_count"]} />
            </Card>
          )}

          {/* Visitation */}
          {visitation && visitation.visitation.length > 0 && (
            <Card title="Visitation Rate">
              <DataTable rows={visitation.visitation} cols={["station_id", "species", "visits_per_night"]} />
            </Card>
          )}

          {/* Export */}
          <div>
            <a
              href="/api/ecological/export"
              className="inline-block px-4 py-2 bg-slate-700 text-white text-sm rounded-lg hover:bg-slate-800"
            >
              Export IDE Summary CSV
            </a>
          </div>
        </div>
      )}
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <h2 className="font-semibold text-slate-700 mb-3">{title}</h2>
      {children}
    </div>
  );
}

function DataTable({ rows, cols }: { rows: Row[]; cols: string[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-slate-50">
          <tr>
            {cols.map((c) => (
              <th key={c} className="text-left px-3 py-2 font-medium text-slate-600 capitalize">
                {c.replace(/_/g, " ")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map((row, i) => (
            <tr key={i} className="hover:bg-slate-50">
              {cols.map((c) => (
                <td key={c} className="px-3 py-2 text-slate-700">
                  {typeof row[c] === "number" ? (row[c] as number).toFixed(4) : String(row[c] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
