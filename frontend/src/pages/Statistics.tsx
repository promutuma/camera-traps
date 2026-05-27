import { useEffect, useState } from "react";
import { getStats } from "../api/client";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from "recharts";

const COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];

export default function Statistics() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getStats().then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-center py-12 text-slate-400">Loading…</div>;
  if (!data || (data.total_images as number) === 0)
    return <div className="text-center py-12 text-slate-400">No data yet. Process images first.</div>;

  const metrics = [
    { label: "Total Images", value: data.total_images },
    { label: "Animals Identified", value: data.animals_identified },
    { label: "Day Images", value: data.day_count },
    { label: "Night Images", value: data.night_count },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Analysis Statistics</h1>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">Summary metrics and activity patterns computed across all sessions.</p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.map((m) => (
          <div key={m.label} className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm">
            <p className="text-sm text-slate-500 dark:text-slate-400 font-medium">{m.label}</p>
            <p className="text-3xl font-bold text-slate-800 dark:text-slate-100 mt-1">{String(m.value)}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Species distribution */}
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm">
          <h2 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">Species Distribution</h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={data.species_distribution as object[]}>
              <XAxis dataKey="species" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Day/Night pie */}
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm">
          <h2 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">Day / Night Distribution</h2>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={data.day_night_distribution as object[]}
                dataKey="count"
                nameKey="label"
                cx="50%"
                cy="50%"
                outerRadius={90}
                label={({ name, percent }: { name?: string; percent?: number }) =>
                  `${name ?? ""} ${((percent ?? 0) * 100).toFixed(0)}%`
                }
              >
                {(data.day_night_distribution as object[]).map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Hourly activity patterns */}
      {Array.isArray(data.hourly_distribution) && data.hourly_distribution.length > 0 && (
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm">
          <h2 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">Daily Hourly Activity Pattern (24-Hour Cycle)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data.hourly_distribution as object[]}>
              <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Confidence series */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm">
        <h2 className="font-semibold text-slate-700 dark:text-slate-300 mb-3">Detection Confidence Distribution</h2>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart
            data={(data.confidence_series as number[]).map((v, i) => ({ i, confidence: v }))}
          >
            <XAxis dataKey="i" hide />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Line type="monotone" dataKey="confidence" stroke="#10b981" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
