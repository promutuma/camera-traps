import { useEffect, useState } from "react";
import { getStats } from "../api/client";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from "recharts";

const COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];

function StatsSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      {/* Title */}
      <div className="space-y-2">
        <div className="h-7 w-48 bg-slate-200 dark:bg-slate-800 rounded-lg"></div>
        <div className="h-4 w-72 bg-slate-150 dark:bg-slate-850 rounded-lg"></div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-24 bg-white/70 dark:bg-slate-900/60 border border-slate-200/50 dark:border-slate-800/50 rounded-xl p-4 space-y-3">
            <div className="h-4 w-20 bg-slate-200 dark:bg-slate-800 rounded"></div>
            <div className="h-8 w-12 bg-slate-250 dark:bg-slate-800 rounded-lg"></div>
          </div>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[1, 2].map((i) => (
          <div key={i} className="h-80 bg-white/70 dark:bg-slate-900/60 border border-slate-200/50 dark:border-slate-800/50 rounded-xl p-4 space-y-3">
            <div className="h-5 w-32 bg-slate-200 dark:bg-slate-800 rounded"></div>
            <div className="h-56 w-full bg-slate-100 dark:bg-slate-800/50 rounded-lg"></div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Statistics() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getStats().then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) return <StatsSkeleton />;
  if (!data || (data.total_images as number) === 0)
    return (
      <div className="text-center py-16 text-slate-400 bg-white/75 dark:bg-slate-900/60 border border-slate-200/50 dark:border-slate-800/50 rounded-2xl p-8 backdrop-blur-md">
        No data yet. Process images first.
      </div>
    );

  const metrics = [
    { label: "Total Images", value: data.total_images },
    { label: "Animals Identified", value: data.animals_identified },
    { label: "Day Images", value: data.day_count },
    { label: "Night Images", value: data.night_count },
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Analysis Statistics</h1>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">Summary metrics and activity patterns computed across all sessions.</p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.map((m) => (
          <div key={m.label} className="bg-white/75 dark:bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-200/50 dark:border-slate-800/50 p-4 shadow-sm">
            <p className="text-sm text-slate-500 dark:text-slate-400 font-medium">{m.label}</p>
            <p className="text-3xl font-bold text-slate-800 dark:text-slate-100 mt-1">{String(m.value)}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Species distribution */}
        <div className="bg-white/75 dark:bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-200/50 dark:border-slate-800/50 p-4 shadow-sm">
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
        <div className="bg-white/75 dark:bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-200/50 dark:border-slate-800/50 p-4 shadow-sm">
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
        <div className="bg-white/75 dark:bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-200/50 dark:border-slate-800/50 p-4 shadow-sm">
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
      {Array.isArray(data.confidence_series) && (data.confidence_series as number[]).length > 0 && (
        <div className="bg-white/75 dark:bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-200/50 dark:border-slate-800/50 p-4 shadow-sm">
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
      )}

      {/* Export section */}
      <div className="bg-white/75 dark:bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-200/50 dark:border-slate-800/50 p-6 space-y-4 shadow-sm">
        <div>
          <h3 className="text-base font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
            <span className="material-symbols-outlined text-emerald-500">ios_share</span>
            Export Analysis Statistics
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Download computed summary statistics, distributions, and activity patterns in standard CSV or Excel spreadsheets.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { label: "Summary Metrics Table", metric: "summary" },
            { label: "Species Distribution", metric: "species" },
            { label: "Day / Night Distribution", metric: "daynight" },
            { label: "Daily Hourly Activity Patterns", metric: "hourly" },
            { label: "Detection Confidence Data", metric: "confidence" },
          ].map((m) => (
            <div
              key={m.metric}
              className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-900/40 border border-slate-200/40 dark:border-slate-800/40 rounded-xl"
            >
              <span className="text-xs font-medium text-slate-700 dark:text-slate-300">{m.label}</span>
              <div className="flex gap-1.5">
                <a
                  href={`/api/stats/export?metric=${m.metric}&format=csv`}
                  className="inline-flex items-center gap-1 px-2.5 py-1 bg-white hover:bg-slate-50 dark:bg-slate-800 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-[10px] font-bold uppercase tracking-wide text-slate-600 dark:text-slate-400 rounded-md transition shadow-sm"
                >
                  CSV
                </a>
                <a
                  href={`/api/stats/export?metric=${m.metric}&format=excel`}
                  className="inline-flex items-center gap-1 px-2.5 py-1 bg-white hover:bg-slate-50 dark:bg-slate-800 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 text-[10px] font-bold uppercase tracking-wide text-green-600 dark:text-green-400 rounded-md transition shadow-sm"
                >
                  EXCEL
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
