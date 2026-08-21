import { useEffect, useState } from "react";
import { getObservations, addObservation, deleteObservation, getCrosscheck } from "../api/client";

type Row = Record<string, unknown>;

const OBS_TYPES = ["Animal", "Track", "Scat", "Camera Malfunction", "Human Activity", "Other"];

export default function Community() {
  const [obs, setObs] = useState<Row[]>([]);
  const [crosscheck, setCrosscheck] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"entry" | "crosscheck">("entry");
  const [form, setForm] = useState({
    observer_name: "", observation_type: "Animal", species: "",
    count: "", latitude: "", longitude: "", date: "", time: "", notes: "",
  });

  const load = async () => {
    setLoading(true);
    const [o, c] = await Promise.all([getObservations(), getCrosscheck()]);
    setObs(Array.isArray(o) ? o : []);
    setCrosscheck(Array.isArray(c) ? c : []);
    setLoading(false);
  };
  useEffect(() => { load(); }, []);

  const handleAdd = async () => {
    if (!form.observer_name) return;
    await addObservation({
      ...form,
      count: form.count ? Number(form.count) : null,
      latitude: form.latitude ? Number(form.latitude) : null,
      longitude: form.longitude ? Number(form.longitude) : null,
    });
    setForm({ observer_name: "", observation_type: "Animal", species: "", count: "", latitude: "", longitude: "", date: "", time: "", notes: "" });
    load();
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Community Observer</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Log human sightings, scats, tracks, and verify community observation cross-checks.
        </p>
      </div>

      <div className="flex gap-2">
        {(["entry", "crosscheck"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition cursor-pointer ${
              tab === t
                ? "bg-emerald-600 text-white shadow-sm"
                : "bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-950/50"
            }`}
          >
            {t === "entry" ? "Data Entry Log" : "Cross-Verification Matrix"}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="text-center py-12 text-slate-400 dark:text-slate-550 animate-pulse">
          Loading observations…
        </div>
      ) : tab === "entry" ? (
        <>
          {/* Form */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 space-y-4 shadow-sm">
            <h2 className="font-bold text-base text-slate-800 dark:text-slate-250">Record Observation Sighting</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
              <input
                placeholder="Observer Name *"
                value={form.observer_name}
                onChange={(e) => setForm((f) => ({ ...f, observer_name: e.target.value }))}
                className="border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-450 transition"
              />
              <select
                value={form.observation_type}
                onChange={(e) => setForm((f) => ({ ...f, observation_type: e.target.value }))}
                className="border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition cursor-pointer"
              >
                {OBS_TYPES.map((t) => (
                  <option key={t} value={t} className="dark:bg-slate-950">
                    {t}
                  </option>
                ))}
              </select>
              <input
                placeholder="Species / Animal"
                value={form.species}
                onChange={(e) => setForm((f) => ({ ...f, species: e.target.value }))}
                className="border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-450 transition"
              />
              <input
                placeholder="Count (Individual qty)"
                type="number"
                value={form.count}
                onChange={(e) => setForm((f) => ({ ...f, count: e.target.value }))}
                className="border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-450 transition"
              />
              <input
                placeholder="Latitude coordinates"
                value={form.latitude}
                onChange={(e) => setForm((f) => ({ ...f, latitude: e.target.value }))}
                className="border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-450 transition"
              />
              <input
                placeholder="Longitude coordinates"
                value={form.longitude}
                onChange={(e) => setForm((f) => ({ ...f, longitude: e.target.value }))}
                className="border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-450 transition"
              />
              <input
                placeholder="Date (YYYY-MM-DD) *"
                value={form.date}
                onChange={(e) => setForm((f) => ({ ...f, date: e.target.value }))}
                className="border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-450 transition"
              />
              <input
                placeholder="Time (HH:MM)"
                value={form.time}
                onChange={(e) => setForm((f) => ({ ...f, time: e.target.value }))}
                className="border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-450 transition"
              />
              <input
                placeholder="Observation Notes"
                value={form.notes}
                onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
                className="border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-450 transition"
              />
            </div>
            <button
              onClick={handleAdd}
              disabled={!form.observer_name}
              className="mt-2 px-5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold rounded-lg shadow-sm hover:shadow transition disabled:bg-slate-300 dark:disabled:bg-slate-800 dark:disabled:text-slate-500 cursor-pointer"
            >
              Add Observation
            </button>
          </div>

          {/* Table */}
          {obs.length === 0 ? (
            <div className="text-center py-12 text-slate-400 dark:text-slate-550 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-850 rounded-xl">
              No observations recorded yet. Fill the form to log.
            </div>
          ) : (
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 dark:bg-slate-950/40 border-b border-slate-200 dark:border-slate-800">
                    <tr>
                      {["observer_name", "observation_type", "species", "count", "date", "notes", ""].map((c, i) => (
                        <th
                          key={i}
                          className="text-left px-4 py-3 font-semibold text-slate-600 dark:text-slate-450 uppercase tracking-wider text-[11px]"
                        >
                          {c.replace(/_/g, " ")}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                    {obs.map((row, i) => (
                      <tr key={i} className="hover:bg-slate-50/50 dark:hover:bg-slate-950/20 transition-colors">
                        {["observer_name", "observation_type", "species", "count", "date", "notes"].map((c) => (
                          <td key={c} className="px-4 py-3 text-slate-700 dark:text-slate-305">
                            {String(row[c] ?? "")}
                          </td>
                        ))}
                        <td className="px-4 py-3 text-right">
                          <button
                            onClick={() => deleteObservation(Number(row.id ?? i)).then(load)}
                            className="text-red-500 hover:text-red-700 font-semibold text-xs cursor-pointer transition"
                          >
                            Remove
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      ) : crosscheck.length === 0 ? (
        <div className="text-center py-16 text-slate-400 dark:text-slate-550 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-850 rounded-xl shadow-sm">
          No cross-verification data. Add community observations and run camera trap pipeline processes first.
        </div>
      ) : (
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 dark:bg-slate-950/40 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  {["species", "camera_detections", "observer_sightings", "match"].map((c) => (
                    <th
                      key={c}
                      className="text-left px-4 py-3 font-semibold text-slate-600 dark:text-slate-455 uppercase tracking-wider text-[11px]"
                    >
                      {c.replace(/_/g, " ")}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {crosscheck.map((row, i) => (
                  <tr key={i} className="hover:bg-slate-50/50 dark:hover:bg-slate-950/20 transition-colors">
                    <td className="px-4 py-3 font-semibold text-slate-800 dark:text-slate-200">
                      {String(row.species ?? "")}
                    </td>
                    <td className="px-4 py-3 text-slate-700 dark:text-slate-350">
                      {String(row.camera_detections ?? 0)}
                    </td>
                    <td className="px-4 py-3 text-slate-700 dark:text-slate-350">
                      {String(row.observer_sightings ?? 0)}
                    </td>
                    <td className="px-4 py-3">
                      {row.match ? (
                        <span className="px-2 py-0.5 bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-250 dark:border-emerald-800/40 text-emerald-600 dark:text-emerald-450 rounded-full text-xs font-bold uppercase">
                          Match
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 bg-amber-50 dark:bg-amber-950/30 border border-amber-250 dark:border-amber-800/40 text-amber-600 dark:text-amber-450 rounded-full text-xs font-bold uppercase">
                          Mismatch
                        </span>
                      )}
                    </td>
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

