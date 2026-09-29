import { useEffect, useState } from "react";
import {
  getProject,
  updateProject,
  resetDatabase,
  listProjects,
  setActiveProject,
  createProject,
  deleteProject,
} from "../api/client";

type ProjectData = {
  id: number;
  project_name: string;
  survey_area: string;
  notes: string;
  thresholds: Record<string, number>;
  speciesnet_lat: number;
  speciesnet_lng: number;
  speciesnet_country: string;
};

type ProjectListItem = {
  id: number;
  name: string;
  survey_area: string;
  notes: string;
  is_active: number;
  created_at: string;
};

const THRESHOLD_DEFS = {
  min_trap_nights: { label: "Min Trap Nights", unit: "nights", min: 1, max: 365, step: 1 },
  min_functionality_pct: { label: "Min Functionality %", unit: "%", min: 0.0, max: 100.0, step: 5 },
  confidence_threshold: { label: "Min Reporting Confidence", unit: "0-1", min: 0.1, max: 1.0, step: 0.05 },
  independence_window_min: { label: "Independence Window", unit: "minutes", min: 5, max: 120, step: 5 },
  rai_alert_below: { label: "RAI Alert Threshold", unit: "IDEs/TN", min: 0.0, max: 10.0, step: 0.1 },
  min_ide_for_richness: { label: "Min IDEs for Richness", unit: "IDEs", min: 1, max: 50, step: 1 },
};

export default function ProjectConfig() {
  const [activeProject, setActiveProjectData] = useState<ProjectData | null>(null);
  const [projects, setProjects] = useState<ProjectListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  // Active project form states
  const [form, setForm] = useState({ project_name: "", survey_area: "", notes: "" });
  const [thresholds, setThresholds] = useState<Record<string, number>>({});
  const [coords, setCoords] = useState({ speciesnet_lat: -1.0, speciesnet_lng: 37.0, speciesnet_country: "KEN" });

  // New project form state
  const [newProjectForm, setNewProjectForm] = useState({ name: "", survey_area: "", notes: "" });

  const loadAllData = async () => {
    try {
      const active = await getProject();
      if (active) {
        setActiveProjectData(active);
        setForm({
          project_name: String(active.project_name ?? ""),
          survey_area: String(active.survey_area ?? ""),
          notes: String(active.notes ?? ""),
        });
        setThresholds(active.thresholds ?? {});
        setCoords({
          speciesnet_lat: active.speciesnet_lat ?? -1.0,
          speciesnet_lng: active.speciesnet_lng ?? 37.0,
          speciesnet_country: active.speciesnet_country ?? "KEN",
        });
      }
      const list = await listProjects() as any;
      setProjects(list);
    } catch (e) {
      console.error("Failed to load project config data", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllData();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      await updateProject({
        ...form,
        thresholds,
        ...coords,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      // Reload list and active details to stay in sync
      const active = await getProject();
      setActiveProjectData(active);
      const list = await listProjects() as any;
      setProjects(list);
    } catch (e: any) {
      alert(e.response?.data?.detail || "Failed to save project config.");
    } finally {
      setSaving(false);
    }
  };

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjectForm.name.trim()) return;
    setSaving(true);
    try {
      await createProject({
        name: newProjectForm.name,
        survey_area: newProjectForm.survey_area,
        notes: newProjectForm.notes,
      });
      setNewProjectForm({ name: "", survey_area: "", notes: "" });
      // Since creating a new project sets it to active, reload page to sync all application state
      window.location.reload();
    } catch (err: any) {
      alert(err.response?.data?.detail || "Failed to create project.");
    } finally {
      setSaving(false);
    }
  };

  const handleSwitchProject = async (id: number) => {
    setLoading(true);
    try {
      await setActiveProject(id);
      window.location.reload();
    } catch (err: any) {
      alert(err.response?.data?.detail || "Failed to switch active project.");
      setLoading(false);
    }
  };

  const handleDeleteProject = async (id: number, name: string) => {
    const confirmDelete = window.confirm(
      `Are you sure you want to delete the project "${name}"? This will permanently remove it from the database along with all associated station, deployment, image, and detection records. This action cannot be undone.`
    );
    if (!confirmDelete) return;
    setLoading(true);
    try {
      await deleteProject(id);
      loadAllData();
    } catch (err: any) {
      alert(err.response?.data?.detail || "Failed to delete project.");
      setLoading(false);
    }
  };

  const handleLockBaseline = async () => {
    await fetch("/api/project/baseline/lock", { method: "POST" });
    alert("Baseline locked.");
  };

  const handleResetDb = async () => {
    const firstConfirm = window.confirm(
      "WARNING: This will permanently delete ALL data in the database (detections, images, stations, deployments, community observations, history, etc.). This action CANNOT be undone.\n\nAre you sure you want to proceed?"
    );
    if (!firstConfirm) return;

    const secondConfirm = window.confirm(
      "FINAL CONFIRMATION: Are you absolutely certain you want to wipe the entire database? All camera trap data and configs will be cleared."
    );
    if (!secondConfirm) return;

    try {
      const res = await resetDatabase();
      alert(res.message || "Database completely cleared and reset.");
      window.location.reload();
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : String(e);
      alert("Failed to reset database: " + errMsg);
    }
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto py-12 text-center text-slate-400 dark:text-slate-500 animate-pulse">
        Loading project configurations…
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6 animate-fade-in px-4">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <span className="material-symbols-outlined text-emerald-500 text-3xl select-none">folder_shared</span>
          Project Hub
        </h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          Manage multiple camera trap survey projects, configure active parameters, and toggle between survey scopes.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Projects Registry & Creator */}
        <div className="lg:col-span-1 space-y-6">
          {/* Projects List Card */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm flex flex-col h-[400px]">
            <h2 className="font-bold text-base text-slate-800 dark:text-slate-200 mb-4 flex items-center gap-1.5">
              <span className="material-symbols-outlined text-slate-500 text-lg select-none font-semibold">list_alt</span>
              Projects Registry
            </h2>
            <div className="flex-1 overflow-y-auto space-y-3 pr-1 custom-scrollbar">
              {projects.map((p) => {
                const isActive = p.is_active === 1;
                return (
                  <div
                    key={p.id}
                    className={`p-4 rounded-xl border transition-all flex flex-col justify-between ${
                      isActive
                        ? "bg-emerald-50/50 dark:bg-emerald-950/20 border-emerald-500/30 ring-1 ring-emerald-500/20"
                        : "bg-slate-50 dark:bg-slate-950/40 border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700"
                    }`}
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <h3 className="font-bold text-sm text-slate-800 dark:text-slate-200 break-words leading-tight">
                          {p.name}
                        </h3>
                        {isActive ? (
                          <span className="shrink-0 flex items-center gap-1 bg-emerald-100 dark:bg-emerald-900/40 text-emerald-800 dark:text-emerald-400 px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider">
                            <span className="material-symbols-outlined text-[10px] select-none leading-none">check_circle</span>
                            Active
                          </span>
                        ) : null}
                      </div>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">
                        {p.survey_area ? `Area: ${p.survey_area}` : "No area specified"}
                      </p>
                      {p.notes && (
                        <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-1.5 line-clamp-1 italic">
                          "{p.notes}"
                        </p>
                      )}
                    </div>

                    <div className="flex items-center gap-2 mt-4 pt-3 border-t border-slate-200/50 dark:border-slate-800/50">
                      {!isActive ? (
                        <>
                          <button
                            onClick={() => handleSwitchProject(p.id)}
                            className="flex-1 py-1.5 bg-slate-200 hover:bg-slate-300 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 text-xs font-semibold rounded-lg shadow-sm transition flex items-center justify-center gap-1 cursor-pointer"
                          >
                            <span className="material-symbols-outlined text-sm select-none">swap_horiz</span>
                            Switch to
                          </button>
                          <button
                            onClick={() => handleDeleteProject(p.id, p.name)}
                            className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-950/20 rounded-lg transition cursor-pointer"
                            title="Delete Project"
                          >
                            <span className="material-symbols-outlined text-sm select-none">delete</span>
                          </button>
                        </>
                      ) : (
                        <span className="text-[11px] text-emerald-600 dark:text-emerald-400 font-medium flex items-center gap-1">
                          <span className="material-symbols-outlined text-xs select-none">info</span>
                          Currently selected workspace
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* New Project Creator Card */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
            <h2 className="font-bold text-base text-slate-800 dark:text-slate-200 mb-4 flex items-center gap-1.5">
              <span className="material-symbols-outlined text-slate-500 text-lg select-none font-semibold">create_new_folder</span>
              New Survey Project
            </h2>
            <form onSubmit={handleCreateProject} className="space-y-4">
              <div>
                <label className="block text-[10px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
                  Project Name *
                </label>
                <input
                  required
                  value={newProjectForm.name}
                  onChange={(e) => setNewProjectForm((f) => ({ ...f, name: e.target.value }))}
                  placeholder="e.g. Serengeti North 2026"
                  className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-400 transition"
                />
              </div>
              <div>
                <label className="block text-[10px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
                  Survey Area
                </label>
                <input
                  value={newProjectForm.survey_area}
                  onChange={(e) => setNewProjectForm((f) => ({ ...f, survey_area: e.target.value }))}
                  placeholder="e.g. Core Zone Sector C"
                  className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-400 transition"
                />
              </div>
              <div>
                <label className="block text-[10px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
                  Notes
                </label>
                <textarea
                  value={newProjectForm.notes}
                  onChange={(e) => setNewProjectForm((f) => ({ ...f, notes: e.target.value }))}
                  rows={2}
                  placeholder="Details about this survey project..."
                  className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-400 transition text-slate-800 dark:text-slate-100"
                />
              </div>
              <button
                type="submit"
                disabled={saving}
                className="w-full py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow-sm hover:shadow transition disabled:bg-slate-300 dark:disabled:bg-slate-800 cursor-pointer flex items-center justify-center gap-1"
              >
                <span className="material-symbols-outlined text-sm select-none">add</span>
                Create & Switch Project
              </button>
            </form>
          </div>
        </div>

        {/* Right Column: Active Project Details & Thresholds */}
        <div className="lg:col-span-2 space-y-6">
          {activeProject ? (
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 space-y-6 shadow-sm">
              <div className="flex items-start justify-between border-b border-slate-200/60 dark:border-slate-800/60 pb-4">
                <div>
                  <h2 className="font-bold text-base text-slate-800 dark:text-slate-200 flex items-center gap-2">
                    <span className="material-symbols-outlined text-emerald-500 select-none">folder_open</span>
                    Active Workspace: {activeProject.project_name}
                  </h2>
                  <p className="text-xs text-slate-400 dark:text-slate-500">
                    Modify active project scope attributes and custom indicators below.
                  </p>
                </div>
              </div>

              {/* Metadata Inputs */}
              <div className="space-y-4">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Project Info
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
                      Project Name
                    </label>
                    <input
                      value={form.project_name}
                      onChange={(e) => setForm((f) => ({ ...f, project_name: e.target.value }))}
                      placeholder="e.g. Gambella Conservation Survey"
                      className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-400 transition"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
                      Survey Area
                    </label>
                    <input
                      value={form.survey_area}
                      onChange={(e) => setForm((f) => ({ ...f, survey_area: e.target.value }))}
                      placeholder="e.g. Sector A & B"
                      className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-400 transition"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-[10px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">
                    Notes / Descriptions
                  </label>
                  <textarea
                    value={form.notes}
                    onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
                    rows={2}
                    placeholder="Provide background information on this survey effort…"
                    className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-400 transition"
                  />
                </div>
              </div>

              {/* SpeciesNet Survey Location */}
              <div className="space-y-3 pt-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1">
                  <span className="material-symbols-outlined text-sm select-none">my_location</span>
                  SpeciesNet Survey Location
                </h3>
                <p className="text-[11px] text-slate-400 dark:text-slate-500">
                  Geographic prior applied to AI species classification. Set to the centre of your survey area for improved accuracy.
                </p>
                <div className="grid grid-cols-3 gap-3 bg-slate-50 dark:bg-slate-950/40 border border-slate-200 dark:border-slate-850 p-4 rounded-xl">
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">Latitude</label>
                    <input
                      type="number"
                      step="0.001"
                      min={-90}
                      max={90}
                      value={coords.speciesnet_lat}
                      onChange={(e) => setCoords((c) => ({ ...c, speciesnet_lat: parseFloat(e.target.value) || 0 }))}
                      className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">Longitude</label>
                    <input
                      type="number"
                      step="0.001"
                      min={-180}
                      max={180}
                      value={coords.speciesnet_lng}
                      onChange={(e) => setCoords((c) => ({ ...c, speciesnet_lng: parseFloat(e.target.value) || 0 }))}
                      className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1">Country Code</label>
                    <input
                      type="text"
                      maxLength={3}
                      value={coords.speciesnet_country}
                      onChange={(e) => setCoords((c) => ({ ...c, speciesnet_country: e.target.value.toUpperCase() }))}
                      placeholder="KEN"
                      className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition font-mono"
                    />
                  </div>
                </div>
              </div>

              {/* Threshold Sliders */}
              <div className="space-y-4 pt-2">
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1">
                    <span className="material-symbols-outlined text-sm select-none">tune</span>
                    Ecological & Quality Thresholds
                  </h3>
                  <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-1">
                    Independence Window and Min Reporting Confidence are applied to the live pipeline when saved.
                  </p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4 bg-slate-50 dark:bg-slate-950/40 border border-slate-200 dark:border-slate-850 p-4 rounded-xl">
                  {Object.entries(THRESHOLD_DEFS).map(([key, def]) => {
                    const val = thresholds[key] !== undefined ? thresholds[key] : 0;
                    return (
                      <div key={key} className="space-y-1">
                        <div className="flex justify-between text-xs text-slate-600 dark:text-slate-400">
                          <span className="font-semibold">{def.label}</span>
                          <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
                            {val} <span className="text-[10px] text-slate-400 font-normal">{def.unit}</span>
                          </span>
                        </div>
                        <input
                          type="range"
                          min={def.min}
                          max={def.max}
                          step={def.step}
                          value={val}
                          onChange={(e) => setThresholds((prev) => ({ ...prev, [key]: Number(e.target.value) }))}
                          className="w-full accent-emerald-500 dark:accent-emerald-400 bg-slate-200 dark:bg-slate-800 cursor-pointer h-1.5 rounded-lg appearance-none"
                        />
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Save Details Button */}
              <div className="flex gap-3 items-center pt-2 border-t border-slate-200/60 dark:border-slate-800/60">
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="px-5 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg shadow-sm hover:shadow transition disabled:bg-slate-300 dark:disabled:bg-slate-800 cursor-pointer"
                >
                  {saving ? "Saving…" : "Save Project Settings"}
                </button>
                {saved && (
                  <span className="text-emerald-600 dark:text-emerald-400 text-xs font-semibold animate-pulse flex items-center gap-1">
                    <span className="material-symbols-outlined text-sm select-none font-bold">check_circle</span>
                    Saved successfully
                  </span>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 text-center text-slate-400 dark:text-slate-500 shadow-sm">
              No active project selected. Switch to or create a project.
            </div>
          )}

          {/* Configuration Schema Preview */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 space-y-3 shadow-sm">
            <h2 className="font-bold text-base text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
              <span className="material-symbols-outlined text-slate-500 text-lg select-none">data_object</span>
              Active Configuration Schema
            </h2>
            <pre className="text-xs bg-slate-50 dark:bg-slate-950/60 rounded-xl border border-slate-100 dark:border-slate-850 p-4 overflow-x-auto max-h-40 text-slate-700 dark:text-slate-400 font-mono">
              {JSON.stringify(activeProject, null, 2)}
            </pre>
          </div>

          {/* Admin Actions */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 space-y-4 shadow-sm">
            <h2 className="font-bold text-base text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
              <span className="material-symbols-outlined text-slate-500 text-lg select-none">admin_panel_settings</span>
              Administrative Actions
            </h2>
            <div className="flex flex-wrap gap-3">
              <button
                onClick={handleLockBaseline}
                className="px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white text-xs font-semibold rounded-lg shadow-sm hover:shadow transition cursor-pointer flex items-center gap-1"
              >
                <span className="material-symbols-outlined text-sm select-none">lock</span>
                Lock Reference Baseline
              </button>
              <a
                href="/api/project/export"
                className="px-4 py-2 bg-slate-700 hover:bg-slate-800 text-white text-xs font-semibold rounded-lg shadow-sm hover:shadow transition cursor-pointer flex items-center gap-1"
              >
                <span className="material-symbols-outlined text-sm select-none">download</span>
                Export Config JSON
              </a>
              <button
                onClick={handleResetDb}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded-lg shadow-sm hover:shadow transition cursor-pointer ml-auto flex items-center gap-1"
              >
                <span className="material-symbols-outlined text-sm select-none">delete_forever</span>
                Wipe & Reset Database
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

