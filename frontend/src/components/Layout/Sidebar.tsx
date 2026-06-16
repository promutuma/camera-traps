import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { useConfigStore } from "../../store/configStore";

const NAV_GROUPS = [
  {
    title: "Core Workflow",
    items: [
      { path: "/", label: "Upload & Process", icon: "upload_file" },
      { path: "/results", label: "Review Results", icon: "preview" },
      { path: "/review-queue", label: "Review Queue", icon: "rate_review" },
      { path: "/diagnostics", label: "Diagnostics", icon: "biotech" },
    ],
  },
  {
    title: "Ecological Analytics",
    items: [
      { path: "/statistics", label: "Statistics", icon: "bar_chart" },
      { path: "/ecological", label: "Ecological", icon: "forest" },
      { path: "/corridor", label: "Corridor Analysis", icon: "insights" },
      { path: "/history", label: "History & Logs", icon: "history" },
    ],
  },
  {
    title: "Data & Assets",
    items: [
      { path: "/stations", label: "Stations", icon: "pin_drop" },
      { path: "/community", label: "Community", icon: "group" },
      { path: "/spatial", label: "Spatial & Map", icon: "map" },
      { path: "/species", label: "Species Library", icon: "menu_book" },
      { path: "/storage", label: "Storage Management", icon: "storage" },
    ],
  },
  {
    title: "Project Config",
    items: [
      { path: "/project", label: "Project Config", icon: "settings" },
      { path: "/arcgis", label: "ArcGIS Sync", icon: "sync" },
    ],
  },
];

export default function Sidebar() {
  const { config, fetch, patch } = useConfigStore();
  const [activeTab, setActiveTab] = useState<"nav" | "config">("nav");
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    return (localStorage.getItem("color-scheme") as "light" | "dark") || 
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  });

  useEffect(() => {
    fetch();
    
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = (e: MediaQueryListEvent) => {
      if (!localStorage.getItem("color-scheme")) {
        const nextTheme = e.matches ? "dark" : "light";
        setTheme(nextTheme);
        if (nextTheme === "dark") {
          document.documentElement.classList.add("dark");
        } else {
          document.documentElement.classList.remove("dark");
        }
      }
    };
    media.addEventListener("change", handler);
    return () => media.removeEventListener("change", handler);
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "light" ? "dark" : "light";
    setTheme(nextTheme);
    localStorage.setItem("color-scheme", nextTheme);
    if (nextTheme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  if (!config) return <div className="p-4 text-sm text-slate-400">Loading config…</div>;

  const toggle = (key: string, val: boolean) => patch({ [key]: val });
  const num = (key: string, val: number) => patch({ [key]: val });
  const str = (key: string, val: string) => patch({ [key]: val });

  return (
    <aside className="w-72 h-full bg-slate-900 text-slate-100 flex flex-col border-r border-slate-800 shrink-0">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800 flex items-center gap-3 bg-slate-950/40">
        <span className="material-symbols-outlined text-emerald-400 text-3xl select-none">
          nature_people
        </span>
        <div>
          <h1 className="text-md font-bold tracking-tight text-white">ViumbeLens</h1>
          <p className="text-xs text-emerald-500 font-medium">Camera Trap Auto-Analyzer</p>
        </div>
      </div>

      {/* Segmented Control */}
      <div className="p-3 border-b border-slate-800">
        <div className="flex bg-slate-950/60 p-1 rounded-lg border border-slate-800/80">
          <button
            onClick={() => setActiveTab("nav")}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center justify-center gap-1.5 ${
              activeTab === "nav"
                ? "bg-emerald-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span className="material-symbols-outlined text-sm select-none">explore</span>
            Navigation
          </button>
          <button
            onClick={() => setActiveTab("config")}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center justify-center gap-1.5 ${
              activeTab === "config"
                ? "bg-emerald-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <span className="material-symbols-outlined text-sm select-none">tune</span>
            Settings
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6">
        {activeTab === "nav" ? (
          <div className="space-y-5">
            {NAV_GROUPS.map((group) => (
              <div key={group.title} className="space-y-1">
                <h3 className="px-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  {group.title}
                </h3>
                <div className="space-y-0.5">
                  {group.items.map((item) => (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      end={item.path === "/"}
                      className={({ isActive }) =>
                        `flex items-center gap-3 px-3 py-2 text-sm font-medium rounded-lg transition-all ${
                          isActive
                            ? "bg-emerald-950/60 text-emerald-400 border-l-2 border-emerald-500 shadow-inner"
                            : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                        }`
                      }
                    >
                      <span className="material-symbols-outlined text-lg select-none">
                        {item.icon}
                      </span>
                      <span>{item.label}</span>
                    </NavLink>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-5">
            {/* Processing Options */}
            <section className="space-y-2">
              <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                Pipeline Processing
              </h3>
              <div className="space-y-2.5 bg-slate-950/30 p-3 rounded-lg border border-slate-800/50">
                {(
                  [
                    ["enable_ocr", "OCR Metadata Extraction"],
                    ["enable_detection", "Animal Detection"],
                    ["enable_day_night", "Day/Night Classification"],
                    ["enable_scrubbing", "Auto-Scrub Person/Vehicle"],
                    ["enable_low_spec", "Low-Spec (Low-Memory)"],
                  ] as [keyof typeof config, string][]
                ).map(([key, label]) => (
                  <label key={key} className="flex items-center gap-2.5 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={config[key] as boolean}
                      onChange={(e) => toggle(key, e.target.checked)}
                      className="rounded border-slate-700 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                    />
                    <span className="text-xs text-slate-300 font-medium">{label}</span>
                  </label>
                ))}
              </div>
            </section>

            {/* Advanced Settings */}
            <section className="space-y-2.5">
              <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                Advanced ML Settings
              </h3>
              <div className="space-y-4 bg-slate-950/30 p-3 rounded-lg border border-slate-800/50">
                <Slider
                  label="Detection Confidence"
                  value={config.detection_confidence}
                  min={0.05}
                  max={1}
                  step={0.05}
                  format={(v) => v.toFixed(2)}
                  onChange={(v) => num("detection_confidence", v)}
                />
                <Slider
                  label="Day/Night Brightness"
                  value={config.brightness_threshold}
                  min={0}
                  max={255}
                  step={5}
                  onChange={(v) => num("brightness_threshold", v)}
                />
                <Slider
                  label="Metadata Strip Height"
                  value={config.ocr_strip_height}
                  min={0.05}
                  max={0.3}
                  step={0.01}
                  format={(v) => (v * 100).toFixed(0) + "%"}
                  onChange={(v) => num("ocr_strip_height", v)}
                />
              </div>
            </section>

            {/* Classifier Fusion */}
            <section className="space-y-2.5">
              <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                Classifier Fusion
              </h3>
              <div className="space-y-4 bg-slate-950/30 p-3 rounded-lg border border-slate-800/50">
                <Slider
                  label="BioClip Weight"
                  value={config.bioclip_weight ?? 0.05}
                  min={0}
                  max={1}
                  step={0.05}
                  format={(v) => v.toFixed(2)}
                  onChange={(v) => num("bioclip_weight", v)}
                />
                <Slider
                  label="SpeciesNet Weight"
                  value={config.speciesnet_weight ?? 0.95}
                  min={0}
                  max={1}
                  step={0.05}
                  format={(v) => v.toFixed(2)}
                  onChange={(v) => num("speciesnet_weight", v)}
                />
                <Slider
                  label="SpeciesNet Bypass Threshold"
                  value={config.speciesnet_bypass_threshold ?? 0.60}
                  min={0}
                  max={1}
                  step={0.05}
                  format={(v) => v === 0 ? "Off" : v.toFixed(2)}
                  onChange={(v) => num("speciesnet_bypass_threshold", v)}
                />
                <p className="text-[10px] text-slate-500 leading-relaxed">
                  Bypass: skip BioClip fusion when SpeciesNet top confidence ≥ threshold. Set to 0 to always fuse.
                </p>
              </div>
            </section>

            {/* Privacy & Review */}
            <section className="space-y-2.5">
              <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                Privacy & Review
              </h3>
              <div className="space-y-4 bg-slate-950/30 p-3 rounded-lg border border-slate-800/50">
                <Slider
                  label="Blur Strength"
                  value={config.blur_strength}
                  min={11}
                  max={101}
                  step={10}
                  onChange={(v) => num("blur_strength", v)}
                />
                <Slider
                  label="Review Queue Threshold"
                  value={config.review_confidence_threshold}
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  format={(v) => v.toFixed(2)}
                  onChange={(v) => num("review_confidence_threshold", v)}
                />
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">
                    Reviewer ID
                  </label>
                  <input
                    type="text"
                    value={config.reviewer_id}
                    onChange={(e) => str("reviewer_id", e.target.value)}
                    className="block w-full rounded border border-slate-700 bg-slate-800/50 text-white px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
                  />
                </div>
              </div>
            </section>

            {/* Station Settings */}
            <section className="space-y-2.5">
              <h3 className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                Station Defaults
              </h3>
              <div className="space-y-4 bg-slate-950/30 p-3 rounded-lg border border-slate-800/50">
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">
                    Default Station ID
                  </label>
                  <input
                    type="text"
                    value={config.default_station_id}
                    onChange={(e) => str("default_station_id", e.target.value)}
                    className="block w-full rounded border border-slate-700 bg-slate-800/50 text-white px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
                  />
                </div>
                <Slider
                  label="Independence Window"
                  value={config.independence_window}
                  min={5}
                  max={120}
                  step={5}
                  format={(v) => `${v} min`}
                  onChange={(v) => num("independence_window", v)}
                />
                <div>
                  <label className="block text-xs font-semibold text-slate-400 mb-1">
                    Default Trap Nights
                  </label>
                  <input
                    type="number"
                    min={1}
                    value={config.trap_nights_default}
                    onChange={(e) => num("trap_nights_default", Number(e.target.value))}
                    className="block w-full rounded border border-slate-700 bg-slate-800/50 text-white px-2.5 py-1.5 text-xs focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
                  />
                </div>
              </div>
            </section>
          </div>
        )}
      </div>

      <div className="p-4 border-t border-slate-800 flex items-center justify-between text-[10px] text-slate-500 bg-slate-950/20">
        <span>Changes apply dynamically</span>
        <button
          onClick={toggleTheme}
          className="flex items-center gap-1 px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition cursor-pointer select-none"
          title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
        >
          <span className="material-symbols-outlined text-[14px] leading-none select-none">
            {theme === "light" ? "dark_mode" : "light_mode"}
          </span>
          <span className="capitalize text-[10px] font-semibold">{theme === "light" ? "dark" : "light"}</span>
        </button>
      </div>
    </aside>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  format = (v) => String(v),
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-400 mb-1.5">
        <span className="font-medium">{label}</span>
        <span className="font-mono text-emerald-400 font-semibold">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-emerald-500 bg-slate-800 cursor-pointer h-1.5 rounded-lg appearance-none"
      />
    </div>
  );
}
