import { useEffect, useMemo, useState } from "react";
import { getHistory, clearHistory, storedThumbUrl, storedThumbUrlById } from "../api/client";
import {
  buildImageGroupsFromRows,
  confidenceOf,
  imageKey,
  primaryDetectionRow,
  detectionSummaryOfGroup,
  speciesSummaryOfGroup,
  type Row,
} from "../utils/imageIdentity";
import { filterVisibleRows, isWildlifeRow } from "../utils/wildlifeFilter";
import { useDisplayStore } from "../store/displayStore";
import { HiddenNonWildlifeBanner, ShowNonWildlifeToggle } from "../components/HiddenNonWildlifeBanner";

export default function History() {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmClear, setConfirmClear] = useState(false);
  const hideNonWildlife = useDisplayStore((s) => s.hideNonWildlife);

  const load = () => getHistory().then(setRows).finally(() => setLoading(false));
  useEffect(() => { load(); }, []);

  const handleClear = async () => {
    await clearHistory();
    setConfirmClear(false);
    load();
  };

  const visibleRows = useMemo(
    () => filterVisibleRows(rows, hideNonWildlife),
    [rows, hideNonWildlife],
  );

  const imageGroups = useMemo(
    () => buildImageGroupsFromRows(visibleRows),
    [visibleRows],
  );

  const uniqueImages = new Set(visibleRows.map(imageKey)).size;
  const uniqueSpecies = new Set(
    visibleRows.filter(isWildlifeRow).map((r) => r.detected_animal)
  ).size;

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Analysis History</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            One row per image — multi-animal frames show all species together.
          </p>
        </div>
        <div className="flex gap-2 flex-wrap items-center">
          <ShowNonWildlifeToggle />
          <a
            href="/api/history/export/csv"
            className="px-4 py-2 bg-slate-700 hover:bg-slate-800 text-white text-sm font-semibold rounded-lg shadow-sm hover:shadow transition"
            title="CSV export includes blank, person, and vehicle records"
          >
            Export CSV
          </a>
          {!confirmClear ? (
            <button
              onClick={() => setConfirmClear(true)}
              className="px-4 py-2 bg-red-50 hover:bg-red-100 dark:bg-red-950/20 dark:hover:bg-red-950/40 text-red-650 dark:text-red-400 text-sm font-semibold rounded-lg border border-red-200 dark:border-red-900/35 transition cursor-pointer"
            >
              Clear Logs
            </button>
          ) : (
            <div className="flex gap-2.5 items-center bg-red-50 dark:bg-red-950/20 border border-red-150 dark:border-red-900/30 rounded-lg px-3 py-1.5">
              <span className="text-xs font-semibold text-red-700 dark:text-red-400">Are you sure?</span>
              <button
                onClick={handleClear}
                className="px-2.5 py-1 bg-red-600 hover:bg-red-700 text-white text-xs font-bold rounded cursor-pointer"
              >
                Yes, clear
              </button>
              <button
                onClick={() => setConfirmClear(false)}
                className="px-2.5 py-1 bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 text-xs font-bold rounded cursor-pointer"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm">
          <p className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Total Images</p>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white mt-1.5">{uniqueImages}</p>
        </div>
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm">
          <p className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Total Detections</p>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white mt-1.5">{visibleRows.length}</p>
        </div>
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm">
          <p className="text-xs font-semibold text-slate-400 dark:text-slate-500 uppercase tracking-wider">Unique Species</p>
          <p className="text-3xl font-extrabold text-slate-900 dark:text-white mt-1.5">{uniqueSpecies}</p>
        </div>
      </div>

      {!loading && <HiddenNonWildlifeBanner rows={rows} />}

      {loading ? (
        <div className="text-center py-12 text-slate-400 dark:text-slate-550 animate-pulse">
          Loading history log list…
        </div>
      ) : imageGroups.length === 0 ? (
        <div className="text-center py-16 text-slate-400 dark:text-slate-550 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-850 rounded-xl">
          {rows.length === 0
            ? "No analysis history found. Process and save images to build your catalog."
            : "No wildlife detections to show. Enable “Show blank / person / vehicle” to view non-wildlife records."}
        </div>
      ) : (
        <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 dark:bg-slate-950/40 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  {["Image", "filename", "station_id", "species", "detection summary", "detections", "confidence", "day_night", "processed_at", "user_notes"].map((c) => (
                    <th
                      key={c}
                      className="text-left px-4 py-3 font-semibold text-slate-600 dark:text-slate-450 uppercase tracking-wider text-[11px]"
                    >
                      {c.replace(/_/g, " ")}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {imageGroups.map((group) => {
                  const row = primaryDetectionRow(group.rows);
                  const multi = group.rows.length > 1;
                  const speciesSummary = speciesSummaryOfGroup(group.rows);
                  const detectionSummary = detectionSummaryOfGroup(group.rows);
                  const confSummary = multi
                    ? group.rows.map((r) => `${r.detected_animal}: ${Math.round(confidenceOf(r) * 100)}%`).join(", ")
                    : `${Math.round(confidenceOf(row) * 100)}%`;

                  return (
                    <tr key={group.imageKey} className="hover:bg-slate-50/50 dark:hover:bg-slate-950/20 transition-colors">
                      <td className="px-4 py-3">
                        <div className="relative w-14 h-11 rounded overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-800">
                          <img
                            src={group.imageId ? storedThumbUrlById(group.imageId, 160) : storedThumbUrl(group.filename, 160)}
                            alt=""
                            className="w-full h-full object-cover"
                            onError={(e) => { (e.target as HTMLElement).style.display = "none"; }}
                          />
                          {multi && (
                            <span className="absolute bottom-0.5 right-0.5 bg-black/70 text-white text-[9px] font-bold px-1 rounded">
                              {group.rows.length}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-slate-750 dark:text-slate-300 max-w-[180px] truncate" title={group.filename}>
                        {group.filename}
                      </td>
                      <td className="px-4 py-3 text-slate-750 dark:text-slate-300">{String(row.station_id ?? "")}</td>
                      <td className="px-4 py-3 text-slate-750 dark:text-slate-300 max-w-[260px] text-xs whitespace-pre-line leading-snug" title={speciesSummary}>
                        {speciesSummary}
                      </td>
                      <td className="px-4 py-3 text-slate-750 dark:text-slate-300 max-w-[320px] text-xs whitespace-pre-line leading-snug" title={detectionSummary}>
                        {detectionSummary}
                      </td>
                      <td className="px-4 py-3 text-slate-750 dark:text-slate-300">{group.rows.length}</td>
                      <td className="px-4 py-3 text-slate-750 dark:text-slate-300 text-xs font-mono" title={confSummary}>
                        {confSummary}
                      </td>
                      <td className="px-4 py-3 text-slate-750 dark:text-slate-300">{String(row.day_night ?? "")}</td>
                      <td className="px-4 py-3 text-slate-750 dark:text-slate-300 whitespace-nowrap">{String(row.processed_at ?? "")}</td>
                      <td className="px-4 py-3 text-slate-750 dark:text-slate-300 max-w-[200px] truncate">{String(row.user_notes ?? "")}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="px-4 py-2.5 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-400 font-medium">
            {imageGroups.length} image(s) · {visibleRows.length} detection(s)
          </div>
        </div>
      )}
    </div>
  );
}
