import { countNonWildlifeRows, summarizeNonWildlife } from "../utils/wildlifeFilter";
import type { Row } from "../utils/imageIdentity";
import { useDisplayStore } from "../store/displayStore";

function describeHidden(count: number, summary: ReturnType<typeof summarizeNonWildlife>): string {
  if (count === 0) return "";
  const parts: string[] = [];
  if (summary.empty) parts.push(`${summary.empty} blank`);
  if (summary.person) parts.push(`${summary.person} person`);
  if (summary.vehicle) parts.push(`${summary.vehicle} vehicle`);
  if (summary.other) parts.push(`${summary.other} other`);
  const detail = parts.length ? ` (${parts.join(", ")})` : "";
  return `${count} non-wildlife detection${count !== 1 ? "s" : ""}${detail}`;
}

export function HiddenNonWildlifeBanner({
  rows,
  className = "",
}: {
  rows: Row[];
  className?: string;
}) {
  const hideNonWildlife = useDisplayStore((s) => s.hideNonWildlife);
  const showNonWildlife = useDisplayStore((s) => s.showNonWildlife);

  if (!hideNonWildlife) return null;

  const count = countNonWildlifeRows(rows);
  if (count === 0) return null;

  const summary = summarizeNonWildlife(rows);

  return (
    <div className={`flex items-center gap-2.5 px-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 text-sm text-slate-500 dark:text-slate-400 ${className}`}>
      <span className="material-symbols-outlined text-base select-none text-slate-400 shrink-0">hide_image</span>
      <span className="flex-1">
        <span className="font-semibold text-slate-600 dark:text-slate-300">{describeHidden(count, summary)}</span> hidden.
        {" "}Exports still include all records.
      </span>
      <button
        onClick={showNonWildlife}
        className="shrink-0 text-xs font-semibold text-emerald-600 dark:text-emerald-400 hover:underline cursor-pointer"
      >
        Show all
      </button>
    </div>
  );
}

export function ShowNonWildlifeToggle({ compact = false }: { compact?: boolean }) {
  const hideNonWildlife = useDisplayStore((s) => s.hideNonWildlife);
  const setHideNonWildlife = useDisplayStore((s) => s.setHideNonWildlife);

  return (
    <label
      className={`flex items-center gap-2 cursor-pointer select-none ${
        compact ? "text-[11px]" : "text-xs"
      } text-slate-600 dark:text-slate-400`}
      title="Blank frames, people, and vehicles are hidden by default. Exports always include every record."
    >
      <input
        type="checkbox"
        checked={!hideNonWildlife}
        onChange={(e) => setHideNonWildlife(!e.target.checked)}
        className="rounded border-slate-300 dark:border-slate-700 text-emerald-600 focus:ring-emerald-400 bg-white dark:bg-slate-950 cursor-pointer"
      />
      <span>Show blank / person / vehicle</span>
    </label>
  );
}
