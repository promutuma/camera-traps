/** Shared helpers for blank / person / vehicle / error detection rows. */

import type { Row } from "./imageIdentity";

/** Labels that are not wildlife species (case-insensitive). */
export const NON_WILDLIFE_LABELS = new Set([
  "empty",
  "blank",
  "person",
  "vehicle",
  "human",
  "error",
  "unidentified",
  "n/a",
  "none",
  "unknown",
]);


export function detectedLabelOf(row: Row): string {
  return String(row.detected_animal ?? "").trim();
}

export function isNonWildlifeLabel(label: string): boolean {
  const lower = label.trim().toLowerCase();
  if (!lower) return true;
  if (NON_WILDLIFE_LABELS.has(lower)) return true;
  if (lower.includes("person") || lower.includes("vehicle") || lower.includes("human")) return true;
  return false;
}

export function isNonWildlifeRow(row: Row): boolean {
  return isNonWildlifeLabel(detectedLabelOf(row));
}

export function isWildlifeRow(row: Row): boolean {
  return !isNonWildlifeRow(row);
}

export function isWildlifeLabel(label: string): boolean {
  return !isNonWildlifeLabel(label);
}

export function filterVisibleRows<T extends Row>(rows: T[], hideNonWildlife: boolean): T[] {
  if (!hideNonWildlife) return rows;
  return rows.filter((r) => !isNonWildlifeRow(r));
}

export function countNonWildlifeRows(rows: Row[]): number {
  return rows.filter(isNonWildlifeRow).length;
}

export function summarizeNonWildlife(rows: Row[]): { empty: number; person: number; vehicle: number; other: number } {
  const summary = { empty: 0, person: 0, vehicle: 0, other: 0 };
  for (const row of rows) {
    if (!isNonWildlifeRow(row)) continue;
    const lower = detectedLabelOf(row).toLowerCase();
    if (lower === "empty") summary.empty += 1;
    else if (lower === "person" || lower === "human" || lower.includes("person")) summary.person += 1;
    else if (lower === "vehicle" || lower.includes("vehicle")) summary.vehicle += 1;
    else summary.other += 1;
  }
  return summary;
}
