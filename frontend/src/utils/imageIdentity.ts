/** Shared helpers for canonical image identity and grouping across Results, History, ReviewQueue. */

export type Row = Record<string, unknown>;

export type ImageGroup = {
  imageKey: string;
  imageId: number;
  fileHash?: string;
  filename: string;
  rows: Row[];
  focusDetectionId?: number;
};

export function imageIdOf(r: Row): number {
  const explicit = Number(r.image_id ?? 0);
  if (explicit) return explicit;
  return Number(r.id ?? 0);
}

export function detIdOf(r: Row): number {
  return Number(r.detection_id ?? 0);
}

export function fileHashOf(r: Row): string {
  return String(r.file_hash ?? "").trim();
}

/** Stable key for grouping/counting unique physical images. Prefer DB PK over hash. */
export function imageKey(r: Row): string {
  const id = imageIdOf(r);
  if (id) return `id:${id}`;
  const hash = fileHashOf(r);
  if (hash) return `hash:${hash}`;
  return `name:${String(r.filename ?? "")}`;
}

export function confidenceOf(r: Row): number {
  const v = typeof r.detection_confidence === "number"
    ? (r.detection_confidence as number)
    : parseFloat(String(r.detection_confidence ?? "0"));
  return isNaN(v) ? 0 : v;
}

/** Highest-confidence detection — used for summary columns when one row represents the image. */
export function primaryDetectionRow(groupRows: Row[]): Row {
  return groupRows.reduce((best, r) => (
    confidenceOf(r) > confidenceOf(best) ? r : best
  ), groupRows[0]);
}

/** Lowest-confidence detection — useful for review queue default focus. */
export function reviewFocusRow(groupRows: Row[]): Row {
  return groupRows.reduce((best, r) => (
    confidenceOf(r) < confidenceOf(best) ? r : best
  ), groupRows[0]);
}

export function detectionIdsOfGroup(groupRows: Row[]): number[] {
  return groupRows.map((r) => detIdOf(r)).filter((id) => id > 0);
}

export function speciesDisplayName(r: Row): string {
  const species = String(r.detected_animal ?? "Unknown");
  const sci = String(r.scientific_name ?? "").trim();
  if (sci && sci.toLowerCase() !== species.toLowerCase()) {
    return `${species} (${sci})`;
  }
  return species;
}

function pctOf(value: unknown): number | null {
  if (typeof value !== "number" || isNaN(value)) return null;
  return Math.round(value * 100);
}

export function speciesListLine(r: Row, index = 1): string {
  const id = detIdOf(r);
  const label = id ? `#${id}` : `#${index}`;
  const conf = Math.round(confidenceOf(r) * 100);
  const md = pctOf(r.md_confidence);
  const sn = pctOf(r.speciesnet_confidence);
  const bits = [`${label} ${speciesDisplayName(r)}`, `${conf}%`];
  if (md != null) bits.push(`MD ${md}%`);
  if (sn != null) bits.push(`SN ${sn}%`);
  const method = String(r.detection_method ?? "").trim();
  if (method) bits.push(method);
  return bits.join(" · ");
}

export function detectionSummaryLine(r: Row, index = 1): string {
  const id = detIdOf(r);
  const label = id ? `#${id}` : `#${index}`;
  const conf = Math.round(confidenceOf(r) * 100);
  const md = pctOf(r.md_confidence);
  const sn = pctOf(r.speciesnet_confidence);
  const confBits = [`${conf}%`];
  if (md != null) confBits.push(`MD ${md}%`);
  if (sn != null) confBits.push(`SN ${sn}%`);
  const method = String(r.detection_method ?? "").trim();
  const ide = String(r.ide_id ?? "").trim();
  const bits = [
    `${index}. [${label}] ${speciesDisplayName(r)}`,
    `confidence ${confBits.join(" / ")}`,
  ];
  if (method) bits.push(`method ${method}`);
  if (ide) bits.push(`IDE ${ide}`);
  return bits.join(" — ");
}

/** Rich species list — one line per detection, aligned with export Species List column. */
export function speciesSummaryOfGroup(groupRows: Row[]): string {
  return groupRows.map((r, i) => speciesListLine(r, i + 1)).join("\n");
}

/** Rich detection summary — aligned with export Detections Summary column. */
export function detectionSummaryOfGroup(groupRows: Row[]): string {
  return groupRows.map((r, i) => detectionSummaryLine(r, i + 1)).join("\n");
}

export function buildImageGroupsFromRows(groupRows: Row[]): ImageGroup[] {
  const map = new Map<string, Row[]>();
  for (const row of groupRows) {
    const key = imageKey(row);
    if (!map.has(key)) map.set(key, []);
    map.get(key)!.push(row);
  }
  return Array.from(map.entries()).map(([key, rowsForImage]) => ({
    imageKey: key,
    imageId: imageIdOf(rowsForImage[0]),
    fileHash: fileHashOf(rowsForImage[0]) || undefined,
    filename: String(rowsForImage[0].filename ?? ""),
    rows: rowsForImage,
  }));
}

export function buildImageGroup(
  allRows: Row[],
  key: string,
  focusDetectionId?: number,
): ImageGroup | null {
  const rowsForImage = allRows.filter((r) => imageKey(r) === key);
  if (!rowsForImage.length) return null;
  return {
    imageKey: key,
    imageId: imageIdOf(rowsForImage[0]),
    fileHash: fileHashOf(rowsForImage[0]) || undefined,
    filename: String(rowsForImage[0].filename ?? ""),
    rows: rowsForImage,
    focusDetectionId,
  };
}

export function rowsMatchGroup(a: Row[], b: Row[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((r, i) => {
    const o = b[i];
    return (
      detIdOf(r) === detIdOf(o)
      && String(r.detected_animal ?? "") === String(o.detected_animal ?? "")
      && String(r.bbox ?? "") === String(o.bbox ?? "")
      && String(r.user_notes ?? "") === String(o.user_notes ?? "")
      && String(r.station_id ?? "") === String(o.station_id ?? "")
      && String(r.scientific_name ?? "") === String(o.scientific_name ?? "")
    );
  });
}
