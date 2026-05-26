import { useEffect, useState } from "react";
import { getSpecies, lookupSpecies, resolveSynonym } from "../api/client";

type Row = Record<string, unknown>;

export default function SpeciesLibrary() {
  const [species, setSpecies] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [lookupQuery, setLookupQuery] = useState("");
  const [lookupResult, setLookupResult] = useState<Row | null>(null);
  const [lookupError, setLookupError] = useState("");
  const [synonymQuery, setSynonymQuery] = useState("");
  const [synonymResult, setSynonymResult] = useState<Row | null>(null);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    getSpecies().then(setSpecies).finally(() => setLoading(false));
  }, []);

  const handleLookup = async () => {
    if (!lookupQuery) return;
    setLookupError("");
    setLookupResult(null);
    try {
      const result = await lookupSpecies(lookupQuery);
      setLookupResult(result);
    } catch {
      setLookupError(`Species "${lookupQuery}" not found in library.`);
    }
  };

  const handleSynonym = async () => {
    if (!synonymQuery) return;
    const result = await resolveSynonym(synonymQuery);
    setSynonymResult(result);
  };

  const filtered = species.filter((s) =>
    !filter || JSON.stringify(s).toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-slate-800">Species Reference Library</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Quick lookup */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-3">
          <h2 className="font-semibold text-slate-700">Quick Species Lookup</h2>
          <div className="flex gap-2">
            <input
              placeholder="Species name…"
              value={lookupQuery}
              onChange={(e) => setLookupQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLookup()}
              className="flex-1 border border-slate-300 rounded px-3 py-2 text-sm"
            />
            <button onClick={handleLookup} className="px-4 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-700">Look up</button>
          </div>
          {lookupError && <p className="text-red-600 text-sm">{lookupError}</p>}
          {lookupResult && (
            <pre className="text-xs bg-slate-50 rounded p-3 overflow-x-auto">{JSON.stringify(lookupResult, null, 2)}</pre>
          )}
        </div>

        {/* Synonym resolver */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-3">
          <h2 className="font-semibold text-slate-700">Synonym Resolver</h2>
          <div className="flex gap-2">
            <input
              placeholder="Common or alternate name…"
              value={synonymQuery}
              onChange={(e) => setSynonymQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSynonym()}
              className="flex-1 border border-slate-300 rounded px-3 py-2 text-sm"
            />
            <button onClick={handleSynonym} className="px-4 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-700">Resolve</button>
          </div>
          {synonymResult && (
            <div className="bg-slate-50 rounded p-3 text-sm">
              <span className="text-slate-500">{String(synonymResult.input ?? "")}</span>
              {" → "}
              <span className="font-semibold text-slate-800">{String(synonymResult.canonical ?? "Not found")}</span>
            </div>
          )}
        </div>
      </div>

      {/* Full library table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-100 flex items-center gap-3">
          <span className="font-semibold text-slate-700">All Species ({species.length})</span>
          <input
            placeholder="Filter…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="ml-auto border border-slate-300 rounded px-3 py-1.5 text-sm"
          />
        </div>
        {loading ? (
          <div className="text-center py-12 text-slate-400">Loading…</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <tbody className="divide-y divide-slate-100">
                {filtered.map((s, i) => (
                  <tr key={i} className="hover:bg-slate-50">
                    <td className="px-4 py-2 font-medium text-slate-700">{String(s.name ?? s.scientific_name ?? String(Object.values(s)[0] ?? ""))}</td>
                    <td className="px-4 py-2 text-slate-500">{String(s.common_name ?? s.family ?? "")}</td>
                    <td className="px-4 py-2 text-slate-400 text-xs">{String(s.order ?? s.category ?? "")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filtered.length === 0 && <div className="text-center py-8 text-slate-400">No species match your filter.</div>}
          </div>
        )}
      </div>
    </div>
  );
}
