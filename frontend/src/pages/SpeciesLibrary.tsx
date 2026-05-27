import { useEffect, useState } from "react";
import { getSpecies, lookupSpecies, resolveSynonym } from "../api/client";

type Row = Record<string, unknown>;

function getTaxonIcon(className?: string): string {
  const c = String(className ?? "").toLowerCase();
  if (c.includes("mammal")) return "🦁";
  if (c.includes("aves") || c.includes("bird")) return "🦅";
  if (c.includes("reptil") || c.includes("snake") || c.includes("lizard")) return "🦎";
  if (c.includes("amphib") || c.includes("frog")) return "🐸";
  if (c.includes("insect") || c.includes("bug") || c.includes("arach")) return "🕷️";
  return "🐾";
}

export default function SpeciesLibrary() {
  const [species, setSpecies] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  
  const [lookupQuery, setLookupQuery] = useState("");
  const [lookupResult, setLookupResult] = useState<Row | null>(null);
  const [lookupError, setLookupError] = useState("");
  
  const [synonymQuery, setSynonymQuery] = useState("");
  const [synonymResult, setSynonymResult] = useState<Row | null>(null);
  
  const [filter, setFilter] = useState("");
  const [selectedSp, setSelectedSp] = useState<Row | null>(null);

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
      setSelectedSp(result); // Also open in detailed drawer
    } catch {
      setLookupError(`Species "${lookupQuery}" not found in library.`);
    }
  };

  const handleSynonym = async () => {
    if (!synonymQuery) return;
    const result = await resolveSynonym(synonymQuery);
    setSynonymResult(result);
  };

  const filtered = species.filter((s) => {
    const term = filter.toLowerCase();
    if (!term) return true;
    const name = String(s.name ?? s.scientific_name ?? "").toLowerCase();
    const common = String(s.common_name ?? "").toLowerCase();
    const order = String(s.order ?? "").toLowerCase();
    const family = String(s.family ?? "").toLowerCase();
    return name.includes(term) || common.includes(term) || order.includes(term) || family.includes(term);
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800 dark:text-white">Species Reference Library</h1>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">Explore baseline taxonomy, common synonyms, and regional classifications.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Quick lookup */}
        <div className="bg-white/75 dark:bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-200/50 dark:border-slate-800/50 p-4 space-y-3 shadow-sm">
          <h2 className="font-semibold text-slate-700 dark:text-slate-300 text-sm">Quick Species Lookup</h2>
          <div className="flex gap-2">
            <input
              placeholder="Enter canonical scientific name…"
              value={lookupQuery}
              onChange={(e) => setLookupQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLookup()}
              className="flex-1 border border-slate-350 dark:border-slate-800 rounded-lg px-3 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100"
            />
            <button onClick={handleLookup} className="px-4 py-2 bg-emerald-600 text-white text-xs font-semibold rounded-lg hover:bg-emerald-700 transition cursor-pointer">Look up</button>
          </div>
          {lookupError && <p className="text-red-500 text-xs font-medium">{lookupError}</p>}
          {lookupResult && (
            <div className="bg-slate-50 dark:bg-slate-950/40 rounded-lg p-3 text-xs border border-slate-100 dark:border-slate-800 font-mono text-slate-700 dark:text-slate-400 max-h-36 overflow-y-auto">
              <pre>{JSON.stringify(lookupResult, null, 2)}</pre>
            </div>
          )}
        </div>

        {/* Synonym resolver */}
        <div className="bg-white/75 dark:bg-slate-900/60 backdrop-blur-md rounded-xl border border-slate-200/50 dark:border-slate-800/50 p-4 space-y-3 shadow-sm">
          <h2 className="font-semibold text-slate-700 dark:text-slate-300 text-sm">Synonym Resolver</h2>
          <div className="flex gap-2">
            <input
              placeholder="Enter common or alternate name…"
              value={synonymQuery}
              onChange={(e) => setSynonymQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSynonym()}
              className="flex-1 border border-slate-350 dark:border-slate-800 rounded-lg px-3 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100"
            />
            <button onClick={handleSynonym} className="px-4 py-2 bg-emerald-600 text-white text-xs font-semibold rounded-lg hover:bg-emerald-700 transition cursor-pointer">Resolve</button>
          </div>
          {synonymResult && (
            <div className="bg-slate-50 dark:bg-slate-950/40 rounded-lg p-3 text-xs border border-slate-100 dark:border-slate-800 flex items-center justify-between">
              <span className="text-slate-500 font-medium">"{String(synonymResult.input ?? "")}"</span>
              <span className="text-slate-400">➔</span>
              <span className="font-bold text-emerald-600 dark:text-emerald-450 bg-emerald-50 dark:bg-emerald-950/40 px-2.5 py-1 rounded-full">
                {String(synonymResult.canonical ?? "Not found")}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Main library section (flex layout for slide-out drawer) */}
      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* Cards Grid */}
        <div className="flex-1 bg-white/75 dark:bg-slate-900/60 backdrop-blur-md rounded-2xl border border-slate-200/50 dark:border-slate-800/50 overflow-hidden shadow-sm w-full">
          <div className="px-4 py-3 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between flex-wrap gap-2">
            <span className="font-bold text-slate-700 dark:text-slate-300 text-sm">All Registered Species ({species.length})</span>
            <input
              placeholder="Search by name, order, family…"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="border border-slate-300 dark:border-slate-800 rounded-lg px-3 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 min-w-[200px]"
            />
          </div>

          {loading ? (
            <div className="text-center py-16 text-slate-400">Loading library cards…</div>
          ) : (
            <div className="p-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {filtered.map((s, i) => {
                  const sciName = String(s.name ?? s.scientific_name ?? "Unknown");
                  const commonName = String(s.common_name ?? "Vernacular name unmapped");
                  const taxonClass = String(s.class ?? s.category ?? "Mammalia");
                  const orderName = String(s.order ?? "");
                  const familyName = String(s.family ?? "");

                  const isSelected = selectedSp && (selectedSp.scientific_name === sciName || selectedSp.name === sciName);

                  return (
                    <div
                      key={i}
                      onClick={() => setSelectedSp(s)}
                      className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col justify-between h-36 ${
                        isSelected
                          ? "bg-emerald-50/50 dark:bg-emerald-950/20 border-emerald-500 dark:border-emerald-700 ring-1 ring-emerald-500"
                          : "bg-white dark:bg-slate-950 border-slate-200 dark:border-slate-850 hover:border-slate-350 dark:hover:border-slate-700 hover:shadow-sm"
                      }`}
                    >
                      <div className="flex justify-between items-start">
                        <div className="space-y-1">
                          <p className="font-bold text-slate-800 dark:text-slate-100 text-sm leading-snug line-clamp-1">{commonName}</p>
                          <p className="text-xs text-slate-550 dark:text-slate-400 italic font-medium line-clamp-1">{sciName}</p>
                        </div>
                        <span className="text-2xl select-none" title={taxonClass}>
                          {getTaxonIcon(taxonClass)}
                        </span>
                      </div>

                      <div className="flex flex-wrap gap-1 mt-3">
                        {orderName && (
                          <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-900 border border-slate-200/40 dark:border-slate-800 text-slate-500 dark:text-slate-400 rounded-full text-[9px] font-bold uppercase tracking-wider">
                            {orderName}
                          </span>
                        )}
                        {familyName && (
                          <span className="px-2 py-0.5 bg-slate-100 dark:bg-slate-900 border border-slate-200/40 dark:border-slate-800 text-slate-500 dark:text-slate-400 rounded-full text-[9px] font-bold uppercase tracking-wider">
                            {familyName}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
              {filtered.length === 0 && (
                <div className="text-center py-16 text-slate-400">No species match your query filter.</div>
              )}
            </div>
          )}
        </div>

        {/* Detailed slide-out profile drawer */}
        {selectedSp && (() => {
          const sciName = String(selectedSp.name ?? selectedSp.scientific_name ?? "Unknown");
          const commonName = String(selectedSp.common_name ?? "Vernacular name unmapped");
          
          return (
            <div className="w-full lg:w-80 xl:w-96 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-200 dark:border-slate-800 p-5 space-y-4 shadow-sm animate-fade-in shrink-0">
              <div className="flex items-start justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                <div className="space-y-1">
                  <h3 className="font-bold text-slate-800 dark:text-slate-100 text-base">{commonName}</h3>
                  <p className="text-xs text-emerald-600 dark:text-emerald-450 italic font-semibold">{sciName}</p>
                </div>
                <button
                  onClick={() => setSelectedSp(null)}
                  className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-250 cursor-pointer text-sm font-semibold"
                >
                  ✕
                </button>
              </div>

              {/* Taxonomy lineage */}
              <div className="space-y-2">
                <p className="text-[10px] uppercase font-bold text-slate-400 dark:text-slate-500 tracking-wider">Taxonomic Hierarchy</p>
                <div className="space-y-1.5 border-l-2 border-slate-100 dark:border-slate-800 pl-3 ml-1 text-xs">
                  {[
                    { rank: "Kingdom", val: selectedSp.kingdom ?? "Animalia" },
                    { rank: "Phylum", val: selectedSp.phylum ?? "Chordata" },
                    { rank: "Class", val: selectedSp.class ?? selectedSp.category ?? "Mammalia" },
                    { rank: "Order", val: selectedSp.order },
                    { rank: "Family", val: selectedSp.family },
                    { rank: "Genus", val: selectedSp.genus },
                    { rank: "Species", val: selectedSp.species ?? sciName.split(" ").slice(-1)[0] },
                  ].map((tax, tIdx) => {
                    if (!tax.val) return null;
                    return (
                      <div key={tIdx} className="flex justify-between items-center py-0.5">
                        <span className="text-slate-400 dark:text-slate-500 font-medium">{tax.rank}</span>
                        <span className="font-bold text-slate-700 dark:text-slate-350">{String(tax.val)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Ecological Status placeholder */}
              <div className="space-y-2 border-t border-slate-100 dark:border-slate-800 pt-3">
                <p className="text-[10px] uppercase font-bold text-slate-400 dark:text-slate-500 tracking-wider">Conservation & Ecology</p>
                <div className="bg-slate-50 dark:bg-slate-950/40 border border-slate-200/50 dark:border-slate-850 p-3 rounded-xl text-xs space-y-2 text-slate-650 dark:text-slate-400">
                  <div className="flex justify-between">
                    <span>Regional Presence</span>
                    <span className="font-bold text-slate-700 dark:text-slate-300">Ethiopia (Gambella)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>IUCN Red List Status</span>
                    <span className="font-bold text-amber-600 dark:text-amber-450">Least Concern (LC)</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
