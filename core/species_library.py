"""
Species Reference Library
Pre-loaded with mammals known or likely to occur in the Gambella wetland
landscape. Supports synonym resolution, search, and Data Manager additions.
"""

import sqlite3
import pandas as pd
from typing import Optional


# ------------------------------------------------------------------
# Pre-loaded species data
# ------------------------------------------------------------------

GAMBELLA_SPECIES = [
    # (common_name, scientific_name, family, order_, iucn_status, kenya_status, notes)
    ("Nile Lechwe",         "Kobus megaceros",           "Bovidae",      "Artiodactyla",  "Vulnerable",    "Vulnerable",    "Semi-aquatic antelope endemic to Gambella floodplains; indicator of wetland health."),
    ("White-eared Kob",     "Kobus kob leucotis",        "Bovidae",      "Artiodactyla",  "Near Threatened","Near Threatened","Undertakes one of Africa's largest migrations through this corridor."),
    ("Tiang",               "Damaliscus lunatus tiang",  "Bovidae",      "Artiodactyla",  "Least Concern", "Least Concern", "Seasonal migrant; large herds use the floodplain grasslands."),
    ("Common Eland",        "Taurotragus oryx",          "Bovidae",      "Artiodactyla",  "Least Concern", "Least Concern", "Largest antelope; uses woodland-grassland ecotones."),
    ("African Buffalo",     "Syncerus caffer",           "Bovidae",      "Artiodactyla",  "Near Threatened","Near Threatened","Herd species; key indicator of intact savanna ecosystem."),
    ("Roan Antelope",       "Hippotragus equinus",       "Bovidae",      "Artiodactyla",  "Least Concern", "Least Concern", "Prefers open woodland and tall grassland."),
    ("Oribi",               "Ourebia ourebi",            "Bovidae",      "Artiodactyla",  "Least Concern", "Least Concern", "Small, territorial antelope of grassy floodplains."),
    ("Common Waterbuck",    "Kobus ellipsiprymnus",      "Bovidae",      "Artiodactyla",  "Least Concern", "Least Concern", "Always close to permanent water; ring marking on rump."),
    ("Bushbuck",            "Tragelaphus scriptus",      "Bovidae",      "Artiodactyla",  "Least Concern", "Least Concern", "Solitary; favours dense riparian vegetation."),
    ("Greater Kudu",        "Tragelaphus strepsiceros",  "Bovidae",      "Artiodactyla",  "Least Concern", "Least Concern", "Tall spiral-horned antelope of woodland and bushland."),
    ("African Elephant",    "Loxodonta africana",        "Elephantidae", "Proboscidea",   "Vulnerable",    "Endangered",    "Keystone species; critical for wetland and woodland ecosystem engineering."),
    ("Common Hippopotamus", "Hippopotamus amphibius",    "Hippopotamidae","Artiodactyla", "Vulnerable",    "Vulnerable",    "Abundant in rivers and lakes; grazes on land at night."),
    ("Plains Zebra",        "Equus quagga",              "Equidae",      "Perissodactyla","Near Threatened","Near Threatened","Migrates with kob and tiang; requires open grassland."),
    ("Grevy's Zebra",       "Equus grevyi",              "Equidae",      "Perissodactyla","Endangered",    "Endangered",    "Larger, narrower-striped; arid and semi-arid areas near corridor."),
    ("Warthog",             "Phacochoerus africanus",    "Suidae",       "Artiodactyla",  "Least Concern", "Least Concern", "Very common; active during the day; uses watering points frequently."),
    ("Lion",                "Panthera leo",              "Felidae",      "Carnivora",     "Vulnerable",    "Vulnerable",    "Apex predator; presence indicates intact prey base."),
    ("Leopard",             "Panthera pardus",           "Felidae",      "Carnivora",     "Vulnerable",    "Vulnerable",    "Nocturnal; uses riparian forest and rocky outcrops."),
    ("Cheetah",             "Acinonyx jubatus",          "Felidae",      "Carnivora",     "Vulnerable",    "Vulnerable",    "Diurnal hunter; open grassland specialist."),
    ("Serval",              "Leptailurus serval",        "Felidae",      "Carnivora",     "Least Concern", "Least Concern", "Medium felid; wetland edges and tall grass."),
    ("Caracal",             "Caracal caracal",           "Felidae",      "Carnivora",     "Least Concern", "Least Concern", "Nocturnal; dry bush and woodland."),
    ("African Wild Cat",    "Felis lybica",              "Felidae",      "Carnivora",     "Least Concern", "Least Concern", "Ancestor of domestic cat; widespread but secretive."),
    ("African Wild Dog",    "Lycaon pictus",             "Canidae",      "Carnivora",     "Endangered",    "Endangered",    "Highly social pack hunter; requires large, connected ranges."),
    ("Spotted Hyena",       "Crocuta crocuta",           "Hyaenidae",    "Carnivora",     "Least Concern", "Least Concern", "Dominant scavenger and predator; nocturnal activity peak."),
    ("Common Jackal",       "Canis aureus",              "Canidae",      "Carnivora",     "Least Concern", "Least Concern", "Generalist; woodland edges and open areas."),
    ("Side-striped Jackal", "Canis adustus",             "Canidae",      "Carnivora",     "Least Concern", "Least Concern", "More forest-associated than golden jackal."),
    ("Honey Badger",        "Mellivora capensis",        "Mustelidae",   "Carnivora",     "Least Concern", "Least Concern", "Notoriously bold; nocturnal; omnivorous."),
    ("Olive Baboon",        "Papio anubis",              "Cercopithecidae","Primates",    "Least Concern", "Least Concern", "Large, adaptable primate; troops common near water."),
    ("Patas Monkey",        "Erythrocebus patas",        "Cercopithecidae","Primates",    "Least Concern", "Least Concern", "Fastest primate; open woodland and grassland specialist."),
    ("Vervet Monkey",       "Chlorocebus pygerythrus",   "Cercopithecidae","Primates",    "Least Concern", "Least Concern", "Common near riparian woodland."),
    ("Aardvark",            "Orycteropus afer",          "Orycteropodidae","Tubulidentata","Least Concern","Least Concern", "Nocturnal; termite specialist; digs burrows used by other species."),
    ("Giant Pangolin",      "Smutsia gigantea",          "Manidae",      "Pholidota",     "Endangered",    "Endangered",    "Nocturnal; heavily trafficked; indicator of low-disturbance areas."),
    ("African Porcupine",   "Hystrix cristata",          "Hystricidae",  "Rodentia",      "Least Concern", "Least Concern", "Nocturnal; woodland and savanna."),
    ("Common Genet",        "Genetta genetta",           "Viverridae",   "Carnivora",     "Least Concern", "Least Concern", "Slender nocturnal carnivore; riparian forest."),
    ("Banded Mongoose",     "Mungos mungo",              "Herpestidae",  "Carnivora",     "Least Concern", "Least Concern", "Highly social; diurnal; common near termite mounds."),
]

# Synonym table: alternative name → accepted common name
SYNONYMS = {
    "hippopotamus":         "Common Hippopotamus",
    "hippo":                "Common Hippopotamus",
    "african wild ass":     "Plains Zebra",
    "burchell's zebra":     "Plains Zebra",
    "common zebra":         "Plains Zebra",
    "painted dog":          "African Wild Dog",
    "painted wolf":         "African Wild Dog",
    "cape hunting dog":     "African Wild Dog",
    "golden jackal":        "Common Jackal",
    "nile lechwi":          "Nile Lechwe",
    "lelwel hartebeest":    "Tiang",
    "topi":                 "Tiang",
    "cape buffalo":         "African Buffalo",
    "savanna elephant":     "African Elephant",
    "african savanna elephant": "African Elephant",
    "kob":                  "White-eared Kob",
    "white eared kob":      "White-eared Kob",
    "cheetah":              "Cheetah",
}


class SpeciesLibrary:
    """
    Searchable species reference library backed by SQLite.
    Pre-seeded with Gambella wetland mammals on first run.
    """

    def __init__(self, db_path: str = "wildlife_data.db"):
        self.db_path = db_path
        self._init_tables()
        self._seed_if_empty()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS species_library (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                common_name     TEXT NOT NULL UNIQUE,
                scientific_name TEXT,
                family          TEXT,
                order_name      TEXT,
                iucn_status     TEXT,
                kenya_status    TEXT,
                notes           TEXT,
                added_by        TEXT DEFAULT 'system',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS species_synonyms (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                synonym         TEXT NOT NULL UNIQUE,
                accepted_name   TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _seed_if_empty(self):
        conn = self._conn()
        count = conn.execute("SELECT COUNT(*) FROM species_library").fetchone()[0]
        if count == 0:
            conn.executemany("""
                INSERT OR IGNORE INTO species_library
                    (common_name, scientific_name, family, order_name,
                     iucn_status, kenya_status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, GAMBELLA_SPECIES)

            conn.executemany("""
                INSERT OR IGNORE INTO species_synonyms (synonym, accepted_name)
                VALUES (?, ?)
            """, [(k, v) for k, v in SYNONYMS.items()])

            conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM species_library ORDER BY common_name", conn
            )
        finally:
            conn.close()

    def search(self, query: str) -> pd.DataFrame:
        """Full-text search across common name, scientific name, and notes."""
        if not query or not query.strip():
            return self.get_all()
        q = f"%{query.strip()}%"
        conn = self._conn()
        try:
            return pd.read_sql_query("""
                SELECT * FROM species_library
                WHERE common_name     LIKE ?
                   OR scientific_name LIKE ?
                   OR family          LIKE ?
                   OR notes           LIKE ?
                ORDER BY common_name
            """, conn, params=(q, q, q, q))
        finally:
            conn.close()

    def get_by_name(self, name: str) -> Optional[dict]:
        """Look up by common name (case-insensitive) or scientific name."""
        resolved = self.resolve_synonym(name) or name
        conn = self._conn()
        try:
            row = conn.execute("""
                SELECT * FROM species_library
                WHERE LOWER(common_name) = LOWER(?)
                   OR LOWER(scientific_name) = LOWER(?)
                LIMIT 1
            """, (resolved, resolved)).fetchone()
            if row:
                cols = [d[0] for d in conn.execute("PRAGMA table_info(species_library)").fetchall()]
                return dict(zip(cols, row))
            return None
        finally:
            conn.close()

    def resolve_synonym(self, name: str) -> Optional[str]:
        """Return the accepted common name for a synonym, or None."""
        conn = self._conn()
        try:
            row = conn.execute("""
                SELECT accepted_name FROM species_synonyms
                WHERE LOWER(synonym) = LOWER(?)
                LIMIT 1
            """, (name.strip(),)).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def get_names_list(self) -> list:
        """Return sorted list of all common names (for autocomplete)."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT common_name FROM species_library ORDER BY common_name"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write (Data Manager additions)
    # ------------------------------------------------------------------

    def add_species(
        self,
        common_name: str,
        scientific_name: str = "",
        family: str = "",
        order_name: str = "",
        iucn_status: str = "",
        kenya_status: str = "",
        notes: str = "",
        added_by: str = "Data Manager",
    ) -> bool:
        """Add a new species. Returns False if common_name already exists."""
        conn = self._conn()
        try:
            conn.execute("""
                INSERT INTO species_library
                    (common_name, scientific_name, family, order_name,
                     iucn_status, kenya_status, notes, added_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (common_name.strip(), scientific_name, family, order_name,
                  iucn_status, kenya_status, notes, added_by))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def add_synonym(self, synonym: str, accepted_name: str) -> bool:
        conn = self._conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO species_synonyms (synonym, accepted_name)
                VALUES (?, ?)
            """, (synonym.strip().lower(), accepted_name.strip()))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()
