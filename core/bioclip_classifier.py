"""
BioCLIP species classifier with full taxonomic output.

predict_list()     – fast flat list of (species, confidence) — used for speed
predict_taxonomy() – single forward pass returning species + family + full path
                     used for ensemble agreement with SpeciesNet hierarchy
"""

import torch
import open_clip
from PIL import Image
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Taxonomy reference table
# Maps every species in AnimalDetector.WILDLIFE_CLASSES to its
# order / family / genus / scientific name.
# ---------------------------------------------------------------------------

SPECIES_TAXONOMY: Dict[str, Dict[str, str]] = {
    # ── Equids ──────────────────────────────────────────────────────────────
    "plains zebra":             {"order": "Perissodactyla", "family": "Equidae",         "genus": "Equus",          "scientific": "Equus quagga"},
    "Grevy's zebra":            {"order": "Perissodactyla", "family": "Equidae",         "genus": "Equus",          "scientific": "Equus grevyi"},
    "mountain zebra":           {"order": "Perissodactyla", "family": "Equidae",         "genus": "Equus",          "scientific": "Equus zebra"},
    "African wild ass":         {"order": "Perissodactyla", "family": "Equidae",         "genus": "Equus",          "scientific": "Equus africanus"},
    # ── Elephants ───────────────────────────────────────────────────────────
    "African elephant":         {"order": "Proboscidea",    "family": "Elephantidae",    "genus": "Loxodonta",      "scientific": "Loxodonta africana"},
    "African bush elephant":    {"order": "Proboscidea",    "family": "Elephantidae",    "genus": "Loxodonta",      "scientific": "Loxodonta africana"},
    "African forest elephant":  {"order": "Proboscidea",    "family": "Elephantidae",    "genus": "Loxodonta",      "scientific": "Loxodonta cyclotis"},
    # ── Felids ──────────────────────────────────────────────────────────────
    "lion":                     {"order": "Carnivora",      "family": "Felidae",         "genus": "Panthera",       "scientific": "Panthera leo"},
    "leopard":                  {"order": "Carnivora",      "family": "Felidae",         "genus": "Panthera",       "scientific": "Panthera pardus"},
    "cheetah":                  {"order": "Carnivora",      "family": "Felidae",         "genus": "Acinonyx",       "scientific": "Acinonyx jubatus"},
    "serval":                   {"order": "Carnivora",      "family": "Felidae",         "genus": "Leptailurus",    "scientific": "Leptailurus serval"},
    "caracal":                  {"order": "Carnivora",      "family": "Felidae",         "genus": "Caracal",        "scientific": "Caracal caracal"},
    "African wildcat":          {"order": "Carnivora",      "family": "Felidae",         "genus": "Felis",          "scientific": "Felis lybica"},
    "African golden cat":       {"order": "Carnivora",      "family": "Felidae",         "genus": "Caracal",        "scientific": "Caracal aurata"},
    # ── Canids ──────────────────────────────────────────────────────────────
    "African wild dog":         {"order": "Carnivora",      "family": "Canidae",         "genus": "Lycaon",         "scientific": "Lycaon pictus"},
    "black-backed jackal":      {"order": "Carnivora",      "family": "Canidae",         "genus": "Lupulella",      "scientific": "Lupulella mesomelas"},
    "side-striped jackal":      {"order": "Carnivora",      "family": "Canidae",         "genus": "Lupulella",      "scientific": "Lupulella adusta"},
    "golden jackal":            {"order": "Carnivora",      "family": "Canidae",         "genus": "Canis",          "scientific": "Canis aureus"},
    "bat-eared fox":            {"order": "Carnivora",      "family": "Canidae",         "genus": "Otocyon",        "scientific": "Otocyon megalotis"},
    "Cape fox":                 {"order": "Carnivora",      "family": "Canidae",         "genus": "Vulpes",         "scientific": "Vulpes chama"},
    "Ethiopian wolf":           {"order": "Carnivora",      "family": "Canidae",         "genus": "Canis",          "scientific": "Canis simensis"},
    # ── Hyaenids ────────────────────────────────────────────────────────────
    "spotted hyena":            {"order": "Carnivora",      "family": "Hyaenidae",       "genus": "Crocuta",        "scientific": "Crocuta crocuta"},
    "striped hyena":            {"order": "Carnivora",      "family": "Hyaenidae",       "genus": "Hyaena",         "scientific": "Hyaena hyaena"},
    "brown hyena":              {"order": "Carnivora",      "family": "Hyaenidae",       "genus": "Parahyaena",     "scientific": "Parahyaena brunnea"},
    "aardwolf":                 {"order": "Carnivora",      "family": "Hyaenidae",       "genus": "Proteles",       "scientific": "Proteles cristata"},
    "hyena":                    {"order": "Carnivora",      "family": "Hyaenidae",       "genus": "",               "scientific": ""},
    # ── Mustelids / Viverrids / Herpestids ──────────────────────────────────
    "honey badger":             {"order": "Carnivora",      "family": "Mustelidae",      "genus": "Mellivora",      "scientific": "Mellivora capensis"},
    "spotted-necked otter":     {"order": "Carnivora",      "family": "Mustelidae",      "genus": "Hydrictis",      "scientific": "Hydrictis maculicollis"},
    "Cape clawless otter":      {"order": "Carnivora",      "family": "Mustelidae",      "genus": "Aonyx",          "scientific": "Aonyx capensis"},
    "African civet":            {"order": "Carnivora",      "family": "Viverridae",      "genus": "Civettictis",    "scientific": "Civettictis civetta"},
    "common genet":             {"order": "Carnivora",      "family": "Viverridae",      "genus": "Genetta",        "scientific": "Genetta genetta"},
    "banded mongoose":          {"order": "Carnivora",      "family": "Herpestidae",     "genus": "Mungos",         "scientific": "Mungos mungo"},
    "dwarf mongoose":           {"order": "Carnivora",      "family": "Herpestidae",     "genus": "Helogale",       "scientific": "Helogale parvula"},
    "meerkat":                  {"order": "Carnivora",      "family": "Herpestidae",     "genus": "Suricata",       "scientific": "Suricata suricatta"},
    "Egyptian mongoose":        {"order": "Carnivora",      "family": "Herpestidae",     "genus": "Herpestes",      "scientific": "Herpestes ichneumon"},
    # ── Giraffids ───────────────────────────────────────────────────────────
    "giraffe":                  {"order": "Artiodactyla",   "family": "Giraffidae",      "genus": "Giraffa",        "scientific": "Giraffa camelopardalis"},
    "reticulated giraffe":      {"order": "Artiodactyla",   "family": "Giraffidae",      "genus": "Giraffa",        "scientific": "Giraffa reticulata"},
    "Masai giraffe":            {"order": "Artiodactyla",   "family": "Giraffidae",      "genus": "Giraffa",        "scientific": "Giraffa tippelskirchi"},
    # ── Bovids — Bovinae ────────────────────────────────────────────────────
    "African buffalo":          {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Syncerus",       "scientific": "Syncerus caffer"},
    "common eland":             {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Taurotragus",    "scientific": "Taurotragus oryx"},
    "greater kudu":             {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Tragelaphus",    "scientific": "Tragelaphus strepsiceros"},
    "lesser kudu":              {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Tragelaphus",    "scientific": "Tragelaphus imberbis"},
    "bushbuck":                 {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Tragelaphus",    "scientific": "Tragelaphus scriptus"},
    "nyala":                    {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Tragelaphus",    "scientific": "Tragelaphus angasii"},
    "sitatunga":                {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Tragelaphus",    "scientific": "Tragelaphus spekii"},
    "bongo":                    {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Tragelaphus",    "scientific": "Tragelaphus eurycerus"},
    # ── Bovids — Antilopinae ────────────────────────────────────────────────
    "impala":                   {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Aepyceros",      "scientific": "Aepyceros melampus"},
    "Thomson's gazelle":        {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Eudorcas",       "scientific": "Eudorcas thomsonii"},
    "Grant's gazelle":          {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Nanger",         "scientific": "Nanger granti"},
    "springbok":                {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Antidorcas",     "scientific": "Antidorcas marsupialis"},
    "gerenuk":                  {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Litocranius",    "scientific": "Litocranius walleri"},
    "dibatag":                  {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Ammodorcas",     "scientific": "Ammodorcas clarkei"},
    "dik-dik":                  {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Madoqua",        "scientific": "Madoqua kirkii"},
    "steenbok":                 {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Raphicerus",     "scientific": "Raphicerus campestris"},
    "oribi":                    {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Ourebia",        "scientific": "Ourebia ourebi"},
    "common duiker":            {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Sylvicapra",     "scientific": "Sylvicapra grimmia"},
    "blue duiker":              {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Philantomba",    "scientific": "Philantomba monticola"},
    "red duiker":               {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Cephalophus",    "scientific": "Cephalophus natalensis"},
    # ── Bovids — Alcelaphinae ───────────────────────────────────────────────
    "blue wildebeest":          {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Connochaetes",   "scientific": "Connochaetes taurinus"},
    "black wildebeest":         {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Connochaetes",   "scientific": "Connochaetes gnou"},
    "topi":                     {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Damaliscus",     "scientific": "Damaliscus lunatus"},
    "hartebeest":               {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Alcelaphus",     "scientific": "Alcelaphus buselaphus"},
    "wildebeest":               {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Connochaetes",   "scientific": ""},
    # ── Bovids — Hippotraginae ──────────────────────────────────────────────
    "roan antelope":            {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Hippotragus",    "scientific": "Hippotragus equinus"},
    "sable antelope":           {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Hippotragus",    "scientific": "Hippotragus niger"},
    "gemsbok":                  {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Oryx",           "scientific": "Oryx gazella"},
    "East African oryx":        {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Oryx",           "scientific": "Oryx beisa"},
    # ── Bovids — Reduncinae ─────────────────────────────────────────────────
    "Nile lechwe":              {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Kobus",          "scientific": "Kobus megaceros"},
    "waterbuck":                {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Kobus",          "scientific": "Kobus ellipsiprymnus"},
    "reedbuck":                 {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Redunca",        "scientific": "Redunca redunca"},
    "lechwe":                   {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Kobus",          "scientific": "Kobus leche"},
    "White-eared kob":          {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Kobus",          "scientific": "Kobus kob leucotis"},
    "Uganda kob":               {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "Kobus",          "scientific": "Kobus kob thomasi"},
    # ── Generic bovid fallbacks ─────────────────────────────────────────────
    "antelope":                 {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "",               "scientific": ""},
    "gazelle":                  {"order": "Artiodactyla",   "family": "Bovidae",         "genus": "",               "scientific": ""},
    # ── Rhinoceroses ────────────────────────────────────────────────────────
    "white rhinoceros":         {"order": "Perissodactyla", "family": "Rhinocerotidae",  "genus": "Ceratotherium",  "scientific": "Ceratotherium simum"},
    "black rhinoceros":         {"order": "Perissodactyla", "family": "Rhinocerotidae",  "genus": "Diceros",        "scientific": "Diceros bicornis"},
    "rhinoceros":               {"order": "Perissodactyla", "family": "Rhinocerotidae",  "genus": "",               "scientific": ""},
    # ── Suids ───────────────────────────────────────────────────────────────
    "common warthog":           {"order": "Artiodactyla",   "family": "Suidae",          "genus": "Phacochoerus",   "scientific": "Phacochoerus africanus"},
    "bushpig":                  {"order": "Artiodactyla",   "family": "Suidae",          "genus": "Potamochoerus",  "scientific": "Potamochoerus larvatus"},
    "giant forest hog":         {"order": "Artiodactyla",   "family": "Suidae",          "genus": "Hylochoerus",    "scientific": "Hylochoerus meinertzhageni"},
    "red river hog":            {"order": "Artiodactyla",   "family": "Suidae",          "genus": "Potamochoerus",  "scientific": "Potamochoerus porcus"},
    # ── Hippos ──────────────────────────────────────────────────────────────
    "common hippopotamus":      {"order": "Artiodactyla",   "family": "Hippopotamidae",  "genus": "Hippopotamus",   "scientific": "Hippopotamus amphibius"},
    "pygmy hippopotamus":       {"order": "Artiodactyla",   "family": "Hippopotamidae",  "genus": "Choeropsis",     "scientific": "Choeropsis liberiensis"},
    "hippopotamus":             {"order": "Artiodactyla",   "family": "Hippopotamidae",  "genus": "Hippopotamus",   "scientific": ""},
    # ── Primates — Cercopithecidae ──────────────────────────────────────────
    "olive baboon":             {"order": "Primates",       "family": "Cercopithecidae", "genus": "Papio",          "scientific": "Papio anubis"},
    "chacma baboon":            {"order": "Primates",       "family": "Cercopithecidae", "genus": "Papio",          "scientific": "Papio ursinus"},
    "hamadryas baboon":         {"order": "Primates",       "family": "Cercopithecidae", "genus": "Papio",          "scientific": "Papio hamadryas"},
    "gelada":                   {"order": "Primates",       "family": "Cercopithecidae", "genus": "Theropithecus",  "scientific": "Theropithecus gelada"},
    "vervet monkey":            {"order": "Primates",       "family": "Cercopithecidae", "genus": "Chlorocebus",    "scientific": "Chlorocebus pygerythrus"},
    "patas monkey":             {"order": "Primates",       "family": "Cercopithecidae", "genus": "Erythrocebus",   "scientific": "Erythrocebus patas"},
    "colobus monkey":           {"order": "Primates",       "family": "Cercopithecidae", "genus": "Colobus",        "scientific": "Colobus guereza"},
    "mandrill":                 {"order": "Primates",       "family": "Cercopithecidae", "genus": "Mandrillus",     "scientific": "Mandrillus sphinx"},
    "drill":                    {"order": "Primates",       "family": "Cercopithecidae", "genus": "Mandrillus",     "scientific": "Mandrillus leucophaeus"},
    "monkey":                   {"order": "Primates",       "family": "Cercopithecidae", "genus": "",               "scientific": ""},
    "baboon":                   {"order": "Primates",       "family": "Cercopithecidae", "genus": "Papio",          "scientific": ""},
    # ── Primates — Hominidae ────────────────────────────────────────────────
    "chimpanzee":               {"order": "Primates",       "family": "Hominidae",       "genus": "Pan",            "scientific": "Pan troglodytes"},
    "bonobo":                   {"order": "Primates",       "family": "Hominidae",       "genus": "Pan",            "scientific": "Pan paniscus"},
    "western gorilla":          {"order": "Primates",       "family": "Hominidae",       "genus": "Gorilla",        "scientific": "Gorilla gorilla"},
    "eastern gorilla":          {"order": "Primates",       "family": "Hominidae",       "genus": "Gorilla",        "scientific": "Gorilla beringei"},
    "mountain gorilla":         {"order": "Primates",       "family": "Hominidae",       "genus": "Gorilla",        "scientific": "Gorilla beringei beringei"},
    "human":                    {"order": "Primates",       "family": "Hominidae",       "genus": "Homo",           "scientific": "Homo sapiens"},
    "person":                   {"order": "Primates",       "family": "Hominidae",       "genus": "Homo",           "scientific": "Homo sapiens"},
    # ── Primates — Lemuridae / Indridae ─────────────────────────────────────
    "ring-tailed lemur":        {"order": "Primates",       "family": "Lemuridae",       "genus": "Lemur",          "scientific": "Lemur catta"},
    "sifaka":                   {"order": "Primates",       "family": "Indridae",        "genus": "Propithecus",    "scientific": "Propithecus verreauxi"},
    # ── Aardvark, pangolins, small mammals ──────────────────────────────────
    "aardvark":                 {"order": "Tubulidentata",  "family": "Orycteropodidae", "genus": "Orycteropus",    "scientific": "Orycteropus afer"},
    "ground pangolin":          {"order": "Pholidota",      "family": "Manidae",         "genus": "Smutsia",        "scientific": "Smutsia temminckii"},
    "giant pangolin":           {"order": "Pholidota",      "family": "Manidae",         "genus": "Smutsia",        "scientific": "Smutsia gigantea"},
    "tree pangolin":            {"order": "Pholidota",      "family": "Manidae",         "genus": "Phataginus",     "scientific": "Phataginus tricuspis"},
    "Cape porcupine":           {"order": "Rodentia",       "family": "Hystricidae",     "genus": "Hystrix",        "scientific": "Hystrix africaeaustralis"},
    "African crested porcupine":{"order": "Rodentia",       "family": "Hystricidae",     "genus": "Hystrix",        "scientific": "Hystrix cristata"},
    "Cape hare":                {"order": "Lagomorpha",     "family": "Leporidae",       "genus": "Lepus",          "scientific": "Lepus capensis"},
    "springhare":               {"order": "Rodentia",       "family": "Pedetidae",       "genus": "Pedetes",        "scientific": "Pedetes capensis"},
    "rock hyrax":               {"order": "Hyracoidea",     "family": "Procaviidae",     "genus": "Procavia",       "scientific": "Procavia capensis"},
    # ── Reptiles ────────────────────────────────────────────────────────────
    "Nile crocodile":           {"order": "Crocodilia",     "family": "Crocodylidae",    "genus": "Crocodylus",     "scientific": "Crocodylus niloticus"},
    "dwarf crocodile":          {"order": "Crocodilia",     "family": "Crocodylidae",    "genus": "Osteolaemus",    "scientific": "Osteolaemus tetraspis"},
    "crocodile":                {"order": "Crocodilia",     "family": "Crocodylidae",    "genus": "",               "scientific": ""},
    "Nile monitor":             {"order": "Squamata",       "family": "Varanidae",       "genus": "Varanus",        "scientific": "Varanus niloticus"},
    "rock python":              {"order": "Squamata",       "family": "Pythonidae",      "genus": "Python",         "scientific": "Python sebae"},
    "leopard tortoise":         {"order": "Testudines",     "family": "Testudinidae",    "genus": "Stigmochelys",   "scientific": "Stigmochelys pardalis"},
    # ── Birds ───────────────────────────────────────────────────────────────
    "ostrich":                  {"order": "Struthioniformes","family": "Struthionidae",  "genus": "Struthio",       "scientific": "Struthio camelus"},
    "kori bustard":             {"order": "Otidiformes",    "family": "Otididae",        "genus": "Ardeotis",       "scientific": "Ardeotis kori"},
    "southern ground hornbill": {"order": "Bucerotiformes", "family": "Bucorvidae",      "genus": "Bucorvus",       "scientific": "Bucorvus leadbeateri"},
    "helmeted guineafowl":      {"order": "Galliformes",    "family": "Numididae",       "genus": "Numida",         "scientific": "Numida meleagris"},
    "crested francolin":        {"order": "Galliformes",    "family": "Phasianidae",     "genus": "Ortygornis",     "scientific": "Ortygornis sephaena"},
    "secretary bird":           {"order": "Accipitriformes","family": "Sagittariidae",   "genus": "Sagittarius",    "scientific": "Sagittarius serpentarius"},
    "marabou stork":            {"order": "Ciconiiformes",  "family": "Ciconiidae",      "genus": "Leptoptilos",    "scientific": "Leptoptilos crumeniferus"},
    # ── Generic / fallback labels ───────────────────────────────────────────
    "animal":                   {"order": "",               "family": "",                "genus": "",               "scientific": ""},
    "vehicle":                  {"order": "",               "family": "",                "genus": "",               "scientific": ""},
}


# ---------------------------------------------------------------------------
# Family-level text prompts for independent family classification.
# One forward pass scores the image against all families simultaneously.
# ---------------------------------------------------------------------------

FAMILY_PROMPTS: Dict[str, str] = {
    "Felidae":          "a photo of a wild cat such as a lion, leopard, cheetah, serval, or caracal",
    "Canidae":          "a photo of a wild dog, jackal, or fox",
    "Hyaenidae":        "a photo of a hyena or aardwolf",
    "Bovidae":          "a photo of an antelope, gazelle, buffalo, wildebeest, impala, or bovid",
    "Equidae":          "a photo of a zebra or wild equid",
    "Elephantidae":     "a photo of an elephant",
    "Giraffidae":       "a photo of a giraffe",
    "Rhinocerotidae":   "a photo of a rhinoceros",
    "Suidae":           "a photo of a warthog, bushpig, or wild pig",
    "Hippopotamidae":   "a photo of a hippopotamus",
    "Cercopithecidae":  "a photo of a monkey or baboon",
    "Hominidae":        "a photo of an ape, chimpanzee, gorilla, or human",
    "Mustelidae":       "a photo of a honey badger or otter",
    "Viverridae":       "a photo of a civet or genet",
    "Herpestidae":      "a photo of a mongoose or meerkat",
    "Crocodylidae":     "a photo of a crocodile",
    "Varanidae":        "a photo of a monitor lizard",
    "Pythonidae":       "a photo of a python",
    "Testudinidae":     "a photo of a tortoise",
    "Struthionidae":    "a photo of an ostrich",
    "Otididae":         "a photo of a bustard",
    "Ciconiidae":       "a photo of a stork",
    "Numididae":        "a photo of a guineafowl",
    "Phasianidae":      "a photo of a francolin or partridge",
    "Bucorvidae":       "a photo of a ground hornbill",
    "Sagittariidae":    "a photo of a secretary bird",
    "Manidae":          "a photo of a pangolin",
    "Orycteropodidae":  "a photo of an aardvark",
    "Hystricidae":      "a photo of a porcupine",
    "Leporidae":        "a photo of a hare or rabbit",
    "Procaviidae":      "a photo of a hyrax",
}

_EMPTY_TAXONOMY = {"top": None, "candidates": [], "family_prediction": None, "family_confidence": 0.0}


class BioClipClassifier:
    """
    Species classifier using BioCLIP (zero-shot foundation model).
    Model: hf-hub:imageomics/bioclip
    """

    def __init__(self, species_list: Optional[List[str]] = None, low_spec: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.species_list: List[str] = species_list or []
        self.text_features: Optional[torch.Tensor] = None
        self.family_features: Optional[torch.Tensor] = None
        self.family_list: List[str] = []
        self.low_spec = low_spec

        self._load_model()
        if self.species_list:
            self.update_species_list(self.species_list)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            if self.low_spec:
                self.device = "cpu"
            print(f"Loading BioClip on {self.device} (Low-Spec: {self.low_spec})...")
            model_name = "hf-hub:imageomics/bioclip"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)

            if self.low_spec and self.model:
                try:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    print("BioClip dynamically quantized (INT8) successfully.")
                except Exception as q_err:
                    print(f"Failed to quantize BioClip: {q_err}")

            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)
            print("BioClip loaded successfully.")
        except Exception as exc:
            print(f"Error loading BioClip: {exc}")
            self.model = None

    # ------------------------------------------------------------------
    # Text feature pre-computation
    # ------------------------------------------------------------------

    def update_species_list(self, species_list: List[str]) -> None:
        """Recompute species text embeddings and family-level embeddings."""
        if not self.model:
            return
        self.species_list = species_list
        prompts = [
            f"a wildlife camera trap photo of a {s} in African savanna or bush"
            for s in species_list
        ]
        try:
            text = self.tokenizer(prompts).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(text)
                self.text_features = feats / feats.norm(dim=-1, keepdim=True)
            print(f"BioClip: Updated text features for {len(species_list)} species.")
        except Exception as exc:
            print(f"Error updating species list: {exc}")

        self._compute_family_features()

    def _compute_family_features(self) -> None:
        """Pre-compute family-level text embeddings for independent classification."""
        if not self.model:
            return
        families = list(FAMILY_PROMPTS.keys())
        prompts = [FAMILY_PROMPTS[f] for f in families]
        try:
            text = self.tokenizer(prompts).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(text)
                self.family_features = feats / feats.norm(dim=-1, keepdim=True)
            self.family_list = families
            print(f"BioClip: Computed family features for {len(families)} families.")
        except Exception as exc:
            print(f"BioClip family features error: {exc}")
            self.family_features = None
            self.family_list = []

    # ------------------------------------------------------------------
    # Image encoding (single forward pass, reused by both predict methods)
    # ------------------------------------------------------------------

    def _encode_image(self, image: Image.Image) -> Optional[torch.Tensor]:
        """Return normalised image feature vector, or None on error."""
        try:
            img_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(img_input)
                return feat / feat.norm(dim=-1, keepdim=True)
        except Exception as exc:
            print(f"BioClip encode error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Prediction — flat list (fast path, backward compatible)
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        results = self.predict_list(image, threshold=0.0, top_k=1)
        return results[0] if results else ("Unknown", 0.0)

    def predict_list(
        self,
        image: Image.Image,
        threshold: float = 0.0,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return list of (species, confidence) tuples above threshold."""
        if not self.model or self.text_features is None:
            return []
        img_feat = self._encode_image(image)
        if img_feat is None:
            return []
        try:
            sim = (100.0 * img_feat @ self.text_features.T).softmax(dim=-1)
            values, indices = sim[0].topk(min(top_k, len(self.species_list)))
            return [
                (self.species_list[i.item()], v.item())
                for v, i in zip(values, indices)
                if v.item() >= threshold
            ]
        except Exception as exc:
            print(f"BioClip predict_list error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Prediction — full taxonomy (rich path, used for ensemble agreement)
    # ------------------------------------------------------------------

    def predict_taxonomy(
        self,
        image: Image.Image,
        threshold: float = 0.0,
        top_k: int = 5,
    ) -> Dict:
        """
        Classify image and return full taxonomic path for each candidate.

        Single image forward pass; species-level and family-level scoring
        both use the same image embedding.

        Returns
        -------
        {
          "top": {
            "species": "lion",
            "scientific_name": "Panthera leo",
            "order": "Carnivora",
            "family": "Felidae",
            "genus": "Panthera",
            "path": "Mammalia > Carnivora > Felidae > Panthera > leo",
            "confidence": 0.45
          },
          "candidates": [ ...same structure, top_k entries... ],
          "family_prediction": "Felidae",   # independent family-level result
          "family_confidence": 0.78
        }
        """
        if not self.model or self.text_features is None:
            return dict(_EMPTY_TAXONOMY)

        img_feat = self._encode_image(image)
        if img_feat is None:
            return dict(_EMPTY_TAXONOMY)

        # ── Species-level scoring ──────────────────────────────────────────
        candidates: List[Dict] = []
        try:
            sim = (100.0 * img_feat @ self.text_features.T).softmax(dim=-1)
            values, indices = sim[0].topk(min(top_k, len(self.species_list)))
            for v, idx in zip(values, indices):
                score = v.item()
                if score < threshold:
                    continue
                name = self.species_list[idx.item()]
                tax = SPECIES_TAXONOMY.get(name, {})
                scientific = tax.get("scientific", "")
                genus = tax.get("genus", "")
                epithet = scientific.split()[-1] if scientific and " " in scientific else ""
                path = " > ".join(
                    p for p in [
                        "Mammalia",
                        tax.get("order", ""),
                        tax.get("family", ""),
                        genus,
                        epithet,
                    ]
                    if p
                )
                candidates.append({
                    "species":         name,
                    "scientific_name": scientific,
                    "order":           tax.get("order", ""),
                    "family":          tax.get("family", ""),
                    "genus":           genus,
                    "path":            path,
                    "confidence":      round(score, 4),
                })
        except Exception as exc:
            print(f"BioClip species scoring error: {exc}")

        # ── Family-level scoring (independent cross-check) ────────────────
        family_prediction: Optional[str] = None
        family_confidence: float = 0.0
        if self.family_features is not None and self.family_list:
            try:
                fam_sim = (100.0 * img_feat @ self.family_features.T).softmax(dim=-1)
                fam_val, fam_idx = fam_sim[0].topk(1)
                family_prediction = self.family_list[fam_idx.item()]
                family_confidence = round(fam_val.item(), 4)
            except Exception as exc:
                print(f"BioClip family scoring error: {exc}")

        return {
            "top":               candidates[0] if candidates else None,
            "candidates":        candidates,
            "family_prediction": family_prediction,
            "family_confidence": family_confidence,
        }
