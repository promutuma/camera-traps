"""
Species Reference Library
Comprehensive African wildlife database covering mammals, reptiles, and
commonly camera-trapped birds. Supports synonym resolution (common ↔ scientific),
search, and user additions via Data Manager.
"""

import sqlite3
import pandas as pd
from typing import Optional


# ------------------------------------------------------------------
# Pre-loaded species data
# Columns: (common_name, scientific_name, family, order_, iucn_status, regional_status, notes)
# regional_status = local / national conservation status where known, else repeats iucn_status
# ------------------------------------------------------------------

SPECIES_DATA = [
    # ══════════════════════════════════════════════════════════
    # BOVIDAE — antelopes, buffalo, sheep relatives
    # ══════════════════════════════════════════════════════════
    ("African Buffalo",         "Syncerus caffer",              "Bovidae",          "Artiodactyla",   "Near Threatened", "Near Threatened", "Herd species; keystone of intact savanna; camera-trap indicator of prey base."),
    ("African Forest Buffalo",  "Syncerus caffer nanus",        "Bovidae",          "Artiodactyla",   "Near Threatened", "Near Threatened", "Smaller red forest form; West/Central Africa."),
    ("Blue Wildebeest",         "Connochaetes taurinus",        "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Mass migratory herds; iconic East African species."),
    ("Black Wildebeest",        "Connochaetes gnou",            "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Endemic to southern African grasslands."),
    ("Common Eland",            "Taurotragus oryx",             "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Largest antelope; woodland-grassland ecotones."),
    ("Giant Eland",             "Taurotragus derbianus",        "Bovidae",          "Artiodactyla",   "Vulnerable",      "Vulnerable",      "Forest-savanna mosaic; West/Central Africa."),
    ("Greater Kudu",            "Tragelaphus strepsiceros",     "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Tall spiral horns; woodland specialist."),
    ("Lesser Kudu",             "Tragelaphus imberbis",         "Bovidae",          "Artiodactyla",   "Near Threatened", "Near Threatened", "Arid bush of East Africa."),
    ("Bushbuck",                "Tragelaphus scriptus",         "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Solitary; dense riparian vegetation."),
    ("Nyala",                   "Tragelaphus angasii",          "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Thicket specialist; strong sexual dimorphism."),
    ("Mountain Nyala",          "Tragelaphus buxtoni",          "Bovidae",          "Endangered",     "Endangered",      "Endangered",      "Ethiopian highland endemic; Bale Mountains."),
    ("Sitatunga",               "Tragelaphus spekii",           "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Semi-aquatic; swamps and papyrus."),
    ("Bongo",                   "Tragelaphus eurycerus",        "Bovidae",          "Near Threatened","Near Threatened", "Near Threatened", "Large forest antelope; rare and elusive."),
    ("Impala",                  "Aepyceros melampus",           "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Most common medium antelope of East/Southern Africa."),
    ("Black-faced Impala",      "Aepyceros melampus petersi",   "Bovidae",          "Artiodactyla",   "Vulnerable",      "Vulnerable",      "Namibian subspecies; smaller range."),
    ("Thomson's Gazelle",       "Eudorcas thomsoni",            "Bovidae",          "Artiodactyla",   "Near Threatened", "Near Threatened", "Abundant East African plains gazelle."),
    ("Grant's Gazelle",         "Nanger granti",                "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Larger gazelle; survives without surface water."),
    ("Springbok",               "Antidorcas marsupialis",       "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "National animal of South Africa; pronking display."),
    ("Gerenuk",                 "Litocranius walleri",          "Bovidae",          "Artiodactyla",   "Near Threatened", "Near Threatened", "Long-necked browser; stands bipedally to feed."),
    ("Dik-dik",                 "Madoqua kirkii",               "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Tiny monogamous antelope; dry bush."),
    ("Steenbok",                "Raphicerus campestris",        "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Small, alert antelope of open country."),
    ("Oribi",                   "Ourebia ourebi",               "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Small floodplain antelope."),
    ("Common Duiker",           "Sylvicapra grimmia",           "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Widespread; forest edges and bush."),
    ("Blue Duiker",             "Philantomba monticola",        "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Tiny forest duiker."),
    ("Red Duiker",              "Cephalophus natalensis",       "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Coastal and montane forest."),
    ("Yellow-backed Duiker",    "Cephalophus silvicultor",      "Bovidae",          "Artiodactyla",   "Near Threatened", "Near Threatened", "Largest duiker; forest specialist."),
    ("Roan Antelope",           "Hippotragus equinus",          "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Open woodland and tall grass."),
    ("Sable Antelope",          "Hippotragus niger",            "Bovidae",          "Least Concern",  "Least Concern",   "Least Concern",   "Curved horns; woodland/grassland edge."),
    ("Gemsbok",                 "Oryx gazella",                 "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Arid and semi-arid southern Africa."),
    ("East African Oryx",       "Oryx beisa",                   "Bovidae",          "Artiodactyla",   "Near Threatened", "Near Threatened", "Dry acacia and semi-arid Horn of Africa."),
    ("Scimitar Oryx",           "Oryx dammah",                  "Bovidae",          "Artiodactyla",   "Extinct in Wild", "Extinct in Wild", "Reintroduction programmes underway in Chad."),
    ("Common Waterbuck",        "Kobus ellipsiprymnus",         "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Always close to permanent water."),
    ("Defassa Waterbuck",       "Kobus ellipsiprymnus defassa", "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "White rump patch (not ring); West/Central Africa."),
    ("Nile Lechwe",             "Kobus megaceros",              "Bovidae",          "Artiodactyla",   "Vulnerable",      "Vulnerable",      "Semi-aquatic; endemic to Gambella/South Sudan floodplains."),
    ("Lechwe",                  "Kobus leche",                  "Bovidae",          "Artiodactyla",   "Near Threatened", "Near Threatened", "Floodplain antelope; Zambia and Botswana."),
    ("Puku",                    "Kobus vardonii",               "Bovidae",          "Artiodactyla",   "Near Threatened", "Near Threatened", "Floodplain grasslands of Central/Southern Africa."),
    ("White-eared Kob",         "Kobus kob leucotis",           "Bovidae",          "Artiodactyla",   "Near Threatened", "Near Threatened", "Mass migration through Gambella/South Sudan."),
    ("Uganda Kob",              "Kobus kob thomasi",            "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Lekking antelope; Uganda national symbol."),
    ("Bohor Reedbuck",          "Redunca redunca",              "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Wetland edges and tall grassland."),
    ("Mountain Reedbuck",       "Redunca fulvorufula",          "Bovidae",          "Least Concern",  "Least Concern",   "Least Concern",   "Rocky hillsides."),
    ("Topi",                    "Damaliscus lunatus jimela",    "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Fast-running; open plains of East Africa."),
    ("Tiang",                   "Damaliscus lunatus tiang",     "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Seasonal migrant through Gambella corridor."),
    ("Tsessebe",                "Damaliscus lunatus lunatus",   "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "One of Africa's fastest antelopes."),
    ("Hartebeest",              "Alcelaphus buselaphus",        "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Long face; several subspecies across Africa."),
    ("Lichtenstein's Hartebeest","Alcelaphus lichtensteinii",   "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Miombo woodland of Central/Southern Africa."),
    ("Klipspringer",            "Oreotragus oreotragus",        "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Rock-dwelling; walks on tips of hooves."),
    ("Suni",                    "Neotragus moschatus",          "Bovidae",          "Artiodactyla",   "Least Concern",   "Least Concern",   "Tiny forest antelope; coastal East Africa."),

    # ══════════════════════════════════════════════════════════
    # ELEPHANTIDAE
    # ══════════════════════════════════════════════════════════
    ("African Bush Elephant",   "Loxodonta africana",           "Elephantidae",     "Proboscidea",    "Vulnerable",      "Endangered",      "Largest land animal; keystone ecosystem engineer."),
    ("African Forest Elephant", "Loxodonta cyclotis",           "Elephantidae",     "Proboscidea",    "Critically Endangered","Critically Endangered","Forest specialist; critical for seed dispersal."),

    # ══════════════════════════════════════════════════════════
    # RHINOCEROTIDAE
    # ══════════════════════════════════════════════════════════
    ("White Rhinoceros",        "Ceratotherium simum",          "Rhinocerotidae",   "Perissodactyla", "Near Threatened", "Near Threatened", "Grazer; two subspecies (southern and northern)."),
    ("Black Rhinoceros",        "Diceros bicornis",             "Rhinocerotidae",   "Perissodactyla", "Critically Endangered","Critically Endangered","Browser; hook-lipped; severely poached."),

    # ══════════════════════════════════════════════════════════
    # EQUIDAE
    # ══════════════════════════════════════════════════════════
    ("Plains Zebra",            "Equus quagga",                 "Equidae",          "Perissodactyla", "Near Threatened", "Near Threatened", "Most common zebra; several subspecies."),
    ("Grevy's Zebra",           "Equus grevyi",                 "Equidae",          "Perissodactyla", "Endangered",      "Endangered",      "Largest wild equid; narrow stripes; arid northeast."),
    ("Mountain Zebra",          "Equus zebra",                  "Equidae",          "Perissodactyla", "Vulnerable",      "Vulnerable",      "South African mountains; gridiron rump pattern."),
    ("African Wild Ass",        "Equus africanus",              "Equidae",          "Perissodactyla", "Critically Endangered","Critically Endangered","Ancestor of domestic donkey; Horn of Africa."),

    # ══════════════════════════════════════════════════════════
    # HIPPOPOTAMIDAE
    # ══════════════════════════════════════════════════════════
    ("Common Hippopotamus",     "Hippopotamus amphibius",       "Hippopotamidae",   "Artiodactyla",   "Vulnerable",      "Vulnerable",      "Semi-aquatic; grazes on land at night."),
    ("Pygmy Hippopotamus",      "Choeropsis liberiensis",       "Hippopotamidae",   "Artiodactyla",   "Endangered",      "Endangered",      "Secretive; West African forest streams."),

    # ══════════════════════════════════════════════════════════
    # SUIDAE
    # ══════════════════════════════════════════════════════════
    ("Common Warthog",          "Phacochoerus africanus",       "Suidae",           "Artiodactyla",   "Least Concern",   "Least Concern",   "Diurnal; common at waterholes."),
    ("Desert Warthog",          "Phacochoerus aethiopicus",     "Suidae",           "Artiodactyla",   "Least Concern",   "Least Concern",   "Arid northeast Africa."),
    ("Bushpig",                 "Potamochoerus larvatus",       "Suidae",           "Artiodactyla",   "Least Concern",   "Least Concern",   "Nocturnal; forest and dense bush."),
    ("Red River Hog",           "Potamochoerus porcus",         "Suidae",           "Artiodactyla",   "Least Concern",   "Least Concern",   "Colourful; West/Central African forests."),
    ("Giant Forest Hog",        "Hylochoerus meinertzhageni",   "Suidae",           "Artiodactyla",   "Least Concern",   "Least Concern",   "Largest wild pig; montane and lowland forest."),

    # ══════════════════════════════════════════════════════════
    # GIRAFFIDAE
    # ══════════════════════════════════════════════════════════
    ("Northern Giraffe",        "Giraffa camelopardalis",       "Giraffidae",       "Artiodactyla",   "Vulnerable",      "Vulnerable",      "Tallest land animal; nine subspecies recognised."),
    ("Southern Giraffe",        "Giraffa giraffa",              "Giraffidae",       "Artiodactyla",   "Vulnerable",      "Vulnerable",      "Southern African populations."),
    ("Okapi",                   "Okapia johnstoni",             "Giraffidae",       "Artiodactyla",   "Endangered",      "Endangered",      "Forest giraffe relative; Congo Basin only."),

    # ══════════════════════════════════════════════════════════
    # FELIDAE
    # ══════════════════════════════════════════════════════════
    ("Lion",                    "Panthera leo",                 "Felidae",          "Carnivora",      "Vulnerable",      "Vulnerable",      "Apex predator; intact prey base indicator."),
    ("Leopard",                 "Panthera pardus",              "Felidae",          "Carnivora",      "Vulnerable",      "Vulnerable",      "Most widespread big cat; nocturnal."),
    ("Cheetah",                 "Acinonyx jubatus",             "Felidae",          "Carnivora",      "Vulnerable",      "Vulnerable",      "Fastest land animal; diurnal hunter."),
    ("Serval",                  "Leptailurus serval",           "Felidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Long legs; hunts in tall grass and wetland edges."),
    ("Caracal",                 "Caracal caracal",              "Felidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Tufted ears; dry bush and rocky terrain."),
    ("African Wildcat",         "Felis lybica",                 "Felidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Ancestor of domestic cat; widespread."),
    ("African Golden Cat",      "Caracal aurata",               "Felidae",          "Carnivora",      "Vulnerable",      "Vulnerable",      "Forest specialist; rarely camera-trapped."),
    ("Sand Cat",                "Felis margarita",              "Felidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Sahara and semi-arid north Africa."),
    ("Black-footed Cat",        "Felis nigripes",               "Felidae",          "Carnivora",      "Vulnerable",      "Vulnerable",      "Smallest African cat; southern Africa arid zones."),

    # ══════════════════════════════════════════════════════════
    # CANIDAE
    # ══════════════════════════════════════════════════════════
    ("African Wild Dog",        "Lycaon pictus",                "Canidae",          "Carnivora",      "Endangered",      "Endangered",      "Pack hunter; requires large connected ranges."),
    ("Black-backed Jackal",     "Canis mesomelas",              "Canidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Adaptable scavenger and predator."),
    ("Side-striped Jackal",     "Canis adustus",                "Canidae",          "Carnivora",      "Least Concern",   "Least Concern",   "More forest-associated; lower elevations."),
    ("Golden Jackal",           "Canis aureus",                 "Canidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Generalist; northern and eastern Africa."),
    ("Ethiopian Wolf",          "Canis simensis",               "Canidae",          "Carnivora",      "Endangered",      "Endangered",      "Rarest canid; Ethiopian highlands only."),
    ("Bat-eared Fox",           "Otocyon megalotis",            "Canidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Huge ears for hearing termites; arid zones."),
    ("Cape Fox",                "Vulpes chama",                 "Canidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Only true fox in sub-Saharan Africa."),
    ("Fennec Fox",              "Vulpes zerda",                 "Canidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Sahara; enormous ears for heat dissipation."),
    ("Pale Fox",                "Vulpes pallida",               "Canidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Sahel zone of West Africa."),
    ("Rüppell's Fox",           "Vulpes rueppelli",             "Canidae",          "Carnivora",      "Least Concern",   "Least Concern",   "Desert and semi-arid north/northeast Africa."),

    # ══════════════════════════════════════════════════════════
    # HYAENIDAE
    # ══════════════════════════════════════════════════════════
    ("Spotted Hyena",           "Crocuta crocuta",              "Hyaenidae",        "Carnivora",      "Least Concern",   "Least Concern",   "Most numerous large carnivore; complex social clans."),
    ("Striped Hyena",           "Hyaena hyaena",                "Hyaenidae",        "Carnivora",      "Near Threatened", "Near Threatened", "Nocturnal scavenger; north/east Africa."),
    ("Brown Hyena",             "Parahyaena brunnea",           "Hyaenidae",        "Carnivora",      "Near Threatened", "Near Threatened", "Southern Africa arid zones."),
    ("Aardwolf",                "Proteles cristata",            "Hyaenidae",        "Carnivora",      "Least Concern",   "Least Concern",   "Insectivore; feeds almost exclusively on termites."),

    # ══════════════════════════════════════════════════════════
    # HERPESTIDAE — mongooses
    # ══════════════════════════════════════════════════════════
    ("Banded Mongoose",         "Mungos mungo",                 "Herpestidae",      "Carnivora",      "Least Concern",   "Least Concern",   "Highly social; diurnal; termite mounds."),
    ("Dwarf Mongoose",          "Helogale parvula",             "Herpestidae",      "Carnivora",      "Least Concern",   "Least Concern",   "Smallest carnivore; termite-mound colonies."),
    ("Egyptian Mongoose",       "Herpestes ichneumon",          "Herpestidae",      "Carnivora",      "Least Concern",   "Least Concern",   "Nile Valley and savanna; snake killer."),
    ("Slender Mongoose",        "Urva flavescens",              "Herpestidae",      "Carnivora",      "Least Concern",   "Least Concern",   "Solitary; sub-Saharan Africa."),
    ("Meerkat",                 "Suricata suricatta",           "Herpestidae",      "Carnivora",      "Least Concern",   "Least Concern",   "Sentinel behaviour; Kalahari specialist."),
    ("White-tailed Mongoose",   "Ichneumia albicauda",          "Herpestidae",      "Carnivora",      "Least Concern",   "Least Concern",   "Nocturnal; widespread sub-Saharan Africa."),

    # ══════════════════════════════════════════════════════════
    # VIVERRIDAE — civets and genets
    # ══════════════════════════════════════════════════════════
    ("African Civet",           "Civettictis civetta",          "Viverridae",       "Carnivora",      "Least Concern",   "Least Concern",   "Large nocturnal omnivore; civetone musk."),
    ("Common Genet",            "Genetta genetta",              "Viverridae",       "Carnivora",      "Least Concern",   "Least Concern",   "Slender spotted carnivore; riparian forest."),
    ("Large-spotted Genet",     "Genetta tigrina",              "Viverridae",       "Carnivora",      "Least Concern",   "Least Concern",   "Forest and bush; larger spots than common genet."),
    ("Servaline Genet",         "Genetta servalina",            "Viverridae",       "Carnivora",      "Least Concern",   "Least Concern",   "Central African forests."),
    ("African Palm Civet",      "Nandinia binotata",            "Nandiniidae",      "Carnivora",      "Least Concern",   "Least Concern",   "Arboreal; forest and dense woodland."),

    # ══════════════════════════════════════════════════════════
    # MUSTELIDAE
    # ══════════════════════════════════════════════════════════
    ("Honey Badger",            "Mellivora capensis",           "Mustelidae",       "Carnivora",      "Least Concern",   "Least Concern",   "Fearless; wide habitat tolerance; nocturnal."),
    ("Spotted-necked Otter",    "Hydrictis maculicollis",       "Mustelidae",       "Carnivora",      "Near Threatened", "Near Threatened", "Fish-eating; clear lakes and rivers."),
    ("Cape Clawless Otter",     "Aonyx capensis",               "Mustelidae",       "Carnivora",      "Near Threatened", "Near Threatened", "Largest freshwater otter; no webbing on forefeet."),
    ("Congo Clawless Otter",    "Aonyx congicus",               "Mustelidae",       "Carnivora",      "Near Threatened", "Near Threatened", "Central African forest rivers."),
    ("Zorilla",                 "Ictonyx striatus",             "Mustelidae",       "Carnivora",      "Least Concern",   "Least Concern",   "Striped skunk-like; nocturnal."),

    # ══════════════════════════════════════════════════════════
    # PRIMATES
    # ══════════════════════════════════════════════════════════
    ("Mountain Gorilla",        "Gorilla beringei beringei",    "Hominidae",        "Primates",       "Endangered",      "Endangered",      "Virunga volcanoes and Bwindi; <1,100 individuals."),
    ("Eastern Gorilla",         "Gorilla beringei",             "Hominidae",        "Primates",       "Critically Endangered","Critically Endangered","Eastern DRC and adjacent countries."),
    ("Western Lowland Gorilla", "Gorilla gorilla gorilla",      "Hominidae",        "Primates",       "Critically Endangered","Critically Endangered","Most numerous gorilla; West/Central Africa."),
    ("Common Chimpanzee",       "Pan troglodytes",              "Hominidae",        "Primates",       "Endangered",      "Endangered",      "Closest living relative to humans."),
    ("Bonobo",                  "Pan paniscus",                 "Hominidae",        "Primates",       "Endangered",      "Endangered",      "Democratic Republic of Congo only."),
    ("Olive Baboon",            "Papio anubis",                 "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Adaptable; widely distributed; near water."),
    ("Chacma Baboon",           "Papio ursinus",                "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Southern Africa; Cape Peninsula population."),
    ("Yellow Baboon",           "Papio cynocephalus",           "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "East Africa savanna."),
    ("Hamadryas Baboon",        "Papio hamadryas",              "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Cliff-dwelling; Horn of Africa."),
    ("Gelada",                  "Theropithecus gelada",         "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Ethiopian highland endemic; grass-grazing primate."),
    ("Vervet Monkey",           "Chlorocebus pygerythrus",      "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Common; riparian woodland; alarm calls."),
    ("Grivet",                  "Chlorocebus aethiops",         "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Northeast Africa savanna."),
    ("Patas Monkey",            "Erythrocebus patas",           "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Fastest primate; open woodland specialist."),
    ("Mandrill",                "Mandrillus sphinx",            "Cercopithecidae",  "Primates",       "Vulnerable",      "Vulnerable",      "Colourful face; West/Central African rainforest."),
    ("Drill",                   "Mandrillus leucophaeus",       "Cercopithecidae",  "Primates",       "Endangered",      "Endangered",      "Nigeria/Cameroon/Bioko; critically declining."),
    ("Eastern Black-and-white Colobus","Colobus guereza",       "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Montane and riparian forests; camera-trapped in trees."),
    ("Red Colobus",             "Piliocolobus badius",          "Cercopithecidae",  "Primates",       "Endangered",      "Endangered",      "West African forests; hunted by chimpanzees."),
    ("Blue Monkey",             "Cercopithecus mitis",          "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Forest canopy; East African mountains."),
    ("Red-tailed Monkey",       "Cercopithecus ascanius",       "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Ugandan and Congo Basin forests."),
    ("De Brazza's Monkey",      "Cercopithecus neglectus",      "Cercopithecidae",  "Primates",       "Least Concern",   "Least Concern",   "Swamp forest; striking white beard."),
    ("Greater Galago",          "Otolemur crassicaudatus",      "Galagidae",        "Primates",       "Least Concern",   "Least Concern",   "Nocturnal; large eyes; loud calls."),
    ("Bushbaby",                "Galago moholi",                "Galagidae",        "Primates",       "Least Concern",   "Least Concern",   "Southern lesser galago; well camera-trapped."),

    # ══════════════════════════════════════════════════════════
    # TUBULIDENTATA
    # ══════════════════════════════════════════════════════════
    ("Aardvark",                "Orycteropus afer",             "Orycteropodidae",  "Tubulidentata",  "Least Concern",   "Least Concern",   "Nocturnal; termite specialist; burrows used by others."),

    # ══════════════════════════════════════════════════════════
    # PHOLIDOTA — pangolins
    # ══════════════════════════════════════════════════════════
    ("Ground Pangolin",         "Smutsia temminckii",           "Manidae",          "Pholidota",      "Vulnerable",      "Vulnerable",      "Most trafficked; southern/eastern Africa."),
    ("Giant Pangolin",          "Smutsia gigantea",             "Manidae",          "Pholidota",      "Endangered",      "Endangered",      "Largest pangolin; West/Central Africa."),
    ("Tree Pangolin",           "Phataginus tricuspis",         "Manidae",          "Pholidota",      "Endangered",      "Endangered",      "Arboreal; Central/West African forests."),
    ("Long-tailed Pangolin",    "Phataginus tetradactyla",      "Manidae",          "Pholidota",      "Endangered",      "Endangered",      "Highly arboreal; very long tail."),

    # ══════════════════════════════════════════════════════════
    # RODENTIA
    # ══════════════════════════════════════════════════════════
    ("Cape Porcupine",          "Hystrix africaeaustralis",     "Hystricidae",      "Rodentia",       "Least Concern",   "Least Concern",   "Largest African rodent; nocturnal."),
    ("African Crested Porcupine","Hystrix cristata",            "Hystricidae",      "Rodentia",       "Least Concern",   "Least Concern",   "North and West Africa."),
    ("South African Springhare","Pedetes capensis",             "Pedetidae",        "Rodentia",       "Least Concern",   "Least Concern",   "Bipedal; open sandy areas; night forager."),
    ("East African Springhare", "Pedetes surdaster",            "Pedetidae",        "Rodentia",       "Least Concern",   "Least Concern",   "East Africa dry zones."),

    # ══════════════════════════════════════════════════════════
    # LAGOMORPHA
    # ══════════════════════════════════════════════════════════
    ("Cape Hare",               "Lepus capensis",               "Leporidae",        "Lagomorpha",     "Least Concern",   "Least Concern",   "Widespread; open grassland and semi-arid."),
    ("African Savanna Hare",    "Lepus victoriae",              "Leporidae",        "Lagomorpha",     "Least Concern",   "Least Concern",   "East African savanna."),
    ("Ethiopian Highland Hare", "Lepus starcki",                "Leporidae",        "Lagomorpha",     "Least Concern",   "Least Concern",   "Ethiopian plateau; montane grassland."),
    ("Rock Hyrax",              "Procavia capensis",            "Procaviidae",      "Hyracoidea",     "Least Concern",   "Least Concern",   "Rocky outcrops; camera-trap favourite."),
    ("Yellow-spotted Rock Hyrax","Heterohyrax brucei",          "Procaviidae",      "Hyracoidea",     "Least Concern",   "Least Concern",   "Rocky habitats; East Africa."),
    ("Tree Hyrax",              "Dendrohyrax arboreus",         "Procaviidae",      "Hyracoidea",     "Least Concern",   "Least Concern",   "Forest; loud nocturnal screaming call."),

    # ══════════════════════════════════════════════════════════
    # REPTILIA — commonly camera-trapped
    # ══════════════════════════════════════════════════════════
    ("Nile Crocodile",          "Crocodylus niloticus",         "Crocodylidae",     "Crocodilia",     "Least Concern",   "Least Concern",   "Widespread; rivers, lakes, and wetlands."),
    ("Dwarf Crocodile",         "Osteolaemus tetraspis",        "Crocodylidae",     "Crocodilia",     "Vulnerable",      "Vulnerable",      "Small; West/Central African forest streams."),
    ("Nile Monitor",            "Varanus niloticus",            "Varanidae",        "Squamata",       "Least Concern",   "Least Concern",   "Large lizard; rivers and wetlands."),
    ("Rock Monitor",            "Varanus albigularis",          "Varanidae",        "Squamata",       "Least Concern",   "Least Concern",   "Savanna and rocky areas; southern Africa."),
    ("African Rock Python",     "Python sebae",                 "Pythonidae",       "Squamata",       "Least Concern",   "Least Concern",   "Largest African snake; ambush predator."),
    ("Puff Adder",              "Bitis arietans",               "Viperidae",        "Squamata",       "Least Concern",   "Least Concern",   "Most common cause of snakebite in Africa."),
    ("African Spurred Tortoise","Centrochelys sulcata",         "Testudinidae",     "Testudines",     "Vulnerable",      "Vulnerable",      "Largest mainland tortoise; Sahel zone."),
    ("Leopard Tortoise",        "Stigmochelys pardalis",        "Testudinidae",     "Testudines",     "Least Concern",   "Least Concern",   "Widespread; camera-trapped at waterholes."),

    # ══════════════════════════════════════════════════════════
    # AVES — birds commonly triggered on camera traps
    # ══════════════════════════════════════════════════════════
    ("Ostrich",                 "Struthio camelus",             "Struthionidae",    "Struthioniformes","Least Concern",  "Least Concern",   "Largest bird; savanna and semi-arid."),
    ("Kori Bustard",            "Ardeotis kori",                "Otididae",         "Otidiformes",    "Near Threatened", "Near Threatened", "Heaviest flying bird; open grassland."),
    ("Secretary Bird",          "Sagittarius serpentarius",     "Sagittariidae",    "Accipitriformes","Vulnerable",      "Vulnerable",      "Terrestrial raptor; walks camera traps."),
    ("Southern Ground Hornbill","Bucorvus leadbeateri",         "Bucorvidae",       "Bucerotiformes", "Vulnerable",      "Vulnerable",      "Terrestrial; slow-breeding; social groups."),
    ("Helmeted Guineafowl",     "Numida meleagris",             "Numididae",        "Galliformes",    "Least Concern",   "Least Concern",   "Common flock; frequent camera-trap trigger."),
    ("Crested Francolin",       "Ortygornis sephaena",          "Phasianidae",      "Galliformes",    "Least Concern",   "Least Concern",   "Bush and thicket; walks in front of traps."),
    ("Vulturine Guineafowl",    "Acryllium vulturinum",         "Numididae",        "Galliformes",    "Least Concern",   "Least Concern",   "Arid northeast Africa."),
    ("Marabou Stork",           "Leptoptilos crumenifer",       "Ciconiidae",       "Ciconiiformes",  "Least Concern",   "Least Concern",   "Scavenger; tall ground-dwelling bird."),
    ("African Jacana",          "Actophilornis africanus",      "Jacanidae",        "Charadriiformes","Least Concern",   "Least Concern",   "Lily-trotter; wetland camera traps."),
    ("Goliath Heron",           "Ardea goliath",                "Ardeidae",         "Pelecaniformes", "Least Concern",   "Least Concern",   "World's largest heron; waterside traps."),
]

# Synonym table: alternative name (lower-case) → accepted common name
SYNONYMS = {
    # hippo
    "hippopotamus":                     "Common Hippopotamus",
    "hippo":                            "Common Hippopotamus",
    # zebra
    "burchell's zebra":                 "Plains Zebra",
    "common zebra":                     "Plains Zebra",
    "quagga zebra":                     "Plains Zebra",
    "imperial zebra":                   "Grevy's Zebra",
    # wild dog
    "painted dog":                      "African Wild Dog",
    "painted wolf":                     "African Wild Dog",
    "cape hunting dog":                 "African Wild Dog",
    "hunting dog":                      "African Wild Dog",
    "lycaon":                           "African Wild Dog",
    # jackals
    "common jackal":                    "Golden Jackal",
    "asiatic jackal":                   "Golden Jackal",
    "silver-backed jackal":             "Black-backed Jackal",
    # elephant
    "savanna elephant":                 "African Bush Elephant",
    "african savanna elephant":         "African Bush Elephant",
    "bush elephant":                    "African Bush Elephant",
    # buffalo
    "cape buffalo":                     "African Buffalo",
    "water buffalo":                    "African Buffalo",
    # kob
    "kob":                              "Uganda Kob",
    "white eared kob":                  "White-eared Kob",
    "nile lechwi":                      "Nile Lechwe",
    # wildebeest
    "gnu":                              "Blue Wildebeest",
    "brindled gnu":                     "Blue Wildebeest",
    "white-tailed gnu":                 "Black Wildebeest",
    # hartebeest / topi
    "topi":                             "Topi",
    "tiang":                            "Tiang",
    "lelwel hartebeest":                "Hartebeest",
    "kongoni":                          "Hartebeest",
    # giraffe
    "masai giraffe":                    "Northern Giraffe",
    "reticulated giraffe":              "Northern Giraffe",
    "angolan giraffe":                  "Southern Giraffe",
    "south african giraffe":            "Southern Giraffe",
    # rhino
    "rhino":                            "White Rhinoceros",
    "white rhino":                      "White Rhinoceros",
    "black rhino":                      "Black Rhinoceros",
    "hook-lipped rhinoceros":           "Black Rhinoceros",
    "square-lipped rhinoceros":         "White Rhinoceros",
    # cheetah
    "cheetah":                          "Cheetah",
    # big cats
    "african leopard":                  "Leopard",
    "african lion":                     "Lion",
    # hyena
    "hyena":                            "Spotted Hyena",
    "laughing hyena":                   "Spotted Hyena",
    # pangolin
    "pangolin":                         "Ground Pangolin",
    "scaly anteater":                   "Ground Pangolin",
    # gorilla
    "gorilla":                          "Western Lowland Gorilla",
    # mongoose
    "meerkat":                          "Meerkat",
    "suricate":                         "Meerkat",
    # warthog
    "warthog":                          "Common Warthog",
    # eland
    "eland":                            "Common Eland",
    # baboon
    "baboon":                           "Olive Baboon",
    "anubis baboon":                    "Olive Baboon",
    # oryx
    "oryx":                             "East African Oryx",
    "beisa oryx":                       "East African Oryx",
    "gemsbok":                          "Gemsbok",
    # otter
    "otter":                            "Cape Clawless Otter",
    # crocodile
    "crocodile":                        "Nile Crocodile",
    # monitor
    "monitor lizard":                   "Nile Monitor",
    "leguaan":                          "Rock Monitor",
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
            """, SPECIES_DATA)

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
