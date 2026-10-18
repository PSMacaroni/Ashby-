"""
Every number the slide analysis uses, in one place, with where it comes from.

How much to trust each number (see the comment at the end of each line):
  doc          from our features doc ("What the Colon Tumor Front Analyzer looks for")
  literature   a published cutoff (named in the comment); still check before it goes on a slide
  PLACEHOLDER  our guess so the code runs. Calibrate with the pathologist before showing results.

Distances are in microns (µm). One tile is 224 px at 0.5 µm/px = 112 µm wide.
"""

# --- Tissue map -----------------------------------------------------------------------------
SMOOTH_SIZE = 3                    # doc: average probabilities over a 3x3 neighborhood
TUMOR_PROB = 0.5                   # doc: tumor = smoothed P(tumor) >= 0.5
SMALL_HOLE_TILES = 4               # PLACEHOLDER: fill holes in the tumor mask up to this many tiles
SECTION_GAP_UM = 300               # PLACEHOLDER: tissue pieces closer than this are one section
LOW_CONFIDENCE = 0.5               # PLACEHOLDER: top probability below this = "unsure here"
TISSUE_MASK_SAT = 12               # measured: on our 5 TCGA slides the old rule (saturation > Otsu, 31-70) dropped
                                   #   pale real tissue (submucosa, loose stroma, fat); > 12 keeps it. Glass that
                                   #   slips in is removed per tile by the background guard (src/detectors.py)
BLUR_MIN = 0.001                   # PLACEHOLDER: Laplacian variance below this = blurry (doc: pick by
                                   #   looking at 20 tiles); used when tiling, see src/slides.py

# --- Zones around the invasive front --------------------------------------------------------
CORE_UM = 500                      # doc: core = more than 500 µm inside the front
FRONT_BAND_UM = 500                # doc: front band = within 500 µm on either side
PERITUMORAL_UM = 2000              # doc: peritumoral = 0.5 to 2 mm outside

# --- Growth pattern -------------------------------------------------------------------------
SOLIDITY_INFILTRATIVE = 0.85       # PLACEHOLDER: doc says pick it from 10 slides the pathologist labels
INFILTRATIVE_MIN_PROTRUSIONS = 3   # PLACEHOLDER: fingers + satellites at or above this = infiltrative
FRACTAL_MIN_BOXES = 8              # PLACEHOLDER: only use box sizes that still hit this many boxes
ROUGHNESS_SMOOTH_UM = 1000         # PLACEHOLDER: "wide moving average" used to measure front roughness
FINGER_DISK_TILES = 3              # doc: fingers = tumor removed by an opening with a 3-tile disk
MIN_FINGER_TILES = 3               # PLACEHOLDER: smaller leftovers are noise, not fingers
SATELLITE_MAX_UM = 1000            # doc: satellites = tumor pieces within 1 mm of the main mass
ARC_LENGTH_UM = 1000               # doc: score the front in arcs of about 1 mm
ARC_NEIGHBOR_UM = 1500             # PLACEHOLDER: fingers/satellites this close to an arc count for it
ARC_ROUGHNESS_INFILTRATIVE_UM = 60 # PLACEHOLDER: arcs rougher than this count as infiltrative
RAY_MAX_UM = 2000                  # doc: look 2 mm ahead of the front ("what it is heading into")

# --- Depth, fat, mucin, necrosis ------------------------------------------------------------
FAT_DEPTH_FLAG_UM = 5000           # doc / literature (rectal cancer, verify before a slide): >5 mm
                                   #   past the muscle carries a worse outlook
NECROSIS_RIM_UM = 200              # doc: tumor within ~200 µm (2 tiles) of necrosis = likely low oxygen
MUCINOUS_FRACTION = 0.5            # literature (WHO): >50% extracellular mucin = mucinous adenocarcinoma
POOL_RING_FAT = 0.5                # PLACEHOLDER: a mucin pool "sits in fat" if >= 50% of its ring is fat

# --- Stroma ---------------------------------------------------------------------------------
TSR_WINDOW_DIAMETER_UM = 2000      # doc: ~2 mm circle, roughly one 10x microscope field
TSR_STROMA_HIGH = 0.5              # literature (Gao et al. 2021): stroma-high = more than 50% stroma
TSR_MIN_TILES = 10                 # PLACEHOLDER: skip windows with fewer tumor + stroma tiles

# --- Immune cells ---------------------------------------------------------------------------
IMMUNE_LOW = 0.05                  # PLACEHOLDER: mean immune score (p_immune if present, else P(lymphocytes))
                                   #   below this = "few lymphocytes"; recalibrate when switching scores
IMMUNE_CORE_TO_MARGIN = 0.5        # PLACEHOLDER: core/margin ratio above this = "lymphocytes get inside"

# --- Lymph nodes, satellites, deposits ------------------------------------------------------
NODE_MIN_DIAMETER_UM = 1000        # doc: lymphocyte nodules over ~1 mm are probably lymph nodes
DEPOSIT_RING_FAT = 0.75            # PLACEHOLDER: "fully surrounded by fat" = >= 75% of its ring is fat

# --- Flags (text phrases + geometry) --------------------------------------------------------
TOP_K = 5                          # doc: show the top 5 "look here" tiles per flag
STRUCTURE_PROB = 0.5               # a nerve/artery detector score above this counts as a nerve/artery
PHRASE_TOP_FRACTION = 0.02         # PLACEHOLDER: without a detector, the top 2% of tiles by phrase score
PHRASE_STRONG_Z = 4.0              # PLACEHOLDER: a phrase hit counts as "strong" (risk map, high-risk
                                   #   list) only if it is this many standard deviations above the slide
VEIN_ISLAND_DIAMETER_UM = (300, 2000)  # doc: a round tumor island 0.3 to 2 mm across
VEIN_ISLAND_MIN_SOLIDITY = 0.9     # PLACEHOLDER: "round, smooth-edged"
VEIN_ISLAND_RING_FAT = 0.5         # PLACEHOLDER: the island sits in fat
ARTERY_NEAR_UM = 500               # doc: tumor island within 0.5 mm of an artery
BUDDING_FIELD_MM2 = 0.785          # literature (ITBCC): the budding hotspot field

# --- Risk map -------------------------------------------------------------------------------
RISK_NEIGHBORHOOD_UM = 1000        # doc: features add points to tiles within 1 mm
RISK_CARD_MIN_POINTS = 3           # PLACEHOLDER: regions with at least this many points get a card
