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
LOW_CONFIDENCE = 0.5               # PLACEHOLDER: top probability below this = "unsure here" (the only rule: MUSK's
                                   #   second opinion, alt_label, is no longer used; see ALT_OVERRULES)
TISSUE_MASK_SAT = 12               # measured: on our 5 TCGA slides the old rule (saturation > Otsu, 31-70) dropped
                                   #   pale real tissue (submucosa, loose stroma, fat); > 12 keeps it. Glass that
                                   #   slips in is removed per tile by the background guard (src/detectors.py)
TISSUE_MASK_TEXTURE_SD = 4.0       # fat-aware mask: pale thumbnail pixels with local brightness changes (fat
TISSUE_MASK_GRAY_MAX = 235         #   membranes / septa; glass is flat, local SD < 1) and not near-white,
TISSUE_MASK_NEAR_PX = 40           #   within this many thumbnail px of stained tissue, also count as tissue.
                                   #   Measured on DC-5337: the saturation rule alone dropped most of a large fat
                                   #   region (pT3 fat contact was invisible); see docs/calibration.md (v2)
GLASS_BLUR_MAX = 0.0003            # measured on 480k tiles from 58 slides: tiles this flat AND this pale are glass
GLASS_COLORFUL_MAX = 0.30          #   (often tinted, so the < 10% colorful guard misses them; labeled necrosis).
                                   #   Real fat almost never qualifies (18 of 12,401 fat tiles)
GLASS_EDGES_MAX = 0.02             # measured (docs/preprocessing_test.md; 980 tiles from 123 TCGA slides, labeled by
GLASS_STRUCTURE_MIN = 0.4          #   eye at full resolution, not by a pathologist): glass = fewer than 2% of pixels
                                   #   on a sharp edge, or structure in fewer than 40% of the tile's 4x4 sub-blocks.
                                   #   Fat has a net of thin membranes, glass is flat. With the ink rule: 71% of glass
                                   #   and ink removed (5% before), 0 of 62 real fat lost, 7 of 427 tissue tiles lost.
                                   #   Applied regardless of the tissue label (incl. fat), only when tiles.csv has the
                                   #   columns (scripts/process_slide.py or scripts/add_texture_measures.py)
INK_BLUEGREEN_MIN = 0.2            # pen ink = more than this share of blue/green pixels, or of near-black pixels,
INK_BLACK_MIN = 0.3                #   or of flat, fully saturated red ones. Ink is skipped as "background" BEFORE
INK_RED_MIN = 0.5                  #   the blood check (the "blood" MUSK found on F4-6808 / F5-6813 was red ink).
                                   #   On the labeled tiles: 0 of 54 real blood tiles hit
BLUR_REL = 0.1                     # blurry = blur score < 10% of the slide's own typical sharpness (median blur of
                                   #   its clearly stained tiles). A fixed cut-off failed: soft scans (AG-A025,
                                   #   median 0.00027) have readable tissue far below sharp scans' junk
ALT_OVERRULES = ()                 # team decision (Oct 18): MUSK is used only for the blood check. Its second opinion
                                   #   (alt_label) changed little on 125 TCGA slides and its disagreements marked 20-28%
                                   #   of the tissue "unsure" (reports/overnight_v4). To bring back the old overrule
                                   #   (CONCH necrosis + MUSK muscle -> muscle; by eye 16 muscle / 0 necrosis):
                                   #   (("necrosis", "muscle"),)
BLOOD_PROB = 0.5                  # PLACEHOLDER: MUSK p_blood at or above this = blood tile, skipped as
                                   #   qc_reason "blood" (src/detectors.py). On CM-4746's clot, 0.5 finds 98% of 376
                                   #   blood tiles and flags 0.5% of random tiles; 0.9 finds 91% and flags 0%
BLOOD_SPARE_LABELS = ("fat",)     # never skip a tile as blood when the tissue model says one of these. Fat: blood
                                   #   models confuse it with blood (16% on TCGA fat). Muscle is NOT spared: the
                                   #   "muscle" MUSK called blood on 125 TCGA slides was pen ink (reports/overnight_v4,
                                   #   section 6), and MUSK called 0 of 200 real CM-4746 muscle tiles blood

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

# --- Depth hint and serosa (CALIBRATE: scripts/calibrate_analysis.py tunes these against TCGA pT;
#     values below are the starting guesses, replaced by calibration/ results when available) -------
WALL_MUSCLE_MIN_MM2 = 2.0          # a muscle region at least this big = the muscle wall (muscularis propria),
                                   #   not the thin muscularis mucosae or stray tiles
WALL_FAT_MIN_MM2 = 0.5             # fat region at least this big = fat around the bowel
DEPTH_CONTACT_MIN_TILES = 3        # raw tumor tiles touching the wall (or fat) needed to say "reaches"
DEPTH_CONTACT_RADIUS_TILES = 1     # "touching" = within this many tiles
WALL_MIN_WIDTH_UM = 600            # calibration v3 (58 TCGA slides): a muscle region must be at least this thick on average
                                   #   to be the wall (muscularis propria ~1-2 mm; the muscularis mucosae is thinner than a
                                   #   tile). Over-calls 7 -> 1 (judged 28 -> 17); chosen on these same slides (docs/calibration.md v3b)
DEPTH_RULE = 'contact'             # PLACEHOLDER: what "at least pT3" means: 'contact' (tumor touches a fat layer; v1/v2),
                                   #   'position' (tumor lies beyond the muscle wall, away from the lining), 'either'
DEPTH_BEYOND_MIN_UM = 300          # PLACEHOLDER: "beyond the wall" = at least this far outside the wall band
DEPTH_BEYOND_MIN_TILES = 30        # PLACEHOLDER: this many raw tumor tiles beyond the wall = "consistent with >= pT3"
# fixed parts of the position rule (src/features.beyond_wall), not tuned by the calibration:
WALL_BAND_CLOSE_TILES = 2          # PLACEHOLDER: close gaps up to ~2 tiles between muscle bundles into one wall band
WALL_CROSS_MIN_UM = 300            # PLACEHOLDER: a line "crosses the wall" if at least this much of it is wall
DEPTH_LINING_REACH_UM = 1000       # PLACEHOLDER: the lining tiles that vote: up to this much farther than the nearest
DEPTH_BEYOND_MAJORITY = 0.5        # PLACEHOLDER: share of those lining tiles that must lie behind the wall
SEROSA_MIN_RUN_UM = 1000           # calibration v1 (58 TCGA slides): false alarms 7 -> 1; tumor must lie on the outer edge this long
SEROSA_NEEDS_WALL = True           # judge the outer edge only in sections that show the muscle wall or fat
SEROSA_MAX_FROM_WALL_UM = 500      # calibration v1. The serosa lies outside the muscle wall: only tissue edges within this
                                   #   distance of the wall or fat count as "outer" (not the luminal surface)

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
