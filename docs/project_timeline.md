# How we built the Colon Tumor Front Analyzer: timeline and reasoning

**Written:** Sun Oct 18, 2026 (state of `main` at `5c1058c`) · **Covers:** Fri Oct 16 evening → Sun Oct 18 ~00:35
**Sources:** commit history on every branch, the write-ups in `docs/`, numbers in `reports/` and `calibration/`, and our
chat sessions with Claude (for decisions that aren't written anywhere else).

All times are local (Central). GitHub handles: **NeelA28** (Delta user `natmakur`; plan, analysis engine, MUSK),
**jhj16** (Delta user `hjeong6`; classifiers, whole-slide pipeline, calibration), **akankshyaswain123** (the app).

Each step follows the same pattern: **why** we did it → **what** we did → **test** → **result** → **decision**.
The detailed write-up for each step is linked in brackets.

---

## 0. The short version

1. **Zero-shot CONCH got ~90%** on clean tiles, but it was weak on exactly the classes our project depends on:
   tumor vs normal mucosa, and stroma vs muscle.
2. **We trained a small classifier** (logistic regression) on frozen CONCH embeddings. Training on one hospital's data
   moved the errors around instead of removing them. **Training on two hospitals together** (NCT + STARC-9) fixed it:
   94.6% average over 4 test sets from 4 sources, against 81.1% zero-shot.
3. **Testing on raw, un-normalized data** found two problems: glass was called necrosis, and scattered lymphocytes were
   missed. We fixed them with **small dedicated checks**, not a bigger model: a background guard and a lymphocyte
   detector (AUROC 0.83 vs 0.75 on independent TCGA data).
4. We **stopped improving the model** and **linked it to the analysis engine**: a whole-slide pipeline running on real
   TCGA slides.
5. On real slides, the **depth hint** was the weak link, and when we traced each miss, it was **the analysis, not the
   model**. We calibrated the depth hint against the real pT stage on **58 TCGA slides**. The tissue map improved
   (4.5× more fat found). The depth rule was the bottleneck, so we rebuilt it. Its claim is now true on **16 of 17**
   judged slides (was 21 of 28), and it says "can't judge" when the section can't show depth.
6. **Blood was being mapped as necrosis.** A MUSK-based blood check fixes it on our raw slides (finds 98% of a clot,
   versus 42% for the CONCH version).
7. **MUSK vs CONCH**, tested fairly: keep **CONCH for the tissue map** (MUSK added only +0.3 points). Use **MUSK for
   blood**.

### Where the numbers stand now

| piece | headline number | tested on |
|---|---|---|
| tissue model `tissue8` (CONCH + LR, NCT + STARC-9) | **94.6%** average, worst 89.1% (zero-shot 81.1%) | 4 test sets, patients never trained on |
| same, raw (un-normalized) tiles | **85.6%** (zero-shot 76.2%) | Kather 2016, 10 patients |
| same, large raw multi-hospital set | **86.5%** (zero-shot 73.8%); ≥ 90% confident → 96.3% right | SPIDER colon test, 7,328 tiles / 381 slides (*not committed yet, see §12*) |
| lymphocyte detector `immune_v1` | AUROC **0.831** (tissue model 0.749) | TCGA-TILs, 4,889 raw patches, 186 patients |
| background guard | removes 99% of glass; wrongly removes 0.02–0.05% of tissue | Kather glass, CRC-VAL, NCT, SPIDER, our 5 slides |
| MUSK blood check | 98% of clot tiles found, 0% of red-stained muscle | 2,237 tiles from our 5 slides (one clot, judged by eye) |
| depth hint (current defaults) | claim true on **16/17** judged sections; 1 over-call (was 7/28) | 58 TCGA slides with known pT (*thresholds chosen on these same slides*) |
| serosal (pT4a) flag | false alarms 7/45 → **1/45**; finds 0/13 pT4 | same 58 slides |
| test suite | **30 tests pass** on `main` | fake slides with planted features + depth-rule strips |

---

## 1. Fri Oct 16, evening: setup and plan

**What we did (NeelA28, `e54d900`, `269bbf6`):** project scaffolding, a `check_setup.py` script, and `CLAUDE.md`, which
holds the plan, the honest-limits rules and the data contract. We renamed the project "Colon Tumor Front Analyzer".

**Why this plan:**
- The challenge asks for multimodal pathology models that make pathology **more precise, reproducible or
  interpretable**.
- The features we picked are ones pathologists find **tedious and inconsistent to judge by eye**: tumor border shape,
  depth of invasion, immune cells at the front versus the center, necrosis, and stage II high-risk features.
- We framed the tool as **decision support, not a diagnosis**, and wrote down the limits first: no CD4/CD8 from H&E,
  one slide is a 2D snapshot, and T stage is only a "consistent with" hint.
- We decided **not to train big models**: zero-shot with prompt ensembles, plus a small classifier only where needed.
  That fits a beginner team, 10 GPU-hours, and two days.

---

## 2. Sat morning: first look, zero-shot CONCH and MUSK [`docs/handoff_small_classifier.md`]

**Why:** before building anything, we needed to know whether zero-shot CONCH was good enough to make the tissue map.

**What (NeelA28, `7489989` → `a31d9fc`):**
- CONCH zero-shot code with 8-class prompt ensembles, and a headless `first_look.py` for Delta.
- Delta problems we hit and fixed along the way:
  - Delta's matplotlib crashes when drawing text. Fix: install a working version, and save results *before* plotting.
  - transformers 5 broke both CONCH's and MUSK's tokenizers. Fix: our own `tokenize()` and `MuskTokenizer`, checked
    identical to the old tokenizer on 139 test strings.
- Then we added MUSK with the same interface as CONCH.

**Test:** the same 400 NCT-CRC-HE-100K tiles (50 per class) through each model.

| | CONCH 224 px | CONCH 448 px | MUSK | average of both |
|---|---|---|---|---|
| accuracy | 90.0% | 88.8% | 90.5% | **92.0%** |
| tumor / stroma | 78% / 70% | 84% / 78% | **92% / 84%** | 88% / 72% |
| log-loss (lower = more honest probabilities) | 0.71 | – | 0.33 | 0.26 |

**Findings:**
- CONCH's two main confusions were **tumor → normal mucosa** and **stroma → muscle**.
- CONCH is overconfident: average confidence 0.88 even when it's wrong.
- The two models make **different** mistakes. They agree on 87% of tiles and are right 96.8% of the time when they
  agree, so **disagreement is a useful "unsure" signal**. That idea later became `alt_label`.
- Blood has no class, so CONCH called a blood patch necrosis with 1.00 confidence.

**Decisions:**
- Run CONCH at **224 px**: 448 px was worse.
- Train a **small classifier on frozen embeddings** next. The handoff doc set the rule: test on **different patients**
  (CRC-VAL-HE-7K), never a random split of NCT, because NCT has no patient IDs.

---

## 3. Sat midday: a better training set, STARC-9 [`docs/dataset_starc9.md`, `reports/starc9_lr/`]

**Why:** NCT-CRC comes from 86 slides at one hospital and has no blood class. We wanted more patients, more of our weak
classes, and honest new-patient test sets.

**What (jhj16):**
- We chose **STARC-9** (NeurIPS 2025): about 200 Stanford patients, our 8 classes plus blood, 70,000 tiles per class,
  and three separate-patient test sets, one of them cut from **30 TCGA slides** like ours.
- We downloaded a seeded subset (90k train + 92k test tiles) at our scale of 0.5 µm/px.
- We considered and rejected three other datasets:
  - NCT-NONORM: same 86 slides, so no new patients;
  - Kather 2016: too small, and it merges muscle into stroma;
  - SemiCOL: unclear access terms.

**Decision on blood (team):** "No 9th class. Identify blood, then ignore it." The classifier learns blood only so the
pipeline can skip those tiles (`qc_reason = blood`). The map stays at 8 classes, and so does the data contract.

**Test:** CONCH + LR trained on STARC-9 only, each test set used once.

| test set | zero-shot | STARC-only classifier |
|---|---|---|
| Stanford VAL-LARGE (48k) | 82.5% | **97.7%** |
| STARC-9 TCGA (18.5k) | 78.7% | **96.9%** |
| CRC-VAL-HE-7K (another hospital) | 86.8% | 86.2% ⚠️ |

- Tumor went from 52–64% to 91–98%; stroma from 32% to 98% on the STARC-9 sets.
- Calibration became excellent: ECE 0.13 → 0.001.
- **But on the Heidelberg test set, overall accuracy didn't improve.** Stroma fell to 12.4%, called muscle or necrosis.
  The two datasets **define and stain "stroma" differently**, and a model trained on one hospital learns that hospital's
  version.

---

## 4. Sat afternoon: moving to Delta, and a nerve/vessel side classifier [`docs/structures_classifier.md`]

**Why Delta:** downloads and embedding were slow on the laptop. We set up SSH control of Delta (account `hjeong6`), so
heavy jobs run there as `sbatch` jobs that end when done.
- Data lives in `/projects`, because `/work/hdd` gave I/O errors.
- Login nodes kill heavy commands, so anything heavy runs as a job.

**Why nerves and vessels:** tumor along a nerve is **perineural invasion**, and tumor inside a vessel is **venous
invasion**. Both are stage II high-risk features that are tedious to find by eye. The team decision was **not to add
them as map classes**: store them as separate probabilities that the risk analysis can look up.

**What:**
- The **SPIDER** dataset (pathologist-labeled, train/test split by slide): 20,941 training tiles from 3,097 slides.
- A logistic regression for nerve / vessel / other.
- MUSK was requested, but our account was still waiting for approval, so this used CONCH.

**Result on unseen slides:**
- It finds **97–98% of vessels and 99% of nerves**.
- False "nerve" calls on colon smooth muscle drop from **50% (zero-shot) to 1.4%**.

**Caveats we wrote down:**
- Nerves are only tested on **skin**; no colon dataset labels nerves.
- At realistic vessel prevalence (2–10% of tiles), only 46–82% of "vessel" calls would be right. They're
  "look here" hints, not counts.
- Artery vs vein is a zero-shot guess, with no labels anywhere.

---

## 5. Sat afternoon: testing on the provided datasets [`docs/eval_provided.md`]

**Why:** check the models on the hackathon's own data, which they were never trained on.

**Test:** all 89,434 NCT-CRC-HE-100K tiles, plus 2,141 PathGen patches. PathGen captions are AI-written, so we measure
agreement with them, not accuracy.

| on NCT-100K | zero-shot | STARC-only classifier |
|---|---|---|
| accuracy | **86.9%** | 80.2% |
| tumor recall | 73.5% | **92.6%** |
| muscle / stroma recall | 87.8% / 67.1% | 60.8% / **25.4%** |

- The classifier removes zero-shot's tumor/normal/mucus confusions, which matter most for tumor % and the border.
- But it adds muscle/stroma/necrosis confusions: the **same hospital-definition problem** as in §3.
- Confidence is meaningful: when it's ≥ 99% sure (58% of tiles), it's right 95.5% of the time.
- The nerve/vessel flags agree with PathGen captions far above chance (nerve AUROC 0.82). They never flag tumor or
  normal mucosa.

**Decision (team, 15:41):** push the work to a **branch, not `main`**, "because overall accuracy is dropping". Try a new
approach: **split into two models**, one that only does the 8 tissues and one for nerves, vessels and blood.
(`e9c9e33` on `starc9-spider-classifiers`)

---

## 6. Sat ~16:15: split models, the accuracy drop is gone [`docs/split_models.md`, `reports/split_v1/`]

**Reasoning:** the drop wasn't caused by nerves or blood sharing a model with the tissues. It came from **training on
one hospital**. So:
- the tissue model trains on **both** hospitals' data;
- blood moves to the structures model, so the tissue model's probabilities always sum to 1 over our 8 classes.

**What (jhj16, `38bd893`):** `tissue8` = LR on 169,434 tiles (NCT + STARC-9), C picked only on tuning sets.
`structures` = nerve / vessel / blood / other.

| test set (new patients) | zero-shot | STARC-only | NCT-only | **tissue8** |
|---|---|---|---|---|
| CRC-VAL-HE-7K (Heidelberg) | 86.8% | 86.2% | 95.8%* | **95.5%** |
| Stanford VAL-LARGE | 82.5% | 97.7%* | 81.5% | **97.6%** |
| STARC-9 TCGA | 78.7% | 96.9%* | 74.4% | **96.4%** |
| SPIDER colon (a hospital never trained on) | 76.5% | 88.0% | 78.6% | **89.1%** |
| **average / worst** | 81.1% / 76.5% | 92.2% / 86.2% | 82.5% / 74.4% | **94.6% / 89.1%** |

\* home advantage: trained on that hospital's data (different patients).

- `tissue8` is the best, or within 0.5 points of the best, on every set, including the hospital it never saw.
- Calibration is good everywhere: ECE 0.003–0.022, versus 0.10–0.18 for zero-shot.

**Extra test, Kather 2016 (raw colors, 10 patients):** 85.6% (zero-shot 76.2%), again the best model. It also revealed
two problems:
1. **Empty glass is called necrosis** 88% of the time. The tissue model has no "background" answer.
2. **Scattered lymphocytes are missed** (66%). In NCT/STARC-9, "lymphocytes" means a tile that is *mostly*
   lymphocytes, so inflamed stroma counts as stroma. That undercounts exactly what the immune score is about.

---

## 7. Sat ~16:30–18:40: background guard and lymphocyte detector [`docs/fixes_background_lymph.md`]

**Why:** both problems have the same cause: asking one 8-class model a question it wasn't built for. The fix follows
the same "separate small checks" idea as §6.

### Glass: a background guard, not a trained glass detector

The trap is **fat**: it's also mostly white, and fat feeds the "tumor touches fat" T3 hint.

| approach | glass removed | fat lost |
|---|---|---|
| pixel rule, < 5% stained | 99.2% | 22.9% |
| zero-shot "background" prompt | 99.0% | 12.8% (also flags 8% of tumor) |
| trained glass detector + pixel rule | 98.6% | 3.0% |
| **guard: < 10% colorful pixels AND label isn't fat** | **99.0%** | **0 fat-labeled tiles** |

- **Our question during the work (21:29):** "is a glass detector worth losing some fat?" We checked, and the answer was
  no.
- The trained glass detector learned from NCT/CRC-VAL `BACK` tiles, which are **purple**: color normalization tints
  empty glass. Real raw slides have white or gray glass, so that detector learned the wrong thing.
- We also asked to be **careful with the necrosis label**, so the guard only overrides non-fat labels on nearly
  colorless tiles. Real necrosis is ≥ 75% colorful.

### Lymphocytes: a separate yes/no detector (`immune_v1`)

- Positives: NCT + STARC-9 lymphocyte tiles plus **9,453 SPIDER "inflammation" tiles** (immune cells scattered in
  tissue). A tile can now be "stroma **and** lymphocyte-rich".
- Kather tiles the tissue model missed: average score **0.04 → 0.44**, versus 0.01 for plain stroma.

### Independent re-tests (data not used to design either fix; jhj16, `9bdeb35`)

- **TCGA-TILs**, chosen because it's **raw TCGA colon/rectum, the same kind of slide as ours**, labeled by a different
  group for exactly our question:

  | score | AUROC [95% CI] |
  |---|---|
  | `tissue8` P(lymphocytes), used before | 0.749 [0.726–0.774] |
  | zero-shot CONCH | 0.761 |
  | **immune_v1** | **0.831 [0.809–0.853]** (intervals don't overlap) |

- **Guard:** wrongly removes 1 tile on CRC-VAL, 43 (0.05%) on NCT-100K, and **0** on our 5 TCGA slides.
- **New finding:** the automatic tissue mask (Otsu threshold) was **throwing away pale real tissue**: loose submucosa,
  pale stroma, and probably fat. A fixed, looser mask (saturation > 12) keeps **12–26% more tissue**. The guard then
  removed the 21 pure-glass tiles that came in with it.
  → Decision: looser mask + guard.

---

## 8. Sat ~18:25: scope check, focus instead of fine-tuning

**Our question:** should we fine-tune CONCH itself, and drop nerves/vessels to stay focused on the 8 tissues?

**Answer and decision:**
- **Don't fine-tune CONCH.** The classifier on top already reaches ~95% on unseen curated data, so the embeddings
  already separate the tissues. What's left is label-definition and stain differences, and retraining CONCH on the same
  data would learn the same quirks. It also risks damaging CONCH's text abilities, costs GPU time we don't have, and the
  plan says "we do NOT train big models".
- **Park nerves and vessels** as experimental. The code and model are kept, but they're not a main result.
- **Keep the lymphocyte detector**, because it directly fixes a core feature.

**Full-system check on the most general dataset (~18:40–18:55):** the complete **SPIDER colon test split**: 13,193
pathologist-labeled, raw-color tiles from 778 slides, never used to train anything.
- Tissue: **86.5% vs 73.8% zero-shot**. When ≥ 90% confident (71% of tiles), it's 96.3% right.
- **Main weakness: high-grade tumor.** Only 55% are called tumor; 30% are called necrosis. Our training tumors are
  mostly gland-forming.
- The model can't tell adenoma (precancer) from invasive cancer: high-grade adenoma is called tumor 99% of the time.
- Guard: 3 of 13,193 tiles wrongly removed. Lymphocyte detector: AUROC 0.992.
- ⚠️ These results are in a **local git stash** on `starc9-spider-classifiers` (`reports/full_system_v1/`,
  `scripts/eval_full_system.py`). They are **not on `main`** yet.

**Decision (team, 18:59):** "at this point trying to improve the model is not practical". Next, link our models to the
teammate's analysis engine.

---

## 9. In parallel, Sat afternoon–evening: the other tracks

### Analysis engine (NeelA28, `eaf2c20`, 15:59)

Everything computed from the tile map on CPU, with no model:
- **Tumor front and zones:** tumor mask, main tumor vs satellites, invasive front, core / front band / peritumoral
  zones.
- **Measurements:** growth pattern, direction, depth hint, fat, tumor-stroma ratio, necrosis + rim, mucin, immune zones,
  lymph nodes, outer edge.
- **Flags:** "look here" hints (perineural, venous, budding, ...) and a risk map.

Every cutoff lives in `src/cutoffs.py`, labeled *doc*, *literature* or *PLACEHOLDER*. Tested on fake slides with planted
features (19 tests). This split, **trained models make the map and plain geometry measures it**, is what keeps the
analysis interpretable.

### App (akankshyaswain123, branch `app-core`, ⚠️ not merged into `main`)

- A Streamlit Slide Box, a slide explorer with a see-through tissue map over the scan, hover and click details, and
  Summary / Spot / Unsure / Risk / Flags panels.
- PDF and PNG exports, search inside a slide, and a model-results page.
- A map redraw with 41k tiles takes 0.4 s, versus 3.6 s before.
- It has a Lumen hook with a check that no numbers are added.

### MUSK backbone experiment (NeelA28, `musk-experiments`)

A fair CONCH vs MUSK vs both comparison. The decision rule was set **before** seeing results (see §14).

---

## 10. Sat ~19:00–19:55: the whole-slide pipeline and the first real slides [`docs/slide_pipeline.md`]

**What (jhj16, `27bf5b8`):** `scripts/process_slide.py`, the missing step:
1. tissue mask (the looser setting from §7);
2. exact 224 px / 0.5 µm grid;
3. guard and blur checks;
4. CONCH embeddings;
5. `tissue8` and `immune_v1` → `tiles.csv`;
6. then the teammate's analysis, unchanged.

The hooks in the teammate's code are small and additive, and all 19 of their tests still pass. We left the nerve/artery
detectors out (parked), so the analysis falls back to text-phrase scores.

**First run:** our 5 TCGA slides, 7.5 min on one A100 (≈ 0.13 GPU-h).

**Our question (19:22):** "what did we miss: the analysis or the data from CONCH? This is very important." We traced
every depth mismatch through the thumbnails, the grids and the actual tile images.

| slide | real pT | analysis said | whose fault | evidence |
|---|---|---|---|---|
| CM-4746 | T2 | "mucosa" | **analysis** | the model correctly mapped **99 tumor tiles touching muscle**. The analysis erased them: smoothing 99 → 9, removing small pieces → 7, "largest piece only" → 0 |
| AH-6547 | T3 | "mucosa" | **the slide** | no muscle wall or fat anywhere in the section |
| A6-3809 | T4 | "mucosa" | **the slide** | three small tumor-only fragments |
| AA-3844 | T3 | "muscle" | the slide (+1 stray tile) | tumor-only pieces |
| AA-3875 | T1 | "mucosa" | right by luck | no wall in the section either |

**Other findings:**
- The serosal "possible pT4a" flag fired on **all 5 slides**, including T1 and T2.
- **Blood is the main model-side problem.** Only ~10–12 of 30 "necrosis" tiles were really necrosis; most were
  red-blood-cell clots.
- Tumor labels looked right (~27 of 30).

**Conclusion:** the most important miss was the **analysis, not CONCH or our classifier**. Three of 5 "misses" are
slides that can't show depth at all. So the next work went into the analysis rules and into blood.

---

## 11. Sat ~20:00–21:30: calibration v1, 58 slides with known pT [`docs/calibration.md`]

**Our idea (19:57):** tune the analysis settings against real results, so it gets more accurate every time we run it.

**What pT is, and why it's a noisy answer key:**
- pT is the depth of invasion on the **whole surgical specimen**: Tis mucosa, T1 submucosa, T2 muscle wall, T3 fat,
  T4 serosa or other organs.
- One diagnostic slide is one section, so it can miss the deepest point.
- So the slide can only say "**at least** X". **Claiming deeper than pT is the real error**; shallower is often just
  sampling.

**What (jhj16):**
- Downloaded **53 more TCGA slides** with known pT from the NCI GDC (open access, untreated, one per patient).
- With our 5, that's **58**: T1/Tis 18, T2 13, T3 14, T4 13.
- Pipeline: 71 min on one A100.
- Grid search over the settings. Scoring: correct +1, deeper than pT −2, shallower −0.5.
- **Leave-one-slide-out** for an honest accuracy number, plus a **learning curve**.

**Our question (20:19):** is it better to calibrate on all 58 at once, or one slide at a time?
- Re-tuning on *all slides so far* ends at the same settings as all at once.
- Nudging toward each new slide **chases label noise** and depends on the order of the slides.
- **Decision:** tune on all slides together, and use "predict first, then add the slide" only to draw the learning
  curve and get an honest accuracy.

**Result:**
- Honest (held-out) depth hint: **25% correct, 33% deeper than pT**, with depth judged on 41% of slides.
- **The learning curve is flat:** more slides won't help.

**Root causes, found by looking at the slides:**
- **57% of sections show no muscle wall or fat.** Those can't be fixed; "can't judge" is the honest answer.
- **The mask still dropped pale fat.** Fixable.
- **Thin muscle inside the lining (muscularis mucosae) counted as the muscle wall.** Partly fixable.
- **A placeholder blur cutoff dropped up to 65% of a slide.** Fixable.
- **Possible label noise** (DC-5337).

**Decision:**
- Apply only the **serosa settings**: false alarms 7 → 1, nothing lost.
- Don't apply the depth settings: the tuned values did worse on held-out slides than the starting ones.

---

## 12. Sat ~21:30–22:56: v2, a better tissue map (`169bf1e`)

**What:**
- **Fat-aware mask.** Fat is pale like glass but has thin membranes, so its brightness changes locally; glass is flat.
- **Glass check:** a tile that is very flat **and** < 30% colorful is glass, whatever the model says.
- **Blur relative to each slide's own typical sharpness.** A fixed cutoff fails, because a soft scan's readable tissue
  scores below a sharp scan's junk.
- We re-tiled **only the newly added grid cells** (+22% tiles, 1 GPU-hour), not whole slides.

| all 58 slides | v1 | v2 |
|---|---|---|
| fat tiles | 12,164 | **55,365 (4.5×)** |
| sections with a muscle wall / fat layer | 23 / 11 | 28 / 18 |
| T3/T4 slides where tumor touches fat | 3 / 27 | 4 / 27 |
| worst blur loss (AG-A025) | 65% of tiles | 0% |

**Result:** **the map got better; the depth hint didn't** (held-out 32% correct / 32% over-calls).
- In T3 tumors, cancer glands in fat are wrapped in scar-like stroma, so tumor tiles rarely sit right next to fat tiles.
- Fat *inside* the wall looks the same as fat outside it.
- **The bottleneck is now the rule, not the map.**

**Decision:** keep the new mask, glass check and blur rule. Don't apply the v2 depth settings. Next idea: judge
**position** ("is tumor beyond the muscle wall?") instead of **contact**.

---

## 13. Sat evening: the blood check with MUSK (NeelA28, `a3f688a`) [`reports/blood_check_v1/`]

**Why:** blood inflated necrosis, and with it the "dead core" and necrosis-rim features. The CONCH blood detector
(trained on STARC-9) missed most of our blood.

**The test that explained it:** 2,237 tiles from our slides, including 376 clot tiles (40 checked by eye) and 200
deep-red muscle tiles.

| blood check | clot found | red muscle wrongly flagged |
|---|---|---|
| CONCH structures model | 42% | 0% |
| CONCH, tiles color-normalized like STARC-9 + STARC-9 zoom | 85% | 2% |
| zero-shot "blood" prompt | 99% | 3.5% (and 31% of other necrosis) |
| **MUSK structures model (chosen)** | **98%** | **0%** |

**The cause was color and zoom:**
- STARC-9 blood is purple, because normalization turns red blood purple.
- STARC-9 tiles are 64 µm wide and get enlarged to 224 px, so cells look 1.75× bigger than in our tiles.
- A simple "bright red" color rule also fails, because CM-4746's muscle is stained deep red.

**Result:**
- CM-4746: necrosis **718 → 255** tiles, with 368 of 376 clot tiles skipped as blood.
- MUSK's tissue model is added as a **second opinion** (`alt_label`); disagreement marks a tile "unsure".
- **One overrule:** where CONCH says necrosis and MUSK says muscle, the analysis uses muscle. On CM-4746, 16 of 20 of
  these tiles checked were plainly muscle.

**Limit:** the only blood "truth" is one clot, judged by eye. `BLOOD_PROB` is still a placeholder.

---

## 14. Sat evening: MUSK vs CONCH, the fair test [`docs/musk_experiment.md`, `reports/musk_v1/`]

**Rule set before the results:** switch away from CONCH only if MUSK or "both" beats it by **≥ 2 points** on the
non-TCGA tests, or clearly fixes a weak spot. TCGA was excluded from the decision because MUSK may have seen TCGA during
pretraining.

| average, decision tests | CONCH | MUSK | both |
|---|---|---|---|
| tissue8 | 92.1% | 90.0% | 92.4% |

(This average uses a different set of tests than §6: Kather is in, TCGA is out. That's why it differs from 94.6%.)

**Decision:**
- **Keep CONCH for the map:** +0.3 points is below the rule.
- **Use MUSK for blood:** it calls 0.1% of TCGA fat "blood", versus 16% for CONCH.

---

## 15. Sat ~22:55 → Sun 00:11: depth hint v3 and v3b (`e8f8740`)

**v3, position rule:** tumor counts as "beyond the wall" if the straight line from it to the lining crosses the muscle
wall. This catches tumor wrapped in stroma inside fat, and ignores fat inside the wall.

**Thick-wall check:** the muscle region must be ≥ 600 µm thick on average, so the thin muscularis mucosae doesn't count
as the wall.

**v3b, our question (23:38):** "why are we checking depth if there is no muscle wall at all?"
- Depth is now judged **only when the wall is in the section**.
- Tumor touching fat without a wall gives a **warning instead of a depth**: "depth unknown, check the fat".

| defaults on 58 slides | judged | exact | **deeper than pT** | claim ("at least X") true | serosal false alarms |
|---|---|---|---|---|---|
| before (wall gate only) | 28 | 8 | **7** | 21/28 (75%) | 2/45 |
| **now (wall gate + 600 µm wall)** | 17 | 7 | **1** | **16/17 (94%)** | 1/45 |

- **Fixed:** six pT1/Tis/pT2 over-calls (AZ-4313, AZ-5407, G4-6323, G4-6628, G5-6641, CM-6163).
- **Cost:** 11 thin-muscle sections now say "can't judge".
- **Left:** DC-5337 (pT1 → "≥ pT3"), possibly a label issue.
- **Applied:** `WALL_MIN_WIDTH_UM = 600`, with a clear anatomical reason. The position rule stays optional
  (`DEPTH_RULE`), because its held-out gain was one slide and it moved with the settings grid.
- **Honest caveat:** 600 µm was chosen on these same 58 slides, so 16/17 is optimistic.

---

## 16. Sun 00:15–00:34: everything into `main`

**What (NeelA28):**
- `slide-pipeline` v2 and v3 merged into `musk-blood-check`. Conflicts were resolved keeping both sides: jhj16's
  `--extend` mode and quality check, plus the MUSK pass and blood check.
- Then `musk-blood-check`, `starc9-spider-classifiers` and `musk-experiments` merged into `main`.
- Repo reorganized: model-building scripts moved to `scripts/models/`, with indexes in `docs/README.md` and
  `reports/README.md`.

**Check:** **30 tests pass**. The analysis was re-run on AA-3844 and CM-4746:
- CM-4746 → "consistent with at least pT2" (real T2) ✅
- AA-3844 → "can't be judged" (honest)
- the serosa false alarm is gone on both.

**Not in `main`:** the app (`app-core`) and the full-SPIDER-test stash (§8).

---

## 17. Sun ~00:20: is a Lumen rewrite realistic?

**Assessment:**
- **Yes, as an optional layer.** Lumen speaks the OpenAI API format, and `summary.json` already has a plain-English
  sentence for every section, so the app can show readable text without an LLM.
- **Risks if we add it:**
  - made-up numbers (fix: check every number against `summary.json`, else fall back to the template);
  - overclaiming ("this is pT3");
  - dropped caveats (fix: caveats always come from the templates);
  - demo reliability (fix: generate once and save, rather than calling Lumen live).

**Decision:** stick with template sentences for now. Lumen is next.

---

## 18. Decision log

| # | decision | alternatives considered | evidence | when |
|---|---|---|---|---|
| 1 | Zero-shot first, then a small LR on frozen embeddings; never fine-tune CONCH | fine-tuning CONCH | LR already ~95% on unseen data; leftover errors are label/stain differences; 10 GPU-h | plan; reconfirmed Sat 18:25 |
| 2 | CONCH at 224 px | 448 px | 90.0% vs 88.8% | Sat AM |
| 3 | Test only on patient-disjoint sets | random split of NCT | NCT has no patient IDs | Sat AM |
| 4 | Blood = identify and ignore, not a 9th class | 9th map class | keeps the map and contract at 8 classes | Sat 12:26 |
| 5 | STARC-9 as the second training set | NCT-NONORM, Kather 2016, SemiCOL | new patients, blood class, TCGA test set | Sat midday |
| 6 | Nerves/vessels as a side model, not map classes; later parked | 9th/10th map classes | skin-only nerve tests, low precision at real prevalence | Sat 13:56; parked 18:25 |
| 7 | STARC-only model kept off `main` | merge it | NCT accuracy 86.9% → 80.2% | Sat 15:41 |
| 8 | Split models; tissue trained on two hospitals | one combined model | 94.6% avg, best/near-best on all 4 sets | Sat 16:14 |
| 9 | Background guard over a glass detector | pixel rules, zero-shot prompt, trained detector | 99% glass, 0 fat-labeled tiles lost; detector trained on purple "glass" | Sat 16:29 |
| 10 | Separate lymphocyte detector | tissue P(lymphocytes) | TCGA-TILs AUROC 0.831 vs 0.749 | Sat 18:38 |
| 11 | Looser (fixed) tissue mask, later fat-aware | Otsu mask | +12–26% tissue; fat 4.5× | Sat 18:38; 22:56 |
| 12 | Stop model tuning; build the pipeline | keep tuning (NONORM, SPIDER train) | time; the model was no longer the bottleneck | Sat 18:59 |
| 13 | Diagnose misses (analysis vs model vs slide) before fixing | tweak settings blindly | the T2 miss was the analysis; 3/5 were the slides | Sat 19:22 |
| 14 | Calibrate on all slides at once; judge by leave-one-out | one-by-one nudging | one-by-one chases label noise and depends on order | Sat 20:19 |
| 15 | Apply only changes that beat held-out tests (serosa yes, v1/v2 depth no) | apply the best-fit settings | tuned depth settings were worse held out | Sat 21:30 / 22:56 |
| 16 | MUSK for blood; CONCH for the map | MUSK or both for the map | +0.3 pts (rule: ≥ 2); blood 98% vs 42% | Sat evening |
| 17 | Depth only with a muscle wall; 600 µm thick-wall default; fat warning | judge every section | over-calls 7 → 1 | Sun 00:04 |
| 18 | Template sentences now, Lumen later | Lumen first | templates already exist; LLM risks | Sun 00:22 |

---

## 19. What didn't work (worth saying on a slide)

- **One-hospital training** moved errors instead of removing them (STARC-only: stroma 12% on Heidelberg data).
- **A trained glass detector:** its training glass was purple, an artifact of color normalization.
- **A "bright red" color rule for blood:** heavily eosin-stained muscle looks red too.
- **Color-normalizing our tiles** to help the CONCH blood model: only partly helped (42% → 85%).
- **Tuning the depth contact rule** on more slides: the learning curve was flat.
- **A 4.5× better fat map** didn't fix depth: tumor in fat sits inside scar-like stroma.
- **A wider settings grid** in v3 made the honest estimate **worse** (11 → 8 correct): overfitting with 58 slides.
- **A simple nucleus counter for lymphocytes:** dense lymphocytes merge together.

---

## 20. Open items and honest limits

**Not on `main` yet**
- The app (`app-core`).
- The full SPIDER test results (`git stash` on `starc9-spider-classifiers`).
- Saved `summary.json` files made before v3 are stale: re-run `analyze_slide.py`.

**Needs the pathologist**
- 29 cutoffs are marked PLACEHOLDER in `src/cutoffs.py`, including `BLOOD_PROB` and the immune cutoffs.
- Check DC-5337 and AZ-5407 (possible label issues).
- Confirm the muscle wall on AZ-6606, F5-6464 and G4-6307.
- Review the ~50 strongest nerve flags.

**Never measured yet**
- Tile accuracy on our own raw TCGA slides (hand-label ~100 tiles; planned in `CLAUDE.md`).
- Normal (-11) slides → near-zero tumor.
- Colon nerves. Artery vs vein.

**Known model weaknesses**
- High-grade (solid) tumor → necrosis.
- Adenoma vs invasive cancer can't be told apart.
- Muscle → stroma on Heidelberg data.

**Depth hint**
- Judged on 17 of 58 sections; 600 µm was picked on the same slides.
- One section ≠ the specimen; "consistent with at least" only.

**GPU budget**
- The jobs documented in the write-ups add up to roughly 3 GPU-hours. The biggest were calibration v1 (~1.2 h) and
  v2 (~1 h).
- That total isn't exact: check the real balance with `accounts` on Delta.

---

## Where to read more

| topic | write-up | numbers |
|---|---|---|
| zero-shot first look | `docs/handoff_small_classifier.md` | `docs/img/` |
| STARC-9 | `docs/dataset_starc9.md` | `reports/starc9_lr/` |
| nerves / vessels | `docs/structures_classifier.md` | `reports/conch_structures_v1/` |
| provided datasets | `docs/eval_provided.md` | `reports/provided_v1/` |
| split models, Kather 2016 | `docs/split_models.md` | `reports/split_v1/`, `reports/kather2016/` |
| guard + lymphocyte detector | `docs/fixes_background_lymph.md` | `reports/immune_v1/`, `tils_v1/`, `guard_check/`, `fixes_existing_v1/` |
| MUSK vs CONCH | `docs/musk_experiment.md` | `reports/musk_v1/` |
| pipeline, first run, blood check | `docs/slide_pipeline.md` | `reports/blood_check_v1/` |
| depth calibration v1–v3b | `docs/calibration.md` | `calibration/` |
