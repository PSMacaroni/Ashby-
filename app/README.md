# The app (Tumor Front Analyzer)

A Streamlit app that runs on a laptop and shows slide results made by the team's pipeline. It never
runs CONCH or MUSK on slides itself; it only reads result files, so it keeps working after Delta
access ends. Light, clinical look; two screens; everything the analysis wrote is on screen somewhere.

## Run it

Once, in a PowerShell terminal (e.g. VS Code's) from the project folder. Keep the environment
**outside OneDrive**, or OneDrive tries to sync thousands of package files:

```powershell
python -m venv $HOME\venvs\ashby
& $HOME\venvs\ashby\Scripts\Activate.ps1
pip install -r app/requirements.txt
```

(Mac: `python3 -m venv ~/venvs/ashby`, then `source ~/venvs/ashby/bin/activate`.)

Every time (activate first, as above):

```powershell
streamlit run app/streamlit_app.py
```

Open **http://127.0.0.1:8501** (use the number, not `localhost`: on some Windows laptops
`localhost` adds ~0.2 s to every request and pages took ~20 s to load). If 8501 is taken, add
`--server.port 8503`. Only this computer can open it. Server settings and the look live in
`app/.streamlit/config.toml`. Starting the server takes 10-40 s on a slow laptop (loading Python
libraries); after that a slide opens in about a second (the 5 showcase slides are loaded in the
background right after the first screen).

## Screens

| Screen | What you can do |
|---|---|
| **Slides** | One row per processed slide: preview, case, site and TCGA report, tumor %, growth pattern, depth hint, number of high-risk features, an **Open** button and a **⋯** menu (rename, duplicate with its own notes and ticks, hide / restore, and the downloads: PDF report, map picture, summary JSON, squares CSV). Filter box and sort menu above. |
| **Slide** | Header: slide, case, site, TCGA report, and a strip of the six key numbers (the "high-risk features" count is the analysis's own list; the Check tab counts every place to look at). Left: the map (below). Right: the panel (below) with **PDF report** and **Map picture** downloads at the bottom. The slide picker in the top bar jumps to another slide; "All slides" goes back to the list. |

### The map (left)

- The slide scan with a see-through colored layer on top. The **layer buttons** and the **opacity slider** sit
  inside the chart and work in the browser without a rerun, so they are instant and keep your zoom. The key sits
  under the drawing area and follows the layer. Four layers, colored exactly like the team's analysis figure: **Tissue**,
  **Zones & front** (core / front band / peritumoral / other tissue, the invasive front as a solid black line, blue
  arrows where the front is infiltrative, blue rings on the satellites, and the main direction of invasion as a
  long blue arrow), **Risk** (points, white to dark red, plus the flag spots with the figure's marker shapes),
  **Certainty** (unsure squares in amber).
- Hover any square for a fixed-size info box (class, score, certainty, zone, risk points); click it to open it in
  the Spot tab. Scroll to zoom, drag to move (you can't zoom out past about half a slide of glass around it); the
  fit button shows the whole slide and clears pointers and the selected square. One square = 112 µm, 9 squares ≈ 1 mm (Plotly can't keep a scale bar right while you zoom, so
  there is none on the live map; the exported map picture has the color key and the square size).
- **Search this slide** above the map ("lymphocytes at the border", "budding", "tumor in a vein", "necrosis in the
  core"): the map is colored by match, the best spots get pointers S1-S5, and a strip under the map says what was
  matched, zooms to a spot on click, switches the coloring, and clears the search (`search.py`, below). The panel
  stays where it is.
- Clicking a square marks it with a blue square (done by Plotly in the browser) and opens it in the Spot tab without
  redrawing the map; the Check tab's pointers stay on the map for that click and go when you switch tabs by hand.
  At opacity 0 only the scan shows: the arrows, rings and flag markers fade with the colored layer.
- **Report**: fits without scrolling. The high-risk features, then one compact line per finding (tumor · growth
  pattern · depth hint · direction · ahead of the front · fat · immune · necrosis · mucin · stroma · outer edge ·
  unsure); hover a line for the analysis's full sentence. With Lumen set up, a plain-language paragraph comes first.
- **Check**: the worklist. Everything to look at, grouped (flags · risk regions · unsure areas · mucin pools ·
  tumor islands · lymph nodes), each with what to do and a **reviewed** tick that survives a restart. The group's
  items are the numbered pointers on the map (F1, R2, U3, ...).
- **Spot**: the clicked square: the real 0.5 µm/px preview (when the slide's `tiles/` folder was copied from
  Delta; otherwise a crop of the overview), where it sits, top class and score, why, certainty, zone and distance
  to the front, the detector scores (lymphocytes, blood, nerve / vessel / artery when present), the position, what
  to check by eye, all 8 scores, and MUSK's second opinion (reference only).
- **Details**: every number in `summary.json`, in pathology-report order (tumor · growth pattern · depth (with fat,
  spread along the lining and the outer edge) · direction and what lies ahead · stroma · necrosis and low oxygen ·
  mucin · immune · tumor islands and lymph nodes · flags · risk map · unsure · tissue mix and method · caveats).
- **Notes**: the pathologist's notes for this slide, saved on this computer when the box loses focus (or Ctrl+Enter)
  and added to the PDF report.

## Where slides come from

Every folder with `summary.json` + `tiles.csv` in:
- `showcase/results/<slide_id>/`: the 5 showcase slides (in git; `showcase/README.md`)
- `data/results/<slide_id>/`: more slides copied down from Delta (git-ignored)

The app reads what the team's analysis writes (format: `CLAUDE.md` → Data contract): `summary.json` (every
section has a ready sentence in `text`), `tile_analysis.csv` (zones, risk points, unsure per square), `tiles.csv`
(scores, `alt_label`, `p_immune`, `p_blood`, `p_nerve`, `p_artery`), `thumbnail.png`,
`phrase_scores.csv` (search), and `tiles/<row>_<col>.jpg` previews if copied. If a slide has `tiles.csv` but no
analysis yet, the app runs `src/analysis.py` in memory (nothing is written to the folder).

**Tile previews** (`tiles/`, 15-30k small JPGs per slide, git-ignored) live on Delta in
`/projects/bfqi/hjeong6/showcase_v1/<slide_id>/tiles/`. Tar each folder there and copy it into
`showcase/results/<slide_id>/`; the Spot tab uses them when present.

## Search in a slide

Two ways (`search.py`):
- **Known words** (works now): words are matched to the 8 tissue classes (lymphocytes use the immune detector),
  the nerve/artery detectors and the analysis's flag phrases (`phrase_scores.csv`); "border", "core", "around"
  limit it to that zone. The panel says exactly what was matched.
- **Live text search** (turns on by itself when a slide has `embeddings.npy` and CONCH is installed on the
  laptop): CONCH text vs tile image embeddings. Not tested yet (embeddings stay on Delta).
Single-cell searches (mitoses, CD8, ...) get an honest "the map can't see that" answer.

## Lumen summaries

Put these in the project's `.env` (never commit it, never paste keys in chat):
`LUMEN_API_KEY`, `LUMEN_BASE_URL`, `LUMEN_MODEL`. Lumen only gets the analysis sentences (no images,
no patient data) and must not add numbers, diagnoses or treatment advice; if its answer contains a
number that wasn't in the input, the app ignores it and shows the findings as computed (`lumen.py`).

## Files

| File | What it does |
|---|---|
| `streamlit_app.py` | Entry point: page setup, background warm-up, picks the screen |
| `page_cases.py` | The slide list and "Manage slides" |
| `page_slide.py` | One slide: header, map, panel (Report · Check · Spot · Search) |
| `report.py` | Key findings, the full report (every number in summary.json), the per-square explanation |
| `worklist.py` | The Check tab's items: flags, risk regions, unsure areas, mucin pools, islands, nodes |
| `search.py` | Search in the open slide (known words now, CONCH text search when embeddings exist) |
| `tile_map.py` | Map picture, hover box, clickable squares, pointers, legend |
| `slide_data.py` | Finds slides, loads them once into memory, the state file (names, hidden, copies, notes, reviewed) |
| `lumen.py` | Plain-language summary from NCSA Lumen, with a number check |
| `export.py` | Map picture (PNG) and the PDF report (findings, worklist with review state, notes, full report, caveats) |
| `ui.py` | Page style, top bar, small HTML pieces |
| `common.py` | Folders, class names, colors (tissue colors come from `src/viz.py`), cut-offs |
| `.streamlit/config.toml` | Theme (light, one blue accent) and server settings |

The app writes only `app/state.json` (git-ignored). Result files are never changed.

## Honest limits shown in the app

- Decision support, not a diagnosis. T stage is a "consistent with" hint from one slide.
- The map gives one answer per 112 µm square (10-15 cells across), so single cells are invisible to it.
- Immune results show where lymphocytes sit, not what they do; H&E can't tell T-cell types apart.
- Direction means where the tumor is invading in this slice, not where it will grow.
- Flags are places to check, not findings; some risk points and cut-offs are PLACEHOLDERS until a pathologist sets them.
- "Unsure here" covers 15-32% of a typical slide (top score below 0.9): honest, and shown in the Certainty layer.
