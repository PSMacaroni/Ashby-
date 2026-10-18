# The app (Tumor Front Analyzer)

A Streamlit app that runs on a laptop and shows slide results made by the team's pipeline. It never
runs CONCH or MUSK on slides itself; it only reads result files, so it keeps working after Delta
access ends.

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
`localhost` adds ~0.2 s to every request and pages took ~20 s to load). Only this computer can open
it. Server settings and the look live in `app/.streamlit/config.toml`. Starting the server takes
10-40 s on a slow laptop (loading Python libraries); after that pages load in about a second.

The first start makes 3 demo slides in `app/demo_data/` (about a minute). To remake them:
`python app/make_demo_data.py`.

## Screens

| Level | Screen | What you can do |
|---|---|---|
| 1 | **Slide Box** | A grid of slide cards (three per row). Hover a card for tumor % and flag count; click it to open. **⋯** opens its options: rename (and copy the details), export (PDF report, map picture, summary JSON, tiles CSV), duplicate, remove. Search and sort above the grid. |
| 2 | **Slide** | Left: the map. Color it by **Tissue**, **Certainty**, **Risk** points or **Zones** (core / front band / peritumoral). In Tissue mode the legend is clickable: pick a class to highlight it. Hover any square for a fixed-size info box; click it to open it in the panel. Scroll to zoom, drag to move, double-click (or the fit button) to reset. Right: the panel (below), with PDF and map-picture downloads at the bottom. |

Panel sections on Level 2:
- **Summary**: flags to check, then the findings (growth, depth, direction, what the front faces, fat,
  immune, necrosis; "Show everything" adds stroma, mucin, spread, outer edge, lymph nodes, tissue mix,
  caveats). With Lumen set up, a short plain-language paragraph comes first.
- **Spot**: the clicked square: picture, where it sits, top class and score, certainty and why, the
  second model's answer, zone and distance to the front, what to check by eye, all 8 scores.
- **Unsure**: the analysis's unsure areas, with pointers on the map.
- **Risk**: risk regions by level (filterable), with pointers, and the points table.
- **Flags**: perineural, venous, lymphatic, budding, etc.: strong spots get pointers; skipped flags say why.

The ← → arrows at the top work like a browser's back and forward.

## Where slides come from

Every folder with `summary.json` + `tiles.csv` in:
- `data/results/<slide_id>/`: real slides, copied down from Delta (format: `CLAUDE.md` → Data contract)
- `app/demo_data/<slide_id>/`: demo slides made from the team's fake slides (`src/fake_slide.py`)

The app reads what the team's analysis writes: `summary.json` (every section has a ready sentence in
`text`), `tile_analysis.csv` (zones, risk points, unsure per square), `tiles.csv` (scores, `alt_label`,
`p_nerve`, `p_artery`), `thumbnail.png`, and `tiles/<row>_<col>.jpg` previews if saved. If a slide has
`tiles.csv` but no analysis yet, the app runs `src/analysis.py` in memory (nothing is written to the folder).

**Real slides: none exist yet.** The whole-slide pipeline (tile + embed + classify on Delta) is still
to build. Once a folder lands in `data/results/`, it shows up in the Slide Box with no app changes.

## Lumen summaries

Put these in the project's `.env` (never commit it, never paste keys in chat):
`LUMEN_API_KEY`, `LUMEN_BASE_URL`, `LUMEN_MODEL`. Lumen only gets the analysis sentences (no images,
no patient data) and must not add numbers, diagnoses or treatment advice; if its answer contains a
number that wasn't in the input, the app ignores it and shows the findings as computed (`lumen.py`).

## Files

| File | What it does |
|---|---|
| `streamlit_app.py` | Entry point: page setup, demo slides, background warm-up, picks the screen |
| `view_box.py` | Level 1, the Slide Box grid and the options window |
| `view_slide.py` | Level 2: map and panel as separate fragments (a click on one side doesn't redraw the other) |
| `tile_map.py` | Map picture, hover box, clickable points, pointers, legend |
| `slide_data.py` | Finds slides, loads them once into memory, remembers renames/copies/removed slides |
| `explain.py` | Picks and orders the analysis sentences; explains a single square |
| `lumen.py` | Plain-language summary from NCSA Lumen, with a number check |
| `export.py` | Map picture (PNG) and the one-page PDF report |
| `ui.py` | Page style, top bar, back/forward history, small HTML pieces |
| `common.py` | Folders, class names, colors (tissue colors come from `src/viz.py`) |
| `make_demo_data.py` | Builds the demo slides with the team's fake slides + analysis |
| `.streamlit/config.toml` | Theme (near-black, one blue accent) and server settings |

The app writes only `app/demo_data/` and `app/slide_box.json` (both git-ignored). Result files are
never changed.

## Design notes

- Dark, near-black background with one blue accent; system font; thin borders; generous spacing.
- Tissue colors are the team's (`src/viz.py`) so the app matches the analysis figures. A color check
  flags that muscle/mucus and stroma/muscle are hard to tell apart (especially with color blindness),
  so the hover box always names the class and the legend can highlight one class.
- Risk uses one blue scale (more points = darker); certainty colors only the doubtful squares.

## Honest limits shown in the app

- Decision support, not a diagnosis. T stage is a "consistent with" hint from one slide.
- The map gives one answer per 112 µm square (10-15 cells across), so single cells are invisible to it.
- Immune results show where lymphocytes sit, not what they do; H&E can't tell T-cell types apart.
- Direction means where the tumor is invading in this slice, not where it will grow.
- Flags are places to check, not findings; risk points include the team's PLACEHOLDER rows.

## Not built yet

- Text search in the open slide (needs `embeddings.npy` + the CONCH text encoder)
- Level 3 tile deep dive (stitched previews)
