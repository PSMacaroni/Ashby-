# The app (Colon Tumor Front Analyzer)

A Streamlit app that runs on a laptop and shows results made on Delta. It never runs CONCH or MUSK
on slides itself; it only reads small result files, so it keeps working after Delta access ends.

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

It opens at http://localhost:8501. The first load after starting takes 10-20 seconds.

The first launch makes 3 fake demo slides in `app/demo_data/` (takes about a minute).
To remake them: `python app/make_demo_data.py`.

## Screens

| Level | Screen | What you can do |
|---|---|---|
| 1 | **Slide Box** | A card per processed slide (thumbnail, name, case ID, date, risk flags). **Open** goes to Level 2. The **⋮** button opens the slide's menu: rename (and copy its details), duplicate, export (summary JSON, tile CSV), remove. Streamlit can't catch right-clicks, so this button replaces the right-click menu. |
| 2 | **Slide explorer** | Left: the slide with the tissue map on top. Turn the map on or off, color it by **tissue type**, **how sure the AI is**, or **risk level**. **Hover** any spot for a tooltip; **click** it for details. Scroll to zoom, drag to move. Right: four panels (below). |

Right-hand panels on Level 2:
- **Summary**: top risk flags and key findings; **Expand summary** shows everything (border, depth, direction, immune, necrosis, stroma, all 8 tissue classes, caveats). An arrow on the map shows which way the front faces *in this slice*.
- **Selected spot**: the clicked square: its picture, where it sits, its top class and score, *why* the AI thinks so, *how sure* it is and why, and what to check by eye when it's unsure. **Expand spot details** shows all 8 probabilities for the combined model, CONCH and MUSK.
- **Look here**: clumps of squares where the AI is unsure (top score under 60%, or CONCH and MUSK disagree), with numbered pointers on the map and a "Show on the map" button for each.
- **Risk regions**: regions grouped by risk level (1 Low to 4 Very high), filterable by level, each with the features nearby and a suggested check.

The ← → arrows at the top work like a browser's back and forward.

## Files

| File | What it does |
|---|---|
| `streamlit_app.py` | Entry point: page setup, picks the screen to show |
| `view_box.py` | Level 1, the Slide Box |
| `view_slide.py` | Level 2, the map and the four right-hand panels |
| `tile_map.py` | Map picture, hover text, clickable points, "unsure" spots |
| `explain.py` | Plain-language text built from the numbers (templates, no AI) |
| `slide_data.py` | Finds slides, reads result files, remembers renames/copies/removed slides |
| `ui.py` | Top bar, back/forward history, badges, page style |
| `common.py` | Folders, the 8 classes, colors, known model mix-ups, caveats |
| `make_demo_data.py` | Makes the fake DEMO slides |
| `assets/logo.svg` | The slide-box logo |

The app writes only two things: `app/demo_data/` (fake slides) and `app/slide_box.json` (your renames,
copies and removed slides). Both are git-ignored. Result files are never changed.

## Where slides come from

The app shows every folder that has a `summary.json` and a `tiles.csv`, from:
- `data/results/<slide_id>/`: real slides, copied down from Delta (`scp -r delta:~/Ashby-/data/results/<slide_id> data/results/`)
- `app/demo_data/<slide_id>/`: fake DEMO slides

## What the app expects from Delta

The base format is the data contract in `CLAUDE.md`. The app also reads these **extra** fields when
present, and still works (showing "not computed yet") when they're missing. **Pipeline owners: please
add these, and copy them into the contract in `CLAUDE.md`.**

`tiles.csv`, extra columns:

| Column | Meaning | Used for |
|---|---|---|
| `p_conch_<class>` ×8 | CONCH's own probabilities | "the models disagree" = unsure; side-by-side table |
| `p_musk_<class>` ×8 | MUSK's own probabilities | same |
| `risk_points` | points from risk features near this tile | risk coloring |
| `risk_level` | 0-4 (points capped at 4) | risk coloring, hover |
| `risk_flags` | short names of the features near it, `; `-separated | hover text |

`p_<class>`, `label` and `confidence` stay the **final** answer (e.g. the classifier on CONCH + MUSK).
Keep `label` as the raw argmax of `p_<class>` so the hover matches the numbers; smooth only for the analysis.

`summary.json`, extra fields:

```json
{
  "display_name": "Optional nicer name for the card",
  "case_id": "TCGA-AA-3844",
  "processed_at": "2026-10-18",
  "slide_mpp": 0.25,
  "invasion": {"direction": {"side": "bottom-right", "clock": 4, "vector_rc": [0.8, 0.6]}},
  "stroma": {"worst_field_stroma_fraction": 0.64, "worst_field_center_rc": [56, 83], "stroma_high": true},
  "satellites": [{"center_rc": [69, 95], "n_tiles": 8, "distance_mm": 0.6, "surrounded_by": "fat"}],
  "risk_regions": [{"id": 1, "level": 4, "points": 5, "n_tiles": 38,
                    "title": "Possible venous invasion candidate",
                    "reasons": ["Possible venous invasion candidate", "Tumor front touches fat"],
                    "suggested_actions": ["Look for a vein wall around this tumor island; consider an elastin stain."],
                    "center_rc": [73, 104], "bbox_rc": [70, 100, 76, 108]}],
  "risk_points_version": "v1"
}
```

Notes:
- `mpp` is read as the microns per pixel **of the tiles** (0.5), so one tile = `tile_size_px × mpp` µm.
  Where a tile sits on the thumbnail comes from its `x`, `y` and `thumbnail_downsample`.
- `*_rc` values are `[row, col]` in the tile grid.
- **Tile previews** `tiles/<row>_<col>.jpg`: please save one for **every** tissue tile (128 px is fine, about 100 MB per
  slide). The Selected spot panel shows it, and the planned Level 3 deep dive will stitch them into a zoomed view.
  Without them, real slides show "no preview".
- If a field isn't computed, leave it out or set it to `null`; don't write 0 for "unknown".

## Demo data

The demo slides are fake: a cartoon colon wall plus a tumor, and fake CONCH/MUSK probabilities with the
mix-ups we measured on NCT-CRC (CONCH: stroma→muscle, tumor→normal; MUSK: normal→mucus, muscle→stroma).
The summary numbers come from simple stand-ins for the analysis code, run on that fake map. They exist
only to build and test the app. Every demo slide carries a DEMO badge and banner.

The risk points table in `make_demo_data.py` (`RISK_FEATURES`) is a **placeholder** until the team's
version (features doc, section 7) is final.

## Honest limits shown in the app

- Decision support, not a diagnosis. T stage is a "consistent with" hint from one slide.
- The map gives one answer per square (112 µm with 224 px tiles, 10-15 cells across), so single cells are invisible to it.
- Immune results show where lymphocytes sit, not what they do; H&E can't tell T-cell types apart.
- Direction means where the tumor is invading in this slice, not where it will grow.
- Flags (venous invasion, budding, deposits) are places to check, not findings.

## Not built yet

- Export of the map picture and a one-page PDF report
- Text search in the open slide (CONCH text encoder on the laptop, with a list of precomputed queries as fallback)
- Lumen rewording of the summaries (template text is shown until then)
- Level 3 tile deep dive (stitched previews), after the core is done
