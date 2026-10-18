"""
Model results: what our models scored in testing, with the team's real figures (confusion charts,
real TCGA and Kather tiles, nerve/vessel examples). Numbers come from the team's reports
(reports/*/results.md, models/README.md, docs/*.md on the starc9-spider-classifiers branch).

Figures are read from the repo's reports/ and docs/img/ folders when they exist (always current);
otherwise from the copies in app/assets/results/ (see SOURCES.md there).
"""
import streamlit as st

import ui
from common import APP_DIR, ROOT
from common import THEME as T

# (big number, what it is, the comparison) - from reports/split_v1/results.md, models/README.md,
# reports/README.md (tils_v1) and docs/fixes_background_lymph.md
HEADLINES = [
    ("94.6%", "tissue model accuracy", "average of 4 test sets from other hospitals; CONCH with no training: 81.1%"),
    ("89.1%", "worst test set", "SPIDER colon slides (tumor 81%, normal mucosa 99%)"),
    ("99.7%", "nerves found", "nerve/vessel detector on SPIDER skin; vessels 97.5-98.7%"),
    ("0.83", "immune detector AUROC", "TCGA-TILs, 4,889 TCGA patches; the tissue model alone: 0.75"),
    ("99%", "empty glass caught", "background guard; no fat tiles removed"),
]
# accuracy per test set: (test set, tiles, zero-shot, trained tissue model)
ACCURACY = [
    ("CRC-VAL-HE-7K (Heidelberg)", "6,333", "86.8%", "95.5%"),
    ("STARC-9 VAL-LARGE (Stanford)", "48,000", "82.5%", "97.6%"),
    ("STARC-9 TCGA slides", "18,500", "78.7%", "96.4%"),
    ("SPIDER colon", "3,500", "76.5%", "89.1%"),
]
# (file name, path in the repo, title, caption)
CHARTS = [
    ("confusion_starc9_val_tcga_zero_shot.png", "reports/starc9_lr/confusion_starc9_val_tcga_zero_shot.png",
     "Before: CONCH with no training", "TCGA tiles (STARC-9 TCGA set). Rows = true class, columns = model answer."),
    ("confusion_tissue8_starc9_val_tcga.png", "reports/split_v1/confusion_tissue8_starc9_val_tcga.png",
     "After: our trained tissue model", "Same TCGA tiles. A small classifier on CONCH features (tissue8, NCT + STARC-9)."),
    ("confusion_tissue8_pooled.png", "reports/split_v1/confusion_tissue8_pooled.png",
     "All four test sets together", "Trained tissue model, pooled over the four test sets above."),
    ("nct_conch224_confusion.png", "docs/img/nct_conch224_confusion.png",
     "First look: CONCH, no training", "Our first test: 400 NCT-CRC tiles, 90.0% (docs/handoff_small_classifier.md)."),
]
SHEETS = [
    ("TCGA-CM-4746-01Z-00-DX1_new_kept.jpg", "reports/guard_check/TCGA-CM-4746-01Z-00-DX1_new_kept.jpg",
     "Real tiles from our TCGA slide CM-4746", "Edge tiles the looser tissue mask now keeps, with our model's label and "
     "confidence under each. Edge tiles are partly glass, so confidence is lower."),
    ("TCGA-AA-3875-01Z-00-DX1_mask_sat12.jpg", "reports/guard_check/TCGA-AA-3875-01Z-00-DX1_mask_sat12.jpg",
     "Tissue mask on our TCGA slide AA-3875", "Blue = background; green = pale tissue the looser mask (saturation > 12) "
     "adds: loose submucosa and edematous stroma the old mask missed."),
    ("patient08_examples.jpg", "reports/kather2016/patient08_examples.jpg",
     "Kather 2016: true label -> our answer", "Unseen, non-normalized tiles from another hospital. Mostly right; debris "
     "is the hard class."),
    ("lymphocyte_examples.jpg", "reports/kather2016/lymphocyte_examples.jpg",
     "Lymphocytes: missed (top) and found (bottom)", "Scattered lymphocytes in stroma or tumor get the tissue label; "
     "that's why the separate immune detector exists."),
    ("nct_top_nerve.jpg", "reports/provided_v1/nct_top_nerve.jpg",
     "Tiles the nerve detector is surest about", "NCT-CRC tiles (labeled muscle or stroma there) scored as nerve: wavy "
     "nerve fibers, which the 8 classes have no name for."),
    ("most_vein_like_colon.jpg", "reports/conch_structures_v1/most_vein_like_colon.jpg",
     "Most vein-like vessels (colon)", "Thin-walled vessels; the artery-vs-vein score is experimental (no labels to check)."),
]


def figure_path(name, repo_path):
    live = ROOT / repo_path
    return live if live.exists() else APP_DIR / "assets" / "results" / name


def show():
    st.html(f"""<style>
      .hl {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 6px 0 4px; }}
      .hl div {{ background: {T['surface']}; border: 1px solid {T['border']}; border-radius: 12px; padding: 14px 16px; }}
      .hl b {{ display: block; font-size: 1.6rem; font-weight: 650; color: {T['text']}; letter-spacing: -0.02em; }}
      .hl span {{ display: block; color: {T['text2']}; font-size: 0.86rem; font-weight: 600; margin-top: 2px; }}
      .hl small {{ display: block; color: {T['muted']}; font-size: 0.78rem; margin-top: 6px; line-height: 1.4; }}
      table.acc {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
      table.acc th, table.acc td {{ text-align: left; padding: 8px 4px; border-bottom: 1px solid {T['border']}; }}
      table.acc th {{ color: {T['muted']}; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em; }}
      table.acc td.up {{ color: {T['accent']}; font-weight: 650; }}
    </style>""")
    st.markdown("#### Model results")
    ui.note("What our models scored on tiles from hospitals and patients they never saw in training. "
            "Numbers and figures are the team's own test outputs (reports/ and models/README.md).")
    cards = "".join(f"<div><b>{ui.esc(v)}</b><span>{ui.esc(k)}</span><small>{ui.esc(c)}</small></div>"
                    for v, k, c in HEADLINES)
    st.html(f'<div class="hl">{cards}</div>')

    ui.section("Accuracy per test set")
    rows = "".join(f"<tr><td>{ui.esc(a)}</td><td>{ui.esc(n)}</td><td>{ui.esc(z)}</td><td class='up'>{ui.esc(t)}</td></tr>"
                   for a, n, z, t in ACCURACY)
    st.html(f"<table class='acc'><tr><th>Test set</th><th>Tiles</th><th>CONCH, no training</th>"
            f"<th>Our tissue model</th></tr>{rows}</table>")
    ui.note("Weak spots to say out loud: muscle vs stroma on Heidelberg data (muscle 72.8% on CRC-VAL-HE-7K), and "
            "SPIDER colon tumor 81.4%. Nerves were only tested on skin (no colon nerve labels exist).")

    ui.section("Before and after training")
    gallery(CHARTS)
    ui.section("Real tiles and slides from our tests")
    gallery(SHEETS)


def gallery(items, columns=2):
    for start in range(0, len(items), columns):
        for col, (name, repo_path, title, caption) in zip(st.columns(columns, gap="medium"), items[start:start + columns]):
            path = figure_path(name, repo_path)
            with col.container(border=True):
                st.markdown(f"**{title}**")
                if path.exists():
                    st.image(str(path), width="stretch")
                else:
                    ui.note(f"Figure not found ({repo_path}).")
                ui.note(f"{caption} Source: {repo_path}")
